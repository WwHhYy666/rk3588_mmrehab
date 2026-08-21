(() => {
  "use strict";

  if (!document.body || document.body.dataset.page !== "train") return;

  const ACTIVE_TRAINING_STATUSES = new Set([
    "running",
    "paused",
    "resting",
    "awaiting_orientation",
    "awaiting_return",
    "awaiting_care_response",
    "awaiting_action_audio",
    "awaiting_rep_feedback",
  ]);
  const FAST_POLL_MS = 400;
  const IDLE_POLL_MS = 1500;
  const TRACK_SECONDS = 180;

  let latestStatus = null;
  let trackedJobId = "";
  let baselineCompletedJobId = "";
  let submittedAfter = 0;
  let trackingUntil = 0;
  let submissionPendingUntil = 0;
  let requestGeneration = 0;
  let timerId = null;
  let lastRenderedJobId = "";
  let lastRenderedUpdatedAt = 0;

  function jobId(job) {
    return String((job && job.job_id) || "");
  }

  function jobTimestamp(job, field = "created_at") {
    const value = Number(job && job[field]);
    return Number.isFinite(value) ? value : 0;
  }

  function voicePayload(status) {
    return (status && status.voice) || {};
  }

  function llmJobs(status) {
    return voicePayload(status).llm_jobs || {};
  }

  function findJob(status, id) {
    if (!id) return null;
    const jobs = llmJobs(status);
    const candidates = [jobs.current_job, jobs.last_completed_job].concat(jobs.recent_jobs || []);
    return candidates.find((job) => jobId(job) === id) || null;
  }

  function providerLabel(provider) {
    if (provider === "local_qwen_rkllm") return "本地千问";
    if (provider === "glm4v_api") return "智谱GLM";
    if (provider === "echo") return "本地规则";
    return String(provider || "");
  }

  function jobSummary(job) {
    const actionName = String((job && (job.action_name || job.actionName)) || window.__SELECTED_VOICE_ACTION_NAME__ || "");
    const action = actionName ? "问答动作：" + actionName + "\n" : "";
    const provider = job && job.active_provider ? "上次回答：" + providerLabel(job.active_provider) + "\n" : "";
    const answer = String((job && (job.answer || job.error)) || "暂无内容。");
    return action + provider + answer;
  }

  function beginTracking() {
    requestGeneration += 1;
    const completed = llmJobs(latestStatus).last_completed_job;
    baselineCompletedJobId = jobId(completed);
    submittedAfter = Date.now() / 1000;
    trackingUntil = Date.now() + TRACK_SECONDS * 1000;
    submissionPendingUntil = Date.now() + 8000;
    trackedJobId = "";
    const answer = document.getElementById("voice-answer");
    if (answer && !answer.textContent.trim()) answer.textContent = "正在等待回答...";
    schedulePoll(0);
  }

  function selectRelevantJob(status) {
    if (trackedJobId) return findJob(status, trackedJobId);

    const jobs = llmJobs(status);
    const recent = Array.isArray(jobs.recent_jobs) ? Array.from(jobs.recent_jobs).reverse() : [];
    const candidates = [jobs.current_job, jobs.last_completed_job].concat(recent).filter(Boolean);
    const selected = candidates.find((job) => {
      const id = jobId(job);
      if (!id || id === baselineCompletedJobId) return false;
      return jobTimestamp(job) >= submittedAfter - 1.0;
    });
    if (selected) trackedJobId = jobId(selected);
    return selected || null;
  }

  function renderCompletedJob(job) {
    if (!job || jobId(job) !== trackedJobId) return false;
    if (!["done", "failed", "blocked_training"].includes(String(job.status || ""))) return false;
    if (!job.answer && !job.error) return false;
    const updatedAt = jobTimestamp(job, "updated_at");
    if (lastRenderedJobId === trackedJobId && updatedAt && updatedAt < lastRenderedUpdatedAt) return false;
    const answer = document.getElementById("voice-answer");
    if (!answer) return false;
    answer.textContent = jobSummary(job);
    lastRenderedJobId = trackedJobId;
    lastRenderedUpdatedAt = updatedAt;
    return true;
  }

  function reportReady() {
    const badge = document.getElementById("voice-current-action");
    if (!badge) return false;
    return !badge.classList.contains("is-missing") && !String(badge.textContent || "").includes("暂无报告");
  }

  function applyAuthoritativeControls(status) {
    const voice = voicePayload(status);
    const current = llmJobs(status).current_job || {};
    const tracked = findJob(status, trackedJobId) || {};
    const assistantBusy = Boolean((voice.assistant_tts || {}).busy);
    const jobBusy = ["queued", "running"].includes(String(current.status || ""));
    const trackedTtsPending = Boolean(tracked.tts && tracked.tts.pending);
    const submissionPending = Date.now() < submissionPendingUntil && !trackedJobId;
    const trainingBusy = ACTIVE_TRAINING_STATUSES.has(String(voice.training_status || ""));
    const allowed = Boolean(voice.qa_allowed) && !assistantBusy && !jobBusy && !trackedTtsPending && !submissionPending && !trainingBusy;
    const askButton = document.getElementById("voice-ask-btn");
    const listenButton = document.getElementById("voice-listen-btn");
    const input = document.getElementById("voice-question");
    const questionReady = Boolean(input && input.value.trim());

    if (askButton) askButton.disabled = !allowed || !questionReady || !reportReady();
    if (listenButton) {
      const stoppingActiveCapture = String(listenButton.textContent || "").includes("结束监听");
      const processingCapture = String(listenButton.textContent || "").includes("正在处理");
      listenButton.disabled = processingCapture ? true : (stoppingActiveCapture ? false : !allowed);
    }
  }

  function consumeStatus(status) {
    latestStatus = status || latestStatus;
    if (!latestStatus) return null;
    const job = selectRelevantJob(latestStatus);
    renderCompletedJob(job);
    applyAuthoritativeControls(latestStatus);
    return job;
  }

  function qaDockVisible() {
    const panel = document.querySelector('[data-dock-panel="qa"] .dock-surface');
    return Boolean(panel && panel.getAttribute("aria-hidden") === "false");
  }

  function activePollingNeeded(status) {
    const voice = voicePayload(status);
    const current = llmJobs(status).current_job || {};
    return (
      Date.now() < trackingUntil
      || qaDockVisible()
      || Boolean((voice.assistant_tts || {}).busy)
      || ["queued", "running"].includes(String(current.status || ""))
    );
  }

  function schedulePoll(delay) {
    if (timerId !== null) window.clearTimeout(timerId);
    timerId = window.setTimeout(pollStatus, Math.max(0, delay));
  }

  async function pollStatus() {
    const generation = requestGeneration;
    try {
      const response = await fetch("/api/voice/status", { cache: "no-store" });
      if (!response.ok) throw new Error("voice status HTTP " + response.status);
      const status = await response.json();
      if (generation === requestGeneration) consumeStatus(status);
    } catch (_) {
      // Keep polling the authoritative snapshot after transient Qwen/network errors.
    } finally {
      schedulePoll(activePollingNeeded(latestStatus) ? FAST_POLL_MS : IDLE_POLL_MS);
    }
  }

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target.closest("#voice-ask-btn") : null;
    if (target && !target.disabled) beginTracking();
  }, true);
  document.addEventListener("input", (event) => {
    if (event.target && event.target.id === "voice-question" && latestStatus) {
      applyAuthoritativeControls(latestStatus);
    }
  });

  window.__REHAB_QA_CONTROLLER__ = {
    beginTracking,
    consumeStatus,
    pollStatus,
    state: () => ({ trackedJobId, baselineCompletedJobId, submittedAfter, submissionPendingUntil, lastRenderedJobId }),
  };

  if (!window.__REHAB_QA_CONTROLLER_DISABLE_AUTOSTART__) schedulePoll(0);
})();
