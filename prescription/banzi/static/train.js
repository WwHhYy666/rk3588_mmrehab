(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");

  const reportActionIds = ["sit_to_stand", "standing_hamstring_curl", "seated_knee_raise"];
  const activeStatuses = new Set([
    "running",
    "paused",
    "resting",
    "awaiting_orientation",
    "awaiting_return",
    "awaiting_care_response",
  ]);
  const trainingBusyStatuses = new Set([
    "running",
    "paused",
    "awaiting_orientation",
    "awaiting_return",
    "awaiting_care_response",
  ]);

  let lastStatus = {};
  let lastSystem = {};
  let voiceJobId = "";
  let voiceQuestionDraft = "";
  let voiceComposing = false;
  let dockState = { hovered: "", pinned: "" };
  let selectedReportActionId = "sit_to_stand";
  let reportActionTouched = false;
  let lastReports = { latest_reports_by_action: {}, recent_reports: [] };
  let lastLLMStatus = { llm: {}, capabilities: {} };
  let lastVoiceStatus = { voice: {} };
  let aiRefreshInFlight = false;
  let systemRefreshInFlight = false;
  let lastAIRefreshAt = 0;
  let lastSystemRefreshAt = 0;

  app.innerHTML = `
    <main class="shell train-shell">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">Patient Training Cockpit</div>
          <h1>患者实时训练</h1>
          <p>摄像头骨架预览、三动作训练、实时计数和纠错优先；AI 建议训练后异步查看。</p>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">首页</a>
          <a class="nav-link" href="/doctor">医生录入</a>
          <a class="nav-link active" href="/train">患者训练</a>
        </nav>
      </header>

      <section class="train-workspace">
        <div class="train-main">
          <section class="train-grid">
            <section class="panel train-preview-panel">
              <div class="panel-header">
                <h3>摄像头预览</h3>
                <span id="train-vision-pill" class="pill info">等待连接</span>
              </div>
              <div class="preview-box">
                <img id="train-preview" src="/assets/placeholder.svg" alt="training preview">
                <div class="preview-overlay">
                  <div class="summary-text" id="train-prompt">等待开始训练</div>
                </div>
              </div>
              <div class="pills train-status-row" id="live-grid"></div>
            </section>
          </section>

          <section class="panel training-hud">
            <div class="section-head">
              <span class="pill info" id="training-status-pill">idle</span>
              <span class="meta" id="playlist-state">single action</span>
            </div>
            <div class="training-hud-grid">
              <section class="training-hero">
                <div class="action-title" id="action-title">准备训练</div>
                <div class="rep-display">
                  <div class="rep-number" id="rep-number">0</div>
                  <div>
                    <div class="eyebrow">REPS</div>
                    <div class="summary-text" id="rep-target">/ 0</div>
                  </div>
                </div>
                <div class="progress-track"><div class="progress-fill" id="rep-progress"></div></div>
              </section>

              <section class="live-metric-panel">
                <div>
                  <div class="eyebrow">实时指标</div>
                  <div class="live-metric-main" id="live-metric-current">-</div>
                </div>
                <div class="live-metric-grid">
                  <article><strong>目标</strong><span id="live-metric-target">-</span></article>
                  <article><strong>保持</strong><span id="live-metric-tut">-</span></article>
                  <article><strong>状态</strong><span id="live-metric-state">-</span></article>
                </div>
              </section>
            </div>

            <div class="field-grid">
              <label>患者编号<input id="patient-id" value="patient_001"></label>
              <label>目标次数<input id="target-reps" type="number" min="1" max="50" value="3"></label>
              <label>侧别模式
                <select id="side-mode">
                  <option value="auto">auto</option>
                  <option value="left">left</option>
                  <option value="right">right</option>
                </select>
              </label>
              <label>动作
                <select id="action-id">
                  <option value="sit_to_stand">坐站训练</option>
                  <option value="standing_hamstring_curl">站姿屈膝后勾腿</option>
                  <option value="seated_knee_raise">坐姿抬膝</option>
                  <option value="seated_knee_extension">坐姿伸膝</option>
                  <option value="knee_flexion">屈膝</option>
                </select>
              </label>
            </div>

            <div class="button-row">
              <button id="playlist-btn">开始完整训练</button>
              <button class="secondary" id="single-btn">开始单动作</button>
              <button class="secondary" id="pause-btn">暂停 / 继续</button>
              <button class="warn" id="stop-btn">结束训练</button>
            </div>
            <div class="message-box" id="train-message">准备就绪，等待开始训练。</div>
          </section>

          <section class="panel">
            <div class="panel-header">
              <h3>训练编排</h3>
              <span class="meta">rehab_demo_plan.yaml</span>
            </div>
            <div class="timeline" id="timeline"></div>
          </section>
        </div>

        <aside class="dock-stack" aria-label="辅助侧栏">
          ${dockMarkup("ai", "AI训练图文建议", "报告 / Qwen")}
          ${dockMarkup("system", "设备运行状态", "CPU / NPU")}
        </aside>
      </section>
    </main>
    <div class="modal-shell hidden" id="care-dialog" role="dialog" aria-modal="true" aria-labelledby="care-dialog-title">
      <section class="modal-card care-card">
        <div class="eyebrow">Care Check</div>
        <h3 id="care-dialog-title">温馨提示</h3>
        <p class="summary-text" id="care-dialog-message">累了吗？要休息吗？</p>
        <div class="care-actions">
          <button id="care-yes-btn" type="button">是</button>
          <button class="secondary" id="care-no-btn" type="button">否</button>
        </div>
      </section>
    </div>
  `;

  function dockMarkup(id, title, hint) {
    return `
      <section class="dock-panel" data-dock-panel="${id}">
        <button class="dock-tab" type="button" data-dock-tab="${id}">
          <span class="dock-tab-title">${title}</span>
          <span class="dock-tab-hint">${hint}</span>
        </button>
        <section class="dock-surface" aria-hidden="true">
          <div class="dock-surface-head">
            <div>
              <div class="eyebrow">${id === "ai" ? "AI Assistant" : "System Monitor"}</div>
              <h3>${title}</h3>
              <p class="summary-text" id="dock-summary-${id}">${id === "ai" ? "训练结束后再生成 GLM 图文建议；无网时用本地 Qwen 文本回答。" : "板端资源和服务状态。"}</p>
            </div>
            <button class="secondary dock-pin" type="button" data-dock-pin="${id}">固定</button>
          </div>
          <div class="dock-surface-body" id="dock-body-${id}"></div>
        </section>
      </section>
    `;
  }

  function payload() {
    return {
      patient_id: document.getElementById("patient-id").value.trim(),
      action_id: document.getElementById("action-id").value,
      side_mode: document.getElementById("side-mode").value,
      target_reps: Number(document.getElementById("target-reps").value || 3),
    };
  }

  function row(label, value) {
    return `<article class="badge-card"><strong>${label}</strong><span class="mono">${UI.safeText(value)}</span></article>`;
  }

  function setMessage(text, tone = "") {
    const node = document.getElementById("train-message");
    node.className = `message-box ${tone}`.trim();
    node.textContent = text;
  }

  function setVoiceMessage(text, tone = "") {
    const node = document.getElementById("voice-message");
    if (!node) return;
    node.className = `message-box ${tone}`.trim();
    node.textContent = text;
  }

  function renderCareDialog(training) {
    const dialog = document.getElementById("care-dialog");
    const title = document.getElementById("care-dialog-title");
    const message = document.getElementById("care-dialog-message");
    const yesBtn = document.getElementById("care-yes-btn");
    const noBtn = document.getElementById("care-no-btn");
    if (!dialog || !title || !message || !yesBtn || !noBtn) return;

    const care = training.care_dialog || {};
    const visible = training.status === "awaiting_care_response" && Boolean(care.visible);
    dialog.classList.toggle("hidden", !visible);
    if (!visible) return;

    title.textContent = UI.safeText(care.title, "温馨提示");
    message.textContent = UI.safeText(care.message, "累了吗？要休息吗？");
    yesBtn.textContent = UI.safeText(care.yes_label, "是");
    noBtn.textContent = UI.safeText(care.no_label, "否");
  }
  function currentVoiceJob(voice) {
    const jobs = voice.llm_jobs || {};
    return jobs.current_job || null;
  }

  function lastCompletedVoiceJob(voice) {
    const jobs = voice.llm_jobs || {};
    if (jobs.last_completed_job) return jobs.last_completed_job;
    const recent = Array.isArray(jobs.recent_jobs) ? jobs.recent_jobs : [];
    for (let index = recent.length - 1; index >= 0; index -= 1) {
      if (["done", "failed", "blocked_training"].includes(recent[index].status)) return recent[index];
    }
    return null;
  }

  function providerSummary(llm, voice) {
    const configured = llm.provider || "auto";
    const expected = llm.active_provider || configured;
    const currentJob = currentVoiceJob(voice || {});
    const lastJob = lastCompletedVoiceJob(voice || {});
    const lastProvider = lastJob?.active_provider || llm.last_active_provider || "暂无";
    const currentText = currentJob ? `${currentJob.status}:${currentJob.active_provider || expected}` : "无";
    const lastText = lastJob ? `${lastJob.status}:${lastProvider}` : lastProvider;
    const lastError = llm.fallback_reason || lastJob?.error || (voice.llm_jobs && voice.llm_jobs.last_error) || "-";
    const qwen = llm.rkllm_server_reachable ? "Qwen ready" : "Qwen off";
    return { configured, expected, lastProvider, currentText, lastText, lastError, qwen };
  }

  function providerLine(llm, voice) {
    const state = providerSummary(llm, voice);
    return `配置：${state.configured} / 预计：${state.expected} / 当前任务：${state.currentText} / 上次完成：${state.lastText} / ${state.qwen}`;
  }

  function renderDockState() {
    document.querySelectorAll(".dock-panel").forEach((panel) => {
      const id = panel.dataset.dockPanel;
      const active = dockState.hovered === id;
      const pinned = dockState.pinned === id;
      panel.classList.toggle("is-active", active);
      panel.classList.toggle("is-pinned", pinned);
      const surface = panel.querySelector(".dock-surface");
      const pin = panel.querySelector(".dock-pin");
      if (surface) surface.setAttribute("aria-hidden", active || pinned ? "false" : "true");
      if (pin) pin.textContent = pinned ? "取消固定" : "固定";
    });
  }

  function bindDockEvents() {
    document.querySelectorAll(".dock-panel").forEach((panel) => {
      const id = panel.dataset.dockPanel || "";
      panel.addEventListener("mouseenter", () => {
        dockState.hovered = id;
        renderDockState();
      });
      panel.addEventListener("mouseleave", () => {
        dockState.hovered = "";
        renderDockState();
      });
    });
    document.querySelectorAll("[data-dock-tab]").forEach((tab) => {
      tab.addEventListener("click", () => {
        const id = tab.getAttribute("data-dock-tab") || "";
        dockState.pinned = dockState.pinned === id ? "" : id;
        renderDockState();
        refreshDock(id, true);
      });
    });
    document.querySelectorAll("[data-dock-pin]").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const id = button.getAttribute("data-dock-pin") || "";
        dockState.pinned = dockState.pinned === id ? "" : id;
        renderDockState();
        refreshDock(id, true);
      });
    });
    renderDockState();
  }

  function actionReportEntries(status) {
    const grouped = lastReports.latest_reports_by_action || status.latest_reports_by_action || {};
    return reportActionIds.map((actionId) => {
      const context = UI.composeReportContext(grouped[actionId]);
      if (context && !context.actionId) context.actionId = actionId;
      if (context && !context.actionName) context.actionName = UI.actionNames[actionId] || actionId;
      return {
        actionId,
        actionName: UI.actionNames[actionId] || actionId,
        context,
        reportId: context ? UI.reportId(context) : `latest:${actionId}`,
      };
    });
  }

  function chooseActiveReportEntry(status, entries) {
    const trainingAction = status.training && status.training.action_id;
    if (!reportActionTouched && trainingAction && reportActionIds.includes(trainingAction)) {
      selectedReportActionId = trainingAction;
    }
    if (!selectedReportActionId || !reportActionIds.includes(selectedReportActionId)) {
      selectedReportActionId = (entries.find((entry) => entry.context) || entries[0] || {}).actionId || "";
    }
    return entries.find((entry) => entry.actionId === selectedReportActionId) || entries[0] || null;
  }

  function latestVoiceJob(voice) {
    return lastCompletedVoiceJob(voice) || currentVoiceJob(voice);
  }

  function voiceJobSummary(job) {
    const provider = job.active_provider ? `上次回答：${job.active_provider}\n` : "";
    const answer = job.answer || job.error || "暂无内容。";
    return `${provider}${answer}`;
  }

  function isTextEditing() {
    const active = document.activeElement;
    return voiceComposing || Boolean(active && (active.id === "voice-question" || active.matches("[data-ai-question]")));
  }

  function ensureAIDockShell() {
    const body = document.getElementById("dock-body-ai");
    if (!body || body.dataset.ready === "1") return;
    body.dataset.ready = "1";
    body.innerHTML = `
      <section class="dock-ai-shell">
        <div class="ai-console">
          <div class="ai-console-head">
            <div>
              <strong>小爱康复问答</strong>
              <div class="summary-text mono" id="voice-provider-line">配置：auto / 预计：- / 上次回答：暂无 / Qwen off</div>
            </div>
            <span class="pill info" id="voice-provider-pill">auto</span>
          </div>
          <div class="metric-matrix" id="voice-provider-matrix"></div>
          <label class="voice-question-label">文字提问
            <textarea id="voice-question" placeholder="例如：屈膝训练要注意什么？"></textarea>
          </label>
          <div class="voice-grid">
            <button class="secondary" id="voice-record-btn" type="button">麦克风到货后测试 ASR</button>
            <button id="voice-ask-btn" type="button">提交问答</button>
          </div>
          <div class="message-box" id="voice-message">空闲、组间休息或训练结束后可以提问；训练进行中会自动阻止。</div>
          <div class="message-box" id="voice-answer">暂无问答结果。</div>
        </div>
        <div id="dock-report-tabs"></div>
        <div id="dock-report-panel"></div>
      </section>
    `;
    bindVoiceEvents();
  }

  function renderAIDock(status) {
    const llm = (lastLLMStatus && lastLLMStatus.llm) || status.llm || {};
    window.__LLM_STATUS__ = llm;
    const voice = (lastVoiceStatus && lastVoiceStatus.voice) || status.voice || {};
    ensureAIDockShell();

    const summary = providerSummary(llm, voice);
    const providerClass = summary.expected === "echo" ? "warn" : "info";
    const qaAllowed = Boolean(voice.qa_allowed);
    const lastJob = latestVoiceJob(voice);
    const providerLineNode = document.getElementById("voice-provider-line");
    const providerPill = document.getElementById("voice-provider-pill");
    const providerMatrix = document.getElementById("voice-provider-matrix");
    const answerNode = document.getElementById("voice-answer");

    if (providerLineNode) providerLineNode.textContent = providerLine(llm, voice);
    if (providerPill) {
      providerPill.className = `pill ${providerClass}`;
      providerPill.textContent = summary.expected;
    }
    if (providerMatrix) {
      providerMatrix.innerHTML = [
        row("配置 provider", summary.configured),
        row("预计 provider", summary.expected),
        row("GLM", summary.glm),
        row("上次回答", summary.lastProvider),
        row("最后错误", summary.lastError),
        row("Qwen", summary.qwen),
      ].join("");
    }
    if (answerNode && lastJob && !voiceJobId) answerNode.textContent = voiceJobSummary(lastJob);

    updateVoiceControls(status);

    const entries = actionReportEntries(status);
    const activeEntry = chooseActiveReportEntry(status, entries);
    const activeContext = activeEntry && activeEntry.context;
    window.__SELECTED_VOICE_REPORT_ID__ = activeContext ? UI.reportId(activeContext) : (activeEntry ? activeEntry.reportId : "latest");
    window.__SELECTED_VOICE_ACTION_NAME__ = activeEntry ? activeEntry.actionName : "训练报告";
    const tabsNode = document.getElementById("dock-report-tabs");
    const reportPanel = document.getElementById("dock-report-panel");
    if (!isTextEditing()) {
      if (tabsNode) {
        tabsNode.innerHTML = `<div class="dock-mini-tabs">${entries.map((entry) => `
          <button class="dock-mini-tab ${entry.actionId === selectedReportActionId ? "is-active" : ""} ${entry.context ? "" : "is-missing"}" type="button" data-report-action="${entry.actionId}">
            <strong>${UI.safeText(entry.actionName)}</strong>
            <span>${entry.context ? UI.safeText(entry.context.reportFile || "") : "暂无该动作报告"}</span>
          </button>
        `).join("")}</div>`;
      }
      if (reportPanel) {
        reportPanel.innerHTML = activeContext
          ? UI.reportCardHtml(activeContext)
          : `<div class="dock-empty">${UI.safeText(activeEntry && activeEntry.actionName, "这个动作")}还没有患者训练报告。完成这个动作的一次训练后，小爱会按该动作最近报告回答。</div>`;
      }
    }
  }

  function renderSystemDock(system, status) {
    const llm = (lastLLMStatus && lastLLMStatus.llm) || status.llm || {};
    const voice = (lastVoiceStatus && lastVoiceStatus.voice) || status.voice || {};
    document.getElementById("dock-body-system").innerHTML = `
      <section class="dock-system-shell">
        <div class="metric-matrix">${UI.renderSystemStats(system || {}) || row("设备状态", "等待 /api/system/status")}</div>
        <div class="metric-matrix">
          ${row("配置 provider", llm.provider || "auto")}
          ${row("预计 provider", llm.active_provider || llm.provider || "-")}
          ${row("GLM", providerSummary(llm, voice).glm)}
          ${row("当前任务", providerSummary(llm, voice).currentText)}
          ${row("上次完成", providerSummary(llm, voice).lastText)}
          ${row("最后错误", providerSummary(llm, voice).lastError)}
          ${row("Qwen proxy", llm.rkllm_server_reachable ? "reachable" : "unreachable")}
          ${row("Voice QA", voice.qa_allowed ? "allowed" : "blocked")}
          ${row("LLM queue", voice.llm_jobs ? voice.llm_jobs.queue_size : "-")}
        </div>
      </section>
    `;
  }

  function updateVoiceControls(status) {
    const training = status.training || {};
    const voice = (lastVoiceStatus && lastVoiceStatus.voice) || status.voice || {};
    const llm = (lastLLMStatus && lastLLMStatus.llm) || status.llm || {};
    const askBtn = document.getElementById("voice-ask-btn");
    const recordBtn = document.getElementById("voice-record-btn");
    const input = document.getElementById("voice-question");
    if (!askBtn || !recordBtn || !input) return;

    if (document.activeElement !== input && !voiceComposing && input.value !== voiceQuestionDraft) {
      input.value = voiceQuestionDraft;
    }
    const trainingBusy = trainingBusyStatuses.has(training.status);
    const activeProvider = String(llm.active_provider || llm.provider || "").toLowerCase();
    const providerReady = Boolean(activeProvider) && activeProvider !== "echo";
    const allowed = Boolean(voice.qa_allowed) && !trainingBusy && providerReady;
    const busy = Boolean(voiceJobId);
    const question = input.value.trim();
    askBtn.disabled = !allowed || busy || !question;
    recordBtn.disabled = true;
    recordBtn.textContent = "麦克风到货后测试 ASR";
    const summaryNode = document.getElementById("dock-summary-ai");
    if (summaryNode) {
      const reason = trainingBusy ? `当前状态 ${training.status || "training"}，问答已阻止。` : (providerReady ? "现在可以文本问答。" : "GLM/Qwen 都未就绪，先检查 Key 或 Qwen 服务。");
      summaryNode.textContent = `${providerLine(llm, voice)}。${reason}`;
    }
    const message = document.getElementById("voice-message");
    if (message && !voiceJobId && !message.classList.contains("bad")) {
      message.className = `message-box ${allowed ? "" : "warn"}`.trim();
      const grouped = lastReports.latest_reports_by_action || {};
      const hasReport = Boolean((window.__SELECTED_VOICE_REPORT_ID__ && !String(window.__SELECTED_VOICE_REPORT_ID__).startsWith("latest:")) || grouped[selectedReportActionId]);
      message.textContent = !hasReport ? `${window.__SELECTED_VOICE_ACTION_NAME__ || "当前动作"}还没有报告，先完成该动作训练。` : (allowed ? "空闲、组间休息或训练结束后可以提问；训练进行中会自动阻止。" : (trainingBusy ? "训练进行中暂不回答，避免抢训练和播报资源。" : "GLM/Qwen 未就绪，不能提交假回答。"));
    }
  }

  function bindVoiceEvents() {
    const recordBtn = document.getElementById("voice-record-btn");
    const askBtn = document.getElementById("voice-ask-btn");
    const input = document.getElementById("voice-question");
    if (recordBtn && recordBtn.dataset.bound !== "1") {
      recordBtn.dataset.bound = "1";
      recordBtn.addEventListener("click", () => {
        setVoiceMessage("麦克风还没到，ASR 先保留接口；现在请直接用文字框测试 Qwen。", "warn");
      });
    }
    if (askBtn && askBtn.dataset.bound !== "1") {
      askBtn.dataset.bound = "1";
      askBtn.addEventListener("click", submitVoiceQuestion);
    }
    if (input && input.dataset.bound !== "1") {
      input.dataset.bound = "1";
      input.addEventListener("compositionstart", () => { voiceComposing = true; });
      input.addEventListener("compositionend", () => {
        voiceComposing = false;
        voiceQuestionDraft = input.value;
        updateVoiceControls(lastStatus);
      });
      input.addEventListener("input", () => {
        voiceQuestionDraft = input.value;
        updateVoiceControls(lastStatus);
      });
    }
  }

  document.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-report-action]");
    if (!tab) return;
    selectedReportActionId = tab.getAttribute("data-report-action") || "";
    reportActionTouched = true;
    renderAIDock(lastStatus || {});
  });

  function render(status, system) {
    lastStatus = status;
    lastSystem = system || {};
    const training = status.training || {};
    const preview = document.getElementById("train-preview");
    const previewSource = UI.streamSource(status);
    if (preview.getAttribute("src") !== previewSource) preview.src = previewSource;

    const streamReady = Boolean(status.stream_ready);
    const visionPill = document.getElementById("train-vision-pill");
    visionPill.className = `pill ${streamReady ? "good" : "warn"}`;
    visionPill.textContent = streamReady ? "实时流已连接" : "视觉链路等待中";

    const actionName = training.current_action_name || UI.actionNames[training.action_id] || training.action_id || "准备训练";
    const completed = Number(training.completed_reps || 0);
    const target = Number(training.target_reps || 0);
    document.getElementById("action-title").textContent = actionName;
    document.getElementById("rep-number").textContent = completed;
    document.getElementById("rep-target").textContent = `/ ${target}`;
    document.getElementById("rep-progress").style.width = `${target > 0 ? Math.min(100, (completed / target) * 100) : 0}%`;
    document.getElementById("live-metric-current").textContent = UI.formatNumber(training.current_metric ?? training.current_angle, 1);
    document.getElementById("live-metric-target").textContent = Array.isArray(training.target_range)
      ? `${UI.formatNumber(training.target_range[0])}-${UI.formatNumber(training.target_range[1])}`
      : "-";
    document.getElementById("live-metric-tut").textContent = `${UI.formatNumber(training.tut_seconds, 1)} / ${UI.formatNumber(training.tut_target, 1)}s`;
    document.getElementById("live-metric-state").textContent = UI.safeText(training.current_state, "-");
    document.getElementById("training-status-pill").className = `pill ${training.status === "running" ? "good" : "warn"}`;
    document.getElementById("training-status-pill").textContent = UI.safeText(training.status, "idle");
    document.getElementById("playlist-state").textContent = training.playlist_mode
      ? `playlist ${Number(training.playlist_index || 0) + 1}/${training.playlist_total || 0}`
      : "single action";
    const promptText = streamReady
      ? UI.safeText(training.prompt, "等待开始训练")
      : UI.safeText(status.vision_boot_error || status.status, "视觉链路未就绪");
    const finishedPrompt = /全部训练完成|训练完成/.test(promptText);
    const finishedStatus = ["finished", "completed", "done"].includes(String(training.status || "").toLowerCase());
    const promptNode = document.getElementById("train-prompt");
    const previewBox = document.querySelector(".train-grid .preview-box");
    promptNode.textContent = promptText;
    promptNode.classList.toggle("is-finished", streamReady && (finishedPrompt || finishedStatus));
    if (previewBox) previewBox.classList.toggle("is-finished", streamReady && (finishedPrompt || finishedStatus));

    const quality = status.pose_quality || {};
    document.getElementById("live-grid").innerHTML = `
      <span class="pill info">${UI.safeText(actionName)}</span>
      <span class="pill info">rep ${completed}/${target}</span>
      <span class="pill ${quality.quality_ok ? "good" : "warn"}">${quality.quality_ok ? "keypoints OK" : "check keypoints"}</span>
    `;
    document.getElementById("timeline").innerHTML = renderTimeline(training.demo_plan, training);
    renderCareDialog(training);
    if (isDockVisible("ai")) renderAIDock(status);
    if (isDockVisible("system")) renderSystemDock(system, status);
    refreshVisibleDocks(false);

    const busy = activeStatuses.has(training.status);
    document.getElementById("single-btn").disabled = busy || !status.active_template;
    document.getElementById("playlist-btn").disabled = busy;
    document.getElementById("pause-btn").disabled = !["running", "paused"].includes(training.status);
    document.getElementById("stop-btn").disabled = !busy;
    renderDockState();
  }

  function isDockVisible(id) {
    return dockState.pinned === id || dockState.hovered === id;
  }

  function isTrainingBusy() {
    const training = lastStatus.training || {};
    return trainingBusyStatuses.has(training.status);
  }

  function refreshDock(id, force = false) {
    if (id === "system") refreshSystem(force);
    if (id === "ai") refreshAIReports(force);
  }

  function refreshVisibleDocks(force = false) {
    if (isDockVisible("system")) refreshSystem(force);
    if (isDockVisible("ai")) refreshAIReports(force);
  }

  async function refreshSystem(force = false) {
    const now = Date.now();
    if (!isDockVisible("system")) return;
    if (systemRefreshInFlight) return;
    if (!force && now - lastSystemRefreshAt < 3000) return;
    systemRefreshInFlight = true;
    try {
      const system = await UI.fetchJSON("/api/system/status");
      lastSystem = system || lastSystem;
      lastSystemRefreshAt = Date.now();
      if (isDockVisible("system")) renderSystemDock(lastSystem || {}, lastStatus || {});
    } catch (error) {
      const body = document.getElementById("dock-body-system");
      if (body && isDockVisible("system")) {
        body.innerHTML = `<div class="message-box warn">${UI.safeText(error.message || String(error))}</div>`;
      }
    } finally {
      systemRefreshInFlight = false;
    }
  }

  async function refreshAIReports(force = false) {
    const now = Date.now();
    const aiVisible = isDockVisible("ai");
    if (!aiVisible) return;
    if (isTrainingBusy()) {
      renderAIDock(lastStatus || {});
      return;
    }
    if (aiRefreshInFlight) return;
    if (!force && now - lastAIRefreshAt < 15000) return;
    aiRefreshInFlight = true;
    try {
      const [reports, llmStatus, voiceStatus] = await Promise.all([
        UI.fetchJSON("/api/reports/latest_by_action"),
        UI.fetchJSON("/api/llm/status"),
        UI.fetchJSON("/api/voice/status"),
      ]);
      lastReports = reports || lastReports;
      lastLLMStatus = llmStatus || lastLLMStatus;
      lastVoiceStatus = voiceStatus || lastVoiceStatus;
      lastAIRefreshAt = Date.now();
      if (aiVisible) renderAIDock(lastStatus || {});
    } catch (error) {
      const message = document.getElementById("voice-message");
      if (message && aiVisible) {
        message.className = "message-box warn";
        message.textContent = error.message || String(error);
      }
    } finally {
      aiRefreshInFlight = false;
    }
  }
  function renderTimeline(plan, training) {
    const actions = plan && Array.isArray(plan.actions) ? plan.actions : [];
    if (!actions.length) return `<div class="empty">暂无训练编排。</div>`;
    return actions.map((action, index) => {
      const active = training.playlist_mode && Number(training.playlist_index) === index && activeStatuses.has(training.status);
      return `<article class="timeline-step ${active ? "active" : ""}">
        <strong>${UI.safeText(action.action_name || UI.actionNames[action.action_id] || action.action_id)}</strong>
        <div class="summary-text">${UI.safeText(action.camera_prompt)}</div>
      </article>`;
    }).join("");
  }

  async function refresh() {
    try {
      const status = await UI.fetchJSON("/status");
      render(status, lastSystem || {});
    } catch (error) {
      setMessage(error.message || String(error), "bad");
      if (isDockVisible("ai")) renderAIDock(lastStatus || {});
      if (isDockVisible("system")) renderSystemDock(lastSystem || {}, lastStatus || {});
    }
  }

  async function submitCareResponse(needsRest) {
    const yesBtn = document.getElementById("care-yes-btn");
    const noBtn = document.getElementById("care-no-btn");
    if (yesBtn) yesBtn.disabled = true;
    if (noBtn) noBtn.disabled = true;
    try {
      await UI.postJSON("/api/realtime/care_response", { needs_rest: Boolean(needsRest) });
      setMessage(needsRest ? "好的，先休息一下。" : "好的，我们继续训练。", needsRest ? "warn" : "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      if (yesBtn) yesBtn.disabled = false;
      if (noBtn) noBtn.disabled = false;
      refresh();
    }
  }

  async function submitVoiceQuestion() {
    const input = document.getElementById("voice-question");
    const question = input ? input.value.trim() : voiceQuestionDraft.trim();
    if (!question) {
      setVoiceMessage("请先在文本框输入问题；麦克风到货后再测试录音识别。", "warn");
      return;
    }
    if (isTrainingBusy()) {
      setVoiceMessage("训练动作进行中暂不回答，避免抢摄像头、计数和播报资源。", "warn");
      return;
    }
    const llm = (lastLLMStatus && lastLLMStatus.llm) || {};
    const activeProvider = String(llm.active_provider || llm.provider || "").toLowerCase();
    if (!activeProvider || activeProvider === "echo") {
      setVoiceMessage("GLM/Qwen 都未就绪，不能提交假回答。请先检查 GLM Key 或 Qwen 服务。", "bad");
      return;
    }
    voiceQuestionDraft = question;
    voiceJobId = "pending";
    updateVoiceControls(lastStatus);
    setVoiceMessage("已提交到异步 LLM worker，正在等待回答...", "warn");
    try {
      const reportId = window.__SELECTED_VOICE_REPORT_ID__ || "latest";
      const result = await UI.postJSON("/api/voice/ask", { question, report_id: reportId, speak: true });
      if (result.status === "blocked_training") {
        voiceJobId = "";
        setVoiceMessage(result.error || result.answer || "训练中暂不回答。", "warn");
        updateVoiceControls(lastStatus);
        return;
      }
      voiceJobId = result.job_id;
      pollVoiceAnswer(result.job_id);
    } catch (error) {
      voiceJobId = "";
      setVoiceMessage(error.message || String(error), "bad");
      updateVoiceControls(lastStatus);
    }
  }

  async function pollVoiceAnswer(jobId) {
    try {
      const result = await UI.fetchJSON(`/api/voice/ask_result?job_id=${encodeURIComponent(jobId)}`);
      if (result.status === "queued" || result.status === "running") {
        window.setTimeout(() => pollVoiceAnswer(jobId), 800);
        return;
      }
      voiceJobId = "";
      if (result.status === "blocked_training") {
        setVoiceMessage(result.error || result.answer || "训练中暂不回答。", "warn");
      } else if (result.status === "done") {
        setVoiceMessage(`回答完成：${UI.safeText(result.active_provider, "-")}`, "good");
      } else {
        throw new Error(result.error || "LLM failed");
      }
      const answer = document.getElementById("voice-answer");
      if (answer) answer.textContent = voiceJobSummary(result);
      updateVoiceControls(lastStatus);
      refresh();
    } catch (error) {
      voiceJobId = "";
      setVoiceMessage(error.message || String(error), "bad");
      updateVoiceControls(lastStatus);
    }
  }

  document.getElementById("single-btn").addEventListener("click", async () => {
    try {
      await UI.postJSON("/api/realtime/start", payload());
      setMessage("单动作训练已开始。", "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  document.getElementById("playlist-btn").addEventListener("click", async () => {
    try {
      await UI.postJSON("/api/realtime/start_playlist", payload());
      setMessage("完整训练已开始。", "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  document.getElementById("pause-btn").addEventListener("click", async () => {
    try {
      await UI.postJSON("/api/realtime/pause", {});
      setMessage("训练状态已切换。", "warn");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  document.getElementById("stop-btn").addEventListener("click", async () => {
    try {
      await UI.postJSON("/api/realtime/stop", {});
      setMessage("训练已结束。", "warn");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  document.getElementById("care-yes-btn").addEventListener("click", () => submitCareResponse(true));
  document.getElementById("care-no-btn").addEventListener("click", () => submitCareResponse(false));

  bindDockEvents();
  refresh();
  setInterval(refresh, 1000);
})();





















