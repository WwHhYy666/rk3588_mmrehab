(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");

  const reportActionIds = ["sit_to_stand", "standing_hamstring_curl", "seated_knee_raise"];
  const actionVoiceAliases = [
    { actionId: "seated_knee_raise", words: ["坐姿抬膝", "抬膝", "髋屈", "屈髋", "提膝"] },
    { actionId: "standing_hamstring_curl", words: ["站姿屈膝后勾腿", "屈膝后勾", "后勾腿", "后勾", "后弯", "小腿后弯"] },
    { actionId: "sit_to_stand", words: ["坐站训练", "坐站", "站起", "坐下", "起立", "站立"] },
  ];

  function actionDisplayName(actionId) {
    return UI.actionNames[actionId] || actionId || "当前动作";
  }

  function inferActionIdFromQuestion(text) {
    const value = String(text || "").replace(/\s+/g, "");
    if (!value) return "";
    const match = actionVoiceAliases.find((item) => item.words.some((word) => value.includes(word)));
    return match ? match.actionId : "";
  }

  function selectedVoiceReportReady() {
    const grouped = lastReports.latest_reports_by_action || {};
    const reportId = String(window.__SELECTED_VOICE_REPORT_ID__ || "");
    return Boolean((reportId && !reportId.startsWith("latest:")) || grouped[selectedReportActionId]);
  }

  function refreshVoiceActionBadge(actionName, hasReport) {
    const node = document.getElementById("voice-current-action");
    if (!node) return;
    node.className = `voice-current-action ${hasReport ? "" : "is-missing"}`.trim();
    node.textContent = `当前问答动作：${actionName || "当前动作"}${hasReport ? "" : "（暂无报告）"}`;
  }

  function applyQuestionActionContext(question) {
    const inferredActionId = inferActionIdFromQuestion(question);
    if (inferredActionId && reportActionIds.includes(inferredActionId)) {
      selectedReportActionId = inferredActionId;
      reportActionTouched = true;
    }
    const { activeEntry, activeContext } = currentReportSelection(lastStatus || {});
    const actionId = (activeEntry && activeEntry.actionId) || selectedReportActionId || "";
    const actionName = (activeEntry && activeEntry.actionName) || actionDisplayName(actionId);
    const reportId = activeContext ? UI.reportId(activeContext) : (activeEntry ? activeEntry.reportId : (actionId ? `latest:${actionId}` : "latest"));
    const hasReport = Boolean(activeContext);
    refreshVoiceActionBadge(actionName, hasReport);
    return { actionId, actionName, reportId, hasReport, switched: Boolean(inferredActionId) };
  }
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
    "resting",
    "awaiting_orientation",
    "awaiting_return",
    "awaiting_care_response",
    "awaiting_action_audio",
    "awaiting_rep_feedback",
  ]);
  let lastStatus = {};
  let lastSystem = {};
  let voiceJobId = "";
  let voicePollToken = 0;
  let displayedVoiceAnswerJobId = "";
  let voiceQuestionDraft = "";
  let voiceComposing = false;
  let voiceListening = false;
  let voiceListenBusy = false;
  let voiceListenSessionId = "";
  let micStream = null;
  let micCaptureMode = "backend";
  let dockState = { hovered: "", pinned: "" };
  let selectedReportActionId = "sit_to_stand";
  let reportActionTouched = false;
  let lastReports = { latest_reports_by_action: {}, recent_reports: [] };
  let lastLLMStatus = { llm: {}, capabilities: {} };
  let lastVoiceStatus = { voice: {} };
  let aiRefreshInFlight = false;
  let systemRefreshInFlight = false;
  let completionRefreshInFlight = false;
  let completionAutoPinned = false;
  let lastAIRefreshAt = 0;
  let lastSystemRefreshAt = 0;
  let lastCompletionRefreshAt = 0;
  const pageParams = new URLSearchParams(window.location.search);
  const kioskMode = pageParams.get("kiosk") === "1";
  const displayMode = pageParams.get("display") === "1";
  document.body.classList.toggle("kiosk-mode", kioskMode);
  document.body.classList.toggle("display-mode", displayMode);

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
          <a class="nav-link active" href="${displayMode ? "/train?display=1" : "/train"}">患者训练</a>
          ${displayMode ? `<a class="nav-link display-exit" href="/">退出展示</a>` : ""}
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

              <section class="quality-score-panel">
                <div class="quality-score-head">
                  <div>
                    <div class="eyebrow">完成度后台</div>
                    <div class="quality-score-main" id="quality-score-main">未评分</div>
                  </div>
                  <span class="pill info" id="quality-score-backend">waiting</span>
                </div>
                <div class="quality-score-meta" id="quality-score-meta">每次动作结束后后台计算，训练结束后到右侧完成度栏查看。</div>
                <div class="quality-score-list" id="quality-score-history"></div>
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
          ${dockMarkup("completion", "动作完成度", "三组汇总")}
          ${dockMarkup("report", "图文报告", "骨架 / 卡片")}
          ${dockMarkup("qa", "康复问答", "Qwen")}
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
    const eyebrow = {
      completion: "Completion",
      report: "Visual Report",
      qa: "Rehab QA",
      system: "System Monitor",
    }[id] || "Panel";
    const summary = {
      completion: "完整训练结束后查看三组动作每次尝试的完成度。",
      report: "训练结束后查看骨架图、报告卡片和 AI 图文建议。",
      qa: "空闲、组间休息或训练结束后，可以咨询康复问题。",
      system: "板端前六项关键运行状态。",
    }[id] || "";
    return `
      <section class="dock-panel" data-dock-panel="${id}">
        <button class="dock-tab" type="button" data-dock-tab="${id}">
          <span class="dock-tab-title">${title}</span>
          <span class="dock-tab-hint">${hint}</span>
        </button>
        <section class="dock-surface" aria-hidden="true">
          <div class="dock-surface-head">
            <div>
              <div class="eyebrow">${eyebrow}</div>
              <h3>${title}</h3>
              <p class="summary-text" id="dock-summary-${id}">${summary}</p>
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
    const value = String(text || "").trim();
    node.hidden = !value;
    node.className = `message-box ${tone}`.trim();
    node.textContent = value;
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

  function qwenStatusText(llm) {
    if (llm.rkllm_server_reachable !== true) return "Qwen 端口不可达";
    const cached = llm.qwen_generate_cached ? ` / 缓存${UI.safeText(llm.qwen_generate_age_seconds, "-")}s` : "";
    const error = llm.qwen_generate_ok === false && llm.qwen_generate_error ? ` / ${UI.safeText(llm.qwen_generate_error).slice(0, 42)}` : "";
    if (llm.qwen_generate_ok === true) return `Qwen 端口可达 / 生成可用${cached}`;
    if (llm.qwen_generate_ok === false) return `Qwen 端口可达 / 生成失败${cached}${error}`;
    return "Qwen 端口可达 / 未测生成";
  }

  function micStatusText(voice) {
    const mic = (voice && voice.mic) || {};
    const asr = (voice && voice.asr) || {};
    const asrMissing = asr.model_available === false
      ? ` / ASR模型缺失：${Array.isArray(asr.missing_files) ? asr.missing_files.join("、") : "model/tokens"}`
      : "";
    const asrImport = asr.sherpa_available === false
      ? ` / sherpa不可用：${UI.safeText(asr.sherpa_import_error || "import failed").slice(0, 36)}`
      : "";
    const asrRuntime = asr.last_error
      ? ` / ASR错误：${UI.safeText(asr.last_error).slice(0, 42)}`
      : (asr.recognizer_api ? ` / ASR:${UI.safeText(asr.recognizer_api)}` : "");
    const asrHint = `${asrMissing}${asrImport}${asrRuntime}`;
    if (mic.usb_audio_capture_detected) return `板端 ALSA：USB 麦克风可见${asrHint}`;
    if (Array.isArray(mic.capture_devices) && mic.capture_devices.length) return `板端 ALSA：仅发现板载录音${asrHint}`;
    if (mic.arecord_available) return `板端 ALSA：未发现录音设备${asrHint}`;
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) return `板端 ALSA 优先；请检查 arecord 录音设备${asrHint}`;
    return `未发现 ALSA/浏览器录音能力${asrHint}`;
  }

  function providerSummary(llm, voice) {
    const configured = llm.provider || "auto";
    const expected = llm.active_provider || configured;
    const currentJob = currentVoiceJob(voice || {});
    const lastJob = lastCompletedVoiceJob(voice || {});
    const lastProvider = lastJob?.active_provider || llm.last_active_provider || "暂无";
    const currentText = currentJob ? `${currentJob.status}:${currentJob.active_provider || expected}` : "无";
    const lastText = lastJob ? `${lastJob.status}:${lastProvider}` : lastProvider;
    const lastError = lastJob?.error || llm.last_error || llm.fallback_reason || (voice.llm_jobs && voice.llm_jobs.last_error) || "-";
    const qwen = qwenStatusText(llm);
    const glm = llm.glm_endpoint_reachable === true ? "GLM ready" : (llm.api_key_configured ? "GLM checking" : "GLM key off");
    const mic = micStatusText(voice || {});
    return { configured, expected, lastProvider, currentText, lastText, lastError, qwen, glm, mic };
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

  function voiceProviderLabel(provider) {
    if (provider === "local_qwen_rkllm") return "本地千问";
    if (provider === "glm4v_api") return "智谱GLM";
    if (provider === "echo") return "本地规则";
    return provider || "";
  }

  function voiceJobSummary(job) {
    const actionName = job.action_name || job.actionName || window.__SELECTED_VOICE_ACTION_NAME__ || "";
    const action = actionName ? `问答动作：${actionName}\n` : "";
    const provider = job.active_provider ? `上次回答：${voiceProviderLabel(job.active_provider)}\n` : "";
    const answer = job.answer || job.error || "暂无内容。";
    return `${action}${provider}${answer}`;
  }

  function isTextEditing() {
    const active = document.activeElement;
    return voiceComposing || Boolean(active && (active.id === "voice-question" || active.matches("[data-ai-question]")));
  }

  function ensureReportDockShell() {
    const body = document.getElementById("dock-body-report");
    if (!body || body.dataset.ready === "1") return;
    body.dataset.ready = "1";
    body.innerHTML = `
      <section class="dock-report-shell visual-report-shell">
        <div id="dock-report-tabs"></div>
        <div id="dock-report-panel"></div>
      </section>
    `;
  }

  function ensureQADockShell() {
    const body = document.getElementById("dock-body-qa");
    if (!body || body.dataset.ready === "1") return;
    body.dataset.ready = "1";
    body.innerHTML = `
      <section class="dock-ai-shell qa-dock-shell">
        <div class="ai-console qa-console">
          <div class="ai-console-head">
            <div>
              <strong>小爱康复问答</strong>
              <div class="summary-text" id="voice-provider-line">训练结束后可以围绕最近报告提问。</div>
            </div>
            <span class="pill info" id="voice-provider-pill">auto</span>
          </div>
          <div class="provider-switch" id="llm-provider-switch" aria-label="LLM provider">
            <button class="secondary" type="button" data-provider-choice="auto">Auto</button>
            <button class="secondary" type="button" data-provider-choice="glm4v_api">GLM</button>
            <button class="secondary" type="button" data-provider-choice="local_qwen_rkllm">Qwen</button>
          </div>
          <div class="voice-current-action" id="voice-current-action">当前问答动作：坐站训练</div>
          <label class="voice-question-label">文字提问
            <textarea id="voice-question" placeholder="例如：坐姿抬膝标准是什么？"></textarea>
          </label>
          <div class="voice-grid">
            <button class="secondary" id="voice-listen-btn" type="button">唤醒监听</button>
            <button id="voice-ask-btn" type="button">提交问答</button>
          </div>
          <div class="message-box" id="voice-message">空闲、组间休息或训练结束后可以提问；训练进行中会自动阻止。</div>
          <div class="message-box qa-answer" id="voice-answer">暂无问答结果。</div>
        </div>
      </section>
    `;
    bindVoiceEvents();
  }

  function currentReportSelection(status) {
    const entries = actionReportEntries(status);
    const activeEntry = chooseActiveReportEntry(status, entries);
    const activeContext = activeEntry && activeEntry.context;
    window.__SELECTED_VOICE_REPORT_ID__ = activeContext ? UI.reportId(activeContext) : (activeEntry ? activeEntry.reportId : "latest");
    window.__SELECTED_VOICE_ACTION_ID__ = activeEntry ? activeEntry.actionId : "";
    window.__SELECTED_VOICE_ACTION_NAME__ = activeEntry ? activeEntry.actionName : "训练报告";
    window.__SELECTED_VOICE_HAS_REPORT__ = Boolean(activeContext);
    refreshVoiceActionBadge(window.__SELECTED_VOICE_ACTION_NAME__, Boolean(activeContext));
    return { entries, activeEntry, activeContext };
  }

  function renderReportDock(status) {
    const llm = (lastLLMStatus && lastLLMStatus.llm) || status.llm || {};
    window.__LLM_STATUS__ = llm;
    ensureReportDockShell();
    const { entries, activeEntry, activeContext } = currentReportSelection(status);
    const tabsNode = document.getElementById("dock-report-tabs");
    const reportPanel = document.getElementById("dock-report-panel");
    if (tabsNode) {
      tabsNode.innerHTML = `<div class="dock-mini-tabs">${entries.map((entry) => `
        <button class="dock-mini-tab ${entry.actionId === selectedReportActionId ? "is-active" : ""} ${entry.context ? "" : "is-missing"}" type="button" data-report-action="${entry.actionId}">
          <strong>${UI.safeText(entry.actionName)}</strong>
          <span>${entry.context ? "最近报告已生成" : "暂无该动作报告"}</span>
        </button>
      `).join("")}</div>`;
    }
    if (reportPanel) {
      reportPanel.innerHTML = activeContext
        ? (UI.visualReportCardHtml ? UI.visualReportCardHtml(activeContext) : UI.reportCardHtml(activeContext))
        : `<div class="dock-empty">${UI.safeText(activeEntry && activeEntry.actionName, "这个动作")}还没有患者训练报告。完成这个动作的一次训练后，这里会显示骨架图和图文报告。</div>`;
    }
  }

  function renderQADock(status) {
    const llm = (lastLLMStatus && lastLLMStatus.llm) || status.llm || {};
    window.__LLM_STATUS__ = llm;
    const voice = (lastVoiceStatus && lastVoiceStatus.voice) || status.voice || {};
    ensureQADockShell();
    const summary = providerSummary(llm, voice);
    const providerClass = summary.expected === "echo" ? "warn" : "info";
    const qaAllowed = Boolean(voice.qa_allowed);
    const lastJob = latestVoiceJob(voice);
    const providerLineNode = document.getElementById("voice-provider-line");
    const providerPill = document.getElementById("voice-provider-pill");
    const answerNode = document.getElementById("voice-answer");
    const selection = currentReportSelection(status);
    const selectedActionName = (selection.activeEntry && selection.activeEntry.actionName) || window.__SELECTED_VOICE_ACTION_NAME__ || "当前动作";

    if (providerLineNode) {
      providerLineNode.textContent = qaAllowed
        ? `当前问答动作：${selectedActionName}。只基于该动作最近一次训练报告回答。`
        : "训练进行中暂不回答，避免影响摄像头、计数和播报。";
    }
    if (providerPill) {
      providerPill.className = `pill ${providerClass}`;
      providerPill.textContent = summary.expected;
    }
    if (answerNode && lastJob && !voiceJobId) {
      const lastJobId = String(lastJob.job_id || "");
      if (!displayedVoiceAnswerJobId || displayedVoiceAnswerJobId === lastJobId) {
        displayedVoiceAnswerJobId = lastJobId;
        answerNode.textContent = voiceJobSummary(lastJob);
      }
    }

    updateVoiceControls(status);
  }
  function completionValue(item) {
    if (!item) return null;
    const value = item.completion_percent ?? item.quality_score ?? item.score;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function completionReasonText(errorCode, reason) {
    const code = String(errorCode || "OK").toUpperCase();
    const map = {
      OK: "动作达标",
      ROM_LOW: "高度不够",
      TUT_LOW: "保持时间不够",
      TOO_FAST: "速度过快",
      EARLY_RETURN: "回落过早",
      SHAPE_BAD: "动作形态需调整",
      VISIBILITY_LOW: "关键点可见性不足",
    };
    if (map[code]) return map[code];
    return reason ? UI.safeText(reason) : "需要复核";
  }

  function normalizeCompletionSummary(actionId, summary, context) {
    const report = context && context.report ? context.report : {};
    const reportCompletion = report.completion_by_action || {};
    const source = summary || reportCompletion[actionId] || null;
    const rawAttempts = Array.isArray(source && source.attempts)
      ? source.attempts
      : (Array.isArray(report.quality_attempts) ? report.quality_attempts : []);
    const attempts = rawAttempts.map((item, index) => ({
      attemptIndex: item.attempt_index || index + 1,
      countable: Boolean(item.countable),
      primaryError: item.primary_error || "OK",
      reason: item.reason,
      completion: completionValue(item),
    }));
    const scores = attempts.map((item) => item.completion).filter((value) => value != null);
    const averageSource = source && source.average_completion != null
      ? source.average_completion
      : (report.overall_completion ?? report.overall_quality);
    const averageNumber = Number(averageSource);
    const averageCompletion = Number.isFinite(averageNumber)
      ? averageNumber
      : (scores.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null);
    return {
      actionId,
      actionName: (source && source.action_name) || (context && context.actionName) || UI.actionNames[actionId] || actionId,
      reportFile: (source && source.report_file) || (context && context.reportFile) || "",
      attempts,
      averageCompletion,
    };
  }

  function completionSummaries(status) {
    const training = status.training || {};
    const grouped = training.completion_by_action || {};
    const reportEntries = actionReportEntries(status);
    return reportActionIds.map((actionId) => {
      const entry = reportEntries.find((item) => item.actionId === actionId) || null;
      return normalizeCompletionSummary(actionId, grouped[actionId], entry && entry.context);
    });
  }

  function hasCompletionData(status) {
    return completionSummaries(status).some((summary) => summary.attempts.length > 0);
  }

  function overallCompletionFromSummaries(summaries) {
    const actionAverages = summaries
      .map((summary) => Number(summary.averageCompletion))
      .filter((value) => Number.isFinite(value));
    if (actionAverages.length) {
      return actionAverages.reduce((sum, value) => sum + value, 0) / actionAverages.length;
    }
    const attemptScores = [];
    summaries.forEach((summary) => {
      summary.attempts.forEach((item) => {
        const score = Number(item.completion);
        if (Number.isFinite(score)) attemptScores.push(score);
      });
    });
    return attemptScores.length ? attemptScores.reduce((sum, value) => sum + value, 0) / attemptScores.length : null;
  }

  function renderCompletionDock(status) {
    const body = document.getElementById("dock-body-completion");
    if (!body) return;
    const training = status.training || {};
    const summaries = completionSummaries(status);
    const hasAny = summaries.some((summary) => summary.attempts.length > 0);
    const summaryNode = document.getElementById("dock-summary-completion");
    if (summaryNode) {
      if (trainingBusyStatuses.has(training.status)) {
        summaryNode.textContent = "训练进行中，完成度后台计算；结束后显示每次尝试和平均值。";
      } else if (hasAny) {
        const overall = overallCompletionFromSummaries(summaries);
        summaryNode.textContent = Number.isFinite(overall)
          ? `训练完成度汇总，三组整体平均 ${UI.formatNumber(overall, 1)}%。`
          : "训练完成度汇总已生成。";
      } else {
        summaryNode.textContent = "完成一次单动作或完整训练后，这里显示每次尝试的完成度。";
      }
    }
    body.innerHTML = `
      <section class="completion-dock-shell">
        ${hasAny ? summaries.map((summary) => completionActionHtml(summary)).join("") : `<div class="dock-empty">暂无完成度结果。完整训练结束后会按三组动作显示每次尝试和平均值。</div>`}
      </section>
    `;
  }

  function completionActionHtml(summary) {
    const attemptsHtml = summary.attempts.length
      ? summary.attempts.map((item) => {
          const tone = item.primaryError === "OK" ? "good" : "warn";
          const completion = item.completion == null ? "未评分" : `${UI.formatNumber(item.completion, 1)}%`;
          return `<div class="completion-attempt-row">
            <strong>#${UI.escapeHtml(item.attemptIndex || "-")}</strong>
            <span class="pill ${tone}">${UI.escapeHtml(item.primaryError || "OK")}</span>
            <span>${UI.escapeHtml(completionReasonText(item.primaryError, item.reason))}</span>
            <span class="mono">${UI.escapeHtml(completion)}</span>
            <small>${item.countable ? "计数" : "未计数"}</small>
          </div>`;
        }).join("")
      : `<div class="completion-empty">暂无本动作 attempt。</div>`;
    const average = summary.averageCompletion == null ? "暂无完成度" : `${UI.formatNumber(summary.averageCompletion, 1)}%`;
    return `<article class="completion-action-card">
      <div class="completion-action-head">
        <div>
          <strong>${UI.escapeHtml(summary.actionName)}</strong>
          <div class="summary-text">本动作完成度明细</div>
        </div>
        <span class="pill info">平均 ${UI.escapeHtml(average)}</span>
      </div>
      <div class="completion-attempt-list">${attemptsHtml}</div>
    </article>`;
  }
  function renderSystemDock(system, status) {
    const body = document.getElementById("dock-body-system");
    if (!body) return;
    body.innerHTML = `
      <section class="dock-system-shell">
        <div class="metric-matrix system-key-matrix">${UI.renderSystemStats(system || {}) || row("设备状态", "等待 /api/system/status")}</div>
      </section>
    `;
  }
  function providerChoiceLabel(provider) {
    if (provider === "glm4v_api") return "GLM";
    if (provider === "local_qwen_rkllm") return "Qwen";
    if (provider === "echo") return "Echo";
    return "Auto";
  }

  function updateProviderSwitch(llm) {
    const provider = String((llm && llm.provider) || "auto");
    document.querySelectorAll("[data-provider-choice]").forEach((button) => {
      const active = button.getAttribute("data-provider-choice") === provider;
      button.classList.toggle("is-active", active);
      button.classList.toggle("good", active);
    });
  }

  async function submitProviderChoice(provider) {
    try {
      setVoiceMessage(`正在切换到 ${providerChoiceLabel(provider)}...`, "warn");
      const result = await UI.postJSON("/api/llm/provider", { provider });
      lastLLMStatus = { llm: result.llm || {} };
      updateProviderSwitch(lastLLMStatus.llm || {});
      renderQADock(lastStatus || {});
      setVoiceMessage(`已切换：${providerChoiceLabel((result.llm || {}).provider || provider)}`, "good");
      window.setTimeout(() => refreshAIReports(true), 200);
    } catch (error) {
      setVoiceMessage(error.message || String(error), "bad");
    }
  }

  function stopMicStream() {
    if (!micStream) return;
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }

  async function getMicStream() {
    if (micStream && micStream.active) return micStream;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("当前浏览器不支持麦克风录音，或页面权限不满足 getUserMedia 要求。");
    }
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false });
    return micStream;
  }

  function flattenAudioBuffers(buffers, totalLength) {
    const output = new Float32Array(totalLength);
    let offset = 0;
    buffers.forEach((buffer) => {
      output.set(buffer, offset);
      offset += buffer.length;
    });
    return output;
  }

  function downsampleBuffer(buffer, inputRate, outputRate) {
    if (outputRate === inputRate) return buffer;
    const ratio = inputRate / outputRate;
    const length = Math.max(1, Math.round(buffer.length / ratio));
    const result = new Float32Array(length);
    for (let i = 0; i < length; i += 1) {
      const start = Math.floor(i * ratio);
      const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      let count = 0;
      for (let j = start; j < end; j += 1) {
        sum += buffer[j];
        count += 1;
      }
      result[i] = count ? sum / count : 0;
    }
    return result;
  }

  function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (offset, value) => {
      for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i += 1, offset += 2) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
    return buffer;
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return window.btoa(binary);
  }

  async function recordWavClip(durationMs = 2600) {
    const stream = await getMicStream();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error("当前浏览器不支持 AudioContext 录音。 ");
    const audioContext = new AudioContextClass();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const buffers = [];
    let totalLength = 0;
    let peak = 0;
    processor.onaudioprocess = (event) => {
      const data = event.inputBuffer.getChannelData(0);
      const copy = new Float32Array(data);
      buffers.push(copy);
      totalLength += copy.length;
      for (let i = 0; i < copy.length; i += 1) peak = Math.max(peak, Math.abs(copy[i]));
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
    await new Promise((resolve) => window.setTimeout(resolve, durationMs));
    processor.disconnect();
    source.disconnect();
    await audioContext.close();
    const merged = flattenAudioBuffers(buffers, totalLength);
    const samples = downsampleBuffer(merged, audioContext.sampleRate, 16000);
    const wav = encodeWav(samples, 16000);
    return { wavB64: arrayBufferToBase64(wav), peak };
  }

  function wait(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  async function pollAsrResult(jobId, maxPolls = 80) {
    for (let index = 0; index < maxPolls; index += 1) {
      const result = await UI.fetchJSON(`/api/voice/asr_result?job_id=${encodeURIComponent(jobId)}`);
      if (result.status === "done") return String(result.text || "").trim();
      if (result.status === "failed") throw new Error(result.error || "ASR failed");
      await wait(150);
    }
    throw new Error("ASR 识别超时。");
  }

  async function submitAsrClip(clip) {
    const result = await UI.postJSON("/api/voice/asr", { audio_b64: `data:audio/wav;base64,${clip.wavB64}` });
    return pollAsrResult(result.job_id);
  }

  async function submitBackendAsrCapture(durationMs = 2600, options = {}) {
    const result = await UI.postJSON("/api/voice/asr_capture", {
      duration_seconds: Math.max(1, durationMs / 1000),
      stop_on_silence: options.stopOnSilence !== false,
      silence_seconds: options.silenceSeconds || 0.75,
      speech_wait_seconds: options.speechWaitSeconds || 1.2,
      min_capture_seconds: options.minCaptureSeconds || 0.6,
    });
    return { text: await pollAsrResult(result.job_id), capture: result.capture || {}, mic: result.mic || {} };
  }

  async function captureAsrText(durationMs = 2600, options = {}) {
    if (micCaptureMode === "backend") {
      const backend = await submitBackendAsrCapture(durationMs, options);
      return { text: backend.text, source: "backend", capture: backend.capture };
    }
    try {
      const clip = await recordWavClip(durationMs);
      return { text: await submitAsrClip(clip), source: "browser", peak: clip.peak };
    } catch (error) {
      micCaptureMode = "backend";
      const backend = await submitBackendAsrCapture(durationMs, options);
      return { text: backend.text, source: "backend", capture: backend.capture, fallbackError: error };
    }
  }

  async function refreshVoiceStatusNow() {
    const result = await UI.fetchJSON("/api/voice/status");
    lastVoiceStatus = result || lastVoiceStatus;
    return (lastVoiceStatus && lastVoiceStatus.voice) || {};
  }

  function setVoiceQuestionText(text = "") {
    const value = String(text || "").trim();
    voiceQuestionDraft = value;
    const input = document.getElementById("voice-question");
    if (input) input.value = value;
  }

  function setManualCaptureStatus(active, sessionId = "") {
    voiceListening = Boolean(active);
    voiceListenSessionId = voiceListening ? String(sessionId || voiceListenSessionId || "") : "";
    const voice = lastVoiceStatus && lastVoiceStatus.voice;
    if (voice) {
      voice.manual_capture = {
        ...(voice.manual_capture || {}),
        active: voiceListening,
        session_id: voiceListenSessionId || null,
      };
    }
  }

  async function startVoiceListening() {
    const voice = await refreshVoiceStatusNow().catch(() => (lastVoiceStatus && lastVoiceStatus.voice) || {});
    if (isTrainingBusy() || !voice.qa_allowed || (voice.assistant_tts || {}).busy) {
      setVoiceMessage("训练固定语音或小爱语音尚未结束，暂不能开始监听。", "warn");
      return;
    }
    if (!selectedVoiceReportReady()) {
      setVoiceMessage("当前动作还没有训练报告，完成训练后再录音提问。", "warn");
      return;
    }
    voiceListenBusy = true;
    updateVoiceControls(lastStatus);
    try {
      setVoiceQuestionText("");
      setVoiceMessage("正在开启监听，请开始说出完整问题；说完后点击“结束监听”。", "good");
      const result = await UI.postJSON("/api/voice/listen_start", {});
      setManualCaptureStatus(true, result.session_id);
      setVoiceMessage("正在监听并录音；说完后点击“结束监听”。", "good");
    } catch (error) {
      setManualCaptureStatus(false);
      setVoiceMessage(error.message || String(error), "bad");
    } finally {
      voiceListenBusy = false;
      updateVoiceControls(lastStatus);
    }
  }

  async function stopVoiceListening() {
    if (!voiceListening || voiceListenBusy) return;
    voiceListenBusy = true;
    updateVoiceControls(lastStatus);
    try {
      setVoiceMessage("正在结束监听并识别整段录音...", "warn");
      const result = await UI.postJSON("/api/voice/listen_stop", { session_id: voiceListenSessionId });
      setManualCaptureStatus(false);
      const text = await pollAsrResult(result.job_id, 240);
      if (text) {
        setVoiceQuestionText(text);
        setVoiceMessage(`识别结果：${text}`, "good");
      } else {
        setVoiceQuestionText("");
        setVoiceMessage("录音完成，但没有识别到有效问题；请靠近麦克风后重试。", "warn");
      }
    } catch (error) {
      const voice = await refreshVoiceStatusNow().catch(() => (lastVoiceStatus && lastVoiceStatus.voice) || {});
      const capture = voice.manual_capture || {};
      setManualCaptureStatus(Boolean(capture.active), capture.session_id || "");
      setVoiceMessage(error.message || String(error), "bad");
    } finally {
      voiceListenBusy = false;
      updateVoiceControls(lastStatus);
    }
  }

  async function toggleVoiceListening() {
    if (voiceListenBusy) return;
    if (voiceListening) await stopVoiceListening();
    else await startVoiceListening();
  }

  function updateVoiceControls(status) {
    const training = status.training || {};
    const voice = (lastVoiceStatus && lastVoiceStatus.voice) || status.voice || {};
    const llm = (lastLLMStatus && lastLLMStatus.llm) || status.llm || {};
    const askBtn = document.getElementById("voice-ask-btn");
    const listenBtn = document.getElementById("voice-listen-btn");
    const input = document.getElementById("voice-question");
    updateProviderSwitch(llm);
    if (!askBtn || !listenBtn || !input) return;

    const manualCapture = voice.manual_capture || {};
    if (manualCapture.active && !voiceListenBusy) {
      setManualCaptureStatus(true, manualCapture.session_id || "");
    }

    if (document.activeElement !== input && !voiceComposing && input.value !== voiceQuestionDraft) {
      input.value = voiceQuestionDraft;
    }
    const trainingBusy = trainingBusyStatuses.has(training.status);
    const activeProvider = String(llm.active_provider || llm.provider || "").toLowerCase();
    const providerReady = Boolean(activeProvider) && activeProvider !== "echo";
    const allowed = Boolean(voice.qa_allowed) && !trainingBusy;
    const busy = Boolean(voiceJobId);
    const question = input.value.trim();
    const grouped = lastReports.latest_reports_by_action || {};
    const hasReport = selectedVoiceReportReady();
    askBtn.disabled = !allowed || busy || voiceListening || voiceListenBusy || !question || !hasReport;
    listenBtn.disabled = voiceListenBusy || (!voiceListening && (!allowed || busy || trainingBusy));
    listenBtn.textContent = voiceListenBusy ? "正在处理..." : (voiceListening ? "结束监听" : "唤醒监听");
    listenBtn.classList.toggle("warn", voiceListening);
    const summaryNode = document.getElementById("dock-summary-qa");
    if (summaryNode) {
      summaryNode.textContent = trainingBusy
        ? "训练进行中暂不回答，避免影响摄像头、计数和播报。"
        : (providerReady ? "现在可以基于最近一次训练报告进行康复问答。" : "训练结束后可通过语音或文字提问。");
    }    const message = document.getElementById("voice-message");
    if (message && !voiceJobId && !voiceListening && !voiceListenBusy && !message.classList.contains("bad")) {
      const idleMessage = !hasReport
        ? `${window.__SELECTED_VOICE_ACTION_NAME__ || "当前动作"}还没有报告，先完成该动作训练。`
        : (trainingBusy ? "训练进行中暂不回答，避免抢训练和播报资源。" : (allowed ? `当前问答动作：${window.__SELECTED_VOICE_ACTION_NAME__ || "当前动作"}。可直接提问，训练进行中会自动阻止。` : ""));
      message.hidden = !idleMessage;
      message.className = `message-box ${allowed ? "" : "warn"}`.trim();
      message.textContent = idleMessage;
    }
  }

  function bindVoiceEvents() {
    const listenBtn = document.getElementById("voice-listen-btn");
    const askBtn = document.getElementById("voice-ask-btn");
    const input = document.getElementById("voice-question");
    if (listenBtn && listenBtn.dataset.bound !== "1") {
      listenBtn.dataset.bound = "1";
      listenBtn.addEventListener("click", toggleVoiceListening);
    }
    document.querySelectorAll("[data-provider-choice]").forEach((button) => {
      if (button.dataset.bound === "1") return;
      button.dataset.bound = "1";
      button.addEventListener("click", () => submitProviderChoice(button.getAttribute("data-provider-choice") || "auto"));
    });
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

  window.__REHAB_RERENDER_REPORTS__ = function () {
    if (isDockVisible("report")) renderReportDock(lastStatus || {});
  };
  document.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-report-action]");
    if (!tab) return;
    selectedReportActionId = tab.getAttribute("data-report-action") || "";
    reportActionTouched = true;
    renderReportDock(lastStatus || {});
    if (isDockVisible("qa")) renderQADock(lastStatus || {});
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
    renderQualityScore(training);
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

    if ((finishedPrompt || finishedStatus) && hasCompletionData(status) && !completionAutoPinned) {
      dockState.pinned = "completion";
      completionAutoPinned = true;
      renderCompletionDock(status);
    }

    const quality = status.pose_quality || {};
    document.getElementById("live-grid").innerHTML = `
      <span class="pill info">${UI.safeText(actionName)}</span>
      <span class="pill info">rep ${completed}/${target}</span>
      <span class="pill ${quality.quality_ok ? "good" : "warn"}">${quality.quality_ok ? "keypoints OK" : "check keypoints"}</span>
      <span class="pill ${training.latest_quality && training.latest_quality.score != null ? "good" : "warn"}">${qualityPillText(training)}</span>
    `;
    document.getElementById("timeline").innerHTML = renderTimeline(training.demo_plan, training);
    renderCareDialog(training);
    if (isDockVisible("completion")) renderCompletionDock(status);
    if (isDockVisible("report")) renderReportDock(status);
    if (isDockVisible("qa")) renderQADock(status);
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
    if (id === "completion") refreshCompletion(force);
    if (id === "system") refreshSystem(force);
    if (id === "report" || id === "qa") refreshAIReports(force);
  }

  function refreshVisibleDocks(force = false) {
    if (isDockVisible("completion")) refreshCompletion(force);
    if (isDockVisible("system")) refreshSystem(force);
    if (isDockVisible("report") || isDockVisible("qa")) refreshAIReports(force);
  }

  async function refreshCompletion(force = false) {
    const now = Date.now();
    if (!isDockVisible("completion")) return;
    renderCompletionDock(lastStatus || {});
    if (completionRefreshInFlight) return;
    if (!force && now - lastCompletionRefreshAt < 15000) return;
    completionRefreshInFlight = true;
    try {
      const reports = await UI.fetchJSON("/api/reports/latest_by_action");
      lastReports = reports || lastReports;
      lastCompletionRefreshAt = Date.now();
      if (isDockVisible("completion")) renderCompletionDock(lastStatus || {});
    } catch (error) {
      const body = document.getElementById("dock-body-completion");
      if (body && isDockVisible("completion")) {
        body.innerHTML = `<div class="message-box warn">${UI.safeText(error.message || String(error))}</div>`;
      }
    } finally {
      completionRefreshInFlight = false;
    }
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
    const reportVisible = isDockVisible("report");
    const qaVisible = isDockVisible("qa");
    if (!reportVisible && !qaVisible) return;
    if (isTrainingBusy()) {
      if (reportVisible) renderReportDock(lastStatus || {});
      if (qaVisible) renderQADock(lastStatus || {});
      return;
    }
    if (aiRefreshInFlight) return;
    if (!force && now - lastAIRefreshAt < 15000) return;
    aiRefreshInFlight = true;
    try {
      const [reports, llmStatus, voiceStatus] = await Promise.all([
        UI.fetchJSON("/api/reports/latest_by_action"),
        UI.fetchJSON(force ? "/api/llm/status?force=1" : "/api/llm/status"),
        UI.fetchJSON("/api/voice/status"),
      ]);
      lastReports = reports || lastReports;
      lastLLMStatus = llmStatus || lastLLMStatus;
      lastVoiceStatus = voiceStatus || lastVoiceStatus;
      lastAIRefreshAt = Date.now();
      if (reportVisible) renderReportDock(lastStatus || {});
      if (qaVisible) renderQADock(lastStatus || {});
    } catch (error) {
      const message = document.getElementById("voice-message");
      if (message && qaVisible) {
        message.className = "message-box warn";
        message.textContent = error.message || String(error);
      }
      const reportBody = document.getElementById("dock-body-report");
      if (reportBody && reportVisible) {
        reportBody.innerHTML = `<div class="message-box warn">${UI.safeText(error.message || String(error))}</div>`;
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
      if (isDockVisible("report")) renderReportDock(lastStatus || {});
      if (isDockVisible("qa")) renderQADock(lastStatus || {});
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

  function qualityPillText(training) {
    const latest = training.latest_quality || {};
    if (latest.score == null) return "完成度后台计算中";
    return `完成度 ${UI.formatNumber(latest.score, 1)}%`;
  }

  function renderQualityScore(training) {
    const latest = training.latest_quality || null;
    const model = training.quality_model || {};
    const scoreNode = document.getElementById("quality-score-main");
    const backendNode = document.getElementById("quality-score-backend");
    const metaNode = document.getElementById("quality-score-meta");
    const historyNode = document.getElementById("quality-score-history");
    if (!scoreNode || !backendNode || !metaNode || !historyNode) return;

    const backend = latest && latest.backend ? latest.backend : (model.backend || "waiting");
    backendNode.className = `pill ${model.available === false ? "warn" : "info"}`;
    backendNode.textContent = UI.safeText(backend, "waiting");

    if (latest && latest.score != null) {
      scoreNode.textContent = `${UI.formatNumber(latest.score, 1)}%`;
      const grade = latest.grade ? ` / ${UI.safeText(latest.grade)}` : "";
      const reason = latest.reason || latest.primary_error || (latest.countable ? "计数" : "纠错");
      metaNode.textContent = `#${latest.attempt_index || "-"}${grade} · ${UI.safeText(reason)} · ${latest.countable ? "计数" : "未计数"}`;
    } else if (model.available === false) {
      scoreNode.textContent = "未评分";
      metaNode.textContent = model.last_error ? `模型不可用：${UI.safeText(model.last_error)}` : "模型不可用，训练仍会正常进行。";
    } else {
      scoreNode.textContent = "等待动作";
      metaNode.textContent = model.queue_size ? `评分队列中：${model.queue_size}` : "每次动作结束后后台计算，训练结束后到右侧完成度栏查看。";
    }

    const rows = Array.isArray(training.quality_score_history) ? training.quality_score_history.slice(-6) : [];
    historyNode.innerHTML = rows.length
      ? rows.map((item) => {
          const score = item.score == null ? "等待" : `${UI.formatNumber(item.score, 1)}%`;
          const grade = item.grade ? ` / ${UI.escapeHtml(item.grade)}` : "";
          const state = item.countable ? "计数" : "纠错";
          return `<div class="quality-score-row">
            <strong>#${UI.escapeHtml(item.attempt_index || "-")}</strong>
            <span>${UI.escapeHtml(score)}${grade}</span>
            <small>${UI.escapeHtml(item.primary_error || state)}</small>
          </div>`;
        }).join("")
      : `<div class="quality-score-empty">暂无完成度后台。</div>`;
  }

  async function submitVoiceQuestion() {
    const input = document.getElementById("voice-question");
    const question = input ? input.value.trim() : voiceQuestionDraft.trim();
    if (!question) {
      setVoiceMessage("请先输入问题，或点击“唤醒监听”录音后再提交。", "warn");
      return;
    }
    if (isTrainingBusy()) {
      setVoiceMessage("训练动作进行中暂不回答，避免抢摄像头、计数和播报资源。", "warn");
      return;
    }
    const context = applyQuestionActionContext(question);
    if (!context.hasReport) {
      setVoiceMessage(`${context.actionName}还没有报告，先完成该动作训练。`, "warn");
      updateVoiceControls(lastStatus);
      return;
    }
    voiceQuestionDraft = question;
    const pollToken = voicePollToken + 1;
    voicePollToken = pollToken;
    voiceJobId = `pending:${pollToken}`;
    displayedVoiceAnswerJobId = voiceJobId;
    const answerNode = document.getElementById("voice-answer");
    if (answerNode) answerNode.textContent = `问答动作：${context.actionName}\n当前问题：${question}\n正在等待回答...`;
    updateVoiceControls(lastStatus);
    setVoiceMessage(`${context.switched ? "已按问题切换动作，" : ""}已提交 ${context.actionName} 问答，正在等待回答...`, "warn");
    try {
      const result = await UI.postJSON("/api/voice/ask", { question, report_id: context.reportId, action_id: context.actionId, action_name: context.actionName, speak: true });
      if (pollToken !== voicePollToken) return;
      if (result.status === "blocked_training") {
        voiceJobId = "";
        setVoiceMessage(result.error || result.answer || "训练中暂不回答。", "warn");
        updateVoiceControls(lastStatus);
        return;
      }
      voiceJobId = result.job_id;
      displayedVoiceAnswerJobId = result.job_id;
      pollVoiceAnswer(result.job_id, pollToken);
    } catch (error) {
      if (pollToken !== voicePollToken) return;
      voiceJobId = "";
      setVoiceMessage(error.message || String(error), "bad");
      updateVoiceControls(lastStatus);
    }
  }

  async function pollVoiceAnswer(jobId, pollToken) {
    if (pollToken !== voicePollToken || voiceJobId !== jobId) return;
    try {
      const result = await UI.fetchJSON(`/api/voice/ask_result?job_id=${encodeURIComponent(jobId)}`);
      if (pollToken !== voicePollToken || voiceJobId !== jobId) return;
      if (result.status === "queued" || result.status === "running") {
        const answer = document.getElementById("voice-answer");
        if (answer && (result.answer || result.error)) answer.textContent = voiceJobSummary(result);
        window.setTimeout(() => pollVoiceAnswer(jobId, pollToken), 400);
        return;
      }
      voiceJobId = "";
      displayedVoiceAnswerJobId = String(result.job_id || jobId || "");
      if (result.status === "blocked_training") {
        setVoiceMessage(result.error || result.answer || "训练中暂不回答。", "warn");
      } else if (result.status === "done") {
        const actionName = result.action_name || window.__SELECTED_VOICE_ACTION_NAME__ || "当前动作";
        setVoiceMessage(`${actionName}回答完成：${UI.safeText(voiceProviderLabel(result.active_provider), "-")}`, "good");
      } else {
        try {
          lastLLMStatus = await UI.fetchJSON("/api/llm/status?force=1");
        } catch (_) {}
        throw new Error(result.error || "LLM failed");
      }
      const answer = document.getElementById("voice-answer");
      if (answer) answer.textContent = voiceJobSummary(result);
      updateVoiceControls(lastStatus);
      refresh();
    } catch (error) {
      if (pollToken !== voicePollToken || voiceJobId !== jobId) return;
      voiceJobId = "";
      setVoiceMessage(error.message || String(error), "bad");
      updateVoiceControls(lastStatus);
    }
  }

  document.getElementById("single-btn").addEventListener("click", async () => {
    try {
      completionAutoPinned = false;
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
      completionAutoPinned = false;
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
