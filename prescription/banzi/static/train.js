(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");
  const activeStatuses = new Set(["running", "paused", "resting", "awaiting_orientation", "awaiting_return", "awaiting_care_response"]);
  const dockState = {
    hovered: "",
    pinned: "",
    activeReportKey: "",
    reportContexts: [],
    collapseTimer: 0,
  };
  const DOCK_COLLAPSE_DELAY_MS = 260;
  let lastRepCount = 0;

  app.innerHTML = `
    <main class="shell train-shell">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">Patient Training Cockpit</div>
          <h1>患者训练</h1>
          <p>聚焦训练主流程，AI 图文建议与设备状态收纳到右侧停靠栏。</p>
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
                  <div class="pills" id="train-pill-row"></div>
                </div>
              </div>
              <div class="status-badges" id="live-grid"></div>
            </section>
          </section>

          <section class="panel training-hud">
            <div class="section-head">
              <span class="pill info" id="training-status-pill">idle</span>
              <span class="meta" id="playlist-state">playlist 0/0</span>
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
                  <article>
                    <strong>目标</strong>
                    <span id="live-metric-target">-</span>
                  </article>
                  <article>
                    <strong>保持</strong>
                    <span id="live-metric-tut">-</span>
                  </article>
                  <article>
                    <strong>状态</strong>
                    <span id="live-metric-state">-</span>
                  </article>
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
                <input id="action-id" value="seated_knee_extension" list="train-action-options">
                <datalist id="train-action-options">
                  <option value="seated_knee_extension">坐姿伸膝</option>
                  <option value="standing_hamstring_curl">站姿屈膝后勾腿</option>
                  <option value="seated_knee_raise">坐姿抬膝</option>
                  <option value="sit_to_stand">坐站训练</option>
                  <option value="knee_flexion">屈膝</option>
                </datalist>
              </label>
            </div>

            <div class="button-row">
              <button id="playlist-btn">开始完整训练</button>
              <button class="secondary" id="single-btn">开始单动作</button>
              <button class="secondary" id="pause-btn">暂停 / 继续</button>
              <button class="warn" id="stop-btn">结束训练</button>
            </div>
            <div class="message-box" id="train-message">准备就绪，等待开始训练。</div>

            <div class="message-box">
              训练报告、AI 图文建议与设备状态已收纳在右侧停靠栏，悬停可展开，点击可固定。
            </div>
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
          ${dockMarkup("ai", "AI训练图文建议", "报告 / 建议 / 问答")}
          ${dockMarkup("system", "设备运行状态", "CPU / 温度 / NPU")}
        </aside>
      </section>
    </main>
    <div id="care-modal" class="modal-shell hidden" aria-hidden="true"></div>
  `;

  const restAudio = new Audio();
  restAudio.preload = "auto";
  restAudio.loop = true;
  restAudio.playsInline = true;

  bindDockEvents();
  applyInitialDockFromHash();

  function setMessage(text, tone = "") {
    const node = document.getElementById("train-message");
    node.className = `message-box ${tone}`.trim();
    node.textContent = text;
  }

  function payload() {
    return {
      patient_id: document.getElementById("patient-id").value.trim(),
      action_id: document.getElementById("action-id").value.trim(),
      side_mode: document.getElementById("side-mode").value,
      target_reps: Number(document.getElementById("target-reps").value || 3),
    };
  }

  function musicUrl(file) {
    try {
      return new URL(file, window.location.origin).toString();
    } catch (error) {
      return file;
    }
  }

  function stopRestMusic() {
    if (!restAudio.paused) restAudio.pause();
    restAudio.currentTime = 0;
    restAudio.volume = 1;
  }

  function syncRestMusic(training) {
    const music = training.rest_music || {};
    if ((music.playback || "browser") === "backend") {
      stopRestMusic();
      return;
    }
    const shouldPlay = training.status === "resting" && music.enabled && music.file;
    if (!shouldPlay) {
      stopRestMusic();
      return;
    }
    const nextSrc = musicUrl(music.file);
    if (restAudio.src !== nextSrc) {
      restAudio.src = nextSrc;
      restAudio.currentTime = 0;
    }
    if (restAudio.paused) restAudio.play().catch(() => {});
    const fadeSeconds = Number(music.fade_seconds || 0);
    if (training.rest_remaining_seconds != null && fadeSeconds > 0 && training.rest_remaining_seconds <= fadeSeconds) {
      restAudio.volume = Math.max(0, Math.min(1, Number(training.rest_remaining_seconds) / fadeSeconds));
    } else {
      restAudio.volume = 1;
    }
  }

  function renderCareDialog(training) {
    const modal = document.getElementById("care-modal");
    const dialog = training.care_dialog || {};
    if (!dialog.visible) {
      modal.className = "modal-shell hidden";
      modal.setAttribute("aria-hidden", "true");
      modal.innerHTML = "";
      return;
    }
    modal.className = "modal-shell";
    modal.setAttribute("aria-hidden", "false");
    modal.innerHTML = `
      <section class="modal-card">
        <div class="eyebrow">Care Check</div>
        <h3>${UI.safeText(dialog.title, "温馨提示")}</h3>
        <p>${UI.safeText(dialog.message)}</p>
        <div class="button-row">
          <button data-care-response="yes">${UI.safeText(dialog.yes_label, "是")}</button>
          <button class="secondary" data-care-response="no">${UI.safeText(dialog.no_label, "否")}</button>
        </div>
      </section>
    `;
  }

  function dockMarkup(id, title, hint) {
    return `
      <section class="dock-panel" data-dock-panel="${id}">
        <button
          class="dock-tab"
          id="dock-tab-${id}"
          type="button"
          data-dock-trigger="${id}"
          aria-expanded="false"
          aria-controls="dock-surface-${id}"
        >
          <span class="dock-tab-title">${title}</span>
          <span class="dock-tab-hint">${hint}</span>
        </button>
        <section class="dock-surface" id="dock-surface-${id}" aria-hidden="true">
          <div class="dock-surface-head">
            <div>
              <div class="eyebrow">辅助信息</div>
              <h3>${title}</h3>
              <p class="summary-text" id="dock-summary-${id}">${hint}</p>
            </div>
            <button class="plain dock-pin" type="button" data-dock-pin="${id}">固定</button>
          </div>
          <div class="dock-surface-body" id="dock-content-${id}"></div>
        </section>
      </section>
    `;
  }

  function contextKey(context) {
    return UI.reportKey ? UI.reportKey(context) : UI.safeText(context && context.reportFile, "latest");
  }

  function collectContexts(status) {
    const sources = [];
    const training = status.training || {};
    if (training.report) {
      sources.push({
        report: training.report,
        report_file: training.report_file,
        summary_bundle: training.report.summary_bundle,
        report_card: training.report.report_card,
      });
    }
    if (status.latest_report) sources.push(status.latest_report);
    if (Array.isArray(status.recent_reports)) sources.push(...status.recent_reports);

    const seen = new Set();
    const contexts = [];
    sources.forEach((source) => {
      const context = UI.composeReportContext(source);
      if (!context) return;
      const key = contextKey(context);
      if (seen.has(key)) return;
      seen.add(key);
      contexts.push(context);
    });
    return contexts.slice(0, 3);
  }

  function dockCurrent() {
    return dockState.pinned || dockState.hovered || "";
  }

  function ensureActiveReportKey() {
    const available = dockState.reportContexts.map((context) => contextKey(context));
    if (!available.length) {
      dockState.activeReportKey = "";
      return;
    }
    if (!available.includes(dockState.activeReportKey)) {
      dockState.activeReportKey = available[0];
    }
  }

  function setHoveredDock(id) {
    if (dockState.pinned) return;
    clearCollapseTimer();
    dockState.hovered = id;
    renderDockState();
  }

  function clearHoveredDock(id) {
    if (dockState.pinned) return;
    if (id && dockState.hovered !== id) return;
    scheduleDockCollapse();
  }

  function togglePinnedDock(id) {
    clearCollapseTimer();
    dockState.pinned = dockState.pinned === id ? "" : id;
    dockState.hovered = dockState.pinned || "";
    renderDockState();
  }

  function clearCollapseTimer() {
    if (dockState.collapseTimer) {
      window.clearTimeout(dockState.collapseTimer);
      dockState.collapseTimer = 0;
    }
  }

  function scheduleDockCollapse() {
    clearCollapseTimer();
    dockState.collapseTimer = window.setTimeout(() => {
      dockState.collapseTimer = 0;
      if (dockState.pinned) return;
      dockState.hovered = "";
      renderDockState();
    }, DOCK_COLLAPSE_DELAY_MS);
  }

  function renderDockState() {
    const activeDock = dockCurrent();
    Array.from(document.querySelectorAll("[data-dock-panel]")).forEach((panel) => {
      const id = panel.getAttribute("data-dock-panel");
      const active = activeDock === id;
      const pinned = dockState.pinned === id;
      panel.classList.toggle("is-active", active);
      panel.classList.toggle("is-pinned", pinned);

      const trigger = panel.querySelector("[data-dock-trigger]");
      const surface = panel.querySelector(".dock-surface");
      const pinButton = panel.querySelector("[data-dock-pin]");
      if (trigger) trigger.setAttribute("aria-expanded", String(active));
      if (surface) surface.setAttribute("aria-hidden", String(!active));
      if (pinButton) {
        pinButton.classList.toggle("active", pinned);
        pinButton.textContent = pinned ? "取消固定" : "固定";
      }
    });
  }

  function applyInitialDockFromHash() {
    if (window.location.hash === "#dock-ai" || window.location.hash === "#ai-rehab") {
      dockState.pinned = "ai";
      dockState.hovered = "ai";
    }
    if (window.location.hash === "#dock-system") {
      dockState.pinned = "system";
      dockState.hovered = "system";
    }
    renderDockState();
  }

  function bindDockEvents() {
    Array.from(document.querySelectorAll("[data-dock-panel]")).forEach((panel) => {
      const id = panel.getAttribute("data-dock-panel");
      panel.addEventListener("mouseenter", () => setHoveredDock(id));
      panel.addEventListener("mouseleave", () => clearHoveredDock(id));
      panel.addEventListener("focusin", () => setHoveredDock(id));
      panel.addEventListener("focusout", () => {
        window.setTimeout(() => {
          if (!panel.contains(document.activeElement)) clearHoveredDock(id);
        }, 0);
      });
    });

    Array.from(document.querySelectorAll(".dock-surface-body")).forEach((body) => {
      body.addEventListener(
        "wheel",
        (event) => {
          const scrollTop = body.scrollTop;
          const maxScrollTop = body.scrollHeight - body.clientHeight;
          if (maxScrollTop <= 0) return;
          const movingDown = event.deltaY > 0;
          const movingUp = event.deltaY < 0;
          const atTop = scrollTop <= 0;
          const atBottom = scrollTop >= maxScrollTop - 1;
          if ((movingUp && atTop) || (movingDown && atBottom)) {
            event.preventDefault();
            return;
          }
          event.stopPropagation();
        },
        { passive: false }
      );
    });
  }

  function systemHealth(system, status) {
    const temp = system.temperature || {};
    const cpu = system.cpu || {};
    const npu = system.npu || {};
    const issues = [];
    let tone = "good";
    if (!status.stream_ready) {
      tone = "warn";
      issues.push("视觉链路等待中");
    }
    if (temp.available && Number(temp.max_celsius) >= 75) {
      tone = "warn";
      issues.push(`温度偏高 ${UI.formatNumber(temp.max_celsius)}°C`);
    }
    if (cpu.available && Number(cpu.percent) >= 90) {
      tone = "warn";
      issues.push(`CPU负载 ${UI.formatNumber(cpu.percent)}%`);
    }
    if (npu.available && npu.percent != null && Number(npu.percent) >= 95) {
      tone = "warn";
      issues.push(`NPU负载 ${UI.formatNumber(npu.percent)}%`);
    }
    if (!issues.length) {
      return { tone, title: "设备运行正常", detail: "链路、温度与核心资源处于可训练状态。" };
    }
    return { tone, title: "请关注设备状态", detail: issues.join(" · ") };
  }

  function renderAIDock(force = false) {
    const content = document.getElementById("dock-content-ai");
    const summary = document.getElementById("dock-summary-ai");
    if (!content || !summary) return;

    ensureActiveReportKey();
    const activeElement = document.activeElement;
    if (!force && activeElement && activeElement.matches("[data-ai-question]")) return;

    const contexts = dockState.reportContexts;
    if (!contexts.length) {
      summary.textContent = "训练完成后会在这里呈现最近报告、图文建议和问答。";
      content.innerHTML = `<div class="dock-empty">暂无训练报告。完成一次患者训练后，这里会自动显示最近一次训练结果。</div>`;
      return;
    }

    const activeContext = contexts.find((context) => contextKey(context) === dockState.activeReportKey) || contexts[0];
    const report = activeContext.report || {};
    const errors = report.errors || {};
    const meta = report.meta || {};
    const activeAction = report.action_name || meta.action_name || UI.actionNames[meta.action_id] || meta.action_id || "最近报告";

    contexts.forEach((context) => UI.registerReportContext && UI.registerReportContext(context));
    window.__LAST_REPORT_CONTEXT__ = activeContext;
    summary.textContent = `${activeAction} · ${errors.primary_error || "OK"} · 最近 ${contexts.length} 份报告`;
    content.innerHTML = `
      <section class="dock-ai-shell">
        <div class="dock-mini-tabs">
          ${contexts.map((context, index) => {
            const itemReport = context.report || {};
            const itemMeta = itemReport.meta || {};
            const itemErrors = itemReport.errors || {};
            const actionName =
              itemReport.action_name ||
              itemMeta.action_name ||
              UI.actionNames[itemMeta.action_id] ||
              itemMeta.action_id ||
              `报告 ${index + 1}`;
            const key = contextKey(context);
            return `
              <button
                type="button"
                class="dock-mini-tab ${dockState.activeReportKey === key ? "is-active" : ""}"
                data-report-tab-key="${UI.escapeHtml(key)}"
              >
                <strong>${UI.escapeHtml(actionName)}</strong>
                <span>${UI.escapeHtml(itemErrors.primary_error || "OK")}</span>
              </button>
            `;
          }).join("")}
        </div>
        <div class="dock-report-shell">
          ${UI.reportCardHtml(activeContext)}
        </div>
      </section>
    `;
  }

  function renderSystemDock(system, status) {
    const content = document.getElementById("dock-content-system");
    const summary = document.getElementById("dock-summary-system");
    if (!content || !summary) return;
    const health = systemHealth(system, status);
    summary.textContent = health.detail;
    content.innerHTML = `
      <section class="dock-system-shell">
        <div class="system-health ${health.tone}">
          <span class="pill ${health.tone === "good" ? "good" : "warn"}">${health.title}</span>
          <div class="summary-text">${health.detail}</div>
        </div>
        <div class="metric-matrix">
          ${UI.renderSystemStats(system)}
        </div>
      </section>
    `;
  }

  function render(status, system) {
    const training = status.training || {};
    const previewSource = UI.streamSource(status);
    const preview = document.getElementById("train-preview");
    if (preview.getAttribute("src") !== previewSource) preview.src = previewSource;

    const streamReady = Boolean(status.stream_ready);
    const visionPill = document.getElementById("train-vision-pill");
    visionPill.className = `pill ${streamReady ? "good" : "warn"}`;
    visionPill.textContent = streamReady ? "实时流已连接" : "视觉链路等待中";

    const actionName = training.current_action_name || UI.actionNames[training.action_id] || training.action_id || "准备训练";
    const completed = Number(training.completed_reps || 0);
    const target = Number(training.target_reps || 0);
    document.getElementById("action-title").textContent = actionName;
    const repNode = document.getElementById("rep-number");
    repNode.textContent = completed;
    if (completed !== lastRepCount) {
      repNode.classList.remove("bump");
      void repNode.offsetWidth;
      repNode.classList.add("bump");
      lastRepCount = completed;
    }
    document.getElementById("rep-target").textContent = `/ ${target}`;
    document.getElementById("rep-progress").style.width = `${target > 0 ? Math.min(100, (completed / target) * 100) : 0}%`;
    document.getElementById("live-metric-current").textContent = formatMetricValue(training.current_metric ?? training.current_angle, training.metric_unit);
    document.getElementById("live-metric-target").textContent = formatTargetRange(training.target_range, training.metric_unit);
    document.getElementById("live-metric-tut").textContent = formatTut(training);
    document.getElementById("live-metric-state").textContent = UI.safeText(training.current_state, "-");
    document.getElementById("training-status-pill").className = `pill ${training.status === "running" ? "good" : "warn"}`;
    document.getElementById("training-status-pill").textContent = UI.safeText(training.status, "idle");
    document.getElementById("playlist-state").textContent =
      training.playlist_mode ? `playlist ${Number(training.playlist_index || 0) + 1}/${training.playlist_total || 0}` : "single action";

    const promptText = streamReady
      ? UI.safeText(training.prompt, "等待开始训练")
      : UI.safeText(status.vision_boot_error || status.status, "视觉链路未就绪");
    document.getElementById("train-prompt").textContent = promptText;
    setMessage(promptText, training.status === "awaiting_return" || training.status === "awaiting_orientation" ? "warn" : "");

    const quality = status.pose_quality || {};
    document.getElementById("train-pill-row").innerHTML = `
      <span class="pill info">${UI.safeText(actionName)}</span>
      <span class="pill info">rep ${completed}/${target}</span>
      <span class="pill info">rest ${training.rest_remaining_seconds == null ? "-" : `${training.rest_remaining_seconds}s`}</span>
      <span class="pill info">${UI.safeText(status.actual_backend)}</span>
      <span class="pill ${quality.quality_ok ? "good" : "warn"}">${quality.quality_ok ? "keypoints OK" : "check keypoints"}</span>
    `;

    document.getElementById("live-grid").innerHTML = [
      badge("Metric", UI.formatNumber(training.current_metric ?? training.current_angle)),
      badge("PoseAngle", formatLivePoseAngle(status)),
      badge("LegVis", formatLegVisibility(status.rknn_live_target_leg_visibility)),
      badge("Pipeline", UI.safeText(status.status_rknn_pipeline || status.rknn_pipeline || status.actual_backend, "-")),
      badge("BBox", formatBBox(status.selected_yolo_bbox)),
      badge("Crop", formatBBox(status.rtmpose_expanded_bbox)),
      badge("Geom", formatGeometry(status)),
      badge("Cam", status.camera_live_ok ? "live" : shortValue(`stall ${status.camera_frame_age_ms ?? "-"}`, 18)),
      badge("Missing", shortValue((status.rknn_live_missing_keypoints || []).join(","), 18)),
      badge("Target", Array.isArray(training.target_range) ? `${UI.formatNumber(training.target_range[0])}-${UI.formatNumber(training.target_range[1])}` : "-"),
    ].join("");

    document.getElementById("timeline").innerHTML = renderTimeline(training.demo_plan, training);
    window.__LLM_STATUS__ = status.llm || {};
    dockState.reportContexts = collectContexts(status);
    ensureActiveReportKey();
    renderAIDock();
    renderSystemDock(system, status);

    renderCareDialog(training);
    syncRestMusic(training);

    const busy = activeStatuses.has(training.status);
    const careVisible = Boolean(training.care_dialog && training.care_dialog.visible);
    document.getElementById("single-btn").disabled = busy || !status.active_template || careVisible;
    document.getElementById("playlist-btn").disabled = busy || careVisible;
    document.getElementById("pause-btn").disabled = !["running", "paused"].includes(training.status) || careVisible;
    document.getElementById("stop-btn").disabled = !busy && !careVisible;
  }

  function badge(label, value) {
    return `<article class="badge-card"><strong>${label}</strong><span class="mono">${UI.safeText(value)}</span></article>`;
  }

  function shortValue(value, max = 18) {
    const text = UI.safeText(value, "-");
    return text.length > max ? `${text.slice(0, max)}...` : text;
  }

  function formatLivePoseAngle(status) {
    const value = status.rknn_live_target_angle_smoothed ?? status.rknn_live_target_angle_raw;
    if (value == null) return "-";
    const suffix = status.training && activeStatuses.has(status.training.status) ? "" : " ready";
    return `${UI.formatNumber(value, 1)}${suffix}`;
  }

  function formatLegVisibility(value) {
    if (!value || typeof value !== "object") return "-";
    const hip = value.hip ?? value.left_hip ?? value.right_hip;
    const knee = value.knee ?? value.left_knee ?? value.right_knee;
    const ankle = value.ankle ?? value.left_ankle ?? value.right_ankle;
    return `H${UI.formatNumber(hip, 2)} K${UI.formatNumber(knee, 2)} A${UI.formatNumber(ankle, 2)}`;
  }

  function formatBBox(value) {
    if (!Array.isArray(value) || value.length < 4) return "-";
    return value.map((item) => UI.formatNumber(item, 0)).join(",");
  }

  function formatGeometry(status) {
    const ok = status.pose_geometry_ok;
    const ratio = status.rtmpose_keypoint_bbox_ratio ?? status.pose_geometry_keypoint_bbox_ratio;
    const prefix = ok === false ? "bad" : ok === true ? "ok" : "-";
    return ratio == null ? prefix : `${prefix} r${UI.formatNumber(ratio, 2)}`;
  }

  function metricUnit(unit) {
    const text = UI.safeText(unit, "");
    if (text === "degree") return "°";
    if (text === "body_ratio") return "";
    return text ? ` ${text}` : "";
  }

  function formatMetricValue(value, unit) {
    const suffix = metricUnit(unit);
    return `${UI.formatNumber(value, 1)}${suffix}`;
  }

  function formatTargetRange(range, unit) {
    if (!Array.isArray(range) || range.length < 2) return "-";
    const suffix = metricUnit(unit);
    return `${UI.formatNumber(range[0], 1)}-${UI.formatNumber(range[1], 1)}${suffix}`;
  }

  function formatTut(training) {
    const current = UI.formatNumber(training.tut_seconds, 1);
    const target = UI.formatNumber(training.tut_target, 1);
    const missing = Number(training.missing_seconds || 0);
    return missing > 0 ? `${current}/${target}s · 还差 ${UI.formatNumber(missing, 1)}s` : `${current}/${target}s`;
  }

  function renderTimeline(plan, training) {
    const actions = plan && Array.isArray(plan.actions) ? plan.actions : [];
    return actions.map((action, index) => {
      const done = Array.isArray(training.playlist_reports) && training.playlist_reports.some((item) => item.action_id === action.action_id);
      const active = training.playlist_mode && Number(training.playlist_index) === index && activeStatuses.has(training.status);
      return `
        <article class="timeline-step ${done ? "done" : active ? "active" : ""}">
          <strong>${UI.safeText(action.action_name || UI.actionNames[action.action_id] || action.action_id)}</strong>
          <div class="summary-text">${UI.safeText(action.camera_prompt)}</div>
          <small>${done ? "已完成并生成报告" : active ? "当前动作" : "等待执行"}</small>
        </article>
      `;
    }).join("");
  }

  async function refresh() {
    try {
      const [status, system] = await Promise.all([
        UI.fetchJSON("/status"),
        UI.fetchJSON("/api/system/status"),
      ]);
      render(status, system);
    } catch (error) {
      setMessage(error.message || String(error), "bad");
      stopRestMusic();
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
      setMessage("完整三动作训练已开始。", "good");
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

  document.addEventListener("click", async (event) => {
    const careButton = event.target.closest("[data-care-response]");
    if (careButton) {
      const needsRest = careButton.getAttribute("data-care-response") === "yes";
      try {
        await UI.postJSON("/api/realtime/care_response", { needs_rest: needsRest });
        setMessage(needsRest ? "已进入休息流程。" : "继续训练。", needsRest ? "warn" : "good");
      } catch (error) {
        setMessage(error.message || String(error), "bad");
      } finally {
        refresh();
      }
      return;
    }
    const accordionButton = event.target.closest("[data-accordion-toggle]");
    if (accordionButton) {
      accordionButton.closest(".accordion-card").classList.toggle("is-collapsed");
      return;
    }
    const dockTrigger = event.target.closest("[data-dock-trigger]");
    if (dockTrigger) {
      togglePinnedDock(dockTrigger.getAttribute("data-dock-trigger"));
      return;
    }
    const dockPin = event.target.closest("[data-dock-pin]");
    if (dockPin) {
      togglePinnedDock(dockPin.getAttribute("data-dock-pin"));
      return;
    }
    const reportTab = event.target.closest("[data-report-tab-key]");
    if (reportTab) {
      dockState.activeReportKey = reportTab.getAttribute("data-report-tab-key");
      renderAIDock(true);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && dockCurrent()) {
      clearCollapseTimer();
      dockState.hovered = "";
      dockState.pinned = "";
      renderDockState();
    }
  });

  window.__REHAB_RERENDER_REPORTS__ = (context) => {
    if (context) {
      dockState.activeReportKey = contextKey(context);
    }
    renderAIDock(true);
  };

  refresh();
  setInterval(refresh, 1000);
})();
