(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");

  app.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">Training Cockpit</div>
          <h1>患者实时训练驾驶舱</h1>
          <p>支持单动作训练和三动作 playlist，实时显示动作提示、计数、休息倒计时和报告卡片。</p>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">首页</a>
          <a class="nav-link" href="/doctor">医生录制</a>
          <a class="nav-link active" href="/train">患者训练</a>
        </nav>
      </header>
      <section class="train-grid">
        <section class="panel">
          <div class="panel-header">
            <h3>实时预览</h3>
            <span id="train-vision-pill" class="pill info">等待连接</span>
          </div>
          <div class="preview-box">
            <img id="train-preview" src="/assets/placeholder.svg" alt="training preview">
            <div class="preview-overlay">
              <div class="big-number" id="train-prompt">等待开始训练</div>
              <div class="pills" id="train-pill-row"></div>
            </div>
          </div>
          <div class="metrics-grid" id="live-grid"></div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <h3>训练控制台</h3>
            <span class="meta">playlist / 单动作 / 暂停 / 结束</span>
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
          <div class="metrics-grid" id="feedback-grid"></div>
        </section>
      </section>
      <section class="panel">
        <div class="panel-header">
          <h3>训练编排时间线</h3>
          <span class="meta">rehab_demo_plan.yaml</span>
        </div>
        <div class="timeline" id="timeline"></div>
      </section>
      <section class="dashboard-grid">
        <section class="panel">
          <div class="panel-header">
            <h3>系统监控</h3>
            <span class="meta">板端运行状态</span>
          </div>
          <div class="system-grid" id="system-grid"></div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <h3>最近报告</h3>
            <span class="meta">训练完成后自动刷新</span>
          </div>
          <div id="report-panel"></div>
        </section>
      </section>
    </main>
  `;

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

  function render(status, system) {
    const training = status.training || {};
    const streamAvailable = Boolean(status.stream_available);
    const streamReady = Boolean(status.stream_ready);
    const visionError = UI.safeText(status.vision_boot_error, "");
    const requestedCameraDevice = UI.safeText(status.camera_device_requested || status.camera_device, "");
    const activeCameraDevice = UI.safeText(status.camera_device_active, "");
    const cameraAttempts = Array.isArray(status.camera_open_attempts) ? status.camera_open_attempts.join(", ") : "";
    const previewSource = UI.streamSource(status);
    const preview = document.getElementById("train-preview");
    if (preview.getAttribute("src") !== previewSource) {
      preview.src = previewSource;
    }
    const visionPill = document.getElementById("train-vision-pill");
    visionPill.className = `pill ${streamReady ? "good" : "warn"}`;
    visionPill.textContent = streamReady ? "实时流已连接" : streamAvailable ? "摄像头已打开，等待首帧" : "视觉链路未连接";
    document.getElementById("train-prompt").textContent = streamReady
      ? UI.safeText(training.prompt, "等待开始训练")
      : streamAvailable
        ? UI.safeText(status.status, `摄像头已打开，等待首帧${activeCameraDevice ? `：${activeCameraDevice}` : ""}`)
        : visionError || `Vision Preview Unavailable${cameraAttempts ? `，已尝试 ${cameraAttempts}` : ""}`;
    document.getElementById("train-pill-row").innerHTML = `
      <span class="pill info">${UI.safeText(training.current_action_name || training.action_id || "未开始")}</span>
      <span class="pill ${training.status === "running" ? "good" : "warn"}">${UI.safeText(training.status, "idle")}</span>
      <span class="pill info">rep ${training.completed_reps || 0}/${training.target_reps || 0}</span>
      <span class="pill info">rest ${training.rest_remaining_seconds == null ? "-" : `${training.rest_remaining_seconds}s`}</span>
    `;
    document.getElementById("live-grid").innerHTML = [
      UI.metricTile("Current Metric", UI.formatNumber(training.current_metric ?? training.current_angle), training.metric && training.metric.metric_name ? training.metric.metric_name : "target"),
      UI.metricTile("Target Range", Array.isArray(training.target_range) ? `${UI.formatNumber(training.target_range[0])} - ${UI.formatNumber(training.target_range[1])}` : "-", "当前阈值"),
      UI.metricTile("Invalid Attempts", UI.safeText(training.invalid_attempts, "0"), "未计数动作"),
      UI.metricTile(
        "Vision",
        streamReady ? "Ready" : streamAvailable ? "Waiting" : "Fallback",
        streamReady || streamAvailable
          ? `${activeCameraDevice || requestedCameraDevice || "camera opened"}${requestedCameraDevice && activeCameraDevice && requestedCameraDevice !== activeCameraDevice ? ` (requested ${requestedCameraDevice})` : ""}`
          : visionError || cameraAttempts || requestedCameraDevice || "请检查摄像头设备",
      ),
    ].join("");
    document.getElementById("feedback-grid").innerHTML = [
      UI.metricTile("TTS", UI.safeText(training.tts_text), "当前播报"),
      UI.metricTile("Motor", UI.safeText(training.motor_mock_pattern), "mock 震动模式"),
      UI.metricTile("Report", UI.safeText(training.report_file), "当前报告路径"),
    ].join("");
    document.getElementById("timeline").innerHTML = renderTimeline(training.demo_plan, training);
    document.getElementById("system-grid").innerHTML = UI.renderSystemStats(system);

    const context = UI.composeReportContext(
      training.report
        ? { report: training.report, report_file: training.report_file, summary_bundle: training.report.summary_bundle, report_card: training.report.report_card }
        : status.latest_report
    );
    window.__LAST_REPORT_CONTEXT__ = context;
    document.getElementById("report-panel").innerHTML = UI.reportCardHtml(context);

    document.getElementById("single-btn").disabled = training.status === "running" || !status.active_template;
    document.getElementById("playlist-btn").disabled = training.status === "running" || training.status === "resting";
    document.getElementById("pause-btn").disabled = !["running", "paused"].includes(training.status);
    document.getElementById("stop-btn").disabled = !["running", "paused", "resting"].includes(training.status);
  }

  function renderTimeline(plan, training) {
    const actions = plan && Array.isArray(plan.actions) ? plan.actions : [];
    return actions.map((action, index) => {
      const done = Array.isArray(training.playlist_reports) && training.playlist_reports.some((item) => item.action_id === action.action_id);
      const active = training.playlist_mode && Number(training.playlist_index) === index && ["running", "resting", "paused", "completed"].includes(training.status);
      const cls = done ? "done" : active ? "active" : "";
      return `
        <article class="timeline-step ${cls}">
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

  refresh();
  setInterval(refresh, 1000);
})();
