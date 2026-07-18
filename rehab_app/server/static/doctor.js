(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");
  let latestStatus = null;
  let evaluating = false;

  const actions = [
    ["sit_to_stand", "坐站训练", "坐站功能模板"],
    ["standing_hamstring_curl", "后勾腿", "腘绳肌屈曲模板"],
    ["seated_knee_raise", "坐姿抬膝", "髋膝抬高模板"],
  ];

  app.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">Doctor Capture Console</div>
          <h1>医生录入</h1>
          <p>录制标准动作模板，保存为 active template，供患者训练实时比对。</p>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">首页</a>
          <a class="nav-link active" href="/doctor">医生录入</a>
          <a class="nav-link" href="/train">患者训练</a>
        </nav>
      </header>

      <section class="doctor-grid">
        <section class="panel">
          <div class="panel-header">
            <h3>视觉采集</h3>
            <span class="pill info" id="vision-pill">等待视觉链路</span>
          </div>
          <div class="preview-box">
            <img id="preview" src="/assets/placeholder.svg" alt="doctor preview">
            <div class="preview-overlay">
              <div class="pills" id="preview-pills"></div>
              <div class="summary-text" id="quality-line">等待关键点质量</div>
            </div>
          </div>
          <div class="metrics-grid" id="preview-stats"></div>
        </section>

        <section class="panel">
          <div class="panel-header">
            <h3>模板录制控制</h3>
            <span class="meta">动作选择 / 录制 / 保存 / 评估</span>
          </div>

          <div class="action-tabs" id="action-tabs"></div>

          <div class="field-grid">
            <label>患者编号<input id="patient-id" value="patient_001"></label>
            <label>动作名称
              <input id="action-name" value="sit_to_stand" list="doctor-action-options">
              <datalist id="doctor-action-options">
                <option value="seated_knee_extension">坐姿伸膝</option>
                <option value="standing_hamstring_curl">站姿屈膝后勾腿</option>
                <option value="seated_knee_raise">坐姿抬膝</option>
                <option value="sit_to_stand">坐站训练</option>
                <option value="knee_flexion">屈膝</option>
              </datalist>
            </label>
            <label>侧别模式
              <select id="side-mode">
                <option value="auto">auto</option>
                <option value="left">left</option>
                <option value="right">right</option>
              </select>
            </label>
            <label>active template
              <input id="active-template" class="mono" value="未设置" readonly>
            </label>
          </div>

          <div class="doctor-command">
            <button id="start-template" class="record-orb" title="录入标准动作"></button>
            <div class="countdown-display">
              <div>
                <div class="countdown-number" id="doctor-countdown">--</div>
                <div class="summary-text" id="doctor-state">待命</div>
              </div>
            </div>
          </div>

          <div class="button-row">
            <button id="save-template">保存 active template</button>
            <button class="secondary" id="start-attempt">录入患者动作</button>
            <button class="secondary" id="save-attempt">保存 patient attempt</button>
            <button class="warn" id="evaluate">结束并评估</button>
          </div>
          <div class="button-row">
            <button class="plain" id="cancel">取消本轮录制</button>
            <button class="plain" id="clear">清空缓存</button>
          </div>

          <div id="message" class="message-box">等待操作。</div>
          <div class="status-badges" id="status-grid"></div>
        </section>
      </section>
    </main>
  `;

  function payload(recordRole) {
    return {
      patient_id: document.getElementById("patient-id").value.trim(),
      action_name: document.getElementById("action-name").value.trim(),
      side_mode: document.getElementById("side-mode").value,
      record_role: recordRole,
    };
  }

  function setMessage(text, tone = "") {
    const box = document.getElementById("message");
    box.className = `message-box ${tone}`.trim();
    box.textContent = text;
  }

  function renderActionTabs() {
    const selected = document.getElementById("action-name").value.trim();
    document.getElementById("action-tabs").innerHTML = actions.map(([id, name, desc]) => `
      <button class="action-tab ${selected === id ? "is-active" : ""}" data-action-tab="${id}">
        <strong>${name}</strong>
        <small>${desc}</small>
      </button>
    `).join("");
  }

  function updateButtons(status) {
    const recording = Boolean(status && status.recording);
    document.getElementById("start-template").disabled = recording || evaluating;
    document.getElementById("start-attempt").disabled = recording || evaluating;
    document.getElementById("save-template").disabled = !recording || status.current_record_role !== "doctor_template" || evaluating;
    document.getElementById("save-attempt").disabled = !recording || status.current_record_role !== "patient_attempt" || evaluating;
    document.getElementById("evaluate").disabled = evaluating || !status.active_template || !status.patient_attempt_file || recording;
    document.getElementById("start-template").classList.toggle("is-recording", recording && status.current_record_role === "doctor_template");
  }

  function render(status) {
    latestStatus = status;
    document.getElementById("preview").src = UI.streamSource(status);
    const streamReady = Boolean(status.stream_ready);
    const visionPill = document.getElementById("vision-pill");
    visionPill.className = `pill ${streamReady ? "good" : "warn"}`;
    visionPill.textContent = streamReady ? "视觉链路已连接" : "视觉链路等待中";

    const quality = status.pose_quality || {};
    const angle = status.smoothed_flexion_angle == null ? null : Number(status.smoothed_flexion_angle);
    const countdown = status.countdown_seconds ?? status.countdown ?? (angle == null ? "--" : `${UI.formatNumber(angle, 0)}°`);
    document.getElementById("doctor-countdown").textContent = countdown;
    document.getElementById("doctor-state").textContent = status.recording ? UI.safeText(status.current_record_role_label, "录制中") : "待命";
    document.getElementById("quality-line").textContent = UI.safeText(quality.quality_message, "等待关键点质量");
    document.getElementById("preview-pills").innerHTML = `
      <span class="pill info">${UI.safeText(status.selected_side_label, "auto")}</span>
      <span class="pill info">${UI.safeText(status.actual_backend, "pose")}</span>
      <span class="pill ${status.recording ? "warn" : "good"}">${status.recording ? "录制中" : "待命"}</span>
      <span class="pill ${quality.quality_ok ? "good" : "warn"}">${quality.quality_ok ? "关键点 OK" : "关键点检查"}</span>
    `;

    document.getElementById("preview-stats").innerHTML = [
      UI.metricTile("ROM", UI.formatNumber(status.current_rom), "当前录制跨度"),
      UI.metricTile("Angle", angle == null ? "-" : `${UI.formatNumber(angle)}°`, "实时角度"),
      UI.metricTile("Pose FPS", UI.formatNumber(status.pose_fps, 2), "识别帧率"),
    ].join("");

    document.getElementById("active-template").value = UI.safeText(status.active_template && status.active_template.template_file, "未设置");
    document.getElementById("status-grid").innerHTML = [
      badge("状态", UI.safeText(status.status)),
      badge("角色", UI.safeText(status.current_record_role_label)),
      badge("后端", UI.safeText(status.actual_backend)),
      badge("attempt", UI.safeText(status.patient_attempt_file)),
      badge("report", UI.safeText(status.evaluation_report_file)),
      badge("缺失点", Array.isArray(quality.missing_keypoints) ? quality.missing_keypoints.join(", ") || "-" : "-"),
    ].join("");

    window.__LLM_STATUS__ = status.llm || {};
    renderActionTabs();
    updateButtons(status);
  }

  function badge(label, value) {
    return `<article class="badge-card"><strong>${label}</strong><span class="mono">${UI.safeText(value)}</span></article>`;
  }

  async function refresh() {
    try {
      const status = await UI.fetchJSON("/status");
      render(status);
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    }
  }

  async function startRecording(role) {
    try {
      const result = await UI.postJSON("/api/start", payload(role));
      setMessage(result.message || "已开始录制。", "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  }

  async function saveRecording(role) {
    try {
      const result = await UI.postJSON("/api/save", { record_role: role });
      setMessage(result.message || "保存完成。", "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  }

  async function evaluateAttempt() {
    evaluating = true;
    updateButtons(latestStatus || {});
    try {
      const result = await UI.postJSON("/api/evaluate", {
        action_id: latestStatus && latestStatus.action_id,
        attempt_file: latestStatus && latestStatus.patient_attempt_file,
      });
      setMessage(`评估完成，报告已生成。\n${UI.safeText(result.report_file)}\n请到患者训练页查看报告、AI 建议和小爱问答。`, "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      evaluating = false;
      refresh();
    }
  }

  document.getElementById("action-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action-tab]");
    if (!button) return;
    document.getElementById("action-name").value = button.getAttribute("data-action-tab");
    renderActionTabs();
  });
  document.getElementById("action-name").addEventListener("input", renderActionTabs);
  document.getElementById("start-template").addEventListener("click", () => startRecording("doctor_template"));
  document.getElementById("start-attempt").addEventListener("click", () => startRecording("patient_attempt"));
  document.getElementById("save-template").addEventListener("click", () => saveRecording("doctor_template"));
  document.getElementById("save-attempt").addEventListener("click", () => saveRecording("patient_attempt"));
  document.getElementById("evaluate").addEventListener("click", evaluateAttempt);
  document.getElementById("cancel").addEventListener("click", async () => {
    try {
      const result = await UI.postJSON("/api/cancel", {});
      setMessage(result.message || "已取消。", "warn");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });
  document.getElementById("clear").addEventListener("click", async () => {
    try {
      const result = await UI.postJSON("/api/clear", { clear_export: true });
      setMessage(result.message || "已清空。", "warn");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  renderActionTabs();
  refresh();
  setInterval(refresh, 1200);
})();
