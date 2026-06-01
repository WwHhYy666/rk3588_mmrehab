(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");
  let latestStatus = null;
  let evaluating = false;

  app.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">Doctor Workspace</div>
          <h1>医生标准动作录制台</h1>
          <p>录入标准模板、录制患者动作、立即评估并导出报告卡片。</p>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">首页</a>
          <a class="nav-link active" href="/doctor">医生录制</a>
          <a class="nav-link" href="/train">患者训练</a>
        </nav>
      </header>
      <section class="doctor-grid">
        <section class="panel">
          <div class="panel-header">
            <h3>实时预览</h3>
            <span class="pill info" id="vision-pill">等待状态</span>
          </div>
          <div class="preview-box">
            <img id="preview" src="/assets/placeholder.svg" alt="preview">
            <div class="preview-overlay">
              <div class="big-number" id="live-angle">-</div>
              <div class="pills" id="preview-pills"></div>
            </div>
          </div>
          <div class="metrics-grid" id="preview-stats"></div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <h3>工作流控制</h3>
            <span class="meta">医生模板 -> 患者动作 -> 评估报告</span>
          </div>
          <div class="field-grid">
            <label>患者编号<input id="patient-id" value="patient_001"></label>
            <label>动作名称
              <input id="action-name" value="seated_knee_extension" list="doctor-action-options">
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
            <label>当前 active template
              <input id="active-template" class="mono" value="未设置" readonly>
            </label>
          </div>
          <div class="button-row">
            <button id="start-template">录入标准动作</button>
            <button id="save-template">保存为 active template</button>
          </div>
          <div class="button-row">
            <button class="secondary" id="start-attempt">录入患者动作</button>
            <button class="secondary" id="save-attempt">保存 patient attempt</button>
            <button class="warn" id="evaluate">结束并评估</button>
          </div>
          <div class="button-row">
            <button class="plain" id="cancel">取消本轮录制</button>
            <button class="plain" id="clear">清空缓存</button>
          </div>
          <div id="message" class="message-box">等待操作。</div>
          <div class="metrics-grid" id="status-grid"></div>
          <div class="panel">
            <div class="panel-header">
              <h3>模板列表</h3>
              <span class="meta">runtime/active_templates.json</span>
            </div>
            <div id="template-list" class="card-grid"></div>
          </div>
        </section>
      </section>
      <section class="dashboard-grid">
        <section class="panel">
          <div class="panel-header">
            <h3>评估结果卡片</h3>
            <span class="meta">真实 report.json + 模板化总结</span>
          </div>
          <div id="report-panel"></div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <h3>依赖与链路状态</h3>
            <span class="meta">缺失时自动降级</span>
          </div>
          <div class="capability-grid" id="cap-grid"></div>
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

  function updateButtons(status) {
    const recording = Boolean(status && status.recording);
    document.getElementById("start-template").disabled = recording || evaluating;
    document.getElementById("start-attempt").disabled = recording || evaluating;
    document.getElementById("save-template").disabled = !recording || status.current_record_role !== "doctor_template" || evaluating;
    document.getElementById("save-attempt").disabled = !recording || status.current_record_role !== "patient_attempt" || evaluating;
    document.getElementById("evaluate").disabled = evaluating || !status.active_template || !status.patient_attempt_file || recording;
  }

  function render(status) {
    latestStatus = status;
    document.getElementById("preview").src = UI.streamSource(status);
    document.getElementById("vision-pill").className = `pill ${status.stream_available ? "good" : "warn"}`;
    document.getElementById("vision-pill").textContent = status.stream_available ? "视觉链路已连接" : "当前使用降级预览";
    document.getElementById("live-angle").textContent = status.smoothed_flexion_angle == null ? "-" : `${UI.formatNumber(status.smoothed_flexion_angle)}°`;
    document.getElementById("preview-pills").innerHTML = `
      <span class="pill info">${UI.safeText(status.side_mode_label)}</span>
      <span class="pill info">${UI.safeText(status.selected_side_label)}</span>
      <span class="pill info">${UI.safeText(status.actual_backend)}</span>
      <span class="pill ${status.recording ? "warn" : "good"}">${status.recording ? "录制中" : "待命"}</span>
    `;
    const quality = status.pose_quality || {};
    document.getElementById("preview-stats").innerHTML = [
      UI.metricTile("ROM", UI.formatNumber(status.current_rom), "当前录制跨度"),
      UI.metricTile("Visibility", UI.formatNumber(status.visibility_avg, 2), "关键点可见性"),
      UI.metricTile("Pose FPS", UI.formatNumber(status.pose_fps, 2), "实时姿态帧率"),
      UI.metricTile("Pose Backend", UI.safeText(status.actual_backend), status.fallback_used ? `fallback: ${UI.safeText(status.backend_error_message)}` : `requested: ${UI.safeText(status.requested_backend)}`),
      UI.metricTile("关键点质量", quality.quality_ok ? "OK" : "Check", UI.safeText(quality.quality_message)),
      UI.metricTile("机位要求", status.actual_backend === "rknn" ? "2D 侧身固定" : "MediaPipe", status.actual_backend === "rknn" ? "单人训练模式" : "默认路线"),
    ].join("");
    document.getElementById("active-template").value = UI.safeText(status.active_template && status.active_template.template_file, "未设置");
    document.getElementById("status-grid").innerHTML = [
      UI.metricTile("状态", UI.safeText(status.status)),
      UI.metricTile("录制角色", UI.safeText(status.current_record_role_label)),
      UI.metricTile("patient attempt", UI.safeText(status.patient_attempt_file), "最近患者动作"),
      UI.metricTile("report", UI.safeText(status.evaluation_report_file), "最近评估报告"),
      UI.metricTile("镜头侧别", UI.safeText(status.selected_side_label)),
      UI.metricTile("角度来源", UI.safeText(status.selected_source_label)),
      UI.metricTile("缺失关键点", Array.isArray(quality.missing_keypoints) ? quality.missing_keypoints.join(", ") || "-" : "-"),
      UI.metricTile("训练者选择", "单人模式", UI.safeText(quality.selected_person_reason, "默认选择训练者")),
      UI.metricTile("NPU 锁定", UI.safeText(quality.locked_side, "-"), UI.safeText(quality.side_lock_reason, "-")),
      UI.metricTile("NPU 稳定", quality.pose_stabilized ? "开启" : "-", `保持 ${(quality.held_keypoints || []).length}，拒跳 ${(quality.jump_rejected || []).length}，换腿 ${quality.side_switch_blocked ? "是" : "否"}`),
    ].join("");
    document.getElementById("template-list").innerHTML = renderTemplateRegistry(status);
    document.getElementById("cap-grid").innerHTML = UI.renderCaps(status.capabilities);
    const latestContext = UI.composeReportContext(status.latest_report);
    window.__LLM_STATUS__ = status.llm || {};
    window.__LAST_REPORT_CONTEXT__ = latestContext;
    UI.renderReportPanel(latestContext);
    updateButtons(status);
  }

  function renderTemplateRegistry(status) {
    const registry = status.active_templates_by_backend || (status.active_templates && status.active_templates.by_backend) || {};
    const labels = { mediapipe: "CPU / MediaPipe", rknn: "NPU / RKNN" };
    const blocks = ["mediapipe", "rknn"].map((backend) => {
      const entries = registry[backend] || {};
      const cards = Object.entries(entries).map(([key, item]) => `
        <article class="summary-card">
          <strong>${UI.safeText(UI.actionNames[key] || key)}</strong>
          <div class="summary-text mono">${UI.safeText(item && item.template_file)}</div>
        </article>
      `).join("") || `<article class="summary-card"><strong>暂无模板</strong><div class="summary-text">切换到 ${UI.safeText(backend)} 后录制一套即可。</div></article>`;
      return `
        <section>
          <div class="section-head">
            <strong>${labels[backend] || backend}</strong>
            <span class="pill ${status.actual_backend === backend ? "good" : "info"}">${status.actual_backend === backend ? "当前后端" : "可复用"}</span>
          </div>
          <div class="card-grid">${cards}</div>
        </section>
      `;
    });
    return blocks.join("");
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
      const context = UI.composeReportContext(result);
      window.__LAST_REPORT_CONTEXT__ = context;
      UI.renderReportPanel(context, { force: true });
      setMessage(`评估完成。\n${UI.safeText(result.report_file)}`, "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      evaluating = false;
      refresh();
    }
  }

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

  refresh();
  setInterval(refresh, 1200);
})();
