(function () {
  const actionNames = {
    knee_flexion: "屈膝",
    seated_knee_extension: "坐姿伸膝",
    seated_knee_raise: "坐姿抬膝",
    standing_hamstring_curl: "站姿屈膝后勾腿",
    sit_to_stand: "坐站训练",
  };

  function safeText(value, fallback = "-") {
    return value === null || value === undefined || value === "" ? fallback : String(value);
  }

  function formatNumber(value, digits = 1, unit = "") {
    const number = Number(value);
    if (!Number.isFinite(number)) return "-";
    return `${number.toFixed(digits)}${unit}`;
  }

  function toneForCapability(item) {
    if (!item) return "warn";
    return item.available ? "good" : "warn";
  }

  function toneForError(errorCode) {
    if (!errorCode || errorCode === "OK") return "good";
    if (errorCode === "ROM_LOW" || errorCode === "TUT_LOW") return "warn";
    return "bad";
  }

  async function fetchJSON(url) {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || `Request failed: ${response.status}`);
    }
    return data;
  }

  async function postJSON(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || `Request failed: ${response.status}`);
    }
    return data;
  }

  function renderCaps(capabilities) {
    return Object.entries(capabilities || {})
      .map(([key, item]) => `
        <article class="capability-card">
          <strong>${key}</strong>
          <span>${item && item.available ? "Ready" : "Fallback"}</span>
          <small>${safeText(item && item.note)}</small>
        </article>
      `)
      .join("");
  }

  function streamSource(status) {
    return status && status.stream_ready ? "/stream.mjpg" : "/assets/placeholder.svg";
  }

  function composeReportContext(source) {
    if (!source) return null;
    const report = source.report || source;
    const reportCard = source.report_card || report.report_card || null;
    const summaryBundle = source.summary_bundle || report.summary_bundle || null;
    if (!report) return null;
    return { report, reportCard, summaryBundle, reportFile: source.report_file || report.report_file || "-" };
  }

  function metricTile(label, value, extra = "", tone = "") {
    return `
      <article class="metric-card ${tone}">
        <strong>${label}</strong>
        <span>${safeText(value)}</span>
        <small>${extra || ""}</small>
      </article>
    `;
  }

  function reportCardHtml(context) {
    if (!context) {
      return `<div class="empty">暂时还没有报告。完成一次患者评估后，这里会显示图文卡片。</div>`;
    }
    const report = context.report || {};
    const reportCard = context.reportCard || {};
    const metrics = report.metrics || {};
    const errors = report.errors || {};
    const summary = context.summaryBundle || {};
    const tone = toneForError(errors.primary_error);
    return `
      <section class="report-card">
        <div class="report-head">
          <div>
            <div class="eyebrow">Report Card</div>
            <h3>${safeText(reportCard.title, "Rehab Report")}</h3>
            <p>${safeText(reportCard.subtitle)} · ${safeText(report.meta && report.meta.evaluated_at)}</p>
          </div>
          <div class="report-status ${tone}">${safeText(errors.primary_error, "OK")}</div>
        </div>
        <div class="report-grid">
          ${metricTile("ROM", `${formatNumber(metrics.rom && metrics.rom.actual)} / ${formatNumber(metrics.rom && metrics.rom.target)}`, "角度幅度")}
          ${metricTile("TUT", `${formatNumber(metrics.tut && metrics.tut.actual)} / ${formatNumber(metrics.tut && metrics.tut.target)}`, "保持时间")}
          ${metricTile("Speed", formatNumber(metrics.speed && metrics.speed.ratio, 2), "速度比例")}
          ${metricTile("DTW", formatNumber(metrics.dtw && metrics.dtw.normalized_distance, 2), "轨迹距离")}
        </div>
        <div class="summary-grid">
          <article class="summary-card">
            <strong>Doctor Summary</strong>
            <div class="summary-text">${safeText(summary.doctor_summary, "暂无总结")}</div>
          </article>
          <article class="summary-card">
            <strong>Patient Summary</strong>
            <div class="summary-text">${safeText(summary.patient_summary, "暂无总结")}</div>
          </article>
        </div>
        <div class="report-actions">
          <span class="pill ${tone}">next: ${safeText(summary.next_step, "继续训练")}</span>
          <span class="pill info mono">${safeText(context.reportFile)}</span>
          <button class="secondary" data-export-report="1">导出报告卡片</button>
        </div>
      </section>
    `;
  }

  function renderSystemStats(system) {
    const cpu = system.cpu || {};
    const memory = system.memory || {};
    const temperature = system.temperature || {};
    const npu = system.npu || {};
    const pose = system.pose_fps || {};
    const backend = system.pose_backend || {};
    const quality = backend.quality || {};
    const perf = backend.performance || {};
    return [
      metricTile("CPU", cpu.available ? `${formatNumber(cpu.percent)}%` : safeText(cpu.note), "处理器占用"),
      metricTile("Memory", memory.available ? `${formatNumber(memory.percent)}%` : safeText(memory.note), memory.available ? `${formatNumber(memory.used_mb, 0)} / ${formatNumber(memory.total_mb, 0)} MB` : ""),
      metricTile("Temp", temperature.available ? `${formatNumber(temperature.max_celsius)} °C` : safeText(temperature.note), "板端温度"),
      metricTile("NPU", npu.available ? safeText(npu.percent == null ? npu.raw : `${formatNumber(npu.percent)}%`) : safeText(npu.note), "加速器"),
      metricTile("Pose FPS", pose.available ? `${formatNumber(pose.fps, 2)} FPS` : safeText(pose.note), "姿态识别帧率"),
      metricTile("Pose Backend", safeText(backend.actual_backend), backend.fallback_used ? `fallback: ${safeText(backend.backend_error_message)}` : `requested: ${safeText(backend.requested_backend)}`),
      metricTile("Infer ms", perf.inference_ms == null ? "-" : `${formatNumber(perf.inference_ms, 2)} ms`, "RKNN NPU inference"),
      metricTile("Post ms", perf.postprocess_ms == null ? "-" : `${formatNumber(perf.postprocess_ms, 2)} ms`, "YOLO postprocess"),
      metricTile("JPEG ms", perf.jpeg_encode_ms == null ? "-" : `${formatNumber(perf.jpeg_encode_ms, 2)} ms`, "stream encode"),
      metricTile("Pose ms", perf.total_pose_ms == null ? "-" : `${formatNumber(perf.total_pose_ms, 2)} ms`, "pre + infer + post + draw"),
      metricTile("Keypoint Quality", quality.quality_ok ? "OK" : "Check", safeText(quality.quality_message)),
      metricTile("Person Count", safeText(quality.person_count, "0"), quality.multi_person_warning ? "请保持训练者单独入镜" : safeText(quality.selected_person_reason)),
    ].join("");
  }

  function exportReportCard(context) {
    if (!context) return;
    const report = context.report || {};
    const summary = context.summaryBundle || {};
    const title = (context.reportCard && context.reportCard.title) || "Rehab Report";
    const error = safeText(report.errors && report.errors.primary_error, "OK");
    const metrics = report.metrics || {};
    const canvas = document.createElement("canvas");
    canvas.width = 1400;
    canvas.height = 900;
    const ctx = canvas.getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, "#102344");
    gradient.addColorStop(0.55, "#0b1224");
    gradient.addColorStop(1, "#07111f");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "rgba(85,214,255,0.12)";
    ctx.fillRect(60, 60, 1280, 780);
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.strokeRect(60, 60, 1280, 780);

    ctx.fillStyle = "#55d6ff";
    ctx.font = "600 22px Segoe UI";
    ctx.fillText("RK3588 Rehab Report", 96, 120);
    ctx.fillStyle = "#edf5ff";
    ctx.font = "700 44px Segoe UI";
    ctx.fillText(title, 96, 180);
    ctx.fillStyle = error === "OK" ? "#bcffe0" : "#ffe0a6";
    ctx.font = "800 84px Segoe UI";
    ctx.fillText(error, 980, 168);

    ctx.fillStyle = "#96adcb";
    ctx.font = "400 22px Segoe UI";
    ctx.fillText(`Patient: ${safeText(report.meta && report.meta.patient_id)}`, 96, 230);
    ctx.fillText(`Action: ${safeText(report.meta && report.meta.action_name)}`, 96, 266);
    ctx.fillText(`Evaluated: ${safeText(report.meta && report.meta.evaluated_at)}`, 96, 302);

    const metricRows = [
      ["ROM", `${formatNumber(metrics.rom && metrics.rom.actual)} / ${formatNumber(metrics.rom && metrics.rom.target)}`],
      ["TUT", `${formatNumber(metrics.tut && metrics.tut.actual)} / ${formatNumber(metrics.tut && metrics.tut.target)}`],
      ["Speed", formatNumber(metrics.speed && metrics.speed.ratio, 2)],
      ["DTW", formatNumber(metrics.dtw && metrics.dtw.normalized_distance, 2)],
    ];
    let x = 96;
    metricRows.forEach(([label, value], index) => {
      const width = 280;
      const y = 370;
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fillRect(x + index * 300, y, width, 150);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.strokeRect(x + index * 300, y, width, 150);
      ctx.fillStyle = "#96adcb";
      ctx.font = "600 18px Segoe UI";
      ctx.fillText(label, x + index * 300 + 20, y + 38);
      ctx.fillStyle = "#edf5ff";
      ctx.font = "700 38px Segoe UI";
      ctx.fillText(value, x + index * 300 + 20, y + 95);
    });

    ctx.fillStyle = "#edf5ff";
    ctx.font = "700 24px Segoe UI";
    ctx.fillText("Doctor Summary", 96, 590);
    ctx.fillText("Patient Summary", 96, 730);
    ctx.font = "400 22px Segoe UI";
    ctx.fillStyle = "#c8d6ea";
    wrapText(ctx, safeText(summary.doctor_summary), 96, 630, 1180, 32);
    wrapText(ctx, safeText(summary.patient_summary), 96, 770, 1180, 32);

    const url = canvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = url;
    link.download = `${safeText(report.meta && report.meta.patient_id, "patient")}_${safeText(report.meta && report.meta.action_id, "report")}.png`;
    link.click();
  }

  function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    const characters = String(text || "").split("");
    let line = "";
    let offset = 0;
    characters.forEach((character) => {
      const next = line + character;
      if (ctx.measureText(next).width > maxWidth) {
        ctx.fillText(line, x, y + offset);
        line = character;
        offset += lineHeight;
      } else {
        line = next;
      }
    });
    if (line) ctx.fillText(line, x, y + offset);
  }

  document.addEventListener("click", (event) => {
    const exportButton = event.target.closest("[data-export-report]");
    if (!exportButton) return;
    exportReportCard(window.__LAST_REPORT_CONTEXT__ || null);
  });

  window.RehabUI = {
    actionNames,
    safeText,
    formatNumber,
    fetchJSON,
    postJSON,
    renderCaps,
    renderSystemStats,
    composeReportContext,
    reportCardHtml,
    exportReportCard,
    streamSource,
    toneForCapability,
    toneForError,
    metricTile,
  };
})();
