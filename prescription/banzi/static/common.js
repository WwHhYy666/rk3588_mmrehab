(function () {
  const actionNames = {
    knee_flexion: "屈膝",
    seated_knee_extension: "坐姿伸膝",
    seated_knee_raise: "坐姿抬膝",
    standing_hamstring_curl: "站姿屈膝后勾腿",
    sit_to_stand: "坐站训练",
  };

  const aiStateByReport = new Map();
  let aiQuestionComposing = false;

  function safeText(value, fallback = "-") {
    return value === null || value === undefined || value === "" ? fallback : String(value);
  }

  function escapeHtml(value, fallback = "-") {
    return safeText(value, fallback)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
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
    if (!response.ok || data.ok === false) throw new Error(data.message || data.error || `Request failed: ${response.status}`);
    return data;
  }

  async function postJSON(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(data.message || data.error || `Request failed: ${response.status}`);
    return data;
  }

  function renderCaps(capabilities) {
    return Object.entries(capabilities || {})
      .map(([key, item]) => `
        <article class="capability-card">
          <strong>${escapeHtml(key)}</strong>
          <span>${item && item.available ? "Ready" : "Fallback"}</span>
          <small>${escapeHtml(item && item.note)}</small>
        </article>
      `)
      .join("");
  }

  function streamSource(status) {
    return status && status.stream_ready ? "/stream.mjpg" : "/assets/placeholder.svg";
  }

  function inferActionIdFromReportFile(file) {
    const normalized = safeText(file, "").replace(/\\/g, "/");
    const name = normalized.split("/").pop() || "";
    const ids = Object.keys(actionNames).sort((a, b) => b.length - a.length);
    return ids.find((actionId) => name.endsWith(".json") && (name.startsWith(`report_${actionId}_`) || name.includes(`_${actionId}_`))) || "";
  }

  function normalizeActionId(value, reportFile = "") {
    const text = safeText(value, "").trim();
    if (text && actionNames[text]) return text;
    return inferActionIdFromReportFile(reportFile);
  }

  function composeReportContext(source) {
    if (!source) return null;
    const report = source.report || source;
    if (!report) return null;
    const meta = report.meta || {};
    const runtimeMeta = report.runtime_meta || {};
    const reportCard = source.report_card || source.reportCard || report.report_card || null;
    const summaryBundle = source.summary_bundle || source.summaryBundle || report.summary_bundle || null;
    const reportFile = source.report_file || source.reportFile || report.report_file || "-";
    const actionId = normalizeActionId(report.action_id || meta.action_id || runtimeMeta.action_id || source.action_id || source.actionId, reportFile);
    const actionName = report.action_name || meta.action_name || source.action_name || source.actionName || actionNames[actionId] || actionId || "训练报告";
    return { report, reportCard, summaryBundle, reportFile, actionId, actionName };
  }

  function reportKey(context) {
    return safeText(context && (context.reportFile || context.actionId), "latest");
  }

  function reportId(context) {
    const file = safeText(context && context.reportFile, "");
    if (!file || file === "-") return context && context.actionId ? `latest:${context.actionId}` : "latest";
    const normalized = file.replace(/\\/g, "/");
    return normalized.split("/").pop() || (context && context.actionId ? `latest:${context.actionId}` : "latest");
  }

  function reportContextStore() {
    if (!window.__REPORT_CONTEXTS_BY_KEY__) window.__REPORT_CONTEXTS_BY_KEY__ = {};
    return window.__REPORT_CONTEXTS_BY_KEY__;
  }

  function registerReportContext(context) {
    if (!context) return;
    reportContextStore()[reportKey(context)] = context;
    window.__LAST_REPORT_CONTEXT__ = context;
  }

  function contextForElement(element) {
    const root = element && element.closest("[data-report-key], [data-ai-report]");
    const key = root && (root.getAttribute("data-report-key") || root.getAttribute("data-ai-report"));
    if (key && reportContextStore()[key]) return reportContextStore()[key];
    return window.__LAST_REPORT_CONTEXT__ || null;
  }

  function aiStateFor(context) {
    const key = reportKey(context);
    if (!aiStateByReport.has(key)) {
      aiStateByReport.set(key, { status: "idle", summary: null, answer: null, error: "", question: "" });
    }
    return aiStateByReport.get(key);
  }

  function metricTile(label, value, extra = "", tone = "") {
    return `
      <article class="metric-card ${tone}">
        <strong>${escapeHtml(label)}</strong>
        <span>${escapeHtml(value)}</span>
        <small>${escapeHtml(extra, "")}</small>
      </article>
    `;
  }

  function listHtml(items) {
    const list = Array.isArray(items) ? items : [];
    if (!list.length) return "暂无";
    return list.map((item) => `<div class="ai-list-item">• ${escapeHtml(item)}</div>`).join("");
  }

  function selectedAttemptLabel(metrics) {
    if (!metrics || metrics.source === "full_session") return "整段汇总";
    const attempt = metrics.attempt_index ? `第 ${metrics.attempt_index} 次尝试` : "选中样本";
    if (metrics.source === "best_correct") {
      const rep = metrics.rep_index ? ` / 第 ${metrics.rep_index} 次计数` : "";
      return `${attempt}${rep} / 最佳正确动作`;
    }
    if (metrics.source === "representative_wrong") return `${attempt} / 代表性错误动作`;
    return attempt;
  }
  function providerBadgeText(llm) {
    const expected = llm.active_provider || llm.provider || "auto";
    const last = llm.last_active_provider || "暂无";
    const qwen = llm.rkllm_server_reachable ? "Qwen ready" : "Qwen off";
    return `配置 ${safeText(llm.provider, "auto")} / 预计 ${expected} / 上次 ${last} / ${qwen}`;
  }

  function reportCardHtml(context) {
    if (!context) return `<div class="empty">暂时还没有报告。完成一次训练后，这里会显示图文卡片。</div>`;
    registerReportContext(context);
    const report = context.report || {};
    const reportCard = context.reportCard || {};
    const metrics = report.report_card_metrics || report.metrics || {};
    const errors = report.errors || {};
    const summary = context.summaryBundle || {};
    const qualityAttempts = Array.isArray(report.quality_attempts) && report.quality_attempts.length ? report.quality_attempts : (Array.isArray(report.reps) ? report.reps : []);
    const tone = toneForError(errors.primary_error);
    const key = reportKey(context);
    const title = reportCard.title || report.action_name || (report.meta && report.meta.action_name) || "康复报告";
    const sampleLabel = selectedAttemptLabel(metrics);
    const subtitle = reportCard.subtitle || `${(report.meta && report.meta.patient_id) || "训练报告"} · ${sampleLabel}`;
    const evaluatedAt = report.evaluated_at || (report.meta && report.meta.evaluated_at) || "";
    return `
      <section class="report-card" data-report-key="${escapeHtml(key)}">
        <div class="report-head">
          <div>
            <div class="eyebrow">训练报告</div>
            <h3>${escapeHtml(title)}</h3>
            <p>${escapeHtml(subtitle)} ${evaluatedAt ? `· ${escapeHtml(evaluatedAt)}` : ""}</p>
          </div>
          <div class="report-status ${tone}">${escapeHtml(errors.primary_error, "OK")}</div>
        </div>
        <div class="report-grid">
          ${metricTile("ROM", `${formatNumber(metrics.rom && metrics.rom.actual)} / ${formatNumber(metrics.rom && metrics.rom.target)}`, "角度幅度")}
          ${metricTile("TUT", `${formatNumber(metrics.tut && metrics.tut.actual)} / ${formatNumber(metrics.tut && metrics.tut.target)}`, "保持时间")}
          ${metricTile("Speed", formatNumber(metrics.speed && metrics.speed.ratio, 2), "速度比例")}
          ${metricTile("DTW", formatNumber(metrics.dtw && metrics.dtw.normalized_distance, 2), "轨迹距离")}
        </div>
        <div class="summary-grid">
          <article class="summary-card"><strong>医生总结</strong><div class="summary-text">${escapeHtml(summary.doctor_summary, "暂无总结")}</div></article>
          <article class="summary-card"><strong>患者总结</strong><div class="summary-text">${escapeHtml(summary.patient_summary, "暂无总结")}</div></article>
        </div>
        <div class="report-actions">
          <span class="pill ${tone}">下一步：${escapeHtml(summary.next_step, "继续训练")}</span>
          <span class="pill info mono">${escapeHtml(context.reportFile)}</span>
          <button class="secondary" data-export-report="1">导出报告卡片</button>
        </div>
        ${aiPanelHtml(context)}
      </section>
    `;
  }


  function qualitySummaryHtml(report, attempts) {
    const overall = report.overall_quality;
    const model = report.quality_model || {};
    const rows = Array.isArray(attempts) ? attempts.slice(0, 12) : [];
    const modelText = model.backend ? `${safeText(model.backend)}${model.last_score_time_ms == null ? "" : ` / ${formatNumber(model.last_score_time_ms, 1)} ms`}` : (model.available === false ? "model unavailable" : "waiting");
    const attemptHtml = rows.length
      ? rows.map((item, index) => {
          const score = item.quality_score == null ? "未评分" : `${formatNumber(item.quality_score, 1)} 分`;
          const grade = item.quality_grade ? ` / ${safeText(item.quality_grade)}` : "";
          const reason = item.reason || item.primary_error || (item.countable ? "通过" : "未通过");
          const label = item.attempt_index || index + 1;
          return `<div class="ai-list-item"><strong>#${escapeHtml(label)}</strong> ${escapeHtml(score)}${escapeHtml(grade)} · ${escapeHtml(reason)} · ${item.countable ? "计数" : "纠错"}</div>`;
        }).join("")
      : `<div class="ai-list-item">暂无逐次质量评分。没有模型文件时训练和报告仍会正常生成。</div>`;
    return `
      <div class="summary-grid">
        <article class="summary-card">
          <strong>动作质量评分</strong>
          <div class="summary-text">平均分：${overall == null ? "未评分" : `${formatNumber(overall, 1)} 分`} · ${escapeHtml(modelText)}</div>
          <div class="summary-text">${attemptHtml}</div>
        </article>
      </div>
    `;
  }

  function aiPanelHtml(context) {
    const state = aiStateFor(context);
    const llm = window.__LLM_STATUS__ || {};
    const busy = state.status === "loading-summary" || state.status === "loading-answer";
    const canSpeakSummary = Boolean(state.summary && state.summary.spoken_text);
    const canSpeakAnswer = Boolean(state.answer && state.answer.spoken_text);
    const providerTone = (llm.active_provider || llm.provider) === "echo" ? "warn" : "info";
    const summary = state.summary || {};
    const calorie = summary.calorie_estimate || {};
    const sourceFile = summary.source_report_file || (state.answer && state.answer.source_report_file) || (context && context.reportFile);
    const sourceTime = summary.source_evaluated_at || summary.source_report_mtime || (state.answer && (state.answer.source_evaluated_at || state.answer.source_report_mtime)) || "";
    return `
      <section class="summary-card ai-console" data-ai-report="${escapeHtml(reportKey(context))}">
        <div class="ai-console-head">
          <div>
            <div class="eyebrow">辅助建议</div>
            <strong>AI 训练建议</strong>
            <div class="summary-text mono">基于 ${escapeHtml(sourceFile, "latest")} ${sourceTime ? `· ${escapeHtml(sourceTime)}` : ""}</div>
          </div>
          <span class="pill ${providerTone}">${escapeHtml(providerBadgeText(llm))}</span>
        </div>
        <div class="button-row">
          <button class="secondary" data-ai-summary="1" ${busy ? "disabled" : ""}>${state.status === "loading-summary" ? "正在生成 AI 建议..." : "生成 AI 建议"}</button>
          <button class="secondary" data-ai-speak-summary="1" ${!canSpeakSummary || busy ? "disabled" : ""}>朗读 AI 建议</button>
        </div>
        ${state.error ? `<div class="message-box bad">${escapeHtml(state.error)}</div>` : ""}
        ${state.summary ? `
          ${aiVisualAdviceHtml(summary)}
          <div class="pill info calorie-pill">热量估计：${escapeHtml(calorie.text, "热量仅为粗略估计，仅供参考。")}</div>
          <div class="ai-detail-grid">
            ${aiDetailCard("患者版总结", escapeHtml(summary.patient_summary, "暂无患者版总结"), true)}
            ${aiDetailCard("医生版总结", escapeHtml(summary.doctor_summary, "暂无医生版总结"), true)}
            ${aiDetailCard("下一步建议", listHtml(summary.next_steps), true)}
            ${aiDetailCard("风险提醒", listHtml(summary.risk_notes), true)}
          </div>
        ` : `<div class="message-box">暂无 AI 建议。完成训练报告后点击生成；AI 只解释报告，不参与实时计数。</div>`}
        <details class="ai-detail-card ai-qa-card" open>
          <summary><span>报告问答</span><small>只基于本次训练报告回答</small></summary>
          <div class="ai-detail-body">
            <div class="field-grid">
              <label>患者问题<input data-ai-question="1" value="${escapeHtml(state.question, "")}" placeholder="例如：我刚才哪里没做好？"></label>
            </div>
            <div class="button-row">
              <button class="secondary" data-ai-ask="1" ${busy ? "disabled" : ""}>${state.status === "loading-answer" ? "正在回答..." : "提问"}</button>
              <button class="secondary" data-ai-speak-answer="1" ${!canSpeakAnswer || busy ? "disabled" : ""}>朗读回答</button>
            </div>
            ${state.answer ? `<div class="message-box">${escapeHtml(state.answer.answer, "暂无回答")}</div>` : ""}
          </div>
        </details>
      </section>
    `;
  }

  function aiDetailCard(title, body, open = false) {
    return `
      <details class="ai-detail-card" ${open ? "open" : ""}>
        <summary><span>${escapeHtml(title)}</span><small>展开 / 收起</small></summary>
        <div class="ai-detail-body summary-text">${body}</div>
      </details>
    `;
  }

  function aiVisualAdviceHtml(summary) {
    const rendered = summary.rendered_images || {};
    const keyframes = Array.isArray(summary.keyframes) ? summary.keyframes : [];
    const imageItems = [];
    const renderedItems = Array.isArray(rendered.items) && rendered.items.length ? rendered.items : [rendered];
    renderedItems.forEach((item, index) => {
      const rep = item.rep_index || (keyframes[index] && keyframes[index].rep_index) || index + 1;
      if (item.comparison_image && item.comparison_image.url) imageItems.push({ title: `图文建议大图 · Rep ${rep}`, url: item.comparison_image.url });
    });
    if (!imageItems.length) {
      keyframes.forEach((item, index) => {
        if (item && item.url) imageItems.push({ title: `骨架关键帧 · Rep ${item.rep_index || index + 1}`, url: item.url });
      });
    }
    return `
      <details class="ai-detail-card ai-visual-card" open>
        <summary><span>AI 图文建议</span><small>放大查看动作关键帧与建议说明</small></summary>
        <div class="ai-detail-body">
          ${imageItems.length ? `
            <div class="ai-image-grid ai-single-image-grid">
              ${imageItems.slice(0, 1).map((item) => `
                <figure class="ai-image-card">
                  <img src="${escapeHtml(item.url)}" alt="${escapeHtml(item.title)}">
                  <figcaption>${escapeHtml(item.title)}</figcaption>
                </figure>
              `).join("")}
            </div>
          ` : `<div class="message-box warn">本次报告没有可展示的图文建议大图，文本建议仍可正常使用。</div>`}
        </div>
      </details>
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
      metricTile("Temp", temperature.available ? `${formatNumber(temperature.max_celsius)} C` : safeText(temperature.note), "板端温度"),
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
    const metrics = report.report_card_metrics || report.metrics || {};
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
    [
      ["ROM", `${formatNumber(metrics.rom && metrics.rom.actual)} / ${formatNumber(metrics.rom && metrics.rom.target)}`],
      ["TUT", `${formatNumber(metrics.tut && metrics.tut.actual)} / ${formatNumber(metrics.tut && metrics.tut.target)}`],
      ["Speed", formatNumber(metrics.speed && metrics.speed.ratio, 2)],
      ["DTW", formatNumber(metrics.dtw && metrics.dtw.normalized_distance, 2)],
    ].forEach(([label, value], index) => {
      const cardX = 96 + index * 300;
      const y = 370;
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fillRect(cardX, y, 280, 150);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.strokeRect(cardX, y, 280, 150);
      ctx.fillStyle = "#96adcb";
      ctx.font = "600 18px Segoe UI";
      ctx.fillText(label, cardX + 20, y + 38);
      ctx.fillStyle = "#edf5ff";
      ctx.font = "700 38px Segoe UI";
      ctx.fillText(value, cardX + 20, y + 95);
    });
    ctx.fillStyle = "#edf5ff";
    ctx.font = "700 24px Segoe UI";
    ctx.fillText("Doctor Summary", 96, 590);
    ctx.fillText("Patient Summary", 96, 730);
    ctx.font = "400 22px Segoe UI";
    ctx.fillStyle = "#c8d6ea";
    wrapText(ctx, safeText(summary.doctor_summary), 96, 630, 1180, 32);
    wrapText(ctx, safeText(summary.patient_summary), 96, 770, 1180, 32);
    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/png");
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

  function preserveActiveQuestion(context = null) {
    const active = document.activeElement;
    if (!active || !active.matches("[data-ai-question]")) return false;
    const activeContext = context || contextForElement(active);
    const state = activeContext ? aiStateFor(activeContext) : null;
    if (state) state.question = active.value;
    return true;
  }

  function rerenderReport(context = null) {
    if (preserveActiveQuestion(context) || aiQuestionComposing) return;
    if (typeof window.__REHAB_RERENDER_REPORTS__ === "function") {
      window.__REHAB_RERENDER_REPORTS__(context);
      return;
    }
    const panel = document.getElementById("report-panel");
    if (panel) panel.innerHTML = reportCardHtml(context || window.__LAST_REPORT_CONTEXT__ || null);
  }

  function renderReportPanel(context, options = {}) {
    const panel = document.getElementById("report-panel");
    if (!panel) return;
    registerReportContext(context);
    if (!options.force && (preserveActiveQuestion(context) || aiQuestionComposing)) return;
    panel.innerHTML = reportCardHtml(context);
  }

  async function generateAISummary(context) {
    const state = aiStateFor(context);
    state.status = "loading-summary";
    state.error = "";
    rerenderReport(context);
    try {
      const result = await postJSON("/api/llm/report_summary", {
        report_id: reportId(context),
        audience: "both",
        include_calorie: true,
        include_keyframes: true,
        render_metric_cards: true,
      });
      state.summary = result;
      state.status = "ready";
    } catch (error) {
      state.status = "error";
      state.error = error.message || String(error);
    }
    rerenderReport(context);
  }

  async function askAIQuestion(context, element = null) {
    const state = aiStateFor(context);
    const root = element ? element.closest("[data-report-key]") || element.closest("[data-ai-report]") : document;
    const input = root ? root.querySelector("[data-ai-question]") : document.querySelector("[data-ai-question]");
    state.question = input ? input.value.trim() : state.question;
    if (!state.question) {
      state.error = "请输入要咨询的问题。";
      rerenderReport(context);
      return;
    }
    state.status = "loading-answer";
    state.error = "";
    rerenderReport(context);
    try {
      const result = await postJSON("/api/llm/ask", { report_id: reportId(context), question: state.question });
      state.answer = result;
      state.status = "ready";
    } catch (error) {
      state.status = "error";
      state.error = error.message || String(error);
    }
    rerenderReport(context);
  }

  async function speakAIText(context, kind) {
    const state = aiStateFor(context);
    const source = kind === "answer" ? state.answer : state.summary;
    const text = source && source.spoken_text;
    if (!text) {
      state.error = "没有可朗读的 AI 文本。";
      rerenderReport(context);
      return;
    }
    try {
      await postJSON("/api/llm/speak", { text, event_type: kind === "answer" ? "llm_qa" : "llm_summary" });
      state.error = "";
    } catch (error) {
      state.error = error.message || String(error);
    }
    rerenderReport(context);
  }

  document.addEventListener("click", (event) => {
    const exportButton = event.target.closest("[data-export-report]");
    if (exportButton) {
      exportReportCard(contextForElement(exportButton));
      return;
    }
    const summaryButton = event.target.closest("[data-ai-summary]");
    const askButton = event.target.closest("[data-ai-ask]");
    const speakSummaryButton = event.target.closest("[data-ai-speak-summary]");
    const speakAnswerButton = event.target.closest("[data-ai-speak-answer]");
    const actionElement = summaryButton || askButton || speakSummaryButton || speakAnswerButton;
    if (!actionElement) return;
    const context = contextForElement(actionElement);
    if (!context) return;
    if (summaryButton) {
      generateAISummary(context);
      return;
    }
    if (askButton) {
      askAIQuestion(context, askButton);
      return;
    }
    if (speakSummaryButton) {
      speakAIText(context, "summary");
      return;
    }
    if (speakAnswerButton) speakAIText(context, "answer");
  });

  document.addEventListener("compositionstart", (event) => {
    const input = event.target.closest("[data-ai-question]");
    if (!input) return;
    aiQuestionComposing = true;
  });

  document.addEventListener("compositionend", (event) => {
    const input = event.target.closest("[data-ai-question]");
    if (!input) return;
    aiQuestionComposing = false;
    const context = contextForElement(input);
    if (!context) return;
    aiStateFor(context).question = input.value;
  });

  document.addEventListener("input", (event) => {
    const input = event.target.closest("[data-ai-question]");
    if (!input) return;
    const context = contextForElement(input);
    if (!context) return;
    aiStateFor(context).question = input.value;
  });

  window.RehabUI = {
    actionNames,
    safeText,
    escapeHtml,
    formatNumber,
    fetchJSON,
    postJSON,
    renderCaps,
    renderSystemStats,
    composeReportContext,
    reportKey,
    reportId,
    registerReportContext,
    reportCardHtml,
    renderReportPanel,
    exportReportCard,
    streamSource,
    toneForCapability,
    toneForError,
    metricTile,
  };
})();

