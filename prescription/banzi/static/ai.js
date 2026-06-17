(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");
  let latestContexts = [];

  app.innerHTML = `
    <main class="shell ai-review-page">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">Post Training Review</div>
          <h1>AI 康复复盘</h1>
          <p>训练结束后查看三动作报告、图文关键帧、AI 建议与报告问答。</p>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">首页</a>
          <a class="nav-link" href="/doctor">医生录入</a>
          <a class="nav-link" href="/train">患者训练</a>
          <a class="nav-link active" href="/ai">AI复盘</a>
        </nav>
      </header>

      <section class="panel ai-review-status">
        <div class="panel-header">
          <h3>复盘状态</h3>
          <span class="pill info" id="ai-llm-pill">LLM loading</span>
        </div>
        <div class="resource-strip" id="ai-resource-strip"></div>
      </section>

      <section class="ai-review-layout">
        <aside class="panel ai-report-list">
          <div class="panel-header">
            <h3>最近报告</h3>
            <span class="meta">latest 3 reports</span>
          </div>
          <div id="ai-report-tabs" class="ai-report-tabs"></div>
        </aside>
        <section id="ai-report-panel" class="ai-report-stack"></section>
      </section>
    </main>
  `;

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

  function renderTabs(contexts) {
    const tabs = document.getElementById("ai-report-tabs");
    if (!tabs) return;
    if (!contexts.length) {
      tabs.innerHTML = `<div class="empty">暂无训练报告。完成一次患者训练后，这里会显示最近三份动作报告。</div>`;
      return;
    }
    tabs.innerHTML = contexts
      .map((context, index) => {
        const report = context.report || {};
        const meta = report.meta || {};
        const errors = report.errors || {};
        const actionName = report.action_name || meta.action_name || UI.actionNames[meta.action_id] || meta.action_id || `报告 ${index + 1}`;
        const keyframes = Array.isArray(report.keyframes) ? report.keyframes.length : 0;
        return `
          <a class="ai-report-tab" href="#${UI.escapeHtml(contextKey(context))}">
            <strong>${UI.escapeHtml(actionName)}</strong>
            <span>${UI.escapeHtml(errors.primary_error || "OK")} · keyframes ${keyframes}</span>
            <small class="mono">${UI.escapeHtml(context.reportFile)}</small>
          </a>
        `;
      })
      .join("");
  }

  function renderReports(contexts) {
    latestContexts = contexts || latestContexts;
    const panel = document.getElementById("ai-report-panel");
    if (!panel) return;
    if (!latestContexts.length) {
      panel.innerHTML = `<section class="panel"><div class="empty">暂无报告。请先到患者训练完成一次三动作训练。</div></section>`;
      renderTabs([]);
      return;
    }
    latestContexts.forEach((context) => UI.registerReportContext && UI.registerReportContext(context));
    renderTabs(latestContexts);
    panel.innerHTML = latestContexts
      .map((context, index) => `
        <article class="panel ai-review-report" id="${UI.escapeHtml(contextKey(context))}">
          <div class="section-head">
            <span class="pill info">动作 ${index + 1}</span>
            <span class="meta mono">${UI.escapeHtml(context.reportFile)}</span>
          </div>
          ${UI.reportCardHtml(context)}
        </article>
      `)
      .join("");
  }

  function renderSystem(system, status) {
    const resource = document.getElementById("ai-resource-strip");
    if (resource) {
      const cpu = system.cpu || {};
      const memory = system.memory || {};
      const temp = system.temperature || {};
      const npu = system.npu || {};
      const pose = system.pose_fps || {};
      resource.innerHTML = [
        badge("CPU", cpu.available ? `${UI.formatNumber(cpu.percent)}%` : shortValue(cpu.note, 12)),
        badge("MEM", memory.available ? `${UI.formatNumber(memory.percent)}%` : shortValue(memory.note, 12)),
        badge("TEMP", temp.available ? `${UI.formatNumber(temp.max_celsius)}°C` : shortValue(temp.note, 12)),
        badge("NPU", npu.available ? (npu.percent == null ? shortValue(npu.raw, 12) : `${UI.formatNumber(npu.percent)}%`) : shortValue(npu.note, 12)),
        badge("POSE", pose.available ? `${UI.formatNumber(pose.fps, 2)} FPS` : shortValue(status.pose_fps, 12)),
        badge("BACKEND", shortValue(status.actual_backend, 12)),
      ].join("");
    }
    const llm = status.llm || {};
    const pill = document.getElementById("ai-llm-pill");
    if (pill) {
      pill.className = `pill ${llm.provider === "echo" ? "warn" : "info"}`;
      pill.textContent = `${UI.safeText(llm.provider, "echo")} / ${UI.safeText(llm.model, "echo")}`;
    }
  }

  function badge(label, value) {
    return `<article class="badge-card"><strong>${label}</strong><span class="mono">${UI.safeText(value)}</span></article>`;
  }

  function shortValue(value, max = 18) {
    const text = UI.safeText(value, "-");
    return text.length > max ? `${text.slice(0, max)}...` : text;
  }

  async function refresh() {
    try {
      const [status, system] = await Promise.all([
        UI.fetchJSON("/status"),
        UI.fetchJSON("/api/system/status"),
      ]);
      window.__LLM_STATUS__ = status.llm || {};
      renderSystem(system, status);
      const active = document.activeElement;
      if (active && active.matches("[data-ai-question]")) return;
      renderReports(collectContexts(status));
    } catch (error) {
      const panel = document.getElementById("ai-report-panel");
      if (panel) {
        panel.innerHTML = `<section class="panel"><div class="message-box bad">${UI.escapeHtml(error.message || String(error))}</div></section>`;
      }
    }
  }

  window.__REHAB_RERENDER_REPORTS__ = () => renderReports(latestContexts);

  refresh();
  setInterval(refresh, 3000);
})();
