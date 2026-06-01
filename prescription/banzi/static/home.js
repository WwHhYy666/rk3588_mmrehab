(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");

  app.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">RK3588 Digital Rehab Station</div>
          <h1>骨科居家康复统一训练台</h1>
          <p>一个入口串起医生模板录制、患者实时训练、评估反馈和报告卡片。</p>
        </div>
        <nav class="nav-links">
          <a class="nav-link active" href="/">首页</a>
          <a class="nav-link" href="/doctor">医生录制</a>
          <a class="nav-link" href="/train">患者训练</a>
        </nav>
      </header>
      <section class="hero">
        <div class="hero-grid">
          <div class="kpi-stack">
            <div>
              <div class="eyebrow">HDMI Dashboard</div>
              <h2>老师要看的功能，都放在这个展台首页</h2>
              <p>实时展示模板状态、训练链路、系统监控与最近报告，适合作为大屏首屏和答辩入口。</p>
            </div>
            <div class="button-row">
              <a class="nav-link active" href="/doctor">进入医生录制</a>
              <a class="nav-link active" href="/train">进入患者训练</a>
            </div>
          </div>
          <div class="panel">
            <div class="panel-header">
              <h3>运行状态</h3>
              <span class="pill info mono">http://板子IP:8082</span>
            </div>
            <div class="metrics-grid" id="hero-metrics"></div>
          </div>
        </div>
      </section>
      <section class="dashboard-grid">
        <section class="panel">
          <div class="panel-header">
            <h3>功能入口</h3>
            <span class="meta">现有模块汇总展示</span>
          </div>
          <div class="card-grid" id="entry-grid"></div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <h3>系统监控</h3>
            <span class="meta">CPU / Memory / Temperature / NPU / Pose FPS</span>
          </div>
          <div class="system-grid" id="system-grid"></div>
        </section>
      </section>
      <section class="dashboard-grid">
        <section class="panel">
          <div class="panel-header">
            <h3>能力状态</h3>
            <span class="meta">依赖缺失时自动降级</span>
          </div>
          <div class="capability-grid" id="cap-grid"></div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <h3>最近报告</h3>
            <span class="meta">从 evaluate/reports 读取</span>
          </div>
          <div id="recent-report"></div>
          <div id="recent-list" class="card-grid"></div>
        </section>
      </section>
    </main>
  `;

  async function refresh() {
    try {
      const [status, system] = await Promise.all([
        UI.fetchJSON("/status"),
        UI.fetchJSON("/api/system/status"),
      ]);
      render(status, system);
    } catch (error) {
      document.getElementById("recent-report").innerHTML = `<div class="message-box bad">${error.message || String(error)}</div>`;
    }
  }

  function render(status, system) {
    const training = status.training || {};
    const latestContext = UI.composeReportContext(status.latest_report);
    window.__LAST_REPORT_CONTEXT__ = latestContext;
    const registry = status.active_templates_by_backend || (status.active_templates && status.active_templates.by_backend) || {};
    const templateCount = Object.values(registry).reduce((total, entries) => total + Object.keys(entries || {}).length, 0);

    document.getElementById("hero-metrics").innerHTML = [
      UI.metricTile("Active Template", templateCount, "CPU/NPU 已配置模板"),
      UI.metricTile("Training", UI.safeText(training.status, "idle"), training.playlist_mode ? `playlist ${Number(training.playlist_index || 0) + 1}/${training.playlist_total || 0}` : "单动作"),
      UI.metricTile("Pose FPS", UI.formatNumber(status.pose_fps, 2), "板端识别速度"),
    ].join("");

    document.getElementById("entry-grid").innerHTML = [
      featureCard("医生模板录制", "录入标准动作并保存为 active template。", "/doctor"),
      featureCard("患者实时训练", "单动作或三动作 playlist 训练。", "/train"),
      featureCard("评估与总结", "生成 report card 和医生/患者总结。", "/doctor"),
    ].join("");

    document.getElementById("system-grid").innerHTML = UI.renderSystemStats(system);
    document.getElementById("cap-grid").innerHTML = UI.renderCaps(status.capabilities);
    document.getElementById("recent-report").innerHTML = UI.reportCardHtml(latestContext);
    document.getElementById("recent-list").innerHTML = (status.recent_reports || []).slice(0, 3).map((item) => {
      const context = UI.composeReportContext(item);
      const report = context && context.report ? context.report : {};
      const meta = report.meta || {};
      const errorCode = report.errors && report.errors.primary_error;
      return `
        <article class="recent-card">
          <strong>${UI.safeText(meta.action_name || meta.action_id)}</strong>
          <span class="pill ${UI.toneForError(errorCode)}">${UI.safeText(errorCode, "OK")}</span>
          <small>${UI.safeText(meta.patient_id)} · ${UI.safeText(meta.evaluated_at)}</small>
          <div class="mono">${UI.safeText(item.report_file)}</div>
        </article>
      `;
    }).join("");
  }

  function featureCard(title, description, href) {
    return `
      <article class="summary-card">
        <strong>${title}</strong>
        <div class="summary-text">${description}</div>
        <div class="button-row">
          <a class="nav-link active" href="${href}">打开</a>
        </div>
      </article>
    `;
  }

  refresh();
  setInterval(refresh, 2000);
})();
