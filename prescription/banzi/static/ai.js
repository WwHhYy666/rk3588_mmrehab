(function () {
  const app = document.getElementById("app");

  app.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">Integrated Review</div>
          <h1>AI 复盘已并入患者训练</h1>
          <p>图文建议、设备状态与报告问答已收纳到患者训练页右侧停靠栏。</p>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">首页</a>
          <a class="nav-link" href="/doctor">医生录入</a>
          <a class="nav-link active" href="/train#dock-ai">患者训练</a>
        </nav>
      </header>

      <section class="panel">
        <div class="panel-header">
          <h3>统一入口说明</h3>
          <span class="pill info">兼容保留</span>
        </div>
        <div class="message-box">
          当前 AI 复盘不再作为独立主流程页面维护，系统会自动带你进入患者训练页并展开 AI 训练图文建议侧栏。
        </div>
        <div class="button-row">
          <a class="nav-link active" href="/train#dock-ai">进入患者训练并展开 AI 侧栏</a>
          <a class="nav-link" href="/train#dock-system">查看设备运行状态侧栏</a>
        </div>
      </section>
    </main>
  `;
  window.setTimeout(() => {
    if (window.location.pathname === "/ai") {
      window.location.replace("/train#dock-ai");
    }
  }, 900);
})();
