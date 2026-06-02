(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");

  app.innerHTML = `
    <main class="shell hmi-launcher">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">RK3588 Orthopedic Rehab HMI</div>
          <h1>智能康复终端</h1>
          <p>医生录入、患者训练、AI 康复建议与系统状态统一入口</p>
        </div>
        <div class="launcher-status">
          <span class="status-dot good" id="system-dot"></span>
          <span class="pill good" id="system-text">系统正常</span>
          <strong class="clock-text" id="clock">--:--:--</strong>
        </div>
      </header>

      <section class="launcher-center">
        <div class="launcher-grid">
          ${entryCard("✚", "医生录入", "录制标准动作模板", "/doctor")}
          ${entryCard("◉", "患者训练", "三动作实时训练驾驶舱", "/train")}
          ${entryCard("◆", "AI 康复建议", "查看报告图文解释", "/train#ai-rehab")}
          ${entryCard("⌁", "系统状态", "CPU / 内存 / 温度 / NPU", "/api/system/status")}
        </div>
      </section>

      <footer class="launcher-footer hmi-card">
        <span class="mono" id="board-ip">Board IP: ${window.location.hostname || "localhost"}</span>
        <span id="llm-state">LLM: loading</span>
        <span id="resource-line">System: loading</span>
      </footer>
    </main>
  `;

  function entryCard(icon, title, desc, href) {
    return `
      <a class="launcher-card" href="${href}">
        <span class="launcher-icon">${icon}</span>
        <h2>${title}</h2>
        <p>${desc}</p>
      </a>
    `;
  }

  function updateClock() {
    const now = new Date();
    document.getElementById("clock").textContent = now.toLocaleTimeString("zh-CN", { hour12: false });
  }

  async function refreshStatus() {
    const dot = document.getElementById("system-dot");
    const text = document.getElementById("system-text");
    try {
      const [system, status] = await Promise.all([
        UI.fetchJSON("/api/system/status"),
        UI.fetchJSON("/status").catch(() => ({})),
      ]);
      const cpu = system.cpu || {};
      const memory = system.memory || {};
      const temp = system.temperature || {};
      const npu = system.npu || {};
      const llm = status.llm || {};
      dot.className = "status-dot good";
      text.className = "pill good";
      text.textContent = "系统正常";
      document.getElementById("llm-state").textContent =
        `LLM: ${UI.safeText(llm.provider, "echo")} / ${UI.safeText(llm.model, "echo")}`;
      document.getElementById("resource-line").textContent =
        `CPU ${UI.formatNumber(cpu.percent)}% · MEM ${UI.formatNumber(memory.percent)}% · TEMP ${UI.formatNumber(temp.max_celsius)}°C · NPU ${npu.percent == null ? UI.safeText(npu.raw, "-") : `${UI.formatNumber(npu.percent)}%`}`;
    } catch (error) {
      dot.className = "status-dot warn";
      text.className = "pill warn";
      text.textContent = "状态降级";
      document.getElementById("resource-line").textContent = error.message || String(error);
    }
  }

  updateClock();
  refreshStatus();
  setInterval(updateClock, 1000);
  setInterval(refreshStatus, 3000);
})();
