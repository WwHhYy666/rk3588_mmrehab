(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");

  app.innerHTML = `
    <main class="shell hmi-launcher">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">RK3588 Orthopedic Rehab HMI</div>
          <h1>智能康复终端</h1>
          <p>标准动作录入与下肢康复训练统一入口</p>
        </div>
        <div class="launcher-status">
          <span class="status-dot good" id="system-dot"></span>
          <span class="pill good" id="system-text">系统正常</span>
          <strong class="clock-text" id="clock">--:--:--</strong>
        </div>
      </header>

      <section class="launcher-center">
        <div class="launcher-board">
          <div class="launcher-doctor-row">
            ${entryCard({
              icon: "+",
              title: "医生录入",
              desc: "录制与更新标准动作模板",
              href: "/doctor",
              tone: "doctor",
            })}
          </div>
          <div class="launcher-rehab-row">
            ${entryCard({
              icon: "L",
              title: "下肢康复",
              desc: "依据下肢闭链与抗重力训练原理，提升髋膝协同、肌力耐力和步行转移能力。",
              href: "/train",
              tone: "featured leg",
            })}
          </div>
          ${window.__REHAB_APP__?.extendedGroups ? `<div class="launcher-rehab-row">
            ${entryCard({ icon: "U", title: "上肢扩展", desc: "坐姿屈肘、前平举与站姿侧平举。独立开关，失败可回退。", href: "/train-upper", tone: "extended upper" })}
            ${entryCard({ icon: "F", title: "全身扩展", desc: "半蹲、侧向迈步与低冲击开合步。独立模板与评分。", href: "/train-full", tone: "extended full" })}
          </div>` : ""}
        </div>
      </section>
    </main>
  `;

  function entryCard({ icon, title, desc, href, tone = "", badge = "" }) {
    const tag = href ? "a" : "button";
    const hrefAttr = href ? `href="${href}"` : `type="button" disabled aria-disabled="true"`;
    return `
      <${tag} class="launcher-card ${tone}" ${hrefAttr}>
        <span class="launcher-icon">${icon}</span>
        ${badge ? `<span class="launcher-badge">${badge}</span>` : ""}
        <h2>${title}</h2>
        <p>${desc}</p>
      </${tag}>
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
      await Promise.all([
        UI.fetchJSON("/api/system/status"),
        UI.fetchJSON("/status").catch(() => ({})),
      ]);
      dot.className = "status-dot good";
      text.className = "pill good";
      text.textContent = "系统正常";
    } catch (error) {
      dot.className = "status-dot warn";
      text.className = "pill warn";
      text.textContent = "状态降级";
    }
  }

  updateClock();
  refreshStatus();
  setInterval(updateClock, 1000);
  setInterval(refreshStatus, 3000);
})();



