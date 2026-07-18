(function () {
  const labels = {
    loading: "NPU 模型加载中",
    pose_active: "NPU 姿态检测中",
    releasing: "NPU 正在释放",
    qwen_available: "NPU 已释放，可问小爱",
    error: "NPU 资源异常",
  };

  function ensurePill() {
    let pill = document.getElementById("npu-resource-pill");
    if (pill) return pill;
    const host = document.querySelector(".topbar-actions, .topbar nav, .panel-head .pills, .panel-head");
    if (!host) return null;
    pill = document.createElement("span");
    pill.id = "npu-resource-pill";
    pill.className = "pill info";
    pill.textContent = "NPU 状态等待中";
    pill.title = "8085 姿态检测与本地 Qwen 的 NPU 资源状态";
    host.appendChild(pill);
    return pill;
  }

  async function refresh() {
    const pill = ensurePill();
    if (!pill) return;
    try {
      const response = await fetch("/status", { cache: "no-store" });
      const status = await response.json();
      const resource = status.npu_resource || {};
      const state = String(resource.state || "qwen_available");
      pill.textContent = labels[state] || `NPU ${state}`;
      pill.className = `pill ${state === "error" ? "bad" : state === "pose_active" ? "good" : "info"}`;
      pill.title = [
        `owner=${resource.owner || "-"}`,
        `models_loaded=${Boolean(resource.models_loaded)}`,
        resource.last_error ? `error=${resource.last_error}` : "",
      ].filter(Boolean).join(" / ");
    } catch (error) {
      pill.textContent = "NPU 状态不可用";
      pill.className = "pill warn";
      pill.title = String(error);
    }
  }

  refresh();
  window.setInterval(refresh, 1000);
})();
