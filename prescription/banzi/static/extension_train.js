(function () {
  window.renderExtensionPage = async function (group, title) {
    const UI = window.RehabUI;
    const app = document.getElementById("app");
    app.innerHTML = `
      <main class="shell">
        <header class="topbar">
          <div class="brand"><div class="eyebrow">8085 独立扩展训练</div><h1>${title}</h1><p>扩展动作与下肢训练互斥运行；种子阈值不是临床标准</p></div>
          <a class="button secondary" href="/">返回首页</a>
        </header>
        <section class="panel"><div class="training-layout">
          <div><img class="preview" id="preview" src="/stream.mjpg" alt="实时姿态画面"></div>
          <div class="training-sidebar">
            <label>动作<select id="action"></select></label>
            <div id="instructions" class="status-card"></div>
            <label>目标次数<input id="target" type="number" min="1" value="5"></label>
            <div class="button-row"><button id="start-group">开始整组</button><button id="start" class="secondary">单动作测试</button></div>
            <div class="button-row"><button id="pause" class="secondary">暂停/继续</button><button id="stop" class="warn">停止</button></div>
            <div class="button-row"><button id="template-start" class="secondary">录制医生模板</button><button id="template-save" class="secondary">保存模板</button></div>
            <div id="message" class="status-card">加载中...</div>
            <div id="template-health" class="status-card"></div>
            <div id="stats" class="metric-grid"></div>
          </div>
        </div></section>
      </main>`;
    const action = document.getElementById("action");
    const preview = document.getElementById("preview");
    const instructions = document.getElementById("instructions");
    const message = document.getElementById("message");
    const healthBox = document.getElementById("template-health");
    const stats = document.getElementById("stats");
    let timer = null;
    let refreshInFlight = false;
    let latestStatus = null;
    let catalogActions = [];
    let streamOpenedAt = Date.now();
    let streamStaleSince = 0;
    let streamReconnectAttempts = 0;
    let streamReconnectTimer = null;
    const STREAM_RECYCLE_MS = 15000;
    const STREAM_STALE_MS = 3000;
    const STREAM_RECONNECT_BASE_MS = 1000;
    const STREAM_RECONNECT_MAX_MS = 8000;
    const statusLabels = {
      idle: "待命", running: "训练中", paused: "已暂停", resting: "组间休息",
      recording_template: "录制模板", complete: "已完成", stopped: "已停止",
    };
    const phaseLabels = { ready: "等待起始位", moving: "动作进行中", rest: "休息" };
    const errorLabels = {
      wrist_visibility: "腕点不可见", ankle_visibility: "踝点不可见", low_visibility: "关键点可见度不足",
      upper_arm_swing: "上臂摆动", trunk_lean: "躯干倾斜", elbow_bent: "肘部弯曲",
      bilateral_asymmetry: "左右不对称", body_side_lean: "身体侧倾", shoulder_tilt: "肩线倾斜",
      hip_drop: "髋部下沉不足", coordination: "上下肢不同步", sudden_jump: "关键点突跳",
      direction_error: "动作方向错误",
    };

    function streamSourceWithNonce() {
      return `/stream.mjpg?extension_reconnect=${Date.now()}`;
    }

    function reconnectPreview(delay) {
      if (streamReconnectTimer) return;
      streamReconnectTimer = window.setTimeout(() => {
        streamReconnectTimer = null;
        preview.src = streamSourceWithNonce();
        streamOpenedAt = Date.now();
        streamReconnectAttempts += 1;
      }, delay);
    }

    function maintainPreview(status) {
      if (!status || status.stream_available !== true) {
        streamStaleSince = 0;
        streamReconnectAttempts = 0;
        return;
      }
      const now = Date.now();
      const ageMs = Number(status.stream_frame_age_ms);
      const stale = Number.isFinite(ageMs) && ageMs >= STREAM_STALE_MS;
      const recycleDue = now - streamOpenedAt >= STREAM_RECYCLE_MS;
      if (!stale && !recycleDue) {
        streamStaleSince = 0;
        streamReconnectAttempts = 0;
        return;
      }
      if (stale && !streamStaleSince) streamStaleSince = now;
      if (stale && now - streamStaleSince < STREAM_STALE_MS) return;
      const delay = recycleDue && !stale
        ? 0
        : Math.min(STREAM_RECONNECT_MAX_MS, STREAM_RECONNECT_BASE_MS * (2 ** Math.min(streamReconnectAttempts, 3)));
      reconnectPreview(delay);
    }

    function renderInstructions() {
      const selected = catalogActions.find((item) => item.action_id === action.value);
      if (!selected) {
        instructions.textContent = "请选择动作";
        return;
      }
      const steps = (selected.instructions || []).map((item, index) => `${index + 1}. ${item}`).join(" ");
      const view = selected.view === "front" ? "正面" : "侧面";
      instructions.textContent = `${selected.label}（${view}）：${steps}`;
    }
    function showError(error) {
      message.textContent = error.message || String(error);
      message.className = "status-card bad";
    }
    function render(status) {
      latestStatus = status;
      const t = status.training || {};
      if (t.extension_action && action.value !== t.extension_action) {
        action.value = t.extension_action;
        renderInstructions();
      }
      const actionItem = catalogActions.find((item) => item.action_id === t.extension_action);
      const metricUnit = actionItem && actionItem.metric_unit === "deg" ? "°" : "";
      const speedUnit = actionItem && actionItem.metric_unit === "deg" ? "°/s" : "/s";
      const groupProgress = t.playlist_total ? ` · 第 ${(t.playlist_index || 0) + 1}/${t.playlist_total} 项` : "";
      const restText = t.status === "resting" ? ` · 休息 ${UI.formatNumber(t.rest_remaining_seconds || 0, 0)}s` : "";
      message.textContent = status.enabled
        ? `${actionItem ? actionItem.label : "待选择"} · ${statusLabels[t.status] || t.status || "待命"} · ${phaseLabels[t.phase] || t.phase || "等待"}${groupProgress}${restText}`
        : "扩展开关未开启，当前下肢演示不受影响";
      message.className = `status-card ${status.enabled ? "good" : "warn"}`;
      const health = t.template_health || {};
      healthBox.textContent = health.ok
        ? `模板健康：${health.valid_frames} 帧 / ${UI.formatNumber(health.duration_seconds, 1)}s / ROM ${UI.formatNumber(health.target_rom, 2)}`
        : `模板未通过：${health.reason || "尚未录制"}（至少 20 有效帧、2 秒、完整返回）`;
      healthBox.className = `status-card ${health.ok ? "good" : "warn"}`;
      stats.innerHTML = [
        ["计数", t.count || 0], ["无效尝试", t.invalid_attempts || 0],
        ["模板录制", `${t.template_valid_frames || 0} 帧 / ${UI.formatNumber(t.template_duration_seconds || 0, 1)}s`],
        ["识别侧", t.selected_side === "right" ? "右侧" : (t.selected_side === "left" ? "左侧" : "双侧/正面")],
        ["当前指标", t.metric == null ? "--" : `${UI.formatNumber(t.metric, 2)}${metricUnit}`],
        ["目标 ROM", t.target_rom == null ? "--" : `${UI.formatNumber(t.target_rom, 2)}${metricUnit}`],
        ["单次 TUT", `${UI.formatNumber(t.tut_seconds || 0, 1)}s`],
        ["速度", `${UI.formatNumber(t.speed || 0, 2)}${speedUnit}`],
        ["质量分", t.quality_score == null ? "--" : UI.formatNumber(t.quality_score, 0)],
        ["补偿", (t.compensation_errors || []).map((item) => errorLabels[item] || item).join("、") || "无"],
        ["报告", t.last_report_file || "--"],
      ].map(([key, value]) => `<div class="metric-card"><span>${key}</span><strong>${value}</strong></div>`).join("");
      maintainPreview(status);
    }
    async function refresh() {
      if (refreshInFlight) return latestStatus;
      refreshInFlight = true;
      try {
        const status = await UI.fetchJSON("/api/extension/status");
        render(status);
        return status;
      } finally {
        refreshInFlight = false;
      }
    }
    try {
      const catalog = await UI.fetchJSON(`/api/extension/catalog?group=${group}`);
      if (!catalog.enabled) throw new Error("扩展开关未开启");
      catalogActions = catalog.actions || [];
      catalogActions.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.action_id;
        option.textContent = item.label;
        action.appendChild(option);
      });
      action.onchange = renderInstructions;
      renderInstructions();
      document.getElementById("start").onclick = async () => {
        try {
          await UI.postJSON("/api/extension/start", { group, action_id: action.value, target_reps: Number(document.getElementById("target").value), require_template: true });
          await refresh();
        } catch (error) { showError(error); }
      };
      document.getElementById("start-group").onclick = async () => {
        try {
          await UI.postJSON("/api/extension/start_group", { group, target_reps: Number(document.getElementById("target").value), require_template: true });
          await refresh();
        } catch (error) { showError(error); }
      };
      document.getElementById("pause").onclick = async () => { try { await UI.postJSON("/api/extension/pause", {}); await refresh(); } catch (error) { showError(error); } };
      document.getElementById("stop").onclick = async () => { try { await UI.postJSON("/api/extension/stop", {}); await refresh(); } catch (error) { showError(error); } };
      document.getElementById("template-start").onclick = async () => { try { await UI.postJSON("/api/extension/template/start", { group, action_id: action.value }); await refresh(); } catch (error) { showError(error); } };
      document.getElementById("template-save").onclick = async () => { try { await UI.postJSON("/api/extension/template/save", {}); await refresh(); } catch (error) { showError(error); } };
      await refresh();
      timer = setInterval(() => refresh().catch(showError), 700);
      window.addEventListener("beforeunload", () => {
        if (timer) clearInterval(timer);
        if (streamReconnectTimer) clearTimeout(streamReconnectTimer);
      });
    } catch (error) { showError(error); }
  };
})();
