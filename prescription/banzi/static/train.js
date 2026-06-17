(function () {
  const UI = window.RehabUI;
  const app = document.getElementById("app");
  const activeStatuses = new Set(["running", "paused", "resting", "awaiting_orientation", "awaiting_return", "awaiting_care_response"]);
  let lastRepCount = 0;

  app.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="eyebrow">Patient Training Cockpit</div>
          <h1>患者训练</h1>
          <p>实时识别动作、计数、纠错、休息提示与 AI 康复建议。</p>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">首页</a>
          <a class="nav-link" href="/doctor">医生录入</a>
          <a class="nav-link active" href="/train">患者训练</a>
          <a class="nav-link" href="/ai">AI复盘</a>
        </nav>
      </header>

      <section class="train-grid">
        <section class="panel">
          <div class="panel-header">
            <h3>摄像头预览</h3>
            <span id="train-vision-pill" class="pill info">等待连接</span>
          </div>
          <div class="preview-box">
            <img id="train-preview" src="/assets/placeholder.svg" alt="training preview">
            <div class="preview-overlay">
              <div class="summary-text" id="train-prompt">等待开始训练</div>
              <div class="pills" id="train-pill-row"></div>
            </div>
          </div>
          <div class="status-badges" id="live-grid"></div>
          <div class="resource-strip train-resource-strip" id="resource-strip"></div>
        </section>

        <aside class="panel training-hud">
          <div class="section-head">
            <span class="pill info" id="training-status-pill">idle</span>
            <span class="meta" id="playlist-state">playlist 0/0</span>
          </div>
          <div class="action-title" id="action-title">准备训练</div>
          <div class="rep-display">
            <div class="rep-number" id="rep-number">0</div>
            <div>
              <div class="eyebrow">REPS</div>
              <div class="summary-text" id="rep-target">/ 0</div>
            </div>
          </div>
          <div class="progress-track"><div class="progress-fill" id="rep-progress"></div></div>
          <section class="live-metric-panel">
            <div>
              <div class="eyebrow">实时指标</div>
              <div class="live-metric-main" id="live-metric-current">-</div>
            </div>
            <div class="live-metric-grid">
              <article>
                <strong>目标</strong>
                <span id="live-metric-target">-</span>
              </article>
              <article>
                <strong>保持</strong>
                <span id="live-metric-tut">-</span>
              </article>
              <article>
                <strong>状态</strong>
                <span id="live-metric-state">-</span>
              </article>
            </div>
          </section>

          <div class="field-grid">
            <label>患者编号<input id="patient-id" value="patient_001"></label>
            <label>目标次数<input id="target-reps" type="number" min="1" max="50" value="3"></label>
            <label>侧别模式
              <select id="side-mode">
                <option value="auto">auto</option>
                <option value="left">left</option>
                <option value="right">right</option>
              </select>
            </label>
            <label>动作
              <input id="action-id" value="seated_knee_extension" list="train-action-options">
              <datalist id="train-action-options">
                <option value="seated_knee_extension">坐姿伸膝</option>
                <option value="standing_hamstring_curl">站姿屈膝后勾腿</option>
                <option value="seated_knee_raise">坐姿抬膝</option>
                <option value="sit_to_stand">坐站训练</option>
                <option value="knee_flexion">屈膝</option>
              </datalist>
            </label>
          </div>

          <div class="button-row">
            <button id="playlist-btn">开始完整训练</button>
            <button class="secondary" id="single-btn">开始单动作</button>
            <button class="secondary" id="pause-btn">暂停 / 继续</button>
            <button class="warn" id="stop-btn">结束训练</button>
          </div>
          <div class="message-box" id="train-message">准备就绪，等待开始训练。</div>

          <div class="message-box">
            训练结束后请进入 <a class="inline-link" href="/ai">AI 康复复盘</a> 查看三动作报告、关键帧图文建议和问答。
          </div>
        </aside>
      </section>

      <section class="panel">
        <div class="panel-header">
          <h3>训练编排</h3>
          <span class="meta">rehab_demo_plan.yaml</span>
        </div>
        <div class="timeline" id="timeline"></div>
      </section>

    </main>
    <div id="care-modal" class="modal-shell hidden" aria-hidden="true"></div>
  `;

  const restAudio = new Audio();
  restAudio.preload = "auto";
  restAudio.loop = true;
  restAudio.playsInline = true;

  function setMessage(text, tone = "") {
    const node = document.getElementById("train-message");
    node.className = `message-box ${tone}`.trim();
    node.textContent = text;
  }

  function payload() {
    return {
      patient_id: document.getElementById("patient-id").value.trim(),
      action_id: document.getElementById("action-id").value.trim(),
      side_mode: document.getElementById("side-mode").value,
      target_reps: Number(document.getElementById("target-reps").value || 3),
    };
  }

  function musicUrl(file) {
    try {
      return new URL(file, window.location.origin).toString();
    } catch (error) {
      return file;
    }
  }

  function stopRestMusic() {
    if (!restAudio.paused) restAudio.pause();
    restAudio.currentTime = 0;
    restAudio.volume = 1;
  }

  function syncRestMusic(training) {
    const music = training.rest_music || {};
    if ((music.playback || "browser") === "backend") {
      stopRestMusic();
      return;
    }
    const shouldPlay = training.status === "resting" && music.enabled && music.file;
    if (!shouldPlay) {
      stopRestMusic();
      return;
    }
    const nextSrc = musicUrl(music.file);
    if (restAudio.src !== nextSrc) {
      restAudio.src = nextSrc;
      restAudio.currentTime = 0;
    }
    if (restAudio.paused) restAudio.play().catch(() => {});
    const fadeSeconds = Number(music.fade_seconds || 0);
    if (training.rest_remaining_seconds != null && fadeSeconds > 0 && training.rest_remaining_seconds <= fadeSeconds) {
      restAudio.volume = Math.max(0, Math.min(1, Number(training.rest_remaining_seconds) / fadeSeconds));
    } else {
      restAudio.volume = 1;
    }
  }

  function renderCareDialog(training) {
    const modal = document.getElementById("care-modal");
    const dialog = training.care_dialog || {};
    if (!dialog.visible) {
      modal.className = "modal-shell hidden";
      modal.setAttribute("aria-hidden", "true");
      modal.innerHTML = "";
      return;
    }
    modal.className = "modal-shell";
    modal.setAttribute("aria-hidden", "false");
    modal.innerHTML = `
      <section class="modal-card">
        <div class="eyebrow">Care Check</div>
        <h3>${UI.safeText(dialog.title, "温馨提示")}</h3>
        <p>${UI.safeText(dialog.message)}</p>
        <div class="button-row">
          <button data-care-response="yes">${UI.safeText(dialog.yes_label, "是")}</button>
          <button class="secondary" data-care-response="no">${UI.safeText(dialog.no_label, "否")}</button>
        </div>
      </section>
    `;
  }

  function render(status, system) {
    const training = status.training || {};
    const previewSource = UI.streamSource(status);
    const preview = document.getElementById("train-preview");
    if (preview.getAttribute("src") !== previewSource) preview.src = previewSource;

    const streamReady = Boolean(status.stream_ready);
    const visionPill = document.getElementById("train-vision-pill");
    visionPill.className = `pill ${streamReady ? "good" : "warn"}`;
    visionPill.textContent = streamReady ? "实时流已连接" : "视觉链路等待中";

    const actionName = training.current_action_name || UI.actionNames[training.action_id] || training.action_id || "准备训练";
    const completed = Number(training.completed_reps || 0);
    const target = Number(training.target_reps || 0);
    document.getElementById("action-title").textContent = actionName;
    const repNode = document.getElementById("rep-number");
    repNode.textContent = completed;
    if (completed !== lastRepCount) {
      repNode.classList.remove("bump");
      void repNode.offsetWidth;
      repNode.classList.add("bump");
      lastRepCount = completed;
    }
    document.getElementById("rep-target").textContent = `/ ${target}`;
    document.getElementById("rep-progress").style.width = `${target > 0 ? Math.min(100, (completed / target) * 100) : 0}%`;
    document.getElementById("live-metric-current").textContent = formatMetricValue(training.current_metric ?? training.current_angle, training.metric_unit);
    document.getElementById("live-metric-target").textContent = formatTargetRange(training.target_range, training.metric_unit);
    document.getElementById("live-metric-tut").textContent = formatTut(training);
    document.getElementById("live-metric-state").textContent = UI.safeText(training.current_state, "-");
    document.getElementById("training-status-pill").className = `pill ${training.status === "running" ? "good" : "warn"}`;
    document.getElementById("training-status-pill").textContent = UI.safeText(training.status, "idle");
    document.getElementById("playlist-state").textContent =
      training.playlist_mode ? `playlist ${Number(training.playlist_index || 0) + 1}/${training.playlist_total || 0}` : "single action";

    const promptText = streamReady
      ? UI.safeText(training.prompt, "等待开始训练")
      : UI.safeText(status.vision_boot_error || status.status, "视觉链路未就绪");
    document.getElementById("train-prompt").textContent = promptText;
    setMessage(promptText, training.status === "awaiting_return" || training.status === "awaiting_orientation" ? "warn" : "");

    const quality = status.pose_quality || {};
    document.getElementById("train-pill-row").innerHTML = `
      <span class="pill info">${UI.safeText(actionName)}</span>
      <span class="pill info">rep ${completed}/${target}</span>
      <span class="pill info">rest ${training.rest_remaining_seconds == null ? "-" : `${training.rest_remaining_seconds}s`}</span>
      <span class="pill info">${UI.safeText(status.actual_backend)}</span>
      <span class="pill ${quality.quality_ok ? "good" : "warn"}">${quality.quality_ok ? "keypoints OK" : "check keypoints"}</span>
    `;

    document.getElementById("live-grid").innerHTML = [
      badge("Metric", UI.formatNumber(training.current_metric ?? training.current_angle)),
      badge("Target", Array.isArray(training.target_range) ? `${UI.formatNumber(training.target_range[0])}-${UI.formatNumber(training.target_range[1])}` : "-"),
      badge("Invalid", UI.safeText(training.invalid_attempts, "0")),
      badge("TTS", shortValue(training.tts_text, 12)),
      badge("Motor", shortValue(training.motor_mock_pattern, 12)),
      badge("Pause", shortValue(training.pause_reason, 12)),
    ].join("");

    document.getElementById("timeline").innerHTML = renderTimeline(training.demo_plan, training);
    document.getElementById("resource-strip").innerHTML = renderResourceStrip(system, status);

    const context = UI.composeReportContext(
      training.report
        ? { report: training.report, report_file: training.report_file, summary_bundle: training.report.summary_bundle, report_card: training.report.report_card }
        : status.latest_report
    );
    window.__LLM_STATUS__ = status.llm || {};
    window.__LAST_REPORT_CONTEXT__ = context;
    UI.renderReportPanel(context);

    renderCareDialog(training);
    syncRestMusic(training);

    const busy = activeStatuses.has(training.status);
    const careVisible = Boolean(training.care_dialog && training.care_dialog.visible);
    document.getElementById("single-btn").disabled = busy || !status.active_template || careVisible;
    document.getElementById("playlist-btn").disabled = busy || careVisible;
    document.getElementById("pause-btn").disabled = !["running", "paused"].includes(training.status) || careVisible;
    document.getElementById("stop-btn").disabled = !busy && !careVisible;
  }

  function badge(label, value) {
    return `<article class="badge-card"><strong>${label}</strong><span class="mono">${UI.safeText(value)}</span></article>`;
  }

  function shortValue(value, max = 18) {
    const text = UI.safeText(value, "-");
    return text.length > max ? `${text.slice(0, max)}...` : text;
  }

  function metricUnit(unit) {
    const text = UI.safeText(unit, "");
    if (text === "degree") return "°";
    if (text === "body_ratio") return "";
    return text ? ` ${text}` : "";
  }

  function formatMetricValue(value, unit) {
    const suffix = metricUnit(unit);
    return `${UI.formatNumber(value, 1)}${suffix}`;
  }

  function formatTargetRange(range, unit) {
    if (!Array.isArray(range) || range.length < 2) return "-";
    const suffix = metricUnit(unit);
    return `${UI.formatNumber(range[0], 1)}-${UI.formatNumber(range[1], 1)}${suffix}`;
  }

  function formatTut(training) {
    const current = UI.formatNumber(training.tut_seconds, 1);
    const target = UI.formatNumber(training.tut_target, 1);
    const missing = Number(training.missing_seconds || 0);
    return missing > 0 ? `${current}/${target}s · 还差 ${UI.formatNumber(missing, 1)}s` : `${current}/${target}s`;
  }

  function renderResourceStrip(system, status) {
    const cpu = system.cpu || {};
    const memory = system.memory || {};
    const temp = system.temperature || {};
    const npu = system.npu || {};
    const pose = system.pose_fps || {};
    const cameraFailures = Number(status.camera_display_failures || 0);
    const poseErrors = Number(status.pose_worker_error_count || 0);
    return [
      badge("CPU", cpu.available ? `${UI.formatNumber(cpu.percent)}%` : shortValue(cpu.note, 12)),
      badge("MEM", memory.available ? `${UI.formatNumber(memory.percent)}%` : shortValue(memory.note, 12)),
      badge("TEMP", temp.available ? `${UI.formatNumber(temp.max_celsius)}°C` : shortValue(temp.note, 12)),
      badge("NPU", npu.available ? (npu.percent == null ? shortValue(npu.raw, 12) : `${UI.formatNumber(npu.percent)}%`) : shortValue(npu.note, 12)),
      badge("POSE", pose.available ? `${UI.formatNumber(pose.fps, 2)} FPS` : shortValue(status.pose_fps, 12)),
      badge("CAMERA", cameraFailures > 0 ? `fail ${cameraFailures}` : "OK"),
      badge("POSEERR", poseErrors > 0 ? `err ${poseErrors}` : "OK"),
      badge("BACKEND", shortValue(status.actual_backend, 12)),
    ].join("");
  }

  function renderTimeline(plan, training) {
    const actions = plan && Array.isArray(plan.actions) ? plan.actions : [];
    return actions.map((action, index) => {
      const done = Array.isArray(training.playlist_reports) && training.playlist_reports.some((item) => item.action_id === action.action_id);
      const active = training.playlist_mode && Number(training.playlist_index) === index && activeStatuses.has(training.status);
      return `
        <article class="timeline-step ${done ? "done" : active ? "active" : ""}">
          <strong>${UI.safeText(action.action_name || UI.actionNames[action.action_id] || action.action_id)}</strong>
          <div class="summary-text">${UI.safeText(action.camera_prompt)}</div>
          <small>${done ? "已完成并生成报告" : active ? "当前动作" : "等待执行"}</small>
        </article>
      `;
    }).join("");
  }

  async function refresh() {
    try {
      const [status, system] = await Promise.all([
        UI.fetchJSON("/status"),
        UI.fetchJSON("/api/system/status"),
      ]);
      render(status, system);
    } catch (error) {
      setMessage(error.message || String(error), "bad");
      stopRestMusic();
    }
  }

  document.getElementById("single-btn").addEventListener("click", async () => {
    try {
      await UI.postJSON("/api/realtime/start", payload());
      setMessage("单动作训练已开始。", "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  document.getElementById("playlist-btn").addEventListener("click", async () => {
    try {
      await UI.postJSON("/api/realtime/start_playlist", payload());
      setMessage("完整三动作训练已开始。", "good");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  document.getElementById("pause-btn").addEventListener("click", async () => {
    try {
      await UI.postJSON("/api/realtime/pause", {});
      setMessage("训练状态已切换。", "warn");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  document.getElementById("stop-btn").addEventListener("click", async () => {
    try {
      await UI.postJSON("/api/realtime/stop", {});
      setMessage("训练已结束。", "warn");
    } catch (error) {
      setMessage(error.message || String(error), "bad");
    } finally {
      refresh();
    }
  });

  document.addEventListener("click", async (event) => {
    const careButton = event.target.closest("[data-care-response]");
    if (careButton) {
      const needsRest = careButton.getAttribute("data-care-response") === "yes";
      try {
        await UI.postJSON("/api/realtime/care_response", { needs_rest: needsRest });
        setMessage(needsRest ? "已进入休息流程。" : "继续训练。", needsRest ? "warn" : "good");
      } catch (error) {
        setMessage(error.message || String(error), "bad");
      } finally {
        refresh();
      }
      return;
    }
    const accordionButton = event.target.closest("[data-accordion-toggle]");
    if (accordionButton) {
      accordionButton.closest(".accordion-card").classList.toggle("is-collapsed");
    }
  });

  if (window.location.hash === "#ai-rehab") {
    window.location.replace("/ai");
  }

  refresh();
  setInterval(refresh, 1000);
})();
