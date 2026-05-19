
    const statusGrid = document.getElementById("status-grid");
    const messageBox = document.getElementById("message");
    const form = document.getElementById("record-form");
    const startBtn = document.getElementById("start-btn");
    const saveBtn = document.getElementById("save-btn");
    const retryBtn = document.getElementById("retry-btn");
    const clearBtn = document.getElementById("clear-btn");
    const cancelBtn = document.getElementById("cancel-btn");
    function formatNumber(value, unit = "") {
      if (value == null || Number.isNaN(Number(value))) {
        return "-";
      }
      return `${Number(value).toFixed(1)}${unit}`;
    }

    function setMessage(text) {
      messageBox.textContent = text;
    }

    function setBusy(isBusy) {
      [startBtn, saveBtn, retryBtn, clearBtn, cancelBtn].forEach((button) => {
        button.disabled = isBusy;
      });
    }

    function displayStatus(status) {
      if (status.recording) return "录制中";
      if (status.awaiting_ack) return "等待板端确认";
      if (status.pending_export) return "等待本机保存";
      return status.status || "等待操作";
    }

    function displaySaveState(status) {
      if (status.awaiting_ack) return "本机已保存，等待板端确认";
      if (status.pending_export) return "板端已生成结果，等待本机保存";
      return "无待处理导出";
    }

    function renderStatus(status) {
      const rows = [
        ["当前状态", displayStatus(status)],
        ["保存状态", displaySaveState(status)],
        ["患者编号", status.patient_id || "-"],
        ["动作名称", status.action_name || "-"],
        ["侧别模式", status.side_mode_label || status.side_mode || "-"],
        ["已录有效帧", String(status.valid_frames ?? "-")],
        ["无效帧", String(status.invalid_frames ?? "-")],
        ["当前选腿", status.selected_side_label || status.selected_side || "-"],
        ["角度来源", status.selected_source_label || status.selected_source || "-"],
        ["最低可见度", formatNumber(status.visibility_min)],
        ["平均可见度", formatNumber(status.visibility_avg)],
        ["当前屈曲角", formatNumber(status.selected_flexion_angle, " 度")],
        ["平滑屈曲角", formatNumber(status.smoothed_flexion_angle, " 度")],
        ["当前 ROM", formatNumber(status.current_rom, " 度")],
        ["待重试导出", status.pending_export ? "有" : "无"],
        ["最近错误", status.last_export_error || "-"],
      ];
      statusGrid.innerHTML = rows.map(([label, value]) => `
        <div class="stat">
          <b>${label}</b>
          <span>${value}</span>
        </div>
      `).join("");
    }

    async function getStatus() {
      try {
        const response = await fetch("/status");
        const status = await response.json();
        renderStatus(status);
      } catch (error) {
        renderStatus({});
        setMessage("状态拉取失败，请确认板端服务仍在运行。");
      }
    }

    function collectPayload() {
      const data = new FormData(form);
      return {
        patient_id: String(data.get("patient_id") || "").trim(),
        action_name: String(data.get("action_name") || "").trim(),
        side_mode: String(data.get("side_mode") || "auto").trim(),
      };
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || `请求失败: ${response.status}`);
      }
      return data;
    }

    async function pushToLocalSink(payload) {
      const response = await fetch("http://127.0.0.1:8090/api/save_result", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || `本地保存失败: ${response.status}`);
      }
      return data;
    }

    async function ackBoardSaved() {
      return await postJson("/api/ack_saved", {});
    }

    function dualSaveMessage(title, boardResult, localResult = null) {
      const lines = [
        title,
        `板端模板：${boardResult.board_saved_path || "-"}`,
        `板端摘要：${boardResult.board_summary_path || "-"}`,
      ];
      if (localResult) {
        lines.push(`Windows模板：${localResult.saved_path || "-"}`);
        lines.push(`Windows摘要：${localResult.summary_path || "-"}`);
      }
      return lines.join("\n");
    }

    startBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const result = await postJson("/api/start", collectPayload());
        setMessage(result.message || "已开始录制。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    clearBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const result = await postJson("/api/clear", { clear_export: true });
        setMessage(result.message || "已清空缓存。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    cancelBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const result = await postJson("/api/cancel", {});
        setMessage(result.message || "已取消本轮录制。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    saveBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const boardResult = await postJson("/api/save", {});
        try {
          const localResult = await pushToLocalSink({
            board_ip: window.location.hostname,
            board_port: window.location.port || "8082",
            source: "record_prescription_http",
            prescription: boardResult.prescription,
          });
          try {
            await ackBoardSaved();
            setMessage(dualSaveMessage("双端保存完成。", boardResult, localResult));
          } catch (ackError) {
            setMessage(`${dualSaveMessage("Windows 已保存，但板端导出清理失败。", boardResult, localResult)}\n${String(ackError.message || ackError)}\n板端仍保留最近一次导出，可稍后重试。`);
          }
        } catch (error) {
          setMessage(`${dualSaveMessage("板端已完成本地保存。", boardResult, null)}\nWindows 同步失败：${String(error.message || error)}\n板端这份记录已保底落盘；如需补同步，请先确认 Windows 本机的 local_result_sink.py 已启动，然后点击“重试导出最近结果”。`);
        }
      } catch (error) {
        setMessage(`板端本地保存失败。\n${String(error.message || error)}`);
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    retryBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const boardResult = await fetch("/api/export_last");
        const exportPayload = await boardResult.json();
        if (!boardResult.ok || exportPayload.ok === false) {
          throw new Error(exportPayload.error || "没有待导出的结果。");
        }
        const localResult = await pushToLocalSink({
          board_ip: window.location.hostname,
          board_port: window.location.port || "8082",
          source: "record_prescription_http_retry",
          prescription: exportPayload.prescription,
        });
        try {
          await ackBoardSaved();
          setMessage(dualSaveMessage("重试成功。", exportPayload, localResult));
        } catch (ackError) {
          setMessage(`${dualSaveMessage("本机已保存，但板端确认失败。", exportPayload, localResult)}\n${String(ackError.message || ackError)}\n板端仍保留最近一次导出，可继续重试。`);
        }
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    getStatus();
    setInterval(getStatus, 1000);
  