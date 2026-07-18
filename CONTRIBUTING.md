# 贡献约定

本仓库的 `latest` 分支聚焦 RK3588 NPU 8085 康复训练闭环。提交前请确认改动属于当前主线，并保持模块职责清晰。

## 基本规则

- 新功能应接入 `rehab_app/` 的 8085 流程，不新增平行的端口入口或重复应用。
- 模型权重放在 `models/<功能>/`，患者数据放在 `data/`，进程状态放在 `runtime/`；三者均不得提交真实内容。
- 模型、数据和密钥不得通过修改 `.gitignore` 强制加入仓库。
- 文件名应描述实现职责；临时、备份、平台专用试验脚本不进入主分支。
- 调整模块路径时同步更新 Python 导入、Shell 脚本、配置文件、文档和契约测试。

## 本地检查

```bash
python -m compileall rehab_app pose_estimation training evaluation action_feedback action_scoring speech llm scripts tests
python -m pytest
git diff --check
```

真实 RKNNLite、摄像头、ALSA、systemd 和性能指标必须在 RK3588 上按 `docs/npu_rehab_8085_guide.md` 验收。
