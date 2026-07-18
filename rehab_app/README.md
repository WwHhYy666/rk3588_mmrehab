# Rehabilitation application

`rehab_app/` 是 8085 康复终端的应用层。

```text
rehab_app/
├─ server/
│  ├─ npu_rehab_server.py  8085 唯一启动入口
│  ├─ rehab_http_server.py HTTP 路由、摄像头与页面服务
│  ├─ camera_preflight.py  摄像头预检
│  └─ static/              医生页、训练页和状态 UI
└─ services/
   ├─ active_templates.py  活动模板注册
   ├─ result_storage.py    模板与 attempt 保存
   ├─ report_paths.py      报告图片安全路径
   ├─ report_visuals.py    报告可视化
   └─ llm_assistant.py     GLM/Qwen 路由
```

算法分别来自 `pose_estimation/`、`training/`、`evaluation/`、`action_feedback/` 和 `action_scoring/`，应用层只负责有序编排。
