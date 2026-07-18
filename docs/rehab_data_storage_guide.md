# 8085 模板、训练数据与报告存储

代码与运行数据严格分离：

```text
rehab_app/       HTTP 服务、页面和存储逻辑
data/npu/        8085 模板、attempt 和摘要
data/reports/    报告与关键帧
runtime/npu/     PID、日志、活动模板索引和临时状态
```

## 数据角色

`rehab_app/services/result_storage.py` 统一保存两类记录：

- `doctor_template`：医生录入的标准动作模板；
- `patient_attempt`：患者训练动作片段。

在 `data/npu/` 下自动创建：

```text
doctor_templates/
patient_attempts/
summaries/
results/          旧格式兼容
results_log.md
```

报告由 `evaluation/report_generator.py` 生成到：

```text
data/reports/npu/
data/reports/npu/keyframes/
```

## 录入流程

1. 启动 `./scripts/start_npu_rehab_8085.sh`。
2. 打开 `http://板子IP:8085/doctor`。
3. 选择动作并录制医生模板。
4. 检查模板健康状态。
5. 将通过检查的模板写入 `runtime/npu/active_templates.json`。
6. 在 `/train` 使用同一 NPU 关键点体系完成患者训练。

NPU 模板不能与其他关键点体系的旧模板混用。

## 隐私与备份

- `data/`、`runtime/` 默认被 Git 忽略。
- 真实患者编号应匿名化。
- 对外发送日志前先检查问题文本、报告和图片。
- 备份时应使用受控存储，不把数据复制到公开仓库。
- 清理数据前先停止 8085 服务，避免与正在写入的任务竞争。
