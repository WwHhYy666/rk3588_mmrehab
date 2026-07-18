# Training evaluation

`evaluation/` 负责把医生模板和患者 attempt 转换为结构化训练报告。

```text
evaluation/
├─ report_generator.py   报告入口与命令行
├─ core/                 ROM、TUT、速度、DTW、错误分类和模板健康度
├─ configs/              通用动作阈值
│  └─ npu/               8085 三动作阈值
└─ tests/                评估与模板契约测试
```

生成报告写入 `data/reports/`，不放在源码包内部，也不进入 Git。

命令行示例：

```bash
python -m evaluation.report_generator \
  --template data/npu/doctor_templates/<template.json> \
  --attempt data/npu/patient_attempts/<attempt.json> \
  --config evaluation/configs/npu/sit_to_stand.yaml \
  --out data/reports/report_sit_to_stand.json
```

8085 训练服务通常通过 `training/training_session.py` 自动调用本模块，无需人工重复生成。
