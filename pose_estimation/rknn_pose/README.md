# YOLOv5n + RTMPose RKNN 后端

本目录只维护 NPU 8085 使用的姿态路线：

```text
yolov5n_raw_fp.rknn
-> person bbox
-> rtmpose_m_256x192_fp.rknn
-> COCO-17
-> 康复关键点适配
```

核心文件：

| 文件 | 作用 |
| --- | --- |
| `yolov5n_rtmpose_backend.py` | 检测、ROI、姿态推理、跟踪与诊断 |
| `rknn_backend.py` | 供共享 HTTP 应用调用的单路线包装器 |
| `pose_result.py` | 推理结果结构 |
| `pose_frame_adapter.py` | COCO-17 到训练帧结构 |
| `coco17_to_rehab.py` | 关键点名称映射 |
| `check_rknn_runtime.py` | 板端 RKNNLite/运行库检查 |
| `test_npu_8085_integration.py` | 8085 集成契约 |
| `test_yolov5n_rtmpose_pipeline.py` | 后处理与跟踪单元测试 |
| `test_npu_debug_8085.py` | 调试页资源互斥测试 |

模型默认放在仓库根目录的 `models/vision/`，但不会被 Git 跟踪。真实 NPU 验收必须在 RK3588 上进行。
