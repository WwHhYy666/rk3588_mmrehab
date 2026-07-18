# Vision models

8085 姿态主线需要：

```text
models/vision/yolov5n_raw_fp.rknn
models/vision/rtmpose_m_256x192_fp.rknn
```

- `yolov5n_raw_fp.rknn`：YOLOv5n raw head 人体检测。
- `rtmpose_m_256x192_fp.rknn`：top-down RTMPose COCO-17 关键点。

两者必须由与板端 RKNNLite2/`librknnrt.so` 匹配的 Rockchip 工具链生成。
