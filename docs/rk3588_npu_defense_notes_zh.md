# RK3588 NPU 康复训练项目答辩补充细节

这份文档补充 60 页答辩 PDF 和原制作计划中没有单独展开的工程细节。

它不重复介绍项目主流程，而是集中回答下面几类问题：

- 为什么当前板端进程可以证明是新代码，而不是旧服务没有重启。
- `/npu-debug` 为什么不会一直占用 NPU。
- 摄像头、推理、渲染、关键帧和评分线程怎样避免互相阻塞。
- YOLO 检测间隔、人体框缓存和关键点保持有什么区别。
- 训练语音、AI 语音、本地 Qwen 和姿态模型怎样协调资源。
- 哪些状态字段最适合现场排障。
- 当前完成度评分模型在重新训练时还有什么需要注意。

本文只讨论当前正式的 `8085 + YOLOv5n raw + RTMPose` 姿态链路。

---

## 1. 8085 不是一个简单的端口副本

8085 的入口是：

```text
rehab_app/server/npu_rehab_server.py
```

它复用公共网页和训练框架，但会在启动时替换以下运行路径：

```text
端口                    8085
运行目录                runtime/npu/
active template         runtime/npu/active_templates.json
医生/患者数据           data/npu/
训练报告                data/reports/npu/
关键帧                  data/reports/npu/keyframes/
训练计划                training/configs/rehab_demo_plan_npu.yaml
实时基础配置            training/configs/training_defaults_npu.yaml
动作评估配置            evaluation/configs/npu/*.yaml
```

因此，8085 的模板、患者 attempt、报告、日志和配置不会和其他实验路线混在一起。

当前训练逻辑版本是：

```text
npu_training_v8_stage2_pipeline
```

这个字段会出现在 `/status` 的：

```text
training.training_logic_version
```

> 答辩说法：8085 不是只换端口，而是有独立入口、独立数据目录、独立配置、独立资源生命周期和独立状态标识。

---

## 2. 如何证明板子运行的是刚上传的新代码

现场最常见的问题之一是：文件已经上传，但 systemd 仍然运行旧进程。

8085 启动时会对关键源文件计算 SHA256，并生成一个 16 位构建标识：

```text
runtime.build_id
runtime.source_hashes
```

检查脚本会重新读取磁盘文件并计算哈希，比较：

```text
运行进程启动时记录的哈希
        是否等于
当前项目目录中文件的哈希
```

关键验收字段：

```text
runtime.project_root
runtime.entrypoint
runtime.pid
runtime.started_at
runtime.build_id
runtime.source_hashes
deployment.root_ok
deployment.hashes_ok
```

如果出现下面任意情况，不能开始正式验收：

- `runtime.build_id` 仍是旧值。
- `deployment.hashes_ok` 不是 `True`。
- `runtime.project_root` 不是当前项目目录。
- `training.logic_version` 仍是旧版本。

正确操作：

```bash
sudo systemctl restart rehab-station-npu-8085.service
./scripts/check_npu_rehab_8085.sh
```

> 答辩说法：我们不仅检查端口是否打开，还通过进程启动时的源码哈希和当前磁盘哈希判断服务是否真正加载了新版本。

---

## 3. `/npu-debug` 页面为什么不会忘记释放 NPU

打开：

```text
http://板子IP:8085/npu-debug
```

页面本身不会立即加载 YOLO 和 RTMPose。

只有点击“开始 NPU 检测”后，调试租约才生效。

关键参数：

```text
页面心跳间隔        5 秒
调试租约有效期      15 秒
```

代码字段：

```text
NPU_DEBUG_HEARTBEAT_SECONDS = 5.0
NPU_DEBUG_LEASE_SECONDS = 15.0
```

正常流程：

```text
点击开始
-> 后端启用 debug lease
-> 页面每 5 秒发送 heartbeat
-> 姿态模型保持加载
```

释放流程包括：

- 点击“停止并释放 NPU”。
- 关闭页面时通过 `sendBeacon` 请求停止。
- 页面崩溃或网络断开，超过 15 秒没有心跳。
- 开始医生录制或患者训练时，自动关闭 debug 状态。
- 提交 Qwen 问答时，停止 debug 并释放姿态模型。

相关状态：

```text
npu_debug.active
npu_debug.heartbeat_seconds
npu_debug.lease_seconds
npu_debug.lease_expires_at
npu_debug.last_heartbeat_at
npu_debug.last_error
```

> 答辩说法：调试页面采用租约而不是永久开关。即使浏览器异常退出，后端也会在租约超时后自动释放 NPU。

---

## 4. NPU 资源状态机

姿态后端内部维护资源状态：

```text
loading
pose_active
releasing
qwen_available
error
```

同时记录资源所有者：

```text
owner = pose
owner = qwen
```

加载过程：

```text
录模板/训练/调试需要姿态
-> det_model.load()
-> pose_model.load()
-> state = pose_active
-> owner = pose
```

释放过程：

```text
训练结束/保存/取消/停止
-> state = releasing
-> det_model.release()
-> pose_model.release()
-> 清空检测缓存和跟踪状态
-> state = qwen_available
-> owner = qwen
```

如果当前没有业务请求姿态，`infer()` 不会偷偷重新加载模型，而是返回一张带有“NPU pose released - Qwen available”提示的预览状态。

关键状态字段：

```text
npu_resource.state
npu_resource.owner
npu_resource.models_loaded
npu_resource.det_model_loaded
npu_resource.pose_model_loaded
npu_resource.det_model_path
npu_resource.pose_model_path
npu_resource.det_core_mask
npu_resource.pose_core_mask
npu_resource.last_loaded_at
npu_resource.last_released_at
npu_resource.last_error
```

---

## 5. 哪些 HTTP 请求会加载或释放姿态模型

8085 对请求进行了资源分类。

需要姿态的接口：

```text
/api/start
/api/realtime/start
/api/realtime/start_playlist
```

明确释放姿态的接口：

```text
/api/save
/api/cancel
/api/clear
/api/realtime/stop
```

需要把资源交给 Qwen 的接口：

```text
/api/voice/ask
/api/llm/report_summary
/api/llm/ask
```

保护规则：

- 如果小爱正在生成回答，开始录入或训练会返回明确错误，不会强行打断 Qwen。
- 开始训练前会关闭独立 debug lease，但不会提前释放即将使用的姿态模型。
- 提交 Qwen 前会关闭 debug，并调用 `release_pose_for_qwen()`。
- 保存、取消、清空或停止训练后，在 `finally` 中再次尝试释放，避免异常分支漏释放。

---

## 6. 摄像头打开方式的真实回退顺序

请求设备：

```text
/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
```

请求规格：

```text
1280 x 720
30 FPS
MJPG
```

打开策略：

```text
Python GI GStreamer + mppjpegdec
-> Python GI GStreamer + jpegdec
-> OpenCV GStreamer（如果环境支持）
-> OpenCV V4L2
```

为什么单独使用 Python GI：

- 某些板端 OpenCV 编译信息会显示 `GStreamer: NO`。
- 这不代表系统不能使用 GStreamer。
- Python GI 可以直接创建 pipeline 和 appsink，不依赖 OpenCV 的 GStreamer 编译开关。

为什么优先 `mppjpegdec`：

- 它使用 Rockchip 媒体硬件解码 JPEG。
- 能减少 CPU JPEG 解码压力。
- 如果插件不可用或协商失败，自动回退到软件 `jpegdec`。

状态字段：

```text
camera_source.kind
camera_source.requested_device
camera_source.active_device
camera_source.open_mode
camera_gstreamer_backend_requested
camera_gstreamer_gi_available
camera_gstreamer_gi_error
camera_open_attempts
camera_open_failures
```

---

## 7. 请求 30 FPS 不代表现场一定有 30 FPS

脚本请求：

```text
RK_CAMERA_FPS=30
```

但是实际摄像头帧率可能受以下因素影响：

- 自动曝光。
- 室内照明不足。
- USB带宽或驱动协商。
- JPEG解码方式。
- 摄像头固件自动降帧。

为了优先保持固定帧率，脚本在摄像头支持时设置：

```text
exposure_auto_priority = 0
```

对应变量：

```text
RK_CAMERA_FIXED_FPS=1
```

代价是暗环境画面可能变暗。正确处理顺序是：

1. 增加现场照明。
2. 查看 `camera.actual_fps` 和 `camera.capture_fps`。
3. 如果画面确实过暗，再将 `RK_CAMERA_FIXED_FPS=0` 做对比。

不要为了让数字好看重复发送旧帧。FPS必须对应真实新采集帧。

---

## 8. 为什么队列只保留最新一帧

实时系统有两种失败方式：

```text
丢帧
延迟积累
```

康复训练更不能接受延迟积累。

如果处理速度稍慢，保留所有旧帧会造成：

```text
摄像头已经看到当前动作
但页面和状态机仍在处理几秒前的动作
```

因此采集队列和渲染队列采用 latest-only：

```text
新任务进入
-> 如果队列已有旧任务，先丢弃旧任务
-> 队列始终保留最新帧
```

对应诊断字段：

```text
frame_queue_drops
render_queue_drops
latest_capture_id
latest_inference_id
latest_rendered_capture_id
latest_rendered_inference_id
```

> 答辩说法：我们允许可控丢弃旧帧，以换取更低的端到端延迟。实时反馈的第一目标是新鲜度，而不是离线视频逐帧完整处理。

---

## 9. 采集、姿态、渲染、关键帧四条线程的关系

### 9.1 摄像头采集线程

职责：

- 从设备读取最新BGR帧。
- 记录读取耗时、采集FPS和连续失败次数。
- 必要时重新打开摄像头。
- 分配 `capture_id`。
- 把最新帧送入姿态队列。

### 9.2 姿态线程

职责：

- 顺序调用YOLO和RTMPose。
- 更新检测缓存和跟踪ROI。
- 生成康复关键点和动作指标。
- 调用训练状态机。
- 把渲染所需数据送入渲染队列。

RKNN Runtime不由多个线程同时调用，降低Runtime并发不确定性。

### 9.3 渲染线程

职责：

- 绘制人体框。
- 绘制稳定后的COCO-17。
- 绘制训练文字和调试信息。
- 缩放为网页流分辨率。
- 编码JPEG。

没有 `/stream.mjpg` 客户端时，它直接跳过缩放、绘制和JPEG编码。

### 9.4 关键帧线程

职责：

- 接收动作最佳帧候选。
- 异步JPEG编码。
- 写入关键帧目录。
- 返回相对路径给训练报告。

慢写盘不会反向阻塞姿态推理。

---

## 10. 关键帧为什么不是每隔几帧就保存一张

当前关键帧策略更节省资源：

```text
动作主指标刷新最佳值
-> 复制一张较低分辨率BGR候选
-> 不立即编码
-> rep结束后只异步编码最佳候选
```

优点：

- 不在每一帧执行JPEG编码。
- 不在姿态线程中写磁盘。
- 每次动作最终只保存有代表性的峰值帧。
- 报告图片更容易对应“动作做到最高/最深的位置”。

`action_generation` 用于防止串动作：

```text
切换动作
-> generation + 1
-> 旧generation的关键帧任务不能写入新动作
```

相关字段：

```text
keyframe_queue_drops
keyframe_encode_count
keyframe_encode_error
keyframe_encode_ms
keyframe_write_ms
```

---

## 11. YOLO检测缓存、姿态跟踪和显示保持不是同一个东西

### 11.1 检测缓存

```text
RKNN_DET_CACHE_SECONDS=1.5
```

用途：短暂YOLO漏检时复用最近人体框。

### 11.2 自适应检测调度

```text
RKNN_DET_REFRESH_SECONDS=0.75
RKNN_DET_RETRY_SECONDS=0.25
RKNN_DET_BAD_POSE_FRAMES=2
```

触发YOLO的原因包括：

```text
initial_or_lost
lost_retry
pose_quality_drop
periodic_refresh
```

### 11.3 关键点生成的跟踪ROI

RTMPose输出的高置信关键点可反算人体范围，作为下一帧ROI。

```text
RKNN_TRACKER_MIN_POINTS=5
RKNN_TRACKER_MARGIN=0.20
RKNN_TRACKER_ALPHA=0.35
```

### 11.4 显示保持

显示层允许短暂保持脸部、肘、腕或人体框，使画面不闪烁。

```text
RKNN_DISPLAY_MAX_HOLD_FRAMES=4
RKNN_DISPLAY_BBOX_HOLD_FRAMES=6
```

显示保持不等于状态机继续相信旧关键点。

> 答辩说法：检测缓存解决人体框短暂丢失，关键点跟踪降低YOLO调用频率，显示保持只改善视觉连续性，三者不能混为一谈。

---

## 12. 训练稳定器与显示稳定器为什么分开

训练稳定器关注：

- 肩、髋、膝、踝是否足够可靠。
- 单帧跳变是否可能影响角度和计数。
- 短暂漏点是否可以有限保持。

显示稳定器关注：

- 17点骨架是否视觉连续。
- 脸、肘、腕短暂低置信度时是否闪烁。
- 人体框是否平滑。

重要原则：

```text
显示层可以更宽松
训练判定必须更严格
```

因此页面上看到一个保持的点，不代表这个点正在参与ROM、TUT或计数。

相关状态：

```text
display.keypoint_count
display.held_keypoints
display.jump_pending
display.bbox_held
display.bbox_jump_pending
pose.jump_pending
pose.jump_recovered
pose.jump_counts
```

---

## 13. 人体框还有哪些隐藏保护

当前框扩展参数：

```text
基础扩展                1.25
顶部额外扩展            0.10
宽人体判断比例          0.65
宽人体最大水平扩展      1.50
```

扩展框可能超出原图，这是允许的，因为 `padded_crop()` 会补边。

但是如果扩展框在图外的面积过大，系统会：

1. 计算 `padding_ratio`。
2. 判断裁剪面积是否异常大。
3. 回退到更保守的安全框。

相关诊断：

```text
selected_yolo_bbox
rtmpose_expanded_bbox
rtmpose_restore_bboxes
rtmpose_roi_fallbacks
rtmpose_padding_ratios
rtmpose_applied_x_scales
```

这些字段适合回答“为什么某次骨架全部挤在画面边缘”。

---

## 14. 模板健康检查不只是检查文件存在

NPU模板开始训练前会验证：

- 文件存在且可解析。
- `pose_backend` 是当前NPU姿态后端。
- 关键点schema匹配。
- 有足够的有效帧。
- 录制持续时间足够。
- 峰值不能出现在过于靠近序列边缘的位置。
- 动作结束时已经回到接近起始位置。
- ROM位于动作配置允许范围。

配置字段示例：

```text
template_validation.min_valid_frames
template_validation.min_duration_seconds
template_validation.peak_edge_fraction
template_validation.return_rom_tolerance_ratio
template_validation.min_rom
template_validation.max_rom
```

状态字段：

```text
training.template_health.ok
training.template_health.rom
training.template_health.reason
training.template_health.message
```

> 答辩说法：系统不是只要有模板文件就开始训练，还会检查这段录制是否形成了完整、合理、可用于比较的标准动作。

---

## 15. 正面确认和侧面确认的具体阈值

朝向判断使用左右肩、左右髋的横向宽度与躯干高度比值。

当前口径：

```text
orientation.ratio >= 0.55    正面
orientation.ratio <= 0.32    侧面
0.32 < ratio < 0.55          不通过
```

流程：

```text
先正对镜头
-> 连续确认
-> 提示侧身
-> 连续确认侧面
-> 播放角度正确
-> 动作提示
-> baseline
```

双侧躯干点不足时，可以使用近侧三点链作为侧面fallback，但它不能跳过第一次正面确认。

状态字段：

```text
orientation.phase
orientation.state
orientation.front_ok
orientation.side_ok
orientation.front_count
orientation.front_need
orientation.side_count
orientation.side_need
orientation.ratio
orientation.visibility
orientation.mode
orientation.message
```

---

## 16. NPU presence与离屏恢复的细节

NPU计划配置：

```text
npu_presence_v2: true
npu_presence_enter_frames: 2
npu_presence_grace_frames: 8
npu_return_core_points_min: 5
npu_return_core_visibility_min: 0.12
return_confirm_frames: 2
return_orientation_required: false
```

含义：

- 需要连续看到人物，而不是单帧看到就认为已返回。
- 允许8帧短暂漏点，不立即进入离屏。
- 返回时至少有5个核心关键点达到0.12可见度。
- 返回确认不强制重新完成完整朝向流程。
- 恢复后仍要重新baseline，不能直接沿用离屏前的运动起点。

状态字段：

```text
presence.raw
presence.stable
presence.npu_hits
presence.npu_misses
return_core_points
reentry.state
reentry.ready
rebaseline.state
rebaseline.pending
rebaseline.cycles
```

---

## 17. 8085训练语音的限制比普通TTS更严格

NPU训练计划启用：

```text
training_fixed_audio_only: true
```

含义：

- 训练中的开始、次数、纠错、休息和完成提示只使用预录WAV。
- 没有对应WAV的提示只显示文字。
- 训练中不会临时初始化自然TTS、pyttsx3或espeak。
- 这样可以减少训练期间的CPU、内存和音频设备不确定性。

音频时序：

```text
动作提示WAV播放
-> 等待播放结束
-> 进入动作baseline
-> RISING/HOLDING/RETURNING内部不播报
-> rep完全结束
-> 播放次数或纠错WAV
-> 等待反馈播放结束
-> 允许下一次动作
```

训练全部完成后，还要等最后一条 `finished.wav` 播放结束，才能开放：

- 问答。
- 唤醒录音。
- AI自然女声。

诊断字段：

```text
training.fixed_audio
training.tts_backend
training.tts_initialized
count.rep_settled_at
count.wav_queued_at
count.wav_started_at
count.settle_delay_s
count.queue_delay_s
```

---

## 18. 完成度评分队列的隐藏行为

每个可识别尝试结束后：

```text
构造 quality_attempt_segment
-> put_nowait 放入评分队列
-> 训练线程立即继续
```

如果队列满：

```text
不等待
不阻塞
增加 dropped_attempts
训练继续
```

后台worker回写前会检查：

```text
item.session_id == current_session_id
item.action_id == current_action_id
```

这可以防止：

- 上一个患者的结果写进新患者。
- 上一个动作的分数写进下一个动作。
- 页面重开后旧异步结果覆盖当前状态。

状态字段：

```text
action_scoring.available
action_scoring.backend
action_scoring.model_path
action_scoring.last_score_time_ms
action_scoring.queue_size
action_scoring.dropped_attempts
action_scoring.scored_attempts
action_scoring.worker_alive
action_scoring.last_error
```

---

## 19. 当前评分文件与重新训练路径注意事项

当前仓库已经存在三个动作的：

```text
best.pt
model.onnx
train_summary.json
```

训练摘要样本数：

```text
sit_to_stand                 68
standing_hamstring_curl      75
seated_knee_raise            61
```

当前在线评分后端是：

```text
onnx_cpu
```

主仓库当前没有这三个动作的 `model.rknn`，因此不要把姿态NPU和完成度评分后端混为一谈。

### 当前检出版本的重训练路径风险

下面两个文件当前把训练数据根目录组合成了：

```text
data/default/docs
```

相关文件：

```text
action_scoring/dataset.py
action_scoring/check_dataset.py
```

而实际项目数据主要位于：

```text
data/default/results/
data/default/patient_attempts/
data/npu/
```

因此当前在Windows检出版本直接运行 `check_dataset.py` 可能显示0样本。

这不会影响已经导出的 `model.onnx` 在线推理，但会影响：

- 重新统计数据集。
- 现场演示重新训练。
- 新增数据后再次导出模型。

答辩现场不建议临时重新训练。后续重训练前应先核对并修正数据根目录，再检查类别平衡。

> 诚实说法：现有模型文件和训练摘要已经存在，在线ONNX评分可用；重新训练的数据扫描路径需要在正式扩充数据前再次整理。

---

## 20. 规则优先完成度还有哪些细节

NPU训练逻辑下，正确动作：

```text
规则分初值约 90
规则分占 75%
ONNX原始分占 25%
最终限制在 70-96
```

规则分会根据以下因素微调：

- ROM超过目标的程度。
- TUT超过目标的程度。
- 速度比是否偏快。
- 是否使用watchdog结束。
- 是否经历可见度恢复。

错误动作封顶：

```text
ROM_LOW          70
TUT_LOW          78
SHAPE_BAD        75
TOO_FAST         82
VISIBILITY_LOW   50
```

报告中建议同时保留：

```text
completion_percent
rule_score
model_score
error_cap
calibration_mode
```

这样评委问“这个分怎么来的”时，可以拆成规则分、模型分和错误上限三个部分解释。

---

## 21. report不是对整个动作随便取平均

报告会区分：

```text
quality_attempts      所有可识别尝试
reps                  可计数动作
invalid_attempts      不可计数动作
overall_quality       可用完成度汇总
selected_attempts     代表性尝试
```

代表性尝试包括：

```text
best_correct          最有代表性的正确尝试
representative_wrong  最适合解释的错误尝试
```

选择时会考虑：

- 是否可计数。
- 完成度分数。
- ROM和TUT表现。
- 错误类型。
- 是否有关键帧。

AI上下文优先使用压缩后的代表性字段，而不是把整份巨大JSON原样交给模型。

---

## 22. Qwen proxy做的不只是转发HTTP

调用链：

```text
8085主服务
-> 127.0.0.1:18080/generate
-> 127.0.0.1:8080/rkllm_chat
-> RKLLM Runtime
```

proxy职责：

- `/health` 健康检查。
- 请求队列和串行化。
- 统一JSON返回结构。
- 解析流式或非流式上游文本。
- 判断上游busy。
- busy退避重试。
- 空回答重试。
- 记录queue wait和retry count。

默认busy退避：

```text
0.8 秒
1.2 秒
1.8 秒
```

本地Qwen请求参数：

```text
temperature = 0.2
max_new_tokens <= 96
timeout = 120 秒
```

如果compact提示词返回空文本，系统再尝试更短的minimal提示词，而不是无限重复同一请求。

---

## 23. Qwen报告上下文为什么要压缩

完整report包含大量字段、关键帧路径和每次尝试细节，直接输入1.5B模型容易：

- 超过上下文能力。
- 增加首token时间。
- 返回空文本或重复内容。
- 让模型注意不到患者真正的问题。

压缩后保留：

```text
动作名称
主指标
ROM
TUT
速度
主要错误
结构化反馈
总体完成度
质量模型后端
最佳正确尝试
代表性错误尝试
```

本地提示词要求：

- 只基于报告回答。
- 不诊断疾病。
- 不替代医生。
- 直接中文回答1到3句。
- 不输出JSON。
- 不重复题目。

回答返回后还会：

- 去除Markdown符号。
- 把内部错误码翻译成中文。
- 去除重复段落。
- 截短口播文本。

---

## 24. 前端如何防止旧回答覆盖新回答

异步问答可能出现：

```text
问题A提交
-> 问题B提交
-> A比B更晚返回
```

如果没有保护，页面会把旧的A答案覆盖B。

前端保护：

```text
voicePollToken
displayedVoiceAnswerJobId
voiceJobId
```

逻辑：

- 每次提交新问题，`voicePollToken + 1`。
- 旧轮询发现token不匹配，立即退出。
- 返回结果的job_id必须等于当前job_id。
- 同一个job_id只允许展示和播报一次。

后端保护：

```text
speech_generation
current_job
last_completed_job
recent_jobs
```

新问题提交时会使旧的待播报语音generation失效，并清理排队中的AI语音。

限制：已经进入系统播放器的旧音频不能保证瞬间硬中断。项目主要保证旧结果不再继续排队或覆盖页面。

---

## 25. ASR还有哪些输入质量保护

默认ASR：

```text
provider              sherpa_paraformer
线程数                2
优先INT8              false
最小RMS               0.002
最小peak              0.015
边缘补零              180 ms
```

处理过程：

```text
读取WAV
-> 转单声道float32
-> 统计RMS和peak
-> 判断是否包含语音
-> 前后补少量静音
-> Paraformer识别
-> 过滤明显幻觉文本
```

为什么补静音：

- 录音切得太紧时，模型可能丢掉第一个或最后一个字。
- 前后补少量零不会改变语音内容，但能改善边界稳定性。

为什么先检查音量：

- 完全静音直接送模型容易产生幻觉文字。
- 可以区分“麦克风没有声音”和“识别模型不准确”。

---

## 26. 系统状态字段应该按层看

### 摄像头层

```text
camera_capture_fps
camera_read_ms
camera_frame_age_ms
camera_read_failures
camera_reopen_count
camera_source.active_device
camera_source.actual_resolution
camera_source.actual_fps
```

### 姿态层

```text
pose_fps
inference_fps
rknn_infer_call_ms
pose_process_ms
det_inference_ms
pose_inference_ms
detector_trigger_reason
detector_age_ms
tracker_roi
tracker_quality
tracker_visible_points
```

### 端到端延迟层

```text
capture_to_inference_age_ms
capture_to_stream_age_ms
stream_frame_age_ms
queue_wait_ms
render_total_ms
jpeg_encode_ms
```

### 训练层

```text
training.status
training.action_id
training.completed_reps
training.invalid_attempts
training.machine_internal_state
training.runtime_thresholds
training.last_invalid_attempt
```

### NPU资源层

```text
npu_resource.state
npu_resource.owner
npu_resource.models_loaded
npu_resource.last_error
```

### 语音和AI层

```text
voice.qa_allowed
voice.current_job
voice.last_completed_job
llm.active_provider
llm.last_latency_ms
llm.last_error
```

---

## 27. FPS与延迟应该怎样验收

使用：

```bash
python3 scripts/benchmark_npu_rehab_8085.py --scenario idle
python3 scripts/benchmark_npu_rehab_8085.py --scenario npu-debug
python3 scripts/benchmark_npu_rehab_8085.py --scenario doctor
python3 scripts/benchmark_npu_rehab_8085.py --scenario train
```

每个场景默认：

```text
持续60秒
每0.5秒读取一次/status
输出P50、P95、mean、min、max
记录温度和错误
```

验收时优先看：

```text
camera_capture_fps
inference_fps
training_update_fps
stream_fps
capture_to_inference_age_ms P95
capture_to_stream_age_ms P95
render_queue_drops
temperature
```

当前文档口径：

- 摄像头实际约15 FPS时，姿态和网页稳定12-15 FPS是合理目标。
- 只有摄像头真实达到25 FPS以上，才要求姿态达到20-25 FPS。
- `capture_to_stream_age_ms` P95建议低于250 ms。
- 如果960x540流不稳，可同场景测试640x360或将流限制到15 FPS。

不能只截取一瞬间最高FPS，也不能通过重复旧帧制造高FPS。

---

## 28. 页面缓存与“上传了但没变化”

8085静态JS/CSS使用文件版本号，并对直接静态资源设置禁止缓存策略。

目的：

- 上传新 `train.js` 后重启服务，浏览器能请求新版本。
- 避免现场仍运行旧JavaScript，后端已经更新但页面行为没变化。

遇到页面仍像旧版本：

1. 先重启8085服务。
2. 检查 `runtime.build_id` 和哈希。
3. 关闭康复专用Chromium窗口并重新打开。
4. 强制刷新页面。
5. 查看开发者工具Network中的JS版本参数。

不要只反复刷新而不确认后端进程是否已经换成新代码。

---

## 29. 当前需要记住的数据目录

```text
runtime/npu/active_templates.json
runtime/npu/logs/
runtime/npu/pids/
runtime/npu/voice/
runtime/npu/benchmarks/

data/npu/results/
data/npu/doctor_templates/
data/npu/patient_attempts/
data/npu/summaries/

data/reports/npu/
data/reports/npu/keyframes/

models/action_quality/sit_to_stand/
models/action_quality/standing_hamstring_curl/
models/action_quality/seated_knee_raise/
```

删除 `runtime/npu/active_templates.json` 后，训练页会失去三个动作的激活模板，必须重新在 `/doctor` 录制并激活。

清理历史数据前先备份，不要在答辩前做全目录清空。

---

## 30. 现场最短检查命令

```bash
cd /home/elf/project

sudo systemctl restart rehab-station-npu-8085.service

./scripts/check_npu_rehab_8085.sh

curl -s http://127.0.0.1:8085/status | python3 -m json.tool

sudo ss -ltnp | grep -E ':8080 |:8085 |:18080 '

tail -f runtime/npu/logs/npu_rehab_8085.log
```

显示屏：

```bash
REHAB_STATION_URL="http://127.0.0.1:8085/train?display=1" \
  ./scripts/open_npu_rehab_8085_kiosk.sh
```

独立姿态调试：

```bash
./scripts/open_npu_debug_8085_kiosk.sh
```

---

## 31. 现场故障按这个顺序查

### 页面打不开

1. `systemctl status`。
2. 检查8085端口。
3. 看启动日志。
4. 检查Python依赖和模型文件。

### 页面打开但没画面

1. 检查摄像头by-id文件是否存在。
2. 检查是否有其他进程占用设备。
3. 看 `camera.open_attempts` 和 `camera.open_failures`。
4. 重启8085服务。

### 有人体框但没骨架

1. 看RTMPose模型是否加载。
2. 看 `pose_output_shapes`。
3. 看 `keypoint_conf_range`。
4. 看ROI是否过度扩展或落在图外。

### 有骨架但不计数

1. 看朝向阶段。
2. 看start pose。
3. 看baseline。
4. 看motion delta。
5. 看状态机内部阶段。
6. 看是否完整返回。

### 训练完成但Qwen不能问

1. 确认最后一条训练WAV结束。
2. 看 `npu_resource.state=qwen_available`。
3. 检查8080和18080。
4. 请求18080 `/health` 和 `/generate`。

---

## 32. 评委可能继续追问的细节

### 为什么网页流限制20 FPS，而摄像头请求30 FPS？

姿态推理需要完整源帧，网页显示不需要每帧都编码。限制网页流可以降低JPEG编码和浏览器压力，不影响状态机处理频率。

### 为什么检测阈值设为0.70？

当前场景是单人、主体清晰、固定距离，较高阈值可减少错误框。它是工程初始值，仍需现场数据验证，不是理论最优值。

### 为什么NMS阈值0.65比常见0.45高？

人体框在单人场景中可能因不同尺度产生接近但不完全重合的候选。当前值配合0.70分数阈值使用，目的是避免过早抑制真实框；最终以现场稳定性为准。

### 为什么完成度模型很小？

输入只是30帧骨架，不是原始图像。两层Conv1D已能提取局部时间模式，小模型更适合异步板端运行和导出。

### 为什么本地Qwen只生成很短回答？

1.5B模型在端侧资源有限。限制到1-3句、96个新token，可以降低延迟、减少重复和空输出，并符合训练后简短指导的场景。

### 为什么不用大模型直接判断动作？

大模型生成慢、结果不确定、不可逐帧解释。实时计数必须使用确定性视觉指标和状态机，大模型只负责训练后解释。

---

## 33. 关键代码阅读地图

| 想理解什么 | 先看文件 | 重点函数/字段 |
| --- | --- | --- |
| 8085如何隔离运行 | `rehab_app/server/npu_rehab_server.py` | `configure_isolated_runtime`、`NpuRealtimeTrainingSession` |
| NPU何时加载释放 | `pose_estimation/rknn_pose/yolov5n_rtmpose_backend.py` | `load`、`release`、`resource_snapshot` |
| YOLO raw解码 | 同上 | `decode_yolov5_raw_heads`、`decode_yolov5_combined_heads` |
| 人体框和NMS | 同上 | `postprocess_yolov5_person`、`nms_numpy` |
| 检测/跟踪调度 | 同上 | `adaptive_detector_decision`、`tracker_bbox_from_keypoints` |
| RTMPose与SimCC | 同上 | `preprocess_rtmpose`、`decode_simcc` |
| 摄像头和四线程 | `rehab_app/server/rehab_http_server.py` | capture、pose、render、keyframe worker |
| 三动作总流程 | `training/training_session.py` | `start_playlist`、`_start_playlist_action` |
| 单次动作状态机 | `training/action_state_machine.py` | `process`、`_finish_rep` |
| NPU动作阈值 | `evaluation/configs/npu/*.yaml` | `realtime`、`template_validation` |
| 异步评分 | `training/training_session.py` | `_enqueue_quality_segment`、`_run_quality_score_worker` |
| 特征与30帧 | `action_scoring/features.py` | `build_feature_sequence`、`_resample_time` |
| 标签公式 | `action_scoring/labels.py` | `label_from_attempt_segment` |
| 显示分校准 | `action_scoring/completion_calibrator.py` | `calibrated_completion_details` |
| 报告生成 | `evaluation/report_generator.py` | `make_report`、`build_quality_attempt_reports` |
| Qwen路由 | `rehab_app/services/llm_assistant.py` | `_auto_answer`、`_local_qwen_answer` |
| Qwen proxy | `llm/rkllm_proxy.py` | `/health`、`/generate`、retry |
| 异步问答 | `speech/llm_worker.py` | `submit`、`result`、`_run` |
| ASR | `speech/asr_worker.py` | `_transcribe`、`_audio_has_speech` |
| 前端旧结果保护 | `rehab_app/server/static/train.js` | `voicePollToken`、job id |
| 板端状态检查 | `scripts/check_npu_rehab_8085.sh` | `/status`字段和哈希校验 |

---

## 34. 答辩前不要做的事情

- 不要临时重新转换YOLO或RTMPose权重。
- 不要临时重新训练完成度模型。
- 不要在没有备份时清空active template和患者数据。
- 不要同时打开多个摄像头服务。
- 不要在训练中连续点击Qwen问答。
- 不要只看瞬时FPS就宣布性能结果。
- 不要把ONNX完成度评分说成当前已经运行在NPU。
- 不要把训练完成度说成临床诊断或治疗效果。
- 不要为了演示数字好看重复旧帧。
- 不要出现问题就先改阈值，应先看状态和日志确定所属模块。

---

## 35. 最后需要记住的五句话

1. `8085` 是独立的NPU康复服务，不只是换了端口。
2. YOLO负责找人，RTMPose负责找17个关键点，状态机负责判断完整动作周期。
3. 采集、推理、渲染、关键帧、评分和语音异步隔离，慢任务不能阻塞实时训练。
4. 姿态模型与本地Qwen按业务阶段轮流使用NPU，训练完成后明确释放RKNN Runtime。
5. 当前完成度评分是规则优先加ONNX辅助，工程可用但不等同于临床准确率。
