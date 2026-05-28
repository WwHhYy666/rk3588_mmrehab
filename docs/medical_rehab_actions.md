# 医学依据动作说明

这份文档用于说明当前演示版固定的低强度骨科康复动作。动作类型固定，但训练标准不固定：医生仍然需要先为每个患者录制标准模板，患者训练时按自己的模板比对。

重要原则：

- 医学资料用于证明动作类型合理、常见、适合康复演示。
- 具体训练目标来自医生录制的患者个性化模板。
- YAML 中的阈值只表示允许偏离模板多少，不表示统一医学标准角度。
- 大模型回答不能作为医学依据，只能辅助整理话术。

## 1. 坐姿伸膝 / 坐姿平举腿

动作 ID：

```text
seated_knee_extension
```

中文名：

```text
坐姿伸膝
```

适合演示的原因：

- 动作强度低，可以坐在椅子上完成。
- 对摄像头要求相对简单，侧对镜头时髋、膝、踝关键点较容易识别。
- 主要观察膝关节伸展程度，适合做“再抬高一点”“保持住”“慢一点”的实时反馈。

医学/康复来源：

- AAOS Knee Conditioning Program: https://orthoinfo.aaos.org/en/recovery/knee-conditioning-program/
- AAOS Knee Exercises: https://orthoinfo.aaos.org/en/staying-healthy/knee-exercises
- Leeds Teaching Hospitals NHS Trust Knee Exercises: https://www.leedsth.nhs.uk/patients/resources/knee-exercises/
- OPAL / South Tees Hospitals NHS Foundation Trust Seated knee extension: https://www.opalreturntowork.nhs.uk/exercises/seated-knee-extension/
- Cambridge University Hospitals General exercises: https://www.cuh.nhs.uk/patient-information/general-exercises/

摄像头姿势：

- 建议患者坐在稳定椅子上。
- 侧对镜头。
- 目标腿一侧的髋、膝、踝尽量无遮挡。

检测指标：

- 主指标：`knee_extension_angle`，用髋-膝-踝三点计算膝关节伸展角。
- 活动度 ROM：患者伸膝角度变化是否接近医生模板。
- 目标区间保持时间 TUT：伸直后是否保持到医生模板比例。
- 峰值角速度 speed：伸腿或放下是否过快。

注意事项：

- 不要求患者抬到统一角度。
- 医生模板中的最大伸展幅度和保持时间是患者本次训练目标。
- 如果患者比模板幅度明显不足，则提示“再伸直一点”。

## 2. 站姿屈膝后勾腿 / Standing Hamstring Curl

动作 ID：

```text
standing_hamstring_curl
```

中文名：

```text
站姿屈膝后勾腿
```

适合演示的原因：

- 动作幅度清楚，侧对镜头时髋、膝、踝关键点更容易识别。
- 可以手扶椅背或墙面，强度较低，适合骨科康复演示。
- 适合演示“不到位不计数”和“小腿再往后弯一点”的实时反馈。

医学/康复来源：

- AAOS Knee Conditioning Program: https://orthoinfo.aaos.org/en/recovery/knee-conditioning-program/
- AAOS Knee Exercises: https://orthoinfo.aaos.org/en/staying-healthy/knee-exercises
- Mayo Clinic hamstring curl: https://www.mayoclinic.org/healthy-lifestyle/fitness/multimedia/hamstring-curl/vid-20084673

摄像头姿势：

- 患者站稳，侧对镜头。
- 手扶椅背或墙面，避免失衡。
- 目标腿一侧的髋、膝、踝尽量无遮挡。

检测指标：

- 主指标：`hamstring_curl_flexion_angle`，用髋-膝-踝计算膝关节屈曲角。
- 活动度 ROM：小腿后勾幅度是否接近医生模板。
- 动作保持时间：后勾到位后是否保持到医生模板比例。
- 动作速度：后勾或放下是否过快。

注意事项：

- 不把“小腿必须弯到某个固定角度”写成硬标准。
- 患者需要保持大腿基本垂直，主要做小腿向后弯。
- 如果幅度不足，应提示“小腿再往后弯一点”。

## 3. 坐站训练

动作 ID：

```text
sit_to_stand
```

中文名：

```text
坐站训练
```

适合演示的原因：

- Sit-to-stand 是常见的下肢力量和功能训练动作。
- 动作对普通观众直观，视频演示效果好。
- 可以展示完整动作识别、自动计数和组间休息提示。

医学/康复来源：

- NHS Strength exercises: https://www.nhs.uk/live-well/exercise/strength-exercises/
- NHS Forth Valley Right Decisions Super Six: https://www.rightdecisions.scot.nhs.uk/exercise-before-and-after-surgery/exercises/the-super-six/
- Newcastle Hospitals osteoarthritis hip and knee exercises: https://www.newcastle-hospitals.nhs.uk/services/newcastle-occupational-health-service/information-for-staff/physiotherapy/self-help-leaflets/osteoarthritis-oa-hip-and-knee/
- University Hospital Southampton knee strengthening exercises PDF: https://www.uhs.nhs.uk/Media/UHS-website-2019/Patientinformation/Medicinestherapiesandanaesthetics/Knee-strengthening-exercises-3923-PIL.pdf

摄像头姿势：

- 建议患者坐在稳定椅子前半部分。
- 侧前方或正侧方拍摄。
- 髋、膝、踝和躯干尽量在画面中。

检测指标：

- 主指标：`hip_rise_height_ratio`，用髋部相对初始坐姿的上升高度除以肩髋距离。
- 辅助指标：`knee_extension_angle`，检查站起过程中膝关节是否有伸展变化。
- 完整站起和坐下的动作周期。
- 动作速度。

注意事项：

- 椅子要稳定，不要带轮。
- 演示时目标次数建议 3 次，保证视频流畅。
- 当前系统仍以医生模板为目标，不设置统一站起高度标准。

## 4. 配置口径

每个动作的 `evaluate/configs/*.yaml` 都采用同一原则：

```yaml
thresholds:
  rom_diff_max: 10.0
  tut_ratio_min: 0.8
  dtw_normalized_max: 0.25
  speed_ratio_max: 1.5
```

含义：

- `rom_diff_max`：患者比医生模板少多少幅度后判定动作不到位。
- `tut_ratio_min`：患者保持时间低于医生模板多少比例后判定保持不足。
- `dtw_normalized_max`：患者动作曲线偏离模板过多时判定轨迹不标准。
- `speed_ratio_max`：患者速度超过模板过多时判定动作过快。

这些阈值不是医学标准角度，而是演示系统的模板偏差阈值。
