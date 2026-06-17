# RKNN YOLOv8-Pose Smoke Test 操作手册

本手册用于独立验证 RK3588 的 RKNN YOLOv8-Pose NPU 推理链路。

本次目标只做官方例程 smoke test，不接入当前项目主流程，不修改 `prescription/`、`realtime/`、`vision/` 的现有 MediaPipe CPU 路线。MediaPipe 仍然是当前 8082 Demo 的默认稳定后端；RKNN/NPU 先作为并行验证支线。

## 1. 参考资料和有效页码

### 1.1 官方下载地址

- RKNN-Toolkit2 2.3.2: https://github.com/airockchip/rknn-toolkit2/releases/tag/v2.3.2
- RKNN Model Zoo 2.3.2: https://github.com/airockchip/rknn_model_zoo/releases/tag/v2.3.2
- YOLOv8-Pose 官方例程: https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8_pose

### 1.2 《基于RK3588的AI模型训练到部署.pdf》

有用页码按 PDF 阅读器页码记录：

- 第 2 页：文档升级到 RKNN 工具 2.3.2。
- 第 3 页：RK3588 对应 NPU driver 0.9.8，RKNN-Toolkit2 2.3.2，RKNN-Toolkit-Lite2 2.3.2。
- 第 14 页：RKNN Model Zoo 支持 `YOLOv8_pose（YOLOv8n-pose）`。
- 第 21-25 页：RKNN-Toolkit2、RKNN-Lite2、Model Zoo、模型转换、推理流程。
- 第 48-50 页：PC/虚拟机侧安装 RKNN-Toolkit2、下载 Model Zoo、执行 `convert.py`。
- 第 50-51 页：板端安装 RKNN-Toolkit-Lite2 wheel，尤其是 `cp310`、`aarch64` wheel 示例。
- 第 53-54 页：多线程 RKNNLite 推理池、`NPU_CORE_0/1/2`、`NPU_CORE_0_1_2`、轮询多核。
- 第 55 页：板端运行、OpenCV、摄像头环境变量。
- 第 56-58 页：常见问题，尤其是网络源、PyTorch 导出权重未内嵌、量化/预处理/后处理导致乱框或 null 框、Toolkit 与 Lite2 版本一致。

### 1.3 《AI 例程测试手册.pdf》

有用页码按 PDF 阅读器页码记录：

- 第 15 页：Model Zoo 支持 `YOLOv8_pose（YOLOv8n-pose）`。
- 第 16-24 页：ResNet / MobileNet / YOLOv5 的通用 Model Zoo 流程：`download_model.sh -> convert.py -> build-linux.sh -> 拷贝到板端 -> 运行 demo`。
- 第 45-48 页：DeepSeek / RKLLM 流程，属于后续大模型路线，不是本次 pose smoke test 主线。
- 第 52-59 页：Qwen2-VL / RKLLM / vision-rknn + rkllm 部署流程，适合后续“康复助教/图文理解”规划，不要混进本次 NPU pose 验证。

## 2. 整体路线

两台机器分工：

```text
Windows 电脑
  -> 安装 WSL2 Ubuntu 22.04
  -> 下载 rknn-toolkit2 2.3.2
  -> 下载 rknn_model_zoo 2.3.2
  -> 下载 yolov8n-pose.onnx
  -> 转换 yolov8n-pose.rknn

RK3588 板端
  -> 检查 NPU 驱动和 Python 版本
  -> 安装匹配的 rknn_toolkit_lite2 wheel
  -> 运行官方 yolov8_pose demo
  -> 观察 /sys/kernel/debug/rknpu/load 是否有非 0 波动
```

本 smoke test 的成功标准：

1. WSL 中生成 `yolov8n-pose.rknn`，且文件非 0 字节。
2. 板端能加载 `.rknn` 模型运行官方例程。
3. 板端推理时 `/sys/kernel/debug/rknpu/load` 的 `Core0/Core1/Core2` 至少一个出现非 0% 波动。

## 3. Windows 安装 WSL2 Ubuntu 22.04

在 Windows PowerShell 管理员窗口运行：

```powershell
wsl --install -d Ubuntu-22.04
```

安装完成后重启 Windows，打开 Ubuntu，按提示创建 Linux 用户名和密码。

检查：

```bash
lsb_release -a
uname -m
```

成功标志：

- `lsb_release -a` 显示 Ubuntu 22.04。
- `uname -m` 显示 `x86_64`。

常见问题：

- `wsl --install` 失败：启用“适用于 Linux 的 Windows 子系统”和“虚拟机平台”，重启后再试。
- Microsoft Store 下载失败：先运行 `wsl --update`，再重新安装。
- 后续 GitHub 下载慢：换网络，或下载 release zip 后手动解压。

## 4. WSL 安装基础依赖

进入 Ubuntu 终端：

```bash
sudo apt update
sudo apt install -y git wget curl unzip python3 python3-dev python3-venv python3-pip build-essential cmake pkg-config
sudo apt install -y libopencv-dev python3-opencv libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 libprotobuf-dev

python3 -V
python3 -c "import cv2; print(cv2.__version__)"
```

成功标志：

- Python 正常打印版本。
- OpenCV 正常打印版本。

常见问题：

- `apt update` 慢或失败：先确认 WSL 能联网；必要时换 Ubuntu 镜像源。
- `import cv2` 失败：确认安装了 `python3-opencv`，或者后续在 venv 中安装 `opencv-python-headless`。

## 5. 下载官方 RKNN 工具和 Model Zoo

建议统一放到 WSL 用户目录，不放进本项目仓库：

```bash
mkdir -p ~/rk3588_rknn_smoke
cd ~/rk3588_rknn_smoke

git clone -b v2.3.2 https://github.com/airockchip/rknn-toolkit2.git
git clone -b v2.3.2 https://github.com/airockchip/rknn_model_zoo.git
```

如果 `v2.3.2` 分支或 tag 拉取失败，改用：

```bash
cd ~/rk3588_rknn_smoke
git clone https://github.com/airockchip/rknn-toolkit2.git
git clone https://github.com/airockchip/rknn_model_zoo.git

cd ~/rk3588_rknn_smoke/rknn-toolkit2
git checkout v2.3.2

cd ~/rk3588_rknn_smoke/rknn_model_zoo
git checkout v2.3.2
```

检查：

```bash
ls ~/rk3588_rknn_smoke/rknn_model_zoo/examples/yolov8_pose
```

成功标志：

- 目录中应看到 `cpp`、`model`、`python`、`README.md`。

## 6. 下载 YOLOv8-Pose ONNX 权重

官方例程的权重下载脚本位于 `examples/yolov8_pose/model/download_model.sh`。

```bash
cd ~/rk3588_rknn_smoke/rknn_model_zoo/examples/yolov8_pose/model
chmod +x download_model.sh
./download_model.sh

ls -lh yolov8n-pose.onnx
test -s yolov8n-pose.onnx && echo "ONNX OK"
```

成功标志：

- `yolov8n-pose.onnx` 存在。
- 文件大小非 0 字节。
- 终端打印 `ONNX OK`。

注意：

- 本步骤不要随便从第三方网站找权重，优先用官方 `download_model.sh`。
- 如果下载失败，打开 `download_model.sh` 查看其中的 URL，用浏览器或 `wget -O yolov8n-pose.onnx URL` 手动下载。

## 7. WSL 安装 PC 端 RKNN-Toolkit2

创建隔离环境：

```bash
cd ~/rk3588_rknn_smoke
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate

python -V
python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
```

Ubuntu 22.04 默认通常是 Python 3.10，也就是 `cp310`。

安装 RKNN-Toolkit2：

```bash
cd ~/rk3588_rknn_smoke/rknn-toolkit2/rknn-toolkit2/packages/x86_64

python -m pip install --upgrade pip
python -m pip install -r requirements_cp310-2.3.2.txt
python -m pip install ./rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

python -c "from rknn.api import RKNN; print('RKNN Toolkit2 OK')"
```

如果 Python 是 3.11 或 3.12，把命令里的 `cp310` 改成 `cp311` 或 `cp312`。

成功标志：

- 终端打印 `RKNN Toolkit2 OK`。

常见问题：

- `No module named rknn`：venv 没激活，或 wheel 没装成功。
- 找不到 wheel：检查当前目录是否为 `rknn-toolkit2/rknn-toolkit2/packages/x86_64`。
- requirements 安装慢：可以加国内源，例如 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 8. WSL 转换 ONNX 为 RKNN

```bash
source ~/rk3588_rknn_smoke/.venv-rknn/bin/activate
cd ~/rk3588_rknn_smoke/rknn_model_zoo/examples/yolov8_pose/python

python convert.py ../model/yolov8n-pose.onnx rk3588 i8 ../model/yolov8n-pose.rknn

ls -lh ../model/yolov8n-pose.rknn
test -s ../model/yolov8n-pose.rknn && echo "RKNN OK"
```

成功标志：

- `../model/yolov8n-pose.rknn` 存在。
- 文件大小非 0 字节。
- 终端打印 `RKNN OK`。

如果 i8 量化失败，先记录完整错误，然后尝试浮点版验证链路：

```bash
python convert.py ../model/yolov8n-pose.onnx rk3588 fp ../model/yolov8n-pose_fp.rknn
ls -lh ../model/yolov8n-pose_fp.rknn
```

如果 `fp` 能成功而 `i8` 失败，说明链路基本通，问题集中在量化或校准数据。

## 9. RK3588 板端基础检查

SSH 登录板端后运行：

```bash
uname -m
python3 -V
ls -l /dev/rknpu*
sudo cat /sys/kernel/debug/rknpu/load
```

成功标志：

- `uname -m` 是 `aarch64`。
- `/dev/rknpu*` 存在。
- `rknpu/load` 能看到 `Core0/Core1/Core2` 或类似 NPU load 输出。

如果 `rknpu/load` 读不到：

```bash
sudo mount -t debugfs debugfs /sys/kernel/debug
sudo cat /sys/kernel/debug/rknpu/load
```

如果仍读不到，记录现象，不改主项目代码。

## 10. 从 WSL 传输 YOLOv8-Pose 示例到板端

在 WSL 中设置板端信息：

```bash
export BOARD_USER=你的板端用户名
export BOARD_IP=你的板端IP
```

传整个官方示例目录：

```bash
cd ~/rk3588_rknn_smoke/rknn_model_zoo/examples
scp -r yolov8_pose ${BOARD_USER}@${BOARD_IP}:~/yolov8_pose
```

检查：

```bash
ssh ${BOARD_USER}@${BOARD_IP} "ls -lh ~/yolov8_pose/model/yolov8n-pose.rknn ~/yolov8_pose/python/yolov8_pose.py"
```

成功标志：

- 板端能看到 `~/yolov8_pose/model/yolov8n-pose.rknn`。
- 板端能看到 `~/yolov8_pose/python/yolov8_pose.py`。

## 11. 板端安装 RKNN-Toolkit-Lite2

先确认板端 Python ABI：

```bash
python3 - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
```

假设输出 `cp310`，在 WSL 里传对应 `aarch64` wheel：

```bash
scp ~/rk3588_rknn_smoke/rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl ${BOARD_USER}@${BOARD_IP}:~/
```

板端安装：

```bash
python3 -m pip install --upgrade pip
python3 -m pip install ~/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
```

如果板端 Python 是 `cp311` 或 `cp312`，把 wheel 文件名中的 `cp310` 改成对应版本。

成功标志：

- 终端打印 `RKNNLite OK`。

常见问题：

- `No module named rknnlite`：wheel 没装成功，或装到了另一个 Python 环境。
- `not a supported wheel on this platform`：`cpXX` 或 `aarch64` 架构不匹配。
- `externally-managed-environment`：可以使用 venv，或按板端系统策略加 `--break-system-packages`，但要先记录环境状态。

## 12. 先运行官方 Python demo

板端运行：

```bash
cd ~/yolov8_pose/python
python3 yolov8_pose.py --model_path ../model/yolov8n-pose.rknn --target rk3588
ls -lh result.jpg out.png 2>/dev/null || true
```

成功标志：

- 终端出现推理日志。
- 输出 `person @ (...) score` 类似结果，或生成结果图。

重要风险：

官方 Python demo 可能使用 `from rknn.api import RKNN`。如果板端只安装了 Lite2，可能报：

```text
No module named rknn
```

遇到这个错误时：

1. 不要改当前主项目代码。
2. 记录完整报错。
3. 优先改跑官方 Linux C demo。
4. 如果必须验证 Python Lite2，只在 `~/yolov8_pose` 这个临时官方 demo 副本里做 `RKNNLite` 适配，不动本项目。

## 13. 更稳的板端验证：编译官方 Linux C demo

如果 Python demo 因 `rknn.api` / Lite2 差异跑不通，优先使用官方 C demo。

你本地资料包里有交叉编译器：

```text
E:\王义龙大学\大学资料习题\嵌赛\01-教程文档\进阶篇之-基于RK3588的AI模型训练到部署\AI例程源码\gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu.tar.xz
```

在 WSL 中解压：

```bash
mkdir -p ~/rk3588_rknn_smoke/toolchain
tar -xf /mnt/e/王义龙大学/大学资料习题/嵌赛/01-教程文档/进阶篇之-基于RK3588的AI模型训练到部署/AI例程源码/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu.tar.xz -C ~/rk3588_rknn_smoke/toolchain
```

编译：

```bash
cd ~/rk3588_rknn_smoke/rknn_model_zoo
export GCC_COMPILER=~/rk3588_rknn_smoke/toolchain/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d yolov8_pose
```

传到板端：

```bash
scp -r install/rk3588_linux_aarch64/rknn_yolov8_pose_demo ${BOARD_USER}@${BOARD_IP}:~/
```

板端运行：

```bash
cd ~/rknn_yolov8_pose_demo
export LD_LIBRARY_PATH=./lib
chmod +x rknn_yolov8_pose_demo
./rknn_yolov8_pose_demo model/yolov8n-pose.rknn model/bus.jpg
ls -lh out.png
```

成功标志：

- 终端输出 `person @ (...)` 类似检测结果。
- 生成 `out.png`。

## 14. 观察 NPU load

板端另开一个 SSH 窗口：

```bash
watch -n 0.5 "sudo cat /sys/kernel/debug/rknpu/load"
```

如果单次推理太快，看不到波动，用循环运行：

```bash
cd ~/rknn_yolov8_pose_demo
export LD_LIBRARY_PATH=./lib
while true; do ./rknn_yolov8_pose_demo model/yolov8n-pose.rknn model/bus.jpg >/tmp/yolov8_pose.log 2>&1; sleep 0.2; done
```

成功标志：

- `Core0/Core1/Core2` 至少一个出现非 0% 波动。

如果一直是 0：

1. 确认运行的是 `.rknn`，不是 `.onnx`。
2. 确认推理命令真的在循环执行。
3. 确认 `/dev/rknpu*` 存在。
4. 记录 driver/runtime 版本，不改主项目代码。

## 15. 失败记录模板

任意一步失败时，按下面格式记录，不要急着改当前项目代码：

```text
步骤：
机器：Windows / WSL / RK3588
命令：
完整报错：
当前判断：
下一步建议：
是否影响主项目：否
```

## 16. 常见报错处理

### 16.1 WSL 安装失败

现象：

```text
wsl --install failed
WslRegisterDistribution failed
```

处理：

1. 启用“适用于 Linux 的 Windows 子系统”。
2. 启用“虚拟机平台”。
3. 重启 Windows。
4. 再运行 `wsl --install -d Ubuntu-22.04`。

### 16.2 GitHub 下载失败

处理：

1. 换网络。
2. 使用浏览器下载 release zip。
3. 确保最终目录仍是：

```text
~/rk3588_rknn_smoke/rknn-toolkit2
~/rk3588_rknn_smoke/rknn_model_zoo
```

### 16.3 `No module named rknn`

可能原因：

- WSL venv 中没有安装 `rknn_toolkit2`。
- 板端官方 Python demo 需要 `rknn.api`，但板端只安装了 `rknn_toolkit_lite2`。

处理：

- WSL 端：重新激活 venv 并安装 PC 端 wheel。
- 板端：优先跑官方 C demo；不要改主项目代码。

### 16.4 `No module named rknnlite`

可能原因：

- 板端 Lite2 wheel 没装。
- wheel 的 Python ABI 不匹配。
- 装到了另一个 Python 环境。

处理：

```bash
python3 -V
python3 - <<'PY'
import sys
print(sys.executable)
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
python3 -m pip list | grep rknn
```

然后重新安装对应 `cpXX`、`aarch64` wheel。

### 16.5 `Invalid RKNN model`

可能原因：

- PC 端转换工具版本和板端 runtime 版本不一致。
- `.rknn` 文件传输不完整。

处理：

1. 统一使用 RKNN-Toolkit2 2.3.2 和 RKNN-Toolkit-Lite2 2.3.2。
2. 检查 `.rknn` 文件大小：

```bash
ls -lh yolov8n-pose.rknn
```

### 16.6 输出乱框或 null 框

可能原因：

- INT8 量化误差。
- 预处理和后处理不一致。
- 转换脚本和模型类型不匹配。

处理：

1. 先使用官方 `bus.jpg` 和官方 demo。
2. 尝试 `fp` 模型确认链路。
3. 后续自训练模型再检查数据集、预处理、后处理、NMS 阈值。

### 16.7 NPU load 一直 0

可能原因：

- 推理太快，watch 没捕捉到。
- 实际没跑 `.rknn`。
- NPU 驱动或 runtime 没正常工作。

处理：

1. 循环运行 demo。
2. 另开窗口 watch load。
3. 检查 `/dev/rknpu*` 和 `/sys/kernel/debug/rknpu/load`。

## 17. 当前项目边界

本 smoke test 成功之前，不做这些事情：

- 不修改 `prescription/banzi/record_prescription_http.py`。
- 不修改 `realtime/` 训练计数逻辑。
- 不替换 `vision/pose_mediapipe_demo.py`。
- 不把 RKNN 设为默认 pose backend。
- 不删除 MediaPipe CPU 回退方案。

smoke test 成功后，下一阶段才考虑：

```text
官方 yolov8_pose 通过
-> 封装 vision/rknn_pose 后端
-> 输出统一 PoseResult
-> COCO17 映射到 rehab keypoints
-> POSE_BACKEND=auto|mediapipe|rknn
-> 默认仍为 mediapipe
```

