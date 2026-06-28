# RK3588 语音助手与 RKLLM / RKNN 部署计划书

这份文档按你当前真实进度写：你已经开始下载 `Qwen2.5-1.5B-Instruct`，所以第一版离线文字问答就继续用 Qwen2.5，不要中途改方向。

当前主线固定为：

```text
已有 WSL + .venv-rknn 复查
-> 新建 .venv-rkllm
-> 使用资料包里的 rknn-llm-main 工具链
-> 第一版用 Qwen2.5-1.5B-Instruct 文本模型转 .rkllm
-> 借用 DeepSeek-R1-Distill-Qwen-1.5B_Demo 里的 export 脚本当模板
-> 板端启动 qwen_rkllm_server 独立进程
-> 8082 有网走 GLM，没网走本地 Qwen2.5
-> 接 Paraformer ASR
-> 后续评分模型做好后再 ONNX -> RKNN 接入
```

核心结论：

- 你现在下载的 `Qwen2.5-1.5B-Instruct` 不白下，继续下完，第一版就用它。
- `DeepSeek-R1-Distill-Qwen-1.5B_Demo` 不是让你必须换成 DeepSeek，而是因为它里面有 RKLLM 转换脚本，可以改路径后给 Qwen2.5 用。
- `DeepSeek-R1-Distill-Qwen-1.5B` 是备用文本模型，不是你现在必须再下载的东西。
- `Qwen2-VL-2B-Instruct` 是后续图片/图文理解，不放进第一版语音助手。
- 8082 主进程不能加载大模型，只能通过 HTTP 调独立进程 `qwen_rkllm_server`。

## 1. 三个名字先分清

| 名字 | 现在做不做 | 用途 |
| --- | --- | --- |
| `Qwen2.5-1.5B-Instruct` | 现在做 | 第一版离线文字问答模型，你已经在下载它 |
| `DeepSeek-R1-Distill-Qwen-1.5B_Demo` | 现在用它的脚本 | 资料包里的 RKLLM demo，可借用 `export/` 转换脚本 |
| `DeepSeek-R1-Distill-Qwen-1.5B` | 备用 | 另一个文本模型；如果以后想换模型再下 |
| `Qwen2-VL-2B-Instruct` | 后续再做 | 图文/图片理解模型，后面和 GLM 图文能力对齐 |

区别不用想复杂：

```text
Qwen2.5-1.5B-Instruct：更像普通问答助手，短回答更干净，适合语音助手。
DeepSeek-R1-Distill-Qwen-1.5B：偏推理模型，可能输出 <think>，回答更长，语音播报反而可能啰嗦。
Qwen2-VL-2B-Instruct：图文模型，不是第一版文字语音助手。
```

所以现在别改去 DeepSeek，继续用你正在下载的 Qwen2.5。

## 2. 你已经有的资料包

你已经确认有这个目录：

```text
E:\王义龙大学\大学资料习题\嵌赛\01-教程文档\进阶篇之-基于RK3588的AI模型训练到部署\AI例程源码\rknn-llm-main
```

里面已经有：

```text
rkllm-toolkit
rkllm-runtime
rknpu-driver
examples\DeepSeek-R1-Distill-Qwen-1.5B_Demo
examples\Qwen2-VL-2B_Demo
examples\rkllm_server_demo
```

还确认有 RKLLM-Toolkit wheel：

```text
rknn-llm-main\rkllm-toolkit\rkllm_toolkit-1.1.4-cp310-...linux_x86_64.whl
rknn-llm-main\rkllm-toolkit\rkllm_toolkit-1.1.4-cp38-...linux_x86_64.whl
```

你的 demo README 写明要求：

```text
rkllm-toolkit==1.1.4
rkllm-runtime==1.1.4
python==3.8 or python==3.10
```

如果 WSL 里是 Python 3.10，就用 `cp310` wheel，不要用 `cp38`。

## 3. 你现在这个下载怎么处理

你截图里正在下载到：

```text
/home/elf/models/Qwen2.5-1.5B-Instruct
```

不要停。等它下载完后检查：

```bash
ls -lh /home/elf/models/Qwen2.5-1.5B-Instruct
```

成功标准：至少能看到：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors 或 model-*.safetensors
```

你截图里 `model.safetensors` 正在下载，等它到 100% 就行。

另外截图里有一个 NumPy warning：

```text
Failed to initialize NumPy: _ARRAY_API not found
```

如果下载能继续，这个 warning 先不用管。后面真正转换或生成量化数据时报错，再处理 NumPy 版本。

## 4. 第 7 步和第 10 步有什么区别

正确关系是：

```text
第 7 步：新建并安装 RKLLM-Toolkit 环境，只做一次。
第 10 步：检查 RKLLM-Toolkit 是否已经能用，不能用才补装。
```

如果你已经看到：

```text
RKLLM Toolkit OK
```

第 10 步就只检查，不重复安装。

## 5. 第 11 步用哪个目录

第 11 步仍然用你资料包里的这个 demo 目录，但只是借它的转换脚本：

```text
E:\王义龙大学\大学资料习题\嵌赛\01-教程文档\进阶篇之-基于RK3588的AI模型训练到部署\AI例程源码\rknn-llm-main\examples\DeepSeek-R1-Distill-Qwen-1.5B_Demo
```

在 WSL 里建议先复制到：

```bash
mkdir -p ~/work
cp -r /mnt/e/王义龙大学/大学资料习题/嵌赛/01-教程文档/进阶篇之-基于RK3588的AI模型训练到部署/AI例程源码/rknn-llm-main ~/work/
ls ~/work/rknn-llm-main/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo
```

成功时能看到：

```text
Readme.md
export
deploy
```

## 6. 复查已有 RKNN 环境

这一步只确认，不重装。

```bash
source ~/rk3588_rknn_smoke/.venv-rknn/bin/activate
python -c "from rknn.api import RKNN; print('RKNN Toolkit2 OK')"
```

成功标准：

```text
RKNN Toolkit2 OK
```

这个环境只用于：

```text
ONNX -> RKNN
```

不要用 `.venv-rknn` 安装 RKLLM-Toolkit。

## 7. 新建 RKLLM-Toolkit 环境

在 WSL 里执行：

```bash
cd ~/rk3588_rknn_smoke
python3 -m venv .venv-rkllm
source .venv-rkllm/bin/activate
python -m pip install --upgrade pip
python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
```

如果输出 `cp310`，就安装 cp310 wheel：

```bash
mkdir -p ~/work
cp -r /mnt/e/王义龙大学/大学资料习题/嵌赛/01-教程文档/进阶篇之-基于RK3588的AI模型训练到部署/AI例程源码/rknn-llm-main ~/work/
cd ~/work/rknn-llm-main/rkllm-toolkit
python -m pip install ./rkllm_toolkit-1.1.4-cp310-*-linux_x86_64.whl
python -c "from rkllm.api import RKLLM; print('RKLLM Toolkit OK')"
```

成功标准：

```text
RKLLM Toolkit OK
```

## 8. 准备 Qwen2.5 模型权重

你现在正在下载到：

```text
/home/elf/models/Qwen2.5-1.5B-Instruct
```

下载完成后检查：

```bash
ls -lh /home/elf/models/Qwen2.5-1.5B-Instruct
```

如果你想统一文档里的路径，也可以复制或软链接到：

```bash
mkdir -p ~/models
ln -s /home/elf/models/Qwen2.5-1.5B-Instruct ~/models/Qwen2.5-1.5B-Instruct
ls -lh ~/models/Qwen2.5-1.5B-Instruct
```

如果提示目标已存在，就不用管。

成功标准：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors 或 model-*.safetensors
```

看不到 `*.safetensors` 就说明下载没完成，不能继续转换。

## 9. 准备 Qwen2.5 量化数据

进入 demo 的 `export` 目录：

```bash
source ~/rk3588_rknn_smoke/.venv-rkllm/bin/activate
cd ~/work/rknn-llm-main/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export
```

这个目录自带一个：

```text
data_quant.json
```

第一版为了先验证“能不能转、能不能在板端跑”，可以直接用这个自带文件，不需要先跑很慢的 `generate_data_quant.py`。

先检查自带文件：

```bash
ls -lh data_quant.json
head -c 200 data_quant.json
```

成功标准：

```text
data_quant.json 存在
文件大小不是 0
能看到 JSON 内容
```

`generate_data_quant.py` 很慢是正常的，因为它会用 Qwen2.5 在 CPU/GPU 上对多条样本逐条推理，生成校准输入输出。如果你只是在 CPU 上跑，半小时甚至更久都可能。第一版不建议卡在这里。

推荐顺序：

```text
先用自带 data_quant.json 转 .rkllm，验证链路跑通。
后面有时间，再生成康复问答版 data_quant_rehab.json，提高量化校准贴合度。
```

如果你已经启动了很久的生成命令，只是想先跑通，可以按 `Ctrl+C` 停掉，然后继续第 10、11 步。

后续要重新生成 Qwen2.5 的量化数据时，再跑：

```bash
python generate_data_quant.py -m ~/models/Qwen2.5-1.5B-Instruct
```

如果你没有建软链接，就用真实路径：

```bash
python generate_data_quant.py -m /home/elf/models/Qwen2.5-1.5B-Instruct
```

### 9.1 后续怎么做康复版 data_quant_rehab.json

这一步不是训练模型，不需要你重新训练 Qwen。它只是给 RKLLM 量化时用的校准样本，让量化过程多见一些你项目里常出现的问法和回答风格。

可以先手写一个很小的 JSON 文件，例如 10-30 条，内容围绕康复语音助手常见问题：

```bash
cd ~/work/rknn-llm-main/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export
vi data_quant_rehab.json
```

写入示例：

```json
[
  {
    "input": "<|im_start|>system\n你是康复训练语音助手，回答要简短、清楚、适合语音播报。<|im_end|>\n<|im_start|>user\n伸膝保持是什么意思？<|im_end|>\n<|im_start|>assistant\n",
    "target": "伸膝保持就是把膝盖尽量伸直后稳定停住，不要晃动，按要求保持几秒。"
  },
  {
    "input": "<|im_start|>system\n你是康复训练语音助手，回答要简短、清楚、适合语音播报。<|im_end|>\n<|im_start|>user\n屈膝训练要注意什么？<|im_end|>\n<|im_start|>assistant\n",
    "target": "屈膝时动作要慢，膝盖按目标方向弯曲，不要用身体代偿，也不要突然用力。"
  },
  {
    "input": "<|im_start|>system\n你是康复训练语音助手，回答要简短、清楚、适合语音播报。<|im_end|>\n<|im_start|>user\n我刚才哪里没做好？<|im_end|>\n<|im_start|>assistant\n",
    "target": "刚才主要问题是动作幅度不够稳定，下一次可以放慢速度，先保证膝盖到位再保持。"
  },
  {
    "input": "<|im_start|>system\n你是康复训练语音助手，回答要简短、清楚、适合语音播报。<|im_end|>\n<|im_start|>user\n保持时腿抖怎么办？<|im_end|>\n<|im_start|>assistant\n",
    "target": "轻微抖动可以先放慢节奏，减少幅度，保持呼吸平稳。如果疼痛明显，应停止并咨询医生。"
  }
]
```

检查 JSON 格式：

```bash
python -m json.tool data_quant_rehab.json >/tmp/data_quant_rehab_check.json
```

如果没有报错，说明格式正确。

然后把 `export_rkllm.py` 里的：

```python
dataset = "./data_quant.json"
```

改成：

```python
dataset = "./data_quant_rehab.json"
```

再执行：

```bash
python export_rkllm.py
```

第一版建议先用自带 `data_quant.json` 跑通。等 `.rkllm`、server、8082 都通了，再换 `data_quant_rehab.json` 重新导出一次，对康复问答更贴近。
## 10. 检查 RKLLM-Toolkit，不重复安装

先检查：

```bash
source ~/rk3588_rknn_smoke/.venv-rkllm/bin/activate
python -c "from rkllm.api import RKLLM; print('RKLLM Toolkit OK')"
```

如果成功，直接进入第 11 步。

如果失败，再回第 7 步安装 wheel。不要在这里重新创建环境。

## 11. 把 Qwen2.5 转成 .rkllm

进入转换脚本目录：

```bash
cd ~/work/rknn-llm-main/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export
```

打开转换脚本：

```bash
vi export_rkllm.py
```

找到这一行：

```python
modelpath = '/path/to/DeepSeek-R1-Distill-Qwen-1.5B'
```

改成 Qwen2.5 的路径。如果你建了软链接，就写：

```python
modelpath = '/home/elf/models/Qwen2.5-1.5B-Instruct'
```

如果你实际用户不是 `elf`，用这个命令查：

```bash
echo $HOME
```

然后写成类似：

```python
modelpath = '/home/你的用户名/models/Qwen2.5-1.5B-Instruct'
```

然后执行：

```bash
python export_rkllm.py
```

成功后检查：

```bash
ls -lh *.rkllm
```

正常会生成类似：

```text
Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm
```

为了后面命令统一，复制成短名字：

```bash
cp Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm qwen1_5b.rkllm
ls -lh qwen1_5b.rkllm
```

这里的 `qwen1_5b.rkllm` 是项目统一命名。实际模型是 `Qwen2.5-1.5B-Instruct`。

## 12. 上传 .rkllm 到 RK3588

你现在的网络情况是：

```text
Windows PowerShell 能 ping 通板子 192.168.137.232
Windows PowerShell 的 22 端口测试成功
WSL ping 不通板子
```

所以第 12 步不要走 WSL 的 `scp`，改成用 Windows PowerShell 传文件。

先注意：必须等 `.rkllm` 真正转换成功后再传。现在如果看到文件大小是 `0`，不要传。

### 12.1 WSL 终端：只负责检查转换结果

这一段在 WSL 终端执行，就是你运行 `export_rkllm.py` 的那个窗口：

```bash
cd ~/work/rknn-llm-main/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export
ls -lh *.rkllm qwen1_5b.rkllm 2>/dev/null
```

如果显示大小是 `0`，说明转换失败：

```text
0 Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm
0 qwen1_5b.rkllm
```

这种情况先不要传。

成功时大小应该不是 `0`，通常是几百 MB 或几 GB。确认成功后，再把 WSL 文件复制到 Windows 的 D 盘中转目录。

在 WSL 执行：

```bash
mkdir -p /mnt/d/rk3588_transfer
cp qwen1_5b.rkllm /mnt/d/rk3588_transfer/
ls -lh /mnt/d/rk3588_transfer/qwen1_5b.rkllm
```

这一步的目的：让 Windows PowerShell 能直接找到文件：

```text
D:\rk3588_transfer\qwen1_5b.rkllm
```

### 12.2 Windows PowerShell：测试板子连接

这一段在 Windows PowerShell 执行，不是在 WSL，也不是在板子终端。

先设置变量：

```powershell
$BOARD_USER = "elf"
$BOARD_IP = "192.168.137.232"
$LOCAL_MODEL = "D:\rk3588_transfer\qwen1_5b.rkllm"
```

检查本地模型文件是不是非 0：

```powershell
Get-Item $LOCAL_MODEL
```

成功标准：`Length` 不是 `0`。

再测试板子网络：

```powershell
ping $BOARD_IP
```

再测试 SSH 端口：

```powershell
Test-NetConnection $BOARD_IP -Port 22
```

成功标准：

```text
TcpTestSucceeded : True
```

### 12.3 Windows PowerShell：在板子上创建目录并修复权限

这一段在 Windows PowerShell 执行。

你的板子上 `/home/elf/models` 可能属于 `root root`，这时普通 `elf` 用户不能在里面创建 `qwen`。所以要用 `sudo` 建目录，并把 `qwen` 目录改回 `elf` 可写。

先设置变量：

```powershell
$BOARD_USER = "elf"
$BOARD_IP = "192.168.137.232"
$LOCAL_MODEL = "D:\rk3588_transfer\qwen1_5b.rkllm"
```

检查本地文件存在且不是 0 字节：

```powershell
Get-Item $LOCAL_MODEL
```

在板子上创建目录并改权限：

```powershell
ssh -tt ${BOARD_USER}@${BOARD_IP} "sudo mkdir -p /home/elf/models/qwen && sudo chown -R elf:elf /home/elf/models/qwen && ls -ld /home/elf/models/qwen"
```

如果要求输入密码，输入板子密码。输入时屏幕不显示字符，这是正常的。这里必须用 `ssh -tt`，否则会报 `sudo: a terminal is required to read the password`。

成功时应该看到类似：

```text
drwxr-xr-x ... elf elf ... /home/elf/models/qwen
```

如果这里没有变成 `elf elf`，后面上传还会失败。

### 12.4 Windows PowerShell：上传模型到板子

这一段还是在 Windows PowerShell 执行。

```powershell
scp $LOCAL_MODEL ${BOARD_USER}@${BOARD_IP}:/home/elf/models/qwen/
```

如果 PowerShell 对变量解析有问题，就用最直白的写法：

```powershell
scp D:\rk3588_transfer\qwen1_5b.rkllm elf@192.168.137.232:/home/elf/models/qwen/
```

注意：PowerShell 里不要写成 `$BOARD_USER@$BOARD_IP:/...`，冒号会让变量解析出错。要么用 `${BOARD_USER}@${BOARD_IP}:/...`，要么直接写 `elf@192.168.137.232:/...`。


如果 `scp` 直接传 `/home/elf/models/qwen/` 仍失败，就走更稳的两段式：先传到 `/home/elf/`，再在板子上用 sudo 移动。

```powershell
scp $LOCAL_MODEL ${BOARD_USER}@${BOARD_IP}:/home/elf/qwen1_5b.rkllm
ssh -tt ${BOARD_USER}@${BOARD_IP} "sudo mkdir -p /home/elf/models/qwen && sudo mv /home/elf/qwen1_5b.rkllm /home/elf/models/qwen/ && sudo chown elf:elf /home/elf/models/qwen/qwen1_5b.rkllm && ls -lh /home/elf/models/qwen/qwen1_5b.rkllm"
```
### 12.5 Windows PowerShell：检查板端文件

这一段仍然在 Windows PowerShell 执行。

```powershell
ssh $BOARD_USER@$BOARD_IP "ls -lh /home/elf/models/qwen/qwen1_5b.rkllm"
```

成功标准：

```text
板端能看到 /home/elf/models/qwen/qwen1_5b.rkllm
文件大小不是 0
```

### 12.6 板子终端：本机检查

这一段在 RK3588 板子的终端执行。如果你已经 SSH 到板子里，也可以在那里执行。

```bash
ls -lh /home/elf/models/qwen/qwen1_5b.rkllm
```

成功标准同样是：文件存在，并且不是 0 字节。

### 12.7 如果命令没反应

不要一次粘贴多行，按这个顺序一条一条测。

Windows PowerShell：

```powershell
ping 192.168.137.232
Test-NetConnection 192.168.137.232 -Port 22
ssh elf@192.168.137.232 "hostname"
```

如果 `ping` 和 `Test-NetConnection` 都成功，但 `ssh` 没反应，通常是在等你输入密码，直接输入板子密码再回车。

如果 WSL 不通板子，但 Windows PowerShell 通板子，就用 PowerShell 上传，这是正常方案。
## 13. 准备 RKLLM Flask Server

第 13 步的目标：让板子上出现一个能加载 `.rkllm` 的 Flask server。

不要在板子 SSH 终端里运行：

```bash
./build_rkllm_server_flask.sh ...
```

这个脚本是给“电脑通过 adb 自动部署板子”用的。你现在是 SSH 到板子里操作，所以会报：

```text
adb: command not found
cp: cannot stat '../../rkllm-runtime/.../librkllmrt.so'
cp: cannot stat '../../scripts/fix_freq_rk3588.sh'
```

正确做法是手动补齐这几个文件，然后直接运行 `flask_server.py`。

### 13.1 Windows PowerShell：准备要上传的 runtime 文件

这一段在 Windows PowerShell 执行。

先设置资料包根目录：

```powershell
$RKLLM_ROOT = "E:\王义龙大学\大学资料习题\嵌赛\01-教程文档\进阶篇之-基于RK3588的AI模型训练到部署\AI例程源码\rknn-llm-main"
$TRANSFER = "D:\rk3588_transfer"
New-Item -ItemType Directory -Force -Path $TRANSFER | Out-Null
```

把 runtime 库和调频脚本复制到 D 盘中转目录：

```powershell
Copy-Item -Force "$RKLLM_ROOT\rkllm-runtime\Linux\librkllm_api\aarch64\librkllmrt.so" "$TRANSFER\librkllmrt.so"
Copy-Item -Force "$RKLLM_ROOT\scripts\fix_freq_rk3588.sh" "$TRANSFER\fix_freq_rk3588.sh"
```

检查：

```powershell
Get-Item "$TRANSFER\librkllmrt.so"
Get-Item "$TRANSFER\fix_freq_rk3588.sh"
```

### 13.2 Windows PowerShell：上传 runtime 文件到板子

这一段在 Windows PowerShell 执行。

```powershell
$BOARD_USER = "elf"
$BOARD_IP = "192.168.137.232"
```

确保板子目录存在：

```powershell
ssh ${BOARD_USER}@${BOARD_IP} "mkdir -p /home/elf/qwen_server/rkllm_server/lib"
```

上传库和脚本：

```powershell
scp D:\rk3588_transfer\librkllmrt.so ${BOARD_USER}@${BOARD_IP}:/home/elf/qwen_server/rkllm_server/lib/
scp D:\rk3588_transfer\fix_freq_rk3588.sh ${BOARD_USER}@${BOARD_IP}:/home/elf/qwen_server/rkllm_server/
```

检查板子上是否存在：

```powershell
ssh ${BOARD_USER}@${BOARD_IP} "ls -lh /home/elf/qwen_server/rkllm_server/lib/librkllmrt.so /home/elf/qwen_server/rkllm_server/fix_freq_rk3588.sh"
```

### 13.3 板子终端：安装 Flask 依赖

这一段在 RK3588 板子终端执行，或者先 SSH 进去：

```powershell
ssh elf@192.168.137.232
```

进入板子后执行：

```bash
python3 -c "import flask; print('flask ok')"
```

如果报 `ModuleNotFoundError: No module named 'flask'`，再安装：

```bash
python3 -m pip install flask==2.2.2 Werkzeug==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages
```

如果系统不支持 `--break-system-packages`，去掉它再试：

```bash
python3 -m pip install flask==2.2.2 Werkzeug==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 13.4 板子终端：检查 server 目录

这一段在 RK3588 板子终端执行。

```bash
cd /home/elf/qwen_server/rkllm_server
ls -lh
ls -lh lib/librkllmrt.so
ls -lh fix_freq_rk3588.sh
chmod +x fix_freq_rk3588.sh
```

成功标准：

```text
flask_server.py 存在
lib/librkllmrt.so 存在且不是 0 字节
fix_freq_rk3588.sh 存在
/home/elf/models/qwen/qwen1_5b.rkllm 存在且不是 0 字节
```

如果这些都满足，就进入第 14 步启动 server。

## 14. 板端启动 RKLLM Flask Server

这一段在 RK3588 板子终端执行。

进入 server 目录：

```bash
cd /home/elf/qwen_server/rkllm_server
```

启动 Flask server：

```bash
python3 flask_server.py \
  --rkllm_model_path /home/elf/models/qwen/qwen1_5b.rkllm \
  --target_platform rk3588
```

它默认监听：

```text
0.0.0.0:8080
POST /rkllm_chat
```

所以第一版先按官方接口测试，不是 `/generate`。

另开一个板子终端，测试：

```bash
curl -s http://127.0.0.1:8080/rkllm_chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"用一句话回答：伸膝保持是什么意思？"}],"stream":false}' \
  | python3 -m json.tool
```

成功标准：

```text
返回 JSON
choices[0].message.content 里有中文回答
启动 server 的终端没有崩溃
```

注意：官方 Flask server 默认是 `8080/rkllm_chat`。后面接入 8082 时，有两个选择：

```text
方案 A：8082 直接适配官方 /rkllm_chat 接口。
方案 B：再写一个 qwen_rkllm_server 包装层，对外提供 18080/generate。
```

为了先验证模型能不能跑，先用官方 `8080/rkllm_chat`，不要卡在包装层。
## 15. 已验证后的当前架构：8080 + 18080 + 8082

你已经完成第 14 步，并且板端 Qwen RKLLM Flask server 已经验证通过：

```text
模型路径：/home/elf/models/qwen/qwen1_5b.rkllm
官方服务：/home/elf/qwen_server/rkllm_server/flask_server.py
官方接口：http://127.0.0.1:8080/rkllm_chat
项目路径：/home/elf/project/project_system
```

从这里开始，日常运行不再使用 `build_rkllm_server_flask.sh`，也不需要每天手动开很多终端。

当前固定架构是：

```text
Qwen RKLLM Flask server  127.0.0.1:8080/rkllm_chat
        ↓
RKLLM proxy wrapper      127.0.0.1:18080/health 和 /generate
        ↓
8082 主服务              http://127.0.0.1:8082/train
        ↓
浏览器 UI                右侧悬浮 AI 面板 / 小爱文本问答
```

关键原则：

- 8082 主进程不加载 `.rkllm`。
- Qwen2.5 只在独立的官方 Flask server 进程里加载。
- 8082 只通过 `http://127.0.0.1:18080/generate` 调用本地 Qwen。
- 有外网 + GLM API Key 时，`auto` 模式优先 GLM。
- 没网、无 Key 或强制本地时，走 `local_qwen_rkllm`。
- 训练中禁止触发 GLM/Qwen 问答；组间休息、空闲、训练结束允许。
- AI 问答异步执行，不阻塞训练线程、摄像头线程、姿态识别线程和 TTS 主播报。
- 训练纠错/计数 TTS 高优先级，AI 问答 TTS 低优先级。

## 16. 最新需要上传到板子的文件

这一段在 Windows PowerShell 执行，不是在 WSL。

先设置变量：

```powershell
$BOARD_USER = "elf"
$BOARD_IP = "192.168.137.232"
$PROJECT = "D:\rk3588\project"
```

确保板子目录存在：

```powershell
ssh ${BOARD_USER}@${BOARD_IP} "mkdir -p /home/elf/project/project_system/llm /home/elf/project/project_system/voice /home/elf/project/project_system/scripts /home/elf/project/project_system/prescription/common /home/elf/project/project_system/prescription/banzi/static /home/elf/project/project_system/docs"
```

上传当前最新运行文件：

```powershell
scp $PROJECT\llm\rkllm_proxy_server.py ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/llm/
scp $PROJECT\voice\llm_worker.py ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/voice/
scp $PROJECT\voice\asr_worker.py ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/voice/
scp $PROJECT\scripts\start_rehab_station_qwen.sh ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/scripts/
scp $PROJECT\scripts\stop_rehab_station_qwen.sh ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/scripts/
scp $PROJECT\scripts\check_llm_status.sh ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/scripts/
scp $PROJECT\prescription\common\llm_assistant.py ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/prescription/common/
scp $PROJECT\prescription\banzi\record_prescription_http.py ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/prescription/banzi/
scp $PROJECT\prescription\banzi\static\train.js ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/prescription/banzi/static/
scp $PROJECT\prescription\banzi\static\doctor.js ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/prescription/banzi/static/
scp $PROJECT\prescription\banzi\static\common.js ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/prescription/banzi/static/
scp $PROJECT\prescription\banzi\static\app.css ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/prescription/banzi/static/
scp $PROJECT\docs\rk3588_qwen_rkllm_rknn_conversion_guide.md ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/docs/
scp $PROJECT\readme.md ${BOARD_USER}@${BOARD_IP}:/home/elf/project/project_system/
```

板子上继续复用这些已经验证过的文件，不需要重复上传：

```text
/home/elf/models/qwen/qwen1_5b.rkllm
/home/elf/qwen_server/rkllm_server/flask_server.py
/home/elf/qwen_server/rkllm_server/lib/librkllmrt.so
/home/elf/qwen_server/rkllm_server/fix_freq_rk3588.sh
```

目前仓库里实际存在的一键脚本是：

```text
scripts/start_rehab_station_qwen.sh
scripts/stop_rehab_station_qwen.sh
scripts/check_llm_status.sh
```

不要再按旧文档找 `install_rehab_station_autostart.sh`。如果后面要做开机自启动，再单独新增 systemd/autostart 脚本。

## 17. 日常启动：只开一个后端终端

进入板子：

```powershell
ssh elf@192.168.137.232
```

板子终端执行：

```bash
cd /home/elf/project/project_system
chmod +x scripts/start_rehab_station_qwen.sh scripts/stop_rehab_station_qwen.sh scripts/check_llm_status.sh
./scripts/stop_rehab_station_qwen.sh
./scripts/start_rehab_station_qwen.sh
```

启动脚本会自动做三件事：

```text
1. 启动官方 Qwen RKLLM Flask server：127.0.0.1:8080/rkllm_chat
2. 启动项目 proxy：127.0.0.1:18080/health 和 /generate
3. 启动 8082 主服务：http://127.0.0.1:8082/train
```

成功时终端重点看：

```text
[OK] Qwen Flask server is listening on 127.0.0.1:8080
[OK] RKLLM proxy health is ok
[START] Rehab 8082 service -> .../logs/rehab_8082.log
[URL] http://127.0.0.1:8082/train
```

日志位置：

```text
/home/elf/project/project_system/logs/qwen_flask.log
/home/elf/project/project_system/logs/qwen_proxy.log
/home/elf/project/project_system/logs/rehab_8082.log
```

如果启动失败，查看：

```bash
cd /home/elf/project/project_system
tail -n 80 logs/qwen_flask.log
tail -n 80 logs/qwen_proxy.log
tail -n 120 logs/rehab_8082.log
```

## 18. 不看长 JSON：用短状态脚本看当前流程

不要再直接盯完整 `/status`：

```bash
curl -s http://127.0.0.1:8082/status | python3 -m json.tool
```

它太长，不适合快速判断 GLM/Qwen。现在用：

```bash
cd /home/elf/project/project_system
./scripts/check_llm_status.sh
```

最新字段含义：

```text
llm.provider                 配置策略，通常是 auto/local_qwen_rkllm/glm4v_api
llm.expected_provider_now    现在如果提交问答，预计会走谁
llm.qwen_ready               18080 proxy + 8080 Qwen 是否可达
llm.api_key_configured       GLM API Key 是否在 8082 启动环境里
llm.fallback_reason          如果退到 echo，这里看原因
voice.qa_allowed             当前训练状态是否允许问答
voice.training_status        当前训练状态
voice.llm_queue_size         异步 LLM 队列长度
current_job.*                当前正在 queued/running 的任务
last_done.*                  上一次完成/失败/blocked 的任务
```

一定要分清：

```text
llm.expected_provider_now 是“现在提交会走谁”。
current_job 是“当前正在跑什么”。
last_done 是“上一条完成记录”。
```

所以如果 `last_done.active_provider = echo`，不一定代表现在还在用 echo。它可能只是历史最后一次失败或 fallback。重新提交一次问答后，再看 `current_job` 和新的 `last_done`。

## 19. 浏览器 UI 里怎么判断 GLM / Qwen

打开：

```text
板子本地浏览器：http://127.0.0.1:8082/train
电脑浏览器：http://192.168.137.232:8082/train
```

右侧悬浮 `AI训练图文建议` 面板会显示：

```text
配置 provider
预计 provider
当前任务
上次完成
最后错误
Qwen ready/off
```

解释：

```text
预计 provider：现在如果点提交，应该走 glm4v_api 或 local_qwen_rkllm。
当前任务：异步队列里正在跑的 job。
上次完成：上一条完成的 job，可能是 glm4v_api/local_qwen_rkllm/echo/blocked_training。
Qwen ready：只表示本地 Qwen 服务可达，不代表本次一定走 Qwen。
最后错误：如果掉到 echo 或失败，优先看这里。
```

当前 UI 已经修成按动作选择报告，不再显示“报告 1 / 报告 2 / 报告 3”。右侧 AI 面板固定三个动作标签：

```text
坐姿伸膝
站姿屈膝后勾腿
坐姿抬膝
```

每个动作只绑定该动作最近一次患者训练报告。点哪个动作，小爱问答就只基于哪个动作最近一次报告回答。

如果某个动作显示：

```text
暂无该动作报告
```

说明不是医生模板缺，而是这个动作还没有患者训练完成后的评估报告。你只需要把这个患者动作跑完一次并生成报告；不需要重录医生标准动作，除非 active template 本来就缺或后端提示模板缺失。


### 19.1 三动作报告和 AI 面板的数据链路

患者页右侧 AI 面板不是直接扫目录，而是依赖 8082 `/status` 返回：

```text
latest_reports_by_action
```

后端必须从这里汇总三动作最近报告：

```text
evaluate/reports/report_seated_knee_extension_*.json
evaluate/reports/report_standing_hamstring_curl_*.json
evaluate/reports/report_seated_knee_raise_*.json
```

兼容历史数据时，可以兜底读取：

```text
prescription/docs/results/*.json
```

但当前主报告来源仍然是 `evaluate/reports/report_*.json`。如果页面提示“当前动作还没有报告”，优先检查 `/status.latest_reports_by_action`，不要只看目录里有没有文件。正确状态是三个动作标签各自绑定最近报告，分别生成 AI 图文建议和小爱问答。

医生页 `/doctor` 只负责录入和评估，不再承载患者报告 AI 区域。患者报告、AI 图文建议、小爱问答和设备运行状态统一放在 `/train`。

如果 AI 一直显示 `echo`，按这个顺序判断：

```text
1. ./scripts/check_llm_status.sh 看 expected_provider_now
2. 有 GLM Key 时 expected_provider_now 应为 glm4v_api
3. 无 Key 但 Qwen ready 时 expected_provider_now 应为 local_qwen_rkllm
4. 只有 GLM 和 Qwen 都不可用，或请求失败，才允许 echo
5. 训练中问答会 blocked_training，不应该触发 GLM/Qwen
```
## 20. 有网 + GLM API Key：验证 GLM 优先

你现在有手机热点外网和 GLM API Key，所以默认 `auto` 模式应该优先走：

```text
glm4v_api
```

确认 Key 在启动 8082 的同一个终端环境里：

```bash
cd /home/elf/project/project_system
echo ${ZHIPUAI_API_KEY:+ZHIPUAI_API_KEY configured}
echo ${GLM_API_KEY:+GLM_API_KEY configured}
./scripts/check_llm_status.sh
```

成功标准：

```text
llm.provider = auto
llm.expected_provider_now = glm4v_api
llm.api_key_configured = True
llm.qwen_ready = True
voice.qa_allowed = True
```

UI 验证：

```text
1. 打开 /train。
2. 右侧 AI 面板选择一个已有报告的动作。
3. 在“小爱康复问答”输入：我刚才哪里没做好？
4. 点击提交问答。
```

成功标准：

```text
当前任务：running:glm4v_api 或 queued:glm4v_api
完成后：上次完成：done:glm4v_api
回答内容基于当前选中动作的最近一次报告
```

有网 + GLM Key 时，GLM 也用于训练报告 AI 建议和图文建议能力。本地 Qwen2.5 仍作为断网/强制本地时的文本问答兜底。

## 21. 推荐方式：不断网，强制单独验证 Qwen2.5

远程调试时不要直接关手机热点。热点一关，SSH 和电脑浏览器访问 `http://板子IP:8082/train` 可能一起断。

推荐用强制 provider 验证本地 Qwen2.5：

```bash
cd /home/elf/project/project_system
./scripts/stop_rehab_station_qwen.sh
export REHAB_LLM_PROVIDER=local_qwen_rkllm
./scripts/start_rehab_station_qwen.sh
```

另开一个板子终端检查：

```bash
cd /home/elf/project/project_system
./scripts/check_llm_status.sh
```

成功标准：

```text
llm.provider = local_qwen_rkllm
llm.expected_provider_now = local_qwen_rkllm
llm.qwen_ready = True
```

浏览器 UI 里：

```text
1. 打开 /train。
2. 右侧 AI 面板选择一个已有报告的动作。
3. 输入：屈膝训练要注意什么？
4. 点击提交问答。
```

成功标准：

```text
当前任务：running:local_qwen_rkllm 或 queued:local_qwen_rkllm
完成后：上次完成：done:local_qwen_rkllm
返回中文回答
```

测完恢复自动模式：

```bash
cd /home/elf/project/project_system
./scripts/stop_rehab_station_qwen.sh
unset REHAB_LLM_PROVIDER
./scripts/start_rehab_station_qwen.sh
```

## 22. 真实断网验证 Qwen2.5

只有在你使用板子 HDMI 屏幕、本地键盘鼠标、本地浏览器时，才建议真实断网验证。

板子本地终端先启动 auto 模式：

```bash
cd /home/elf/project/project_system
./scripts/stop_rehab_station_qwen.sh
unset REHAB_LLM_PROVIDER
./scripts/start_rehab_station_qwen.sh
```

然后关闭手机热点，或者让板子断开外网。

板子本地浏览器打开：

```text
http://127.0.0.1:8082/train
```

在小爱康复问答里提问。

成功标准：

```text
Qwen ready
完成后：上次完成：done:local_qwen_rkllm
返回中文回答
```

如果你是电脑远程访问板子，不要用这个方法。直接用第 21 步强制 `local_qwen_rkllm`。

## 23. 直接验证 Qwen2.5 proxy

这个测试不经过 8082，只验证本地 Qwen2.5 服务链路：

```bash
curl -s http://127.0.0.1:18080/health | python3 -m json.tool
```

成功标准：

```text
ok = true
upstream_reachable = true
upstream = http://127.0.0.1:8080/rkllm_chat
```

再测生成：

```bash
curl -s http://127.0.0.1:18080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"用一句话回答：屈膝训练要注意什么？","max_new_tokens":64,"temperature":0.2}' \
  | python3 -m json.tool
```

成功标准：

```text
ok = true
text = 中文回答
model = qwen2.5-1.5b-rkllm
```

如果这里成功，但 UI 没走 Qwen，说明 Qwen server 没问题，问题在 8082 provider 配置、训练保护状态或当前动作没有报告。用：

```bash
./scripts/check_llm_status.sh
```

重点看：

```text
llm.expected_provider_now
voice.qa_allowed
current_job.*
last_done.error
```

## 24. UI 完整流程验收

第一版完整流程按这个顺序验收：

```text
医生录入页保存三个标准动作
-> 患者训练页执行完整训练
-> 和医生模板比较
-> 训练结束后输出 ROM、TUT、DTW、速度、错误类型和反馈建议
-> 右侧 AI 面板按动作显示最近一次报告
-> 有网 + GLM Key 时，AI 建议和问答走 GLM
-> 强制本地或真实断网时，小爱问答走 Qwen2.5
```

当前三个动作是：

```text
seated_knee_extension       坐姿伸膝
standing_hamstring_curl     站姿屈膝后勾腿
seated_knee_raise           坐姿抬膝
```

成功标准：

```text
小爱康复问答能输入中文，不清空、不跳光标、不退出候选框
报告问答能输入中文，不清空、不跳光标、不退出候选框
右侧标签显示三个动作，不显示“报告 1/2/3”
每个动作的回答基于该动作最近一次报告
训练中提问返回 blocked_training，不触发 GLM/Qwen 推理
休息、空闲、训练结束后允许问答
AI 问答 TTS 低优先级，不抢训练计数/纠错播报
```

训练保护状态：

```text
running
paused
awaiting_orientation
awaiting_return
awaiting_care_response
```

这些状态下问答必须 blocked。`resting / idle / finished` 允许问答。

## 25. ASR 语音识别后续怎么接

麦克风没到之前，不验收 ASR，只保留接口和 UI 文案。

后续麦克风到了再测：

```text
浏览器录音
-> POST /api/voice/asr
-> sherpa-onnx Paraformer 工作线程识别
-> 文本回填到问题框
-> 点击问答
```

Paraformer 模型目录默认是：

```text
/home/elf/models/sherpa-onnx-paraformer-zh
```

ASR 不转 RKNN，也不转 RKLLM。

## 26. NPU 怎么确认 Qwen2.5 真的在用

Qwen RKLLM 推理会用 NPU。姿态检测当前仍先走 CPU/MediaPipe 稳定路线，避免为了切 NPU 姿态检测影响已跑通训练主流程。

板子终端 1：启动服务。

```bash
cd /home/elf/project/project_system
./scripts/start_rehab_station_qwen.sh
```

板子终端 2：看 NPU load。

```bash
watch -n 0.5 "sudo cat /sys/kernel/debug/rknpu/load"
```

板子终端 3：触发 Qwen。

```bash
curl -s http://127.0.0.1:18080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"简单解释伸膝保持是什么意思","max_new_tokens":128}'
```

成功标准：

```text
curl 返回中文
rknpu/load 有波动
8082 /status 仍正常
训练线程、摄像头线程没有卡死
```

## 27. Qwen2-VL 和姿态评分模型先不接

当前“Qwen2”在这份文档里指已经验证的：

```text
Qwen2.5-1.5B-Instruct RKLLM 文本模型
```

不是 `Qwen2-VL`。`Qwen2-VL` 是后续图文模型，等文字问答稳定后再单独做。

姿态评分模型也先不写。等同学交付 ONNX 后再走：

```text
ONNX
-> 用现有 ~/rk3588_rknn_smoke/.venv-rknn 转 RKNN
-> 板端单独 smoke test
-> 异步接入 8082
```

接入原则：

```text
不替换当前 MediaPipe 计数主流程
不阻塞摄像头/姿态线程
不在每帧都跑评分模型
只做训练后评分或低频异步评分
```

## 28. 最小验收顺序

现在按这个顺序验收：

1. `/home/elf/models/qwen/qwen1_5b.rkllm` 已存在且不是 0 字节。
2. 官方 `8080/rkllm_chat` 已经返回过中文。
3. 上传 `rkllm_proxy_server.py`、启动脚本、`check_llm_status.sh`、8082 后端和前端文件到板子。
4. 执行 `./scripts/start_rehab_station_qwen.sh`。
5. `./scripts/check_llm_status.sh` 能显示短状态，且没有 `PY` here-doc 报错。
6. `curl 127.0.0.1:18080/health` 返回 `ok=true`。
7. `curl 127.0.0.1:18080/generate` 返回中文。
8. 打开 `http://127.0.0.1:8082/train`。
9. 右侧 AI 标签显示三个动作，不显示“报告 1/2/3”。
10. 小爱问答和报告问答都能输入中文，不被刷新打断。
11. 有网 + GLM Key 时，UI 完成后显示 `done:glm4v_api`。
12. 强制 `local_qwen_rkllm` 时，UI 完成后显示 `done:local_qwen_rkllm`。
13. 每个动作问答基于该动作最近一次报告；没有报告的动作显示“暂无该动作报告”。
14. 训练中问答被 `blocked_training`，不触发 GLM/Qwen 推理。
15. AI 问答 TTS 是 low priority，不抢训练纠错。
16. 调用本地 Qwen 时 `rknpu/load` 有波动。

