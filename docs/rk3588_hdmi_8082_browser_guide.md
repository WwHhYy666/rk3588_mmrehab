# RK3588 接显示屏调出 8082 浏览器画面流程

本文给新手说明：如何把 RK3588 开发板接到自己的显示屏上，在板子本机打开浏览器，并看到 8082 统一训练台首页。

推荐路线：

```text
独立显示屏 -> HDMI -> RK3588 HDMI OUT
USB 键盘鼠标 -> RK3588
摄像头 -> RK3588
板子浏览器 -> http://127.0.0.1:8082
```

注意：不要把 RK3588 的 HDMI 插到笔记本电脑 HDMI 口。大多数笔记本 HDMI 是输出口，不能当显示器输入。

## 1. HDMI 线应该怎么插

如果你的 HDMI 线现在是：

```text
独立显示屏/电视 <-> 笔记本电脑
```

可以把插在笔记本电脑上的那一头拔下来，插到 RK3588 板子的 HDMI OUT 口。

正确连接是：

```text
显示屏 HDMI 口 -> RK3588 HDMI OUT
```

错误连接是：

```text
RK3588 HDMI OUT -> 笔记本电脑 HDMI 口
```

笔记本电脑通常不能接收 HDMI 画面，所以不要把笔记本当作开发板显示屏。

## 2. 接好硬件

按下面顺序准备：

1. HDMI 线：显示屏接 RK3588 HDMI OUT。
2. 鼠标键盘：插到 RK3588 USB 口。
3. 摄像头：插到 RK3588 USB 口。
4. 网络：接 Wi-Fi 或网线。真实 GLM API 需要外网。
5. 电源：最后给 RK3588 接电源开机。

显示屏打开后，用显示屏按钮或遥控器选择对应 HDMI 输入源，例如 HDMI1 或 HDMI2。

如果没有画面：

- 换另一个 HDMI 输入源试一下。
- 检查 HDMI 是否插到 RK3588 的 HDMI OUT。
- 重启显示屏和 RK3588。

## 3. 进入板子系统

开机后可能出现三种情况：

- 出现桌面：可以继续。
- 出现登录界面：输入板子账号密码。
- 只出现命令行：也可以继续，只是打开浏览器可能要用命令。

如果完全没有画面，先不要管 8082，优先排查 HDMI、显示屏输入源和板子供电。

## 4. 打开终端

如果有桌面：

- 找 Terminal、终端或命令行图标。

如果只有命令行：

- 直接输入后面的命令。

## 5. 进入项目根目录

默认假设项目在：

```bash
cd /home/elf/project
```

如果这个目录不存在，用下面命令查找：

```bash
find /home/elf -maxdepth 4 -name record_prescription_http.py
```

找到文件后，进入它所在项目的根目录。项目根目录里应该能看到：

```text
prescription/
realtime/
evaluate/
docs/
```

如果看不到这些目录，说明还没有进入项目根目录。

## 6. 真实 GLM API 模式启动

如果你要真实大模型个性化建议，不要用 echo。先在启动 8082 的同一个终端里设置环境变量：

```bash
export ZHIPUAI_API_KEY="c8c07b936f6347c59fa0a6c78b4ead00.gExUfiic3MM5JQyX"
export REHAB_LLM_PROVIDER=glm4v_api
export REHAB_LLM_MODEL=glm-4v-flash
export REHAB_LLM_TIMEOUT=60
export RK_CAMERA_DEVICE=auto
export RK_CAMERA_WIDTH=640
export RK_CAMERA_HEIGHT=360
```

注意：

- API Key 只在终端里输入。
- 不要把 API Key 写进代码文件。
- 不要把 API Key 发到聊天记录或截图里。

确认不是 echo：

```bash
echo $REHAB_LLM_PROVIDER
echo $REHAB_LLM_MODEL
```

应该看到：

```text
glm4v_api
glm-4v-flash
```

## 7. 启动 8082

在项目根目录执行：

```bash
python3 prescription/banzi/record_prescription_http.py
```

看到类似输出，说明服务启动成功：

```text
8082 统一训练台已启动: http://板子IP:8082
请求摄像头设备: auto
实际摄像头设备: ...
姿态后端: requested=..., actual=...
```

这个终端不要关闭。8082 服务靠它运行。

如果要停止 8082，回到这个终端按：

```text
Ctrl + C
```

## 8. 在板子浏览器打开页面

如果有桌面浏览器，打开 Chromium、Browser 或浏览器。

地址栏输入：

```text
http://127.0.0.1:8082
```

也可以输入：

```text
http://localhost:8082
```

看到“智能康复终端”首页，就说明显示屏、本机浏览器和 8082 已经打通。

继续检查：

```text
http://127.0.0.1:8082/
http://127.0.0.1:8082/doctor
http://127.0.0.1:8082/train
```

预期结果：

- `/`：显示“智能康复终端”启动页。
- `/doctor`：显示医生录入页和摄像头预览区域。
- `/train`：显示患者训练 HUD、摄像头区域、AI 康复建议区域。

## 9. 检查真实 GLM 状态

另开一个终端，执行：

```bash
curl -s http://127.0.0.1:8082/status | python3 -m json.tool
```

找到 `llm` 字段，应类似：

```json
{
  "provider": "glm4v_api",
  "model": "glm-4v-flash",
  "api_key_configured": true
}
```

如果看到：

```json
{
  "provider": "echo"
}
```

说明真实大模型环境变量没有生效。处理方法：

1. 回到启动 8082 的终端。
2. 按 `Ctrl + C` 停止服务。
3. 重新 export `ZHIPUAI_API_KEY` 和 `REHAB_LLM_PROVIDER=glm4v_api`。
4. 重新启动 8082。

## 10. 常见问题

### 显示屏没画面

优先检查：

- 显示屏输入源是否选对 HDMI。
- HDMI 是否插到 RK3588 HDMI OUT。
- 板子是否正常供电。
- 线材是否松动。

不要把 RK3588 插到笔记本 HDMI 口。

### 浏览器打不开 `127.0.0.1:8082`

优先检查：

- 启动 8082 的终端是否还在运行。
- 终端是否报 Python 错误。
- 当前命令是否在项目根目录执行。
- 端口是否被占用。

### 摄像头没画面

先确认页面能打开。页面能打开说明 HDMI、浏览器和 8082 已经通了。

摄像头再单独查：

- 摄像头 USB 是否插好。
- 终端输出的实际摄像头设备是什么。
- `/stream.mjpg` 是否能打开。

### 真实 GLM 不工作

先检查：

```bash
curl -s http://127.0.0.1:8082/status | python3 -m json.tool
```

如果 `api_key_configured=false`，说明没有设置 API Key。

如果 `provider=echo`，说明没有用真实 GLM 模式启动。

检查外网：

```bash
getent hosts open.bigmodel.cn
python3 -c "import urllib.request; print(urllib.request.urlopen('https://open.bigmodel.cn', timeout=5).status)"
```

如果外网不通，先修网络、DNS、网关、代理或系统时间。

### 板子没有桌面浏览器

仍然可以启动 8082，然后在同一局域网的电脑浏览器访问：

```bash
hostname -I
```

假设板子 IP 是 `192.168.1.23`，电脑浏览器输入：

```text
http://192.168.1.23:8082
```

这不是板子本机浏览器调试，但可以临时确认 8082 服务和页面是否正常。

## 11. 最小成功标准

完成后至少应满足：

- 显示屏能显示 RK3588 桌面或命令行。
- 8082 终端显示“统一训练台已启动”。
- 板子浏览器能打开 `http://127.0.0.1:8082`。
- `/doctor` 和 `/train` 页面能进入。
- 如果调真实 GLM，`/status` 中 `provider=glm4v_api` 且 `api_key_configured=true`。
