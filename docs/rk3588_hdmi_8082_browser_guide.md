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


## 10. 开机自启动和全屏展示模式

如果你想拍视频时完全不用命令行，可以安装一次自启动。之后 RK3588 开机后会自动启动 8082/Qwen 服务，桌面登录后自动打开全屏浏览器：

```bash
cd /home/elf/project/project_system
chmod +x scripts/install_rehab_station_autostart.sh
./scripts/install_rehab_station_autostart.sh
```

如果项目实际不在这个目录，就进入你实际上传的项目根目录后执行同一条安装命令。脚本会自动使用当前项目路径。

安装后可以不重启，先手动启动服务检查：

```bash
sudo systemctl start rehab-station-qwen.service
sudo systemctl status rehab-station-qwen.service --no-pager
```

浏览器全屏展示地址是：

```text
http://127.0.0.1:8082/train?kiosk=1
```

`kiosk=1` 会进入拍视频展示布局：隐藏顶部导航，压缩页面边距，让训练画面和状态区域尽量铺满显示屏。普通调试仍然打开：

```text
http://127.0.0.1:8082/train
```

如果自启动浏览器没有弹出，先确认板子系统确实进入了桌面环境；只有纯命令行系统时，后台 8082 服务仍会自启动，但浏览器需要桌面或窗口管理器。

这几条安装命令可以在电脑 SSH 到板子后输入，不要求你在显示屏上接键盘输入。它们的作用是给板子安装自启动配置；安装完成后，显示屏展示时不再需要电脑参与。

拍视频时是否完全不用键盘，取决于两个条件：

```text
1. 板子系统能自动进入桌面，或你已经手动登录到桌面。
2. WiFi 热点密码、GLM Key、项目路径、自启动服务都已经提前配置好。
```

满足后，键盘一般不用再接。鼠标或触摸屏仍然有用，因为当前训练页还需要点“开始完整训练”、切换右侧完成度/报告/问答页签、必要时选择 GLM/Qwen。若你只拍固定流程，默认患者编号和目标次数已经够用，通常一只鼠标就能操作。

键盘最有用的场景是：第一次连 WiFi、第一次安装自启动、浏览器没有自动打开、需要临时改热点密码、查看 `systemctl status` 或 `/status`。这些都可以提前通过电脑 SSH 做完。

## 11. WiFi 手机热点连接和外网测试

真实 GLM 需要外网，手机热点可行；本地 Qwen 可以断网展示。建议先在电脑 SSH 里把 WiFi 配好并保存，正式接显示屏时就不用键盘重新输入热点密码。

### 11.1 先安装 WiFi 模块硬件

如果你的 WiFi 模块还没有装，先做硬件安装。不要带电插拔 WiFi 模块。

准备工具：

```text
小十字螺丝刀
WiFi 模块
配套天线或天线延长线
防静电注意：先摸一下金属桌脚/机箱放电，手尽量不要摸金手指
```

安装步骤：

```text
1. 关闭 RK3588，拔掉 12V 电源，HDMI/USB 也可以先拔掉，确保板子完全断电。
2. 找到板子上的 WiFi/BT 模块插槽。常见是 M.2 E-Key 小插槽，旁边通常有固定螺丝孔；如果你的模块是厂家配套的 CF-AX200-M，就按这个槽位安装。
3. 拆下插槽末端的小固定螺丝，先收好，别掉进板子缝里。
4. WiFi 模块金手指对准插槽缺口，斜着约 20-30 度插进去。不要硬怼，方向对时会比较顺。
5. 插到底后，把模块轻轻压平，让模块尾部螺丝孔对准板子螺丝柱。
6. 拧回固定螺丝，拧到稳即可，不要过紧。
7. 接天线。天线小圆扣对准模块上的 MAIN/AUX 小座，垂直轻按，听到或感觉到轻微卡住即可。
8. 天线尽量贴到外壳边缘或空旷位置，避开 USB3.0 摄像头线、电源线、金属屏蔽和散热片。
9. 再接电源开机。
```

天线注意：

```text
MAIN 至少要接一根天线，否则能识别 WiFi 也可能信号很差。
AUX 有第二根就接上，没有也可以先用 MAIN 测试。
小圆扣很脆，按的时候要垂直向下，别斜着撬。
```

装完先确认系统识别：

```bash
nmcli dev status
ip link
rfkill list
```

成功时应能看到类似：

```text
wlP4p65s0
wlan0
```

如果看不到无线网卡，继续查：

```bash
lspci | grep -i -E "wifi|wireless|network|intel"
lsusb
```

判断：

```text
能看到 wlP4p65s0/wlan0：硬件和驱动基本可用，继续连手机热点。
能看到 Intel/AX200/Network 但没有 wlan：可能驱动或 rfkill 问题，先看 rfkill list。
完全看不到模块：优先检查模块是否插到底、方向是否正确、固定螺丝是否压住、板子是否断电重插过。
能扫到热点但信号很弱：优先检查 MAIN 天线有没有扣好。
```

如果你不确定插槽或方向，先不要硬插；拍一张板子 WiFi 插槽和模块正反面照片，再对照确认。

### 11.2 打开手机热点

手机上先打开个人热点，建议临时设置成简单英文名称，避免中文 SSID 或特殊符号导致命令行转义麻烦，例如：

```text
热点名称：rehab-demo
热点密码：12345678
频段：优先 2.4GHz；如果手机只开 5GHz 也可以先试
```

如果现场网络复杂，建议先关闭“隐藏热点”，并确认手机允许新设备加入。

### 11.3 桌面方式连接 WiFi

如果显示屏已经接上并进入桌面，可以打开系统里的 `WiFi` 应用：

```text
选择无线网卡 wlP4p65s0
SSID 输入手机热点名称
PAWD 输入手机热点密码
点击 connect
等待 5 秒后点 status 或 ping
```

ELF 2 手册里 Desktop 系统的 WiFi 工具就是这个流程。连接成功后，系统通常会保存这个热点配置，下次开机可自动重连。

### 11.4 SSH 命令行方式连接 WiFi

电脑先 SSH 到板子，然后执行：

```bash
nmcli radio wifi on
nmcli dev status
nmcli dev wifi rescan
nmcli dev wifi list
```

如果能看到手机热点，用下面命令连接。把 `rehab-demo` 和 `12345678` 换成你的手机热点名称和密码：

```bash
sudo nmcli dev wifi connect "rehab-demo" password "12345678" ifname wlP4p65s0
```

如果系统提示没有 `wlP4p65s0`，先查真实无线网卡名：

```bash
ip link
nmcli dev status
```

然后把命令里的 `wlP4p65s0` 换成实际 WiFi 网卡名。

连接后检查：

```bash
nmcli -f GENERAL.STATE,GENERAL.CONNECTION dev show wlP4p65s0
ip addr show wlP4p65s0
ip route
```

成功标准：

```text
GENERAL.STATE 显示 connected
ip addr 里有 192.168.x.x、172.20.x.x 或 10.x.x.x 这类热点分配的 IP
ip route 里有 default via ...
```

### 11.5 测试能不能访问外网

先测网络连通：

```bash
ping -I wlP4p65s0 -c 3 223.5.5.5
```

再测 DNS：

```bash
getent hosts open.bigmodel.cn
ping -I wlP4p65s0 -c 3 open.bigmodel.cn
```

最后测 HTTPS，也就是 GLM 这类 API 最关心的路径：

```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('https://open.bigmodel.cn', timeout=8).status)"
```

成功标准：

```text
ping 223.5.5.5 有回复      说明手机热点网络通
getent hosts 有 IP          说明 DNS 可用
python3 返回 200/301/302/403 说明 HTTPS 能连到外网服务
```

如果 IP 能 ping 通，但域名不通，多半是 DNS 问题。可以给这个热点连接指定 DNS：

```bash
sudo nmcli con mod "rehab-demo" ipv4.dns "223.5.5.5 119.29.29.29" ipv4.ignore-auto-dns yes
sudo nmcli con up "rehab-demo"
```

如果连热点后电脑 SSH 断了，说明你原来的 SSH 网络和热点网络不是同一个链路。此时不要慌，显示屏本地仍可继续；也可以把电脑也连到同一个手机热点，再用 `hostname -I` 或手机热点设备列表找到板子 IP。

### 11.6 测试 8082 和 GLM/Qwen 状态

WiFi 通了以后，在板子上检查：

```bash
curl -s http://127.0.0.1:8082/status | python3 -m json.tool
./scripts/check_llm_status.sh
```

重点看：

```text
llm.provider / llm.active_provider
llm.api_key_configured
voice.qa_allowed
qwen_generate_ok
```

如果要拍“联网 GLM”效果，必须满足：

```text
手机热点外网可用
ZHIPUAI_API_KEY 已写入 runtime/llm.env 或当前启动环境
/api/status 里 api_key_configured=true
```

如果外网失败，展示时可以切到 Qwen，本地回答仍然能支撑“端侧智能问答”的卖点。

## 12. 常见问题

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

## 13. 最小成功标准

完成后至少应满足：

- 显示屏能显示 RK3588 桌面或命令行。
- 8082 终端显示“统一训练台已启动”。
- 板子浏览器能打开 `http://127.0.0.1:8082`。
- `/doctor` 和 `/train` 页面能进入。
- 如果调真实 GLM，`/status` 中 `provider=glm4v_api` 且 `api_key_configured=true`。






