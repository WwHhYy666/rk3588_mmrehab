import cv2

# 这些常量对应当前在 RK3588 板子上已经验证通过的摄像头参数。
CAMERA_DEVICE = "/dev/video21"
CAMERA_BACKEND = cv2.CAP_V4L2
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FOURCC = "MJPG"


def open_camera() -> cv2.VideoCapture:
    """按当前板端已验证的参数打开摄像头。"""
    cap = cv2.VideoCapture(CAMERA_DEVICE, CAMERA_BACKEND)
    # 设备节点、后端、编码格式和分辨率都尽量显式指定，避免板端默认值不一致。
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


print("摄像头测试程序启动。")
print(f"正在打开摄像头设备：{CAMERA_DEVICE}")

cap = open_camera()

if not cap.isOpened():
    print("摄像头打开失败。")
    print("请确认 /dev/video21 存在，并且支持 MJPG 1280x720。")
    raise SystemExit(1)

print("摄像头打开成功。")
print(f"当前请求格式：{FOURCC}")
print(f"当前请求分辨率：{FRAME_WIDTH}x{FRAME_HEIGHT}")
print("按 ESC 或 q 退出。")

while True:
    success, frame = cap.read()

    if not success:
        print("读取摄像头画面失败。")
        break

    # 实时显示当前摄像头画面，用来确认板端预览链路已经跑通。
    cv2.imshow("RK3588 摄像头测试 - 按 ESC 或 Q 退出", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q"):
        print("摄像头测试结束。")
        break

cap.release()
cv2.destroyAllWindows()
