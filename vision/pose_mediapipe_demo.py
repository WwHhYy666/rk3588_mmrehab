import cv2
import mediapipe as mp

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


print("MediaPipe Pose 演示程序启动。")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print("MediaPipe 模块加载成功。")
print(f"正在打开摄像头设备：{CAMERA_DEVICE}")

cap = open_camera()

if not cap.isOpened():
    print("摄像头打开失败。")
    print("请确认 /dev/video21 存在，并且支持 MJPG 1280x720。")
    raise SystemExit(1)

print("摄像头打开成功。")
print(f"当前请求格式：{FOURCC}")
print(f"当前请求分辨率：{FRAME_WIDTH}x{FRAME_HEIGHT}")
print("正在创建 Pose 模型...")

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

print("Pose 模型创建成功。")
print("按 ESC 或 q 退出。")

frame_count = 0

while True:
    success, frame = cap.read()

    if not success:
        print("读取摄像头画面失败。")
        break

    frame_count += 1

    # OpenCV 读出来的是 BGR，MediaPipe 需要 RGB 格式输入。
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 对当前帧执行姿态识别，结果会包含是否检测到人体以及关键点坐标。
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

        landmarks = result.pose_landmarks.landmark
        left_knee = landmarks[25]

        # 每 30 帧打印一次，避免终端刷屏过快，同时便于确认关键点数值在持续更新。
        if frame_count % 30 == 0:
            print("检测到人体姿态。")
            print(
                "左膝关键点 left_knee:",
                "x =", round(left_knee.x, 3),
                "y =", round(left_knee.y, 3),
                "visibility =", round(left_knee.visibility, 3),
            )
    else:
        cv2.putText(
            frame,
            "未检测到人体",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imshow("RK3588 MediaPipe Pose 演示 - 按 ESC 或 Q 退出", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q"):
        print("Pose 演示结束。")
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
