import cv2
import mediapipe as mp
import numpy as np
import csv

# =========================
# 计算 ABC 三点夹角
# =========================
def calculate_angle(a, b, c):
    """
    a, b, c: 三个点的 (x, y)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


# =========================
# MediaPipe 初始化
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# CSV 初始化
# =========================
csv_file = open("pushup_dataset.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# CSV 表头
csv_writer.writerow([
    "left_elbow_angle",
    "right_elbow_angle",
    "shoulder_hip_dist",
    "view_type",   # 0: front, 1: left-side, 2: right-side
    "label"        # 0: DOWN, 1: UP, -1: OTER
])

# =========================
# 摄像头
# =========================
cap = cv2.VideoCapture(1)

write_mode = False  # 是否写入 CSV
print("按 空格 开始/停止 采集，按 Q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        def get_point(idx):
            return (
                int(lm[idx].x * w),
                int(lm[idx].y * h)
            )

        # =========================
        # 关键点
        # =========================
        left_shoulder = get_point(11)
        left_elbow    = get_point(13)
        left_wrist    = get_point(15)

        right_shoulder = get_point(12)
        right_elbow    = get_point(14)
        right_wrist    = get_point(16)

        left_hip  = get_point(23)
        right_hip = get_point(24)

        # =========================
        # 特征 1 & 2：肘角度
        # =========================
        left_elbow_angle = calculate_angle(
            left_shoulder, left_elbow, left_wrist
        )
        right_elbow_angle = calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )

        # =========================
        # 特征 3：肩-髋垂直距离（归一化）
        # =========================
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        shoulder_hip_dist = (hip_y - shoulder_y) / h

        #判断是否进入俯卧撑形态
        is_horizontal = shoulder_hip_dist < 0.35

        # =========================
        # 视角判定（核心改进）
        # =========================
        z_diff = abs(lm[11].z - lm[12].z)

        if z_diff > 0.2:
            # 侧面
            if lm[11].z > lm[12].z:
                view_type = 1  # 左侧
                main_elbow_angle = left_elbow_angle
            else:
                view_type = 2  # 右侧
                main_elbow_angle = right_elbow_angle
        else:
            # 正面
            view_type = 0
            main_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

        # =========================
        # 按视角区分的 label 规则
        # =========================
        label = -1


        if is_horizontal:
            if view_type == 0:
                # 正面：双臂都可信
                if left_elbow_angle > 165 and right_elbow_angle > 165:
                    label = 1  # UP
                elif left_elbow_angle < 50 and right_elbow_angle < 50:
                    label = 0  # DOWN
            else:
                # 侧面：只看主臂
                if main_elbow_angle > 150:
                    label = 1
                elif main_elbow_angle < 75:
                    label = 0

        # =========================
        # CSV 写入控制
        # =========================
        if key == ord(' '):
            write_mode = not write_mode
            print("CSV 写入状态：", write_mode)

        if write_mode:
            csv_writer.writerow([
                round(left_elbow_angle, 2),
                round(right_elbow_angle, 2),
                round(shoulder_hip_dist, 4),
                view_type,
                label
            ])

        # =========================
        # 可视化
        # =========================
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        view_text = ["Front", "Left", "Right"][view_type]
        cv2.putText(
            frame,
            f"L:{int(left_elbow_angle)} R:{int(right_elbow_angle)} "
            f"View:{view_text} Label:{label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Push-up Data Collection", frame)

    if key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
