import cv2
import time
import numpy as np
import pose_module as pm

# 自动尝试摄像头索引：先试 1（外接），不行就试 0（自带）
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

print("正在启动摄像头，请稍候...")

# ... 前面导入和初始化保持不变 ...

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # --- 核心逻辑 1：利用 Z 轴深度差判定视角 ---
        # 11: 左肩, 12: 右肩
        z_left_shoulder = lmList[11][3]
        z_right_shoulder = lmList[12][3]
        z_diff = abs(z_left_shoulder - z_right_shoulder)

        # 获取置信度用于辅助判断侧面时哪只手在前
        vis_left = lmList[13][4]
        vis_right = lmList[14][4]

        # 根据你观察到的数据：正对时 0.0x，侧对时 0.35。取 0.2 作为阈值。
        is_side_view = z_diff > 0.2

        # --- 核心逻辑 2：计算 3D 角度 ---
        # 开启 use_3d=True 可以在俯拍时获得更真实的物理角度
        angleLeft = detector.findAngle(img, 11, 13, 15, use_3d=True, draw=True)
        angleRight = detector.findAngle(img, 12, 14, 16, use_3d=True, draw=True)

        # 映射逻辑 (根据 3D 角度调整阈值，3D 角度通常在 30-160 之间)
        perLeft = np.interp(angleLeft, (40, 150), (100, 0))
        perRight = np.interp(angleRight, (40, 150), (100, 0))

        if is_side_view:
            view_mode = "SIDE (3D Z-Detect)"
            # 侧面模式下，选 Z 轴更小（离镜头更近/更清晰）的那只手
            main_per = perLeft if z_left_shoulder < z_right_shoulder else perRight
            current_ready_to_count = main_per > 85
            current_ready_to_relax = main_per < 15
        else:
            view_mode = "FRONT (3D Z-Detect)"
            current_ready_to_count = perLeft > 85 and perRight > 85
            current_ready_to_relax = perLeft < 15 and perRight < 15

        # --- 核心逻辑 3：状态机计数 ---
        if current_ready_to_count and dir == 0:
            count += 0.5
            dir = 1
        if current_ready_to_relax and dir == 1:
            count += 0.5
            dir = 0

        # --- 绘制 UI (修正了你代码中左右手进度条画反的小 Bug) ---
        barLeft = np.interp(angleLeft, (40, 150), (100, 650))
        barRight = np.interp(angleRight, (40, 150), (100, 650))

        # 左侧 UI (显示左手数据)
        cv2.rectangle(img, (50, 100), (125, 650), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(barLeft)), (125, 650), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'L:{int(perLeft)}%', (50, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # 右侧 UI (显示右手数据)
        cv2.rectangle(img, (1150, 100), (1225, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1150, int(barRight)), (1225, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'R:{int(perRight)}%', (1150, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.putText(img, view_mode, (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        cv2.putText(img, str(int(count)), (580, 150), cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), 15)

    # ... FPS 和显示逻辑 ...
    # 计算 FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # 关键：显示窗口
    cv2.imshow("Pose Trainer", img)

    # 按 'q' 键可以主动退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord(' '):
        count =0

cap.release()
cv2.destroyAllWindows()