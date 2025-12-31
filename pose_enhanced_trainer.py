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
        # --- 核心逻辑 1：视角判定 ---
        # 计算左右肩膀的水平像素距离
        shoulder_dist = abs(lmList[11][1] - lmList[12][1])
        # 获取左右手的置信度
        vis_left = lmList[13][3]  # 左肘置信度
        vis_right = lmList[14][3]  # 右肘置信度

        # 判定标准：肩膀距离小于某个阈值（例如画面宽度的12%）或者某一侧置信度极低
        is_side_view = shoulder_dist < 150 or abs(vis_left - vis_right) > 0.3

        # --- 核心逻辑 2：获取当前有效百分比 ---
        angleLeft = detector.findAngle(img, 11, 13, 15, draw=True)
        angleRight = detector.findAngle(img, 12, 14, 16, draw=True)

        perLeft = np.interp(angleLeft, (50, 150), (100, 0))
        perRight = np.interp(angleRight, (50, 150), (100, 0))

        if is_side_view:
            # 侧面模式：选更清晰的那只手作为主控
            view_mode = "SIDE VIEW"
            main_per = perLeft if vis_left > vis_right else perRight
            current_ready_to_count = main_per > 90
            current_ready_to_relax = main_per < 10
        else:
            # 正面模式：双臂必须同时达标
            view_mode = "FRONT VIEW"
            current_ready_to_count = perLeft > 90 and perRight > 90
            current_ready_to_relax = perLeft < 10 and perRight < 10

        # --- 核心逻辑 3：状态机计数 ---
        if current_ready_to_count:
            if dir == 0:
                count += 0.5
                dir = 1
        if current_ready_to_relax:
            if dir == 1:
                count += 0.5
                dir = 0

        # --- 绘制 UI ---
        # 进度条逻辑保持不变 ...
        # 计算进度条的 y 坐标映射
        barLeft = np.interp(angleLeft, (50, 160), (100, 650))
        barRight = np.interp(angleRight, (50, 160), (100, 650))

        # 绘制左侧进度条 (Blue)
        cv2.rectangle(img, (50, 100), (125, 650), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(barRight)), (125, 650), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(perRight)}%', (50, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # 绘制右侧进度条 (Green)
        cv2.rectangle(img, (1150, 100), (1225, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1150, int(barLeft)), (1225, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(perLeft)}%', (1150, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # 绘制视角提示和总计数
        cv2.putText(img, view_mode, (540, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
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