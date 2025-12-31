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

while True:
    success, img = cap.read()
    if not success: break

    # --- 第一步：镜像翻转 ---
    # flipCode=1 表示水平翻转。这样你的左手就会出现在屏幕左边
    img = cv2.flip(img, 1)

    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # --- 第二步：重新定义左右（翻转后，11/13/15确实就在左边了） ---
        angleLeft = detector.findAngle(img, 11, 13, 15, draw=True)
        angleRight = detector.findAngle(img, 12, 14, 16, draw=True)

        # 调试：在控制台打印角度，看看右手为什么不跳动
        # 如果你发现右手的角度一直在 160 以上，说明 150 的上限设低了
        # print(f"L: {int(angleLeft)}  R: {int(angleRight)}")

        # --- 第三步：放宽映射范围，防止“卡死”在 100% ---
        # 建议先用 (60, 165) 测试，确保动作能覆盖整个量程
        perLeft = np.interp(angleLeft, (50, 160), (100, 0))
        perRight = np.interp(angleRight, (50, 160), (100, 0))

        barLeft = np.interp(angleLeft, (50, 160), (100, 650))
        barRight = np.interp(angleRight, (50, 160), (100, 650))

        # --- 绘制逻辑保持不变，但坐标要对应 ---
        # 左侧柱状图 (对应 Left)
        cv2.rectangle(img, (50, 100), (125, 650), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(barLeft)), (125, 650), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'L {int(perLeft)}%', (50, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # 右侧柱状图 (对应 Right)
        cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1100, int(barRight)), (1175, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'R {int(perRight)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        color = (255, 0, 255)
        if perLeft > 90 and perRight > 90:
            color = (0, 255, 0)  # 变绿提示到位
            if dir == 0:
                count += 0.5
                dir = 1  # 状态切换：现在期待用户“放下”手臂

        # 2. 检测到达底部（伸展）
        if perLeft < 10 and perRight < 10:
            color = (255, 0, 0)  # 变红提示完全放下
            if dir == 1:
                count += 0.5
                dir = 0  # 状态切换：现在期待用户“提起”手臂
        # 6. 显示总计数
        cv2.putText(img, str(int(count)), (550, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 20)

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