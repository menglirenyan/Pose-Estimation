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

#初始化部分
show_all = False       #显示骨架
hand_start_pos = None  # 记录手掌起始位置
warning_list = []

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, draw=show_all)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        form_feedback = []  # 每一帧重置提示信息

        # 1. 视角判定 (11:左肩, 12:右肩)
        z_left_shoulder = lmList[11][3]
        z_right_shoulder = lmList[12][3]
        z_diff = abs(z_left_shoulder - z_right_shoulder)
        is_side_view = z_diff > 0.3     #0.3是基于观测到的正侧面阈值

        # 2. 基础运动角度 (手臂弯曲度)
        angleLeft = detector.findAngle(img, 11, 13, 15, use_3d=True, draw=True)
        angleRight = detector.findAngle(img, 12, 14, 16, use_3d=True, draw=True)

        # 映射进度
        if is_side_view:
            # 侧面视角：由于遮挡和角度，伸直可能只有 145度，弯曲可能到 60度
            up_threshold, down_threshold = 145, 65
        else:
            # 正面视角：角度更明显
            up_threshold, down_threshold = 160, 45
        perLeft = np.interp(angleLeft, (down_threshold, up_threshold), (100, 0))
        perRight = np.interp(angleRight, (down_threshold, up_threshold), (100, 0))

        # 1：手掌固定检测
        # 当手臂开始弯曲（下压开始）时，记录手掌位置作为锚点
        if perLeft > 10 or perRight > 10:
            if hand_start_pos is None:
                hand_l_anchor = lmList[15][1:3]  # 左右手的 (x, y)
                hand_r_anchor = lmList[16][1:3]

            # 计算当前手腕与锚点的位移
            dist_l = detector.getDistance(lmList[15][1:3], hand_l_anchor)
            dist_r = detector.getDistance(lmList[16][1:3], hand_r_anchor)

            # 阈值设为 40 像素（可根据相机远近调整）
            if dist_l > 40 or dist_r > 40:
                form_feedback.append("手掌不要动")
        else:
            # 回到最高点时重置锚点，允许微调手位
            hand_l_anchor = None
            hand_r_anchor = None

        # 2：大臂与躯干夹角 45°
        # 计算 肘部-肩部-胯部 的夹角: 左(13-11-23), 右(14-12-24)
        shoulder_angle_l = detector.findAngle(img, 13, 11, 23, use_3d=True, draw=False)
        shoulder_angle_r = detector.findAngle(img, 14, 12, 24, use_3d=True, draw=False)

        # 俯卧撑标准：大臂不要向外张开太大（通常建议 45° 左右，若大于 70° 则视为不标准）
        if perLeft > 50:  # 只在下压到一定程度时检测
            if shoulder_angle_l > 70 or shoulder_angle_r > 70:
                form_feedback.append("不标准夹紧大臂")

        # 标准 3：核心收紧:不塌腰
        # 侧面视角最准确：检测 肩-胯-膝 是否呈直线: 11-23-27 (左) 或 12-24-28 (右)
        if is_side_view:
            # 根据哪边离镜头近选哪边
            if z_left_shoulder < z_right_shoulder:
                back_angle = detector.findAngle(img, 11, 23, 25, use_3d=True, draw=show_all)
            else:
                back_angle = detector.findAngle(img, 12, 24, 26, use_3d=True, draw=show_all)

            # 180度为直线，若小于 155度 说明腰部下塌或臀部过高
            if back_angle < 150:
                form_feedback.append("请收紧核心，不要塌腰或撅屁股")
            if back_angle > 175:
                form_feedback.append("请收紧核心，不要塌腰或撅屁股")

        #状态机计数逻辑
        if is_side_view:
            main_per = perLeft if z_left_shoulder < z_right_shoulder else perRight
            current_ready_to_count = main_per > 85
            current_ready_to_relax = main_per < 15
        else:
            current_ready_to_count = perLeft > 85 and perRight > 85
            current_ready_to_relax = perLeft < 15 and perRight < 15

        if current_ready_to_count and dir == 0:
            count += 0.5
            dir = 1
        if current_ready_to_relax and dir == 1:
            count += 0.5
            dir = 0

        # UI 绘制
        # 绘制进度条
        barLeft = np.interp(angleLeft, (45, 155), (100, 650))
        barRight = np.interp(angleRight, (45, 155), (100, 650))
        cv2.rectangle(img, (50, 100), (85, 650), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(barLeft)), (85, 650), (255, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (1195, 100), (1230, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1195, int(barRight)), (1230, 650), (0, 255, 0), cv2.FILLED)

        # 绘制计数
        cv2.putText(img, str(int(count)), (580, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 20)

        # 绘制提示信息 (Feedback)
        #img = detector.putText_chinese(img, f"完成次数: {int(count)}", (550, 50), fontSize=60, color=(255, 255, 255))

        y_offset = 150
        for msg in list(set(form_feedback)):
            img = detector.putText_chinese(img, f"提示: {msg}", (400, y_offset), fontSize=35, color=(0, 0, 255))
            y_offset += 50

        # 视角显示
        view_text = "SIDE VIEW" if is_side_view else "FRONT VIEW"
        cv2.putText(img, view_text, (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    # FPS 显示
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Push-up Analyzer", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord(' '): count = 0
    if key == ord('v'): show_all = not show_all

cap.release()
cv2.destroyAllWindows()