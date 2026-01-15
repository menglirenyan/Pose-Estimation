import cv2
import time
import numpy as np
from src.first_ import pose_module as pm

# 状态常量定义
STATE_IDLE = "待机"  # 人未趴下
STATE_READY = "已就绪"  # 已趴下，手臂伸直
STATE_DOWN = "下压中"  # 正在向下
STATE_UP = "起撑中"  # 正在向上

# 初始化
cap = cv2.VideoCapture(1)  # 默认摄像头
detector = pm.poseDetector(detectionCon=0.7)
count = 0
current_state = STATE_IDLE
hand_l_anchor = None
hand_r_anchor = None
pTime = 0

show_skel = False  # 是否显示骨架

print("系统启动成功，按 'Q' 退出，' ' 重置计数...")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    h, w, _ = img.shape

    img = detector.findPose(img, draw=show_skel)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        form_feedback = []

        # 1. 判定用户是否趴下了
        # 11: 左肩, 23: 左胯
        y_shoulder = lmList[11][2]
        y_hip = lmList[23][2]
        # 在俯卧撑姿态中，肩和胯的垂直高度差会远小于身高
        is_horizontal = abs(y_shoulder - y_hip) < (h * 0.2)

        # 2. 视角判定
        z_diff = abs(lmList[11][3] - lmList[12][3])
        is_side_view = z_diff > 0.3

        # 3. 角度与映射计算
        angleL = detector.findAngle(img, 11, 13, 15, use_3d=True, draw=True)
        angleR = detector.findAngle(img, 12, 14, 16, use_3d=True, draw=True)

        if is_side_view:
            up_limit, down_limit = 150, 75  # 侧面伸直要求低一些
            main_per = np.interp(angleL, (down_limit, up_limit), (100, 0)) \
                if lmList[11][3] < lmList[12][3] \
                else np.interp(angleR, (down_limit, up_limit), (100, 0))
        else:
            up_limit, down_limit = 165, 50
            main_per = (np.interp(angleL, (down_limit, up_limit), (100, 0)) + np.interp(angleR, (down_limit, up_limit),
                                                                                        (100, 0))) / 2

        # 4. 有限状态机阶段逻辑
        if not is_horizontal:
            current_state = STATE_IDLE
            hand_l_anchor = None
        else:
            # 已趴下，根据手臂动作切换状态
            if current_state == STATE_IDLE:
                current_state = STATE_READY  # 趴下瞬间进入就绪

            if current_state == STATE_READY:
                if main_per > 15:  # 开始下潜
                    current_state = STATE_DOWN
                    # 记录动作开始时的手掌位置
                    hand_l_anchor = lmList[15][1:3]
                    hand_r_anchor = lmList[16][1:3]

            elif current_state == STATE_DOWN:
                if main_per > 85:  # 到达底部
                    current_state = STATE_UP
                elif main_per < 5:  # 没做完又站起来了
                    current_state = STATE_READY

            elif current_state == STATE_UP:
                if main_per < 15:  # 回到顶端，完成一次
                    count += 1
                    current_state = STATE_READY

        # 5. 标准动作检测 (仅在动作过程中检测)
        if current_state in [STATE_DOWN, STATE_UP]:
            # 检测手掌固定
            if hand_l_anchor:
                dist_l = detector.getDistance(lmList[15][1:3], hand_l_anchor)
                if dist_l > 30: form_feedback.append("手掌请固定不动")

            # 检测夹角
            shoulder_angle = detector.findAngle(img, 13, 11, 23, use_3d=True, draw=False)
            if shoulder_angle > 75: form_feedback.append("大臂张开过大")

            # 检测塌腰 (侧面)
            if is_side_view:

                back_angle = detector.findAngle(img, 11, 23, 27, use_3d=True, draw=False)
                if back_angle < 155: form_feedback.append("请勿塌腰或拱背")

        # 6. UI 渲染
        # 状态显示
        state_color = (0, 255, 0) if current_state != STATE_IDLE else (0, 0, 255)
        img = detector.putText_chinese(img, f"状态: {current_state}", (50, 100), fontSize=40, color=state_color)

        # 计数显示
        cv2.putText(img, str(int(count)), (w // 2 - 50, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)

        # 提示显示
        y_off = 200
        for msg in list(set(form_feedback)):
            img = detector.putText_chinese(img, f"× {msg}", (w - 400, y_off), fontSize=30, color=(0, 0, 255))
            y_off += 50

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Push-up Pro Trainer", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord(' '): count = 0
    if key == ord('v'): show_skel = not show_skel

cap.release()
cv2.destroyAllWindows()