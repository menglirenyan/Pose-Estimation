import cv2
import mediapipe as mp
import math
import numpy as np

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # 建议使用复杂度 1，兼顾性能与 Z 轴预测
        self.pose = self.mpPose.Pose(static_image_mode=mode, model_complexity=1,
                                     smooth_landmarks=smooth,
                                     min_detection_confidence=detectionCon,
                                     min_tracking_confidence=trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 核心改进：lmList 现在存储 [id, x, y, z, visibility]
                self.lmList.append([id, cx, cy, lm.z, lm.visibility])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, use_3d=False, draw=True):
        # 1. 提取坐标
        # 注意：lmList[p][1:3] 是像素坐标 (x, y)，lmList[p][3] 是深度 z
        # 为了 3D 计算准确，建议此处使用归一化坐标计算，或者给 Z 轴一个权重缩放
        # 这里我们直接取 lmList 中的数据
        p1_data = self.lmList[p1]
        p2_data = self.lmList[p2]
        p3_data = self.lmList[p3]

        # 获取像素坐标用于绘图
        x1, y1 = p1_data[1], p1_data[2]
        x2, y2 = p2_data[1], p2_data[2]
        x3, y3 = p3_data[1], p3_data[2]

        if use_3d:
            # --- 3D 向量夹角计算 ---
            # 构造 numpy 向量 [x, y, z]
            v1 = np.array([p1_data[1], p1_data[2], p1_data[3]]) - np.array(
                [p2_data[1], p2_data[2], p2_data[3] ])
            v2 = np.array([p3_data[1], p3_data[2], p3_data[3]]) - np.array(
                [p2_data[1], p2_data[2], p2_data[3]])

            # 计算点积和模长
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            # 避免除零错误
            if norm_v1 == 0 or norm_v2 == 0:
                return 0

            # 余弦值
            cosine_angle = dot_product / (norm_v1 * norm_v2)
            # 裁剪范围在 [-1, 1] 防止浮点误差导致 arccos 报错
            angle = math.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        else:
            # --- 原有 2D 角度计算 (基于 atan2) ---
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            if angle < 0: angle += 360
            if angle > 180: angle = 360 - angle

        # 绘制可视化信息
        if draw:
            # 绘制连接线 (像素平面)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            # 绘制关键点圆圈
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            # 在关节处显示角度
            # 使用白色文字，带一点偏移防止遮挡
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        return angle