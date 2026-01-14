import cv2
import time
import numpy as np
import pandas as pd
import joblib
import os
from model_wrapper import LogisticPushupModel, STATE_DOWN, STATE_UP, STATE_UNCERTAIN

import mediapipe as mp

# ------------------ 配置 ------------------
MODEL_PATH = "models/logistic.joblib"
SCALER_PATH = "models/scaler.joblib"
THRESH = 0.75            # 概率阈值（可根据结果微调）
MIN_CONSISTENT = 3      # 连续帧确认阈值
OUTPUT_CSV = "realtime_results_single_model.csv"
CAM_INDEX = 1           # 摄像头索引

# ------------------ 检查模型 ------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler not found. Run your training script to produce models/scaler.joblib and models/logistic.joblib")

model_wrapper = LogisticPushupModel(MODEL_PATH, SCALER_PATH)
print("Loaded LogisticPushupModel.")

# ------------------ MediaPipe 初始化 ------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.dot(ba, bc) / denom
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def extract_features_from_landmarks(lm, frame_shape):
    h, w = frame_shape[:2]
    def get_point(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))
    left_shoulder = get_point(11); left_elbow=get_point(13); left_wrist=get_point(15)
    right_shoulder = get_point(12); right_elbow=get_point(14); right_wrist=get_point(16)
    left_hip=get_point(23); right_hip=get_point(24)
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0
    shoulder_hip_dist = (hip_y - shoulder_y) / h
    # return feature vector in same order as training
    return np.array([left_elbow_angle, right_elbow_angle, shoulder_hip_dist])

# ------------------ 计数状态（单模型） ------------------
last_stable_state = STATE_UNCERTAIN
current_candidate = STATE_UNCERTAIN
candidate_count = 0
count = 0

records = []  # 每帧记录

cap = cv2.VideoCapture(CAM_INDEX)
print("按下空格开始录入CSV, 再次按住取消； 按'Q'退出")
write_mode = False

p_time = time.time()
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_idx += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    state_str = "NoPose"
    p_down = p_up = 0.0

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        feats = extract_features_from_landmarks(lm, frame.shape)
        # is_horizontal gate (optional): ensure we are roughly parallel to floor before trusting model
        is_horizontal = feats[2] < 0.35

        # model prediction
        state, p_down, p_up = model_wrapper.predict_state_from_features(feats, thresh=THRESH)

        # if not horizontal, force uncertain
        if not is_horizontal:
            state = STATE_UNCERTAIN

        # debounce
        if state == current_candidate:
            candidate_count += 1
        else:
            current_candidate = state
            candidate_count = 1

        # accept new stable state only if candidate_count >= MIN_CONSISTENT and it's not UNCERTAIN
        if candidate_count >= MIN_CONSISTENT and current_candidate != STATE_UNCERTAIN:
            # check transition DOWN -> UP to count one push-up
            if last_stable_state == STATE_DOWN and current_candidate == STATE_UP:
                count += 1
            last_stable_state = current_candidate

        state_str = {STATE_UNCERTAIN:"-1", STATE_DOWN:"DOWN", STATE_UP:"UP"}.get(state, str(state))

        # draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # record
        records.append({
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "left": float(feats[0]),
            "right": float(feats[1]),
            "dist": float(feats[2]),
            "p_down": float(p_down),
            "p_up": float(p_up),
            "observed_state": int(state),
            "count": int(count)
        })

    # overlay info
    h, w = frame.shape[:2]
    info = f"Cnt={count}  State={state_str}  p_up={p_up:.2f} p_down={p_down:.2f}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Thresh={THRESH} MinCon={MIN_CONSISTENT}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    if write_mode:
        cv2.putText(frame, "LOGGING ON", (w-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # FPS
    c_time = time.time()
    fps = 1.0 / (c_time - p_time + 1e-6)
    p_time = c_time
    cv2.putText(frame, f"FPS:{int(fps)}", (w-120, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)

    cv2.imshow("Pushup Logistic Runner", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        write_mode = not write_mode
        print("Logging:", write_mode)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# save records
if len(records) > 0 and write_mode:
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print("Wrote realtime log to", OUTPUT_CSV)
else:
    print("No realtime records saved (logging was off or no records).")

print("Final count:", count)
