"""
main_realtime.py

实时主控脚本：可以同时加载多个 sklearn 模型（logistic/mlp）并对同一帧进行并行推理与计数对比。

用法示例：
    python main_realtime.py --models logistic mlp --cam 1
"""

import os
import time
import argparse
import joblib
import math
import csv

import numpy as np
import cv2
import mediapipe as mp

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 常量状态
STATE_DOWN = 0
STATE_UP = 1
STATE_UNCERTAIN = -1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["logistic", "mlp"],
                   help="List of model names to load (expects models/{name}.joblib and models/scaler_{name}.joblib)")
    p.add_argument("--models_dir", default=os.path.join(BASE_DIR, "src", "models"), help="Directory where models and scalers are saved")
    p.add_argument("--cam", type=int, default=1, help="Camera index")
    p.add_argument("--thresh", type=float, default=0.75, help="Probability threshold for confident prediction")
    p.add_argument("--min_consistent", type=int, default=3, help="Debounce: consecutive frames required")
    p.add_argument("--log", action="store_true", help="If set, write realtime records CSV on exit ")
    return p.parse_args()

# --------------- 一个通用的 sklearn wrapper（兼容 logistic/mlp 等） ---------------
class SklearnWrapper:
    """
    Generic wrapper for sklearn models saved with joblib.
    Exposes:
      - predict_proba_from_features(feat) -> np.array([p_class0, p_class1])
      - predict_state_from_features(feat, thresh) -> (state, p_down, p_up)
    """
    def __init__(self, model_path, scaler_path):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or scaler not found: {model_path}, {scaler_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        # classes_ must exist
        self.classes_ = getattr(self.model, "classes_", None)
        if self.classes_ is None:
            raise ValueError("Loaded sklearn model has no classes_ attribute.")
        # find index positions
        cls = list(self.classes_)
        if 0 not in cls or 1 not in cls:
            raise ValueError("Model classes_ must contain both 0 and 1.")
        self.idx0 = int(cls.index(0))
        self.idx1 = int(cls.index(1))

    def predict_proba_from_features(self, feat):
        x = np.array(feat).reshape(1, -1)
        x_s = self.scaler.transform(x)
        proba_all = self.model.predict_proba(x_s)[0]
        p0 = float(proba_all[self.idx0])
        p1 = float(proba_all[self.idx1])
        return np.array([p0, p1])

    def predict_state_from_features(self, feat, thresh=0.75):
        p = self.predict_proba_from_features(feat)
        p_down = float(p[0]); p_up = float(p[1])
        if p_up >= thresh and p_up > p_down:
            return STATE_UP, p_down, p_up
        elif p_down >= thresh and p_down > p_up:
            return STATE_DOWN, p_down, p_up
        else:
            return STATE_UNCERTAIN, p_down, p_up

# -------------------- 帮助函数：角度、特征提取 --------------------
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
    return np.array([left_elbow_angle, right_elbow_angle, shoulder_hip_dist])

def valid_pose(lm, vis_thresh=0.6):
    need = [11,12,13,14,15,16,23,24]
    for idx in need:
        if lm[idx].visibility < vis_thresh:
            return False
    return True

# -------------------- 主流程 --------------------
def main():
    args = parse_args()

    # load models
    wrappers = {}
    stats = {}
    for name in args.models:
        model_path = os.path.join(args.models_dir, f"{name}.joblib")
        scaler_path = os.path.join(args.models_dir, f"scaler_{name}.joblib")
        try:
            wrap = SklearnWrapper(model_path, scaler_path)
        except Exception as e:
            print(f"[WARN] Failed to load model '{name}': {e}")
            continue
        wrappers[name] = wrap
        # init per-model state for counting and debounce
        stats[name] = {
            "last_stable_state": STATE_UNCERTAIN,
            "current_candidate": STATE_UNCERTAIN,
            "candidate_count": 0,
            "count": 0
        }
        print(f"Loaded model '{name}'")

    if len(wrappers) == 0:
        raise SystemExit("No models loaded. Place models/*.joblib and scalers in models/ and pass --models accordingly.")

    # mediapipe init
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.cam)
    print("Press SPACE to toggle logging; Press Q to quit.")
    write_mode = False
    records = []
    p_time = time.time()
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_idx += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # defaults for display
            per_model_texts = []
            # probabilities default
            model_probs = {name: (0.0, 0.0) for name in wrappers.keys()}

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w = frame.shape[:2]

                # validate pose
                if not valid_pose(lm):
                    # show "NO_POSE" for all models
                    for name in wrappers.keys():
                        per_model_texts.append(f"{name}: NO_POSE")
                else:
                    feats = extract_features_from_landmarks(lm, frame.shape)
                    # optional horizontal gate
                    is_horizontal = feats[2] < 0.35

                    # iterate models
                    for name, wrap in wrappers.items():
                        state, p_down, p_up = wrap.predict_state_from_features(feats, thresh=args.thresh)
                        # if not horizontal_force uncertain
                        if not is_horizontal:
                            state = STATE_UNCERTAIN

                        model_probs[name] = (p_down, p_up)

                        # debounce logic per model
                        st = stats[name]
                        if state == st["current_candidate"]:
                            st["candidate_count"] += 1
                        else:
                            st["current_candidate"] = state
                            st["candidate_count"] = 1

                        if st["candidate_count"] >= args.min_consistent and st["current_candidate"] != STATE_UNCERTAIN:
                            # count transition DOWN -> UP
                            if st["last_stable_state"] == STATE_DOWN and st["current_candidate"] == STATE_UP:
                                st["count"] += 1
                            st["last_stable_state"] = st["current_candidate"]

                        s_str = {STATE_UNCERTAIN:"? ", STATE_DOWN:"DOWN", STATE_UP:"UP"}.get(state, str(state))
                        per_model_texts.append(f"{name}: {s_str} pU={p_up:.2f} pD={p_down:.2f} c={st['count']}")

                    # draw landmarks
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # record (optional per-frame)
                    records.append({
                        "frame_idx": frame_idx,
                        "timestamp": time.time(),
                        "left": float(feats[0]),
                        "right": float(feats[1]),
                        "dist": float(feats[2]),
                        **{f"{name}_p_down": model_probs[name][0] for name in wrappers.keys()},
                        **{f"{name}_p_up": model_probs[name][1] for name in wrappers.keys()},
                        **{f"{name}_count": int(stats[name]["count"]) for name in wrappers.keys()}
                    })

            else:
                per_model_texts.append("No landmarks")

            # overlay per-model text
            y0 = 30
            for txt in per_model_texts:
                cv2.putText(frame, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                y0 += 26

            # overlay global info
            info = f"Thresh={args.thresh} MinCons={args.min_consistent}"
            cv2.putText(frame, info, (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            # FPS
            c_time = time.time()
            fps = 1.0 / (c_time - p_time + 1e-6)
            p_time = c_time
            cv2.putText(frame, f"FPS:{int(fps)}", (frame.shape[1]-120, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)

            if write_mode:
                cv2.putText(frame, "LOGGING ON", (frame.shape[1]-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Pushup Multi-Model Runner", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                write_mode = not write_mode
                print("Logging:", write_mode)
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 保存记录（如果启用）
    if args.log and len(records) > 0 and write_mode:
        ts = int(time.time())
        out_csv = f"realtime_results_{ts}.csv"
        # ensure consistent column ordering
        keys = list(records[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        print("Wrote realtime log to", out_csv)
    else:
        if args.log:
            print("Logging was off or no records; no CSV written.")

    # print final counts per model
    print("Final counts:")
    for name in wrappers.keys():
        print(f"  {name}: {stats[name]['count']}")

if __name__ == "__main__":
    main()
