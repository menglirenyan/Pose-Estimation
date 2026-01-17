#!/usr/bin/env python3
"""
main_realtime.py

Realtime runner:
- load sklearn models: logistic/mlp
- load pytorch LSTM model: lstm (3-class: DOWN/UP/UNCERTAIN)
- run multi-model inference on each frame and compare outputs

Usage:
  # sklearn baselines
  python src/runners/main_realtime.py --models logistic mlp --cam 1

  # plain LSTM (3-class: DOWN/UP/UNCERTAIN)
  python src/runners/main_realtime.py --models lstm --cam 1 --lstm_seq 16 --device cpu

  # Attention-LSTM (3-class, trained by train_lstm_attn.py)
  python src/runners/main_realtime.py --models lstm_attn --cam 1 --lstm_seq 16 --device cpu
"""
import os
import time
import argparse
import joblib
import csv
from collections import deque

import numpy as np
import cv2
import mediapipe as mp

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STATE_DOWN = 0
STATE_UP = 1
STATE_UNCERTAIN = -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["logistic", "mlp"],
                   help="List of model names to load (e.g. logistic mlp lstm lstm_attn/attn)")
    p.add_argument("--models_dir", default=os.path.join(BASE_DIR, "src", "models"),
                   help="Directory where models and scalers are saved")
    p.add_argument("--cam", type=int, default=1, help="Camera index")
    p.add_argument("--thresh", type=float, default=0.75, help="Threshold for sklearn confident prediction")
    p.add_argument("--min_consistent", type=int, default=3, help="Debounce: consecutive frames required")
    p.add_argument("--log", action="store_true",
                   help="If set, write realtime records CSV on exit (ONLY if logging is ON when you quit)")
    p.add_argument("--lstm_seq", type=int, default=16, help="Sequence length expected by LSTM")
    p.add_argument("--device", type=str, default="cpu", help="Device for LSTM (cpu or cuda)")
    return p.parse_args()


# -------------------- Wrappers --------------------
class SklearnWrapper:
    def __init__(self, model_path, scaler_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"sklearn model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"sklearn scaler not found: {scaler_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        self.classes_ = getattr(self.model, "classes_", None)
        if self.classes_ is None:
            raise ValueError("Loaded sklearn model has no classes_ attribute.")
        cls = list(self.classes_)
        if 0 not in cls or 1 not in cls:
            raise ValueError(f"Model classes_ must contain both 0 and 1, got {cls}")
        self.idx0 = int(cls.index(0))
        self.idx1 = int(cls.index(1))

    def predict_proba_from_features(self, feat):
        x = np.array(feat).reshape(1, -1)
        x_s = self.scaler.transform(x)
        proba_all = self.model.predict_proba(x_s)[0]
        p0 = float(proba_all[self.idx0])
        p1 = float(proba_all[self.idx1])
        return np.array([p0, p1], dtype=np.float32)

    def predict_state_from_features(self, feat, thresh=0.75):
        p = self.predict_proba_from_features(feat)
        p_down, p_up = float(p[0]), float(p[1])
        if p_up >= thresh and p_up > p_down:
            return STATE_UP, p_down, p_up
        elif p_down >= thresh and p_down > p_up:
            return STATE_DOWN, p_down, p_up
        else:
            return STATE_UNCERTAIN, p_down, p_up


class PushupLSTM(nn.Module):
    # must match train_lstm.py (num_classes=3)
    def __init__(self, input_size=3, hidden_size=64, num_layers=1,
                 bidirectional=False, dropout=0.1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        if self.lstm.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        return self.fc(last_hidden)




class TemporalAttention(nn.Module):
    """Additive attention over time steps.

    Given encoder outputs H: (B,T,H), produce context: (B,H).
    """
    def __init__(self, hidden_dim, attn_dim=64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H):
        # H: (B,T,H)
        scores = self.v(torch.tanh(self.W(H))).squeeze(-1)  # (B,T)
        alpha = torch.softmax(scores, dim=1)                # (B,T)
        context = (H * alpha.unsqueeze(-1)).sum(dim=1)      # (B,H)
        return context, alpha


class PushupAttnLSTM(nn.Module):
    """LSTM + Temporal Attention for 3-class classification."""
    def __init__(self, input_size=3, hidden_size=64, num_layers=1,
                 bidirectional=False, dropout=0.1, num_classes=3,
                 attn_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = TemporalAttention(self.hidden_dim, attn_dim=attn_dim)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)               # (B,T,H)
        context, alpha = self.attn(out)    # (B,H), (B,T)
        logits = self.fc(context)          # (B,C)
        return logits
class LSTMWrapper3:
    """
    3-class LSTM wrapper: probs = [DOWN, UP, UNCERTAIN]
    """
    def __init__(self, model_path, scaler_path, seq_len=16, device="cpu", arch="lstm"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"LSTM scaler not found: {scaler_path}")

        self.seq_len = seq_len
        self.scaler = joblib.load(scaler_path)

        # device safety: auto fallback
        if str(device).startswith("cuda") and (not torch.cuda.is_available()):
            print("[WARN] --device cuda requested but torch.cuda.is_available() is False. Falling back to CPU for LSTM.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.arch = str(arch).lower()

        if self.arch in ("attn", "lstm_attn", "attention"):
            self.model = PushupAttnLSTM(num_classes=3)
        else:
            self.model = PushupLSTM(num_classes=3)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.buffer = deque(maxlen=self.seq_len)

    def _append_and_get_seq(self, feat):
        x = np.array(feat).reshape(1, -1)
        x_s = self.scaler.transform(x).reshape(-1).astype(np.float32)
        self.buffer.append(x_s)
        if len(self.buffer) < self.seq_len:
            return None
        return np.stack(list(self.buffer), axis=0)  # (T, F)

    def predict_proba_from_features(self, feat):
        seq = self._append_and_get_seq(feat)
        if seq is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        x = torch.from_numpy(seq).unsqueeze(0).float().to(self.device)  # (1, T, F)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (3,)
        return probs.astype(np.float32)

    def predict_state_from_features(self, feat):
        probs = self.predict_proba_from_features(feat)
        if float(probs.sum()) == 0.0:
            return STATE_UNCERTAIN, 0.0, 0.0, 0.0

        p_down = float(probs[0])
        p_up = float(probs[1])
        p_unc = float(probs[2])

        cls = int(np.argmax(probs))  # 0/1/2
        if cls == 0:
            state = STATE_DOWN
        elif cls == 1:
            state = STATE_UP
        else:
            state = STATE_UNCERTAIN
        return state, p_down, p_up, p_unc


# -------------------- Feature extraction --------------------
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

    left_shoulder = get_point(11); left_elbow = get_point(13); left_wrist = get_point(15)
    right_shoulder = get_point(12); right_elbow = get_point(14); right_wrist = get_point(16)
    left_hip = get_point(23); right_hip = get_point(24)

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0
    shoulder_hip_dist = (hip_y - shoulder_y) / h

    return np.array([left_elbow_angle, right_elbow_angle, shoulder_hip_dist], dtype=np.float32)


def valid_pose(lm, vis_thresh=0.6):
    need = [11, 12, 13, 14, 15, 16, 23, 24]
    for idx in need:
        if lm[idx].visibility < vis_thresh:
            return False
    return True


# -------------------- Main --------------------
def main():
    args = parse_args()

    print("Requested models:", args.models)
    print("models_dir:", args.models_dir)

    wrappers = {}  # loaded model name -> wrapper instance
    stats = {}     # loaded model name -> counting state
    load_errors = {}  # requested model name -> error string

    for name in args.models:
        try:
            lname = name.lower()
            if lname in ("lstm_attn", "attn"):
                model_path = os.path.join(args.models_dir, "lstm_attn.pt")
                scaler_path = os.path.join(args.models_dir, "scaler_lstm_attn.joblib")
                wrap = LSTMWrapper3(model_path, scaler_path, seq_len=args.lstm_seq, device=args.device, arch="attn")
            elif lname == "lstm":
                model_path = os.path.join(args.models_dir, "lstm.pt")
                scaler_path = os.path.join(args.models_dir, "scaler_lstm.joblib")
                wrap = LSTMWrapper3(model_path, scaler_path, seq_len=args.lstm_seq, device=args.device, arch="lstm")
            else:
                model_path = os.path.join(args.models_dir, f"{name}.joblib")
                scaler_path = os.path.join(args.models_dir, f"scaler_{name}.joblib")
                wrap = SklearnWrapper(model_path, scaler_path)

            wrappers[name] = wrap
            stats[name] = {
                "last_stable_state": STATE_UNCERTAIN,
                "current_candidate": STATE_UNCERTAIN,
                "candidate_count": 0,
                "count": 0
            }
            print(f"[OK] Loaded model '{name}'")

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            load_errors[name] = err
            print(f"[FAIL] Model '{name}' not loaded -> {err}")

    print("Loaded models:", list(wrappers.keys()))
    if load_errors:
        print("Not loaded:", load_errors)

    if len(wrappers) == 0:
        raise SystemExit("No models loaded. Fix paths/files and retry.")

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

            per_model_texts = []
            model_probs = {}  # name -> tuple

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                if not valid_pose(lm):
                    # 按“用户请求顺序”显示，没加载的也显示 NOT_LOADED
                    for name in args.models:
                        if name in wrappers:
                            per_model_texts.append(f"{name}: NO_POSE")
                        else:
                            per_model_texts.append(f"{name}: NOT_LOADED")
                else:
                    feats = extract_features_from_landmarks(lm, frame.shape)
                    is_horizontal = feats[2] < 0.35

                    # 推理：按 args.models 顺序走，避免“少一个模型就不显示”
                    for name in args.models:
                        if name not in wrappers:
                            per_model_texts.append(f"{name}: NOT_LOADED")
                            continue

                        wrap = wrappers[name]

                        if name.lower() in ("lstm", "lstm_attn", "attn"):
                            state, p_down, p_up, p_unc = wrap.predict_state_from_features(feats)
                            model_probs[name] = (p_down, p_up, p_unc)
                        else:
                            state, p_down, p_up = wrap.predict_state_from_features(feats, thresh=args.thresh)
                            model_probs[name] = (p_down, p_up)

                        if not is_horizontal:
                            state = STATE_UNCERTAIN

                        # debounce + count
                        st = stats[name]
                        if state == st["current_candidate"]:
                            st["candidate_count"] += 1
                        else:
                            st["current_candidate"] = state
                            st["candidate_count"] = 1

                        if st["candidate_count"] >= args.min_consistent and st["current_candidate"] != STATE_UNCERTAIN:
                            if st["last_stable_state"] == STATE_DOWN and st["current_candidate"] == STATE_UP:
                                st["count"] += 1
                            st["last_stable_state"] = st["current_candidate"]

                        s_str = {STATE_UNCERTAIN: "? ", STATE_DOWN: "DOWN", STATE_UP: "UP"}.get(state, str(state))

                        if name.lower() in ("lstm", "lstm_attn", "attn"):
                            pD, pU, pQ = model_probs[name]
                            per_model_texts.append(f"{name}: {s_str} pU={pU:.2f} pD={pD:.2f} p?={pQ:.2f} c={st['count']}")
                        else:
                            per_model_texts.append(f"{name}: {s_str} pU={p_up:.2f} pD={p_down:.2f} c={st['count']}")

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # record
                    rec = {
                        "frame_idx": frame_idx,
                        "timestamp": time.time(),
                        "left": float(feats[0]),
                        "right": float(feats[1]),
                        "dist": float(feats[2])
                    }
                    for name in args.models:
                        if name not in wrappers:
                            continue
                        if name.lower() in ("lstm", "lstm_attn", "attn"):
                            pD, pU, pQ = model_probs.get(name, (0.0, 0.0, 0.0))
                            rec[f"{name}_p_down"] = float(pD)
                            rec[f"{name}_p_up"] = float(pU)
                            rec[f"{name}_p_uncertain"] = float(pQ)
                        else:
                            pD, pU = model_probs.get(name, (0.0, 0.0))
                            rec[f"{name}_p_down"] = float(pD)
                            rec[f"{name}_p_up"] = float(pU)
                        rec[f"{name}_count"] = int(stats[name]["count"])
                    records.append(rec)

            else:
                for name in args.models:
                    if name in wrappers:
                        per_model_texts.append(f"{name}: NO_LANDMARKS")
                    else:
                        per_model_texts.append(f"{name}: NOT_LOADED")

            # overlay
            y0 = 30
            for txt in per_model_texts:
                cv2.putText(frame, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y0 += 26

            info = f"Thresh={args.thresh} MinCons={args.min_consistent}"
            cv2.putText(frame, info, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            c_time = time.time()
            fps = 1.0 / (c_time - p_time + 1e-6)
            p_time = c_time
            cv2.putText(frame, f"FPS:{int(fps)}", (frame.shape[1] - 120, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

            if write_mode:
                cv2.putText(frame, "LOGGING ON", (frame.shape[1] - 220, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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

    # write csv
    if args.log and len(records) > 0 and write_mode:
        ts = int(time.time())
        out_csv = f"realtime_results_{ts}.csv"
        keys = list(records[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        print("Wrote realtime log to", out_csv)
    else:
        if args.log:
            print("Logging was off or no records; no CSV written.")

    # final counts: 按 args.models 顺序输出（你想要“结束后也显示 mlp”，这里保证）
    print("Final counts (requested order):")
    for name in args.models:
        if name in stats:
            print(f"  {name}: {stats[name]['count']}")
        else:
            print(f"  {name}: NOT_LOADED")


if __name__ == "__main__":
    main()
