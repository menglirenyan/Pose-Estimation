#!/usr/bin/env python3
"""
main_realtime.py

Realtime runner (camera):
- MediaPipe Pose -> 3 features per frame
  1) left_elbow_angle
  2) right_elbow_angle
  3) shoulder_hip_dist (normalized vertical distance)

- Multi-model inference:
  - sklearn baselines: logistic / mlp  (2-class + threshold-gated UNCERTAIN)
  - torch seq models:  lstm / lstm_attn / tcn (3-class: DOWN/UP/UNCERTAIN)

- Per-model counting:
  - Debounce via min_consistent
  - Count increments on stable DOWN->UP transition by default

Usage examples:
  python src/runners/main_realtime.py --models logistic mlp --cam 1
  python src/runners/main_realtime.py --models lstm --cam 1 --seq_len 16 --device cpu
  python src/runners/main_realtime.py --models lstm_attn tcn --cam 1 --seq_len 16 --device cpu
"""

import os
import time
import argparse
import csv
from collections import deque

import numpy as np
import cv2
import mediapipe as mp
import joblib

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
                   help="Model names: logistic mlp lstm lstm_attn/attn tcn")
    p.add_argument("--models_dir", default=os.path.join(BASE_DIR, "src", "models"),
                   help="Directory holding models/scalers")
    p.add_argument("--cam", type=int, default=1, help="Camera index")
    p.add_argument("--thresh", type=float, default=0.75, help="Sklearn threshold for confident prediction")
    p.add_argument("--min_consistent", type=int, default=3, help="Debounce: consecutive frames required")
    p.add_argument("--seq_len", type=int, default=16, help="Sequence length expected by seq models")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--log_csv", action="store_true", help="Press SPACE to start/stop logging; save CSV on exit if ON.")
    p.add_argument("--count_transition", default="down2up", choices=["down2up", "up2down"],
                   help="Which stable transition increments the rep counter")
    return p.parse_args()


# =========================
# Wrappers
# =========================
class SklearnWrapper:
    """2-class proba + threshold gating to UNCERTAIN."""
    def __init__(self, model_path, scaler_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"sklearn model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"sklearn scaler not found: {scaler_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        cls = list(getattr(self.model, "classes_", []))
        if 0 not in cls or 1 not in cls:
            raise ValueError(f"Sklearn model classes_ must include 0 and 1, got {cls}")
        self.idx_down = int(cls.index(0))
        self.idx_up = int(cls.index(1))

    def predict_state(self, feat, thresh=0.75):
        x = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        x_s = self.scaler.transform(x)
        proba = self.model.predict_proba(x_s)[0]
        p_down = float(proba[self.idx_down])
        p_up = float(proba[self.idx_up])

        if p_up >= thresh and p_up > p_down:
            return STATE_UP, p_down, p_up
        if p_down >= thresh and p_down > p_up:
            return STATE_DOWN, p_down, p_up
        return STATE_UNCERTAIN, p_down, p_up


class PushupLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1,
                 bidirectional=False, dropout=0.1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)      # (B,T,H)
        last = out[:, -1, :]       # (B,H)
        return self.fc(last)       # (B,C)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):
        # h: (B,T,H)
        u = torch.tanh(self.proj(h))          # (B,T,A)
        scores = self.v(u).squeeze(-1)        # (B,T)
        alpha = torch.softmax(scores, dim=1)  # (B,T)
        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B,H)
        return context, alpha


class PushupAttnLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1,
                 bidirectional=False, dropout=0.1, num_classes=3, attn_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = TemporalAttention(self.hidden_dim, attn_dim=attn_dim)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)              # (B,T,H)
        context, _ = self.attn(out)        # (B,H)
        return self.fc(context)            # (B,C)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        y = self.drop(self.relu(self.conv1(x)))
        y = self.drop(self.relu(self.conv2(y)))
        res = x if self.downsample is None else self.downsample(x)
        return y + res


class PushupTCN(nn.Module):
    def __init__(self, input_size=3, channels=(32, 64, 64), kernel_size=3, dropout=0.1, num_classes=3):
        super().__init__()
        layers = []
        in_ch = input_size
        dilation = 1
        for out_ch in channels:
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            in_ch = out_ch
            dilation *= 2
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: (B,T,F) -> (B,F,T)
        x = x.permute(0, 2, 1)
        h = self.net(x)          # (B,C,T)
        h = h.mean(dim=2)        # (B,C)
        return self.fc(h)        # (B,C)


class TorchSeqWrapper3:
    """Generic 3-class seq wrapper with a fixed-length feature buffer."""
    def __init__(self, model, model_path, scaler_path, seq_len=16, device="cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Torch model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        self.seq_len = int(seq_len)
        self.scaler = joblib.load(scaler_path)

        if str(device).startswith("cuda") and (not torch.cuda.is_available()):
            print("[WARN] cuda requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        state = torch.load(model_path, map_location=self.device)
        # support raw state_dict or {"state_dict":...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

        self.buffer = deque(maxlen=self.seq_len)

    def _append_and_get_seq(self, feat):
        x = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        x_s = self.scaler.transform(x).reshape(-1).astype(np.float32)
        self.buffer.append(x_s)
        if len(self.buffer) < self.seq_len:
            return None
        return np.stack(list(self.buffer), axis=0)  # (T,F)

    def predict(self, feat):
        seq = self._append_and_get_seq(feat)
        if seq is None:
            return STATE_UNCERTAIN, 0.0, 0.0, 0.0

        x = torch.from_numpy(seq).unsqueeze(0).float().to(self.device)  # (1,T,F)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (3,)

        p_down, p_up, p_unc = float(probs[0]), float(probs[1]), float(probs[2])
        cls = int(np.argmax(probs))
        state = STATE_DOWN if cls == 0 else (STATE_UP if cls == 1 else STATE_UNCERTAIN)
        return state, p_down, p_up, p_unc


# =========================
# Feature extraction
# =========================
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def valid_pose(lm, vis_thresh=0.6, min_ok=8):
    # same set as your data collection
    need = [11, 12, 13, 14, 15, 16, 23, 24]
    ok = sum(1 for i in need if lm[i].visibility >= vis_thresh)
    return ok >= min_ok


def extract_features_from_landmarks(lm, frame_shape):
    h, w = frame_shape[:2]

    def get_point(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    ls, le, lw = get_point(11), get_point(13), get_point(15)
    rs, re, rw = get_point(12), get_point(14), get_point(16)
    lh, rh = get_point(23), get_point(24)

    left_elbow_angle = calculate_angle(ls, le, lw)
    right_elbow_angle = calculate_angle(rs, re, rw)

    shoulder_y = (ls[1] + rs[1]) / 2.0
    hip_y = (lh[1] + rh[1]) / 2.0
    shoulder_hip_dist = (hip_y - shoulder_y) / h

    return np.array([left_elbow_angle, right_elbow_angle, shoulder_hip_dist], dtype=np.float32)


# =========================
# Counting (debounce FSM)
# =========================
def update_counter(stat, new_state, min_consistent=3, transition="down2up"):
    """
    stat: dict with keys last_stable_state, current_candidate, candidate_count, count
    returns updated stat and a bool flag "count_incremented"
    """
    inc = False

    if new_state == stat["current_candidate"]:
        stat["candidate_count"] += 1
    else:
        stat["current_candidate"] = new_state
        stat["candidate_count"] = 1

    if stat["candidate_count"] >= min_consistent and new_state != stat["last_stable_state"]:
        prev = stat["last_stable_state"]
        stat["last_stable_state"] = new_state

        if transition == "down2up" and prev == STATE_DOWN and new_state == STATE_UP:
            stat["count"] += 1
            inc = True
        elif transition == "up2down" and prev == STATE_UP and new_state == STATE_DOWN:
            stat["count"] += 1
            inc = True

    return stat, inc


def main():
    args = parse_args()

    requested = [m.lower() for m in args.models]
    print("Requested models:", requested)
    print("models_dir:", args.models_dir)

    wrappers = {}
    stats = {}
    load_errors = {}

    for name in requested:
        try:
            if name in ("logistic", "mlp"):
                model_path = os.path.join(args.models_dir, f"{name}.joblib")
                scaler_path = os.path.join(args.models_dir, f"scaler_{name}.joblib")
                wrappers[name] = SklearnWrapper(model_path, scaler_path)

            elif name == "lstm":
                model_path = os.path.join(args.models_dir, "lstm.pt")
                scaler_path = os.path.join(args.models_dir, "scaler_lstm.joblib")
                wrappers[name] = TorchSeqWrapper3(
                    PushupLSTM(num_classes=3),
                    model_path, scaler_path,
                    seq_len=args.seq_len, device=args.device
                )

            elif name in ("lstm_attn", "attn"):
                model_path = os.path.join(args.models_dir, "lstm_attn.pt")
                scaler_path = os.path.join(args.models_dir, "scaler_lstm_attn.joblib")
                wrappers[name] = TorchSeqWrapper3(
                    PushupAttnLSTM(num_classes=3),
                    model_path, scaler_path,
                    seq_len=args.seq_len, device=args.device
                )

            elif name == "tcn":
                model_path = os.path.join(args.models_dir, "tcn.pt")
                scaler_path = os.path.join(args.models_dir, "scaler_tcn.joblib")
                wrappers[name] = TorchSeqWrapper3(
                    PushupTCN(num_classes=3),
                    model_path, scaler_path,
                    seq_len=args.seq_len, device=args.device
                )

            else:
                raise ValueError(f"Unknown model name: {name}")

            stats[name] = {
                "last_stable_state": STATE_UNCERTAIN,
                "current_candidate": STATE_UNCERTAIN,
                "candidate_count": 0,
                "count": 0
            }
            print(f"[OK] Loaded: {name}")

        except Exception as e:
            load_errors[name] = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {name} -> {load_errors[name]}")

    if not wrappers:
        raise SystemExit("No models loaded. Fix paths/files and retry.")

    if load_errors:
        print("Not loaded:", load_errors)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.cam)
    print("Press SPACE to toggle logging; Press Q to quit.")
    write_mode = False
    records = []
    frame_idx = 0
    p_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_idx += 1

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            per_model_lines = []
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if valid_pose(lm):
                    feats = extract_features_from_landmarks(lm, frame.shape)

                    for name in requested:
                        if name not in wrappers:
                            per_model_lines.append(f"{name}: NOT_LOADED")
                            continue

                        wrap = wrappers[name]
                        if isinstance(wrap, SklearnWrapper):
                            st, p_down, p_up = wrap.predict_state(feats, thresh=args.thresh)
                            # counting on st
                            stats[name], _ = update_counter(stats[name], st, args.min_consistent, args.count_transition)
                            per_model_lines.append(f"{name}: {st}  pD={p_down:.2f} pU={p_up:.2f}  cnt={stats[name]['count']}")
                        else:
                            st, p_down, p_up, p_unc = wrap.predict(feats)
                            stats[name], _ = update_counter(stats[name], st, args.min_consistent, args.count_transition)
                            per_model_lines.append(f"{name}: {st}  pD={p_down:.2f} pU={p_up:.2f} pX={p_unc:.2f}  cnt={stats[name]['count']}")

                        if write_mode:
                            records.append({
                                "frame": frame_idx,
                                "model": name,
                                "state": st,
                                "feat0": float(feats[0]),
                                "feat1": float(feats[1]),
                                "feat2": float(feats[2]),
                            })
                else:
                    for name in requested:
                        per_model_lines.append(f"{name}: NO_POSE")
            else:
                for name in requested:
                    per_model_lines.append(f"{name}: NO_LANDMARKS")

            # draw pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # FPS
            c_time = time.time()
            fps = 1.0 / max(1e-6, (c_time - p_time))
            p_time = c_time

            y0 = 30
            cv2.putText(frame, f"FPS: {fps:.1f}  logging:{write_mode}", (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y = y0 + 30
            for line in per_model_lines[:10]:
                cv2.putText(frame, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 24

            cv2.imshow("Push-up Realtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                write_mode = not write_mode
                print("[LOG] write_mode:", write_mode)

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if args.log_csv and records:
            out_dir = os.path.join(BASE_DIR, "realtime_logs")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"realtime_{int(time.time())}.csv")

            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                w.writeheader()
                w.writerows(records)

            print("[OK] Saved realtime log:", out_path)


if __name__ == "__main__":
    main()
