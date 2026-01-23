#!/usr/bin/env python3
"""
eval_video_offline_v2.py

Offline evaluation from a recorded video:
  - MediaPipe Pose -> 3 features per frame
  - Weak GT label_rule() -> y_true per frame (0/1/-1)
  - Evaluate selected models:
      * sklearn (logistic/mlp) in frame mode
      * torch seq (lstm/lstm_attn/tcn) in window mode (sliding buffer)

Outputs per model:
  1) Classification metrics:
     - confusion matrix
     - classification report
  2) Counting metrics (NEW):
     - count_true / count_pred / abs_error
     - counting uses a debounce FSM and a configurable transition (down2up or up2down)

Important note (thesis wording):
  GT here is pseudo-label / weak supervision (rule-based), not manual annotation.

Examples
  python src/eval/eval_video_offline_v2.py --video demo.mp4
  python src/eval/eval_video_offline_v2.py --video demo.mp4 --models lstm_attn tcn
  python src/eval/eval_video_offline_v2.py --video demo.mp4 --include_uncertain
"""

import os
import argparse
from collections import deque

import cv2
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import mediapipe as mp


DOWN = 0
UP = 1
UNCERTAIN = -1


# =========================
# Feature extraction
# =========================
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def valid_pose(lm, vis_thresh=0.6, min_ok=8):
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
# Weak GT rule (pseudo label)
# =========================
def label_rule(lm, feats, horiz_thresh=0.35, zdiff_thresh=0.2):
    left_elbow_angle, right_elbow_angle, shoulder_hip_dist = feats.tolist()
    is_horizontal = shoulder_hip_dist < horiz_thresh

    z_diff = abs(lm[11].z - lm[12].z)
    if z_diff > zdiff_thresh:
        if lm[11].z > lm[12].z:
            view_type = 1
            main_elbow_angle = left_elbow_angle
        else:
            view_type = 2
            main_elbow_angle = right_elbow_angle
    else:
        view_type = 0
        main_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

    label = UNCERTAIN
    if is_horizontal:
        if view_type == 0:
            if left_elbow_angle > 165 and right_elbow_angle > 165:
                label = UP
            elif left_elbow_angle < 50 and right_elbow_angle < 50:
                label = DOWN
        else:
            if main_elbow_angle > 150:
                label = UP
            elif main_elbow_angle < 75:
                label = DOWN
    return label


# =========================
# Counting (debounce FSM)
# =========================
def count_reps(states, min_consistent=4, transition="down2up", ignore_uncertain=True):
    """
    states: iterable of {0,1,-1} (or torch-class {0,1,2} mapped beforehand)
    Counting uses stable-state debounce.

    ignore_uncertain=True:
      - UNCERTAIN frames do not contribute to candidate stability.
    """
    last_stable = UNCERTAIN
    cand = UNCERTAIN
    cand_cnt = 0
    reps = 0

    for s in states:
        if ignore_uncertain and s == UNCERTAIN:
            continue

        if s == cand:
            cand_cnt += 1
        else:
            cand = s
            cand_cnt = 1

        if cand_cnt >= min_consistent and s != last_stable:
            prev = last_stable
            last_stable = s

            if transition == "down2up" and prev == DOWN and s == UP:
                reps += 1
            elif transition == "up2down" and prev == UP and s == DOWN:
                reps += 1

    return reps


# =========================
# Model loaders
# =========================
def load_sklearn(models_dir, model_name, thresh=0.75):
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    scaler_path = os.path.join(models_dir, f"scaler_{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing sklearn model: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    classes_ = list(getattr(model, "classes_", [0, 1]))
    if 0 not in classes_ or 1 not in classes_:
        raise ValueError(f"Sklearn model classes_ must include 0 and 1, got {classes_}")
    idx0 = classes_.index(0)
    idx1 = classes_.index(1)

    def predict_frame(feat3):
        x = scaler.transform(np.asarray(feat3, dtype=np.float32).reshape(1, -1))
        proba = model.predict_proba(x)[0]
        p_down = float(proba[idx0])
        p_up = float(proba[idx1])
        if p_up >= thresh and p_up > p_down:
            return UP
        if p_down >= thresh and p_down > p_up:
            return DOWN
        return UNCERTAIN

    return predict_frame


def load_torch_seq(models_dir, model_name, device="cpu", seq_len=16):
    """
    Loads a 3-class torch seq model and returns a per-frame predictor using an internal buffer.

    Expected filenames:
      - lstm:      lstm.pt + scaler_lstm.joblib
      - lstm_attn: lstm_attn.pt + scaler_lstm_attn.joblib
      - tcn:       tcn.pt + scaler_tcn.joblib
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
            hdim = hidden_size * (2 if bidirectional else 1)
            self.fc = nn.Linear(hdim, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class TemporalAttention(nn.Module):
        def __init__(self, hidden_dim: int, attn_dim: int = 64):
            super().__init__()
            self.proj = nn.Linear(hidden_dim, attn_dim)
            self.v = nn.Linear(attn_dim, 1, bias=False)

        def forward(self, h):
            u = torch.tanh(self.proj(h))
            scores = self.v(u).squeeze(-1)
            alpha = torch.softmax(scores, dim=1)
            context = torch.sum(h * alpha.unsqueeze(-1), dim=1)
            return context

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
            hdim = hidden_size * (2 if bidirectional else 1)
            self.attn = TemporalAttention(hdim, attn_dim=attn_dim)
            self.fc = nn.Linear(hdim, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            ctx = self.attn(out)
            return self.fc(ctx)

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
            x = x.permute(0, 2, 1)
            h = self.net(x)
            h = h.mean(dim=2)
            return self.fc(h)

    if str(device).startswith("cuda") and (not torch.cuda.is_available()):
        print("[WARN] cuda requested but not available. Falling back to cpu.")
        device = "cpu"
    dev = torch.device(device)

    if model_name == "lstm":
        ckpt_path = os.path.join(models_dir, "lstm.pt")
        scaler_path = os.path.join(models_dir, "scaler_lstm.joblib")
        net = PushupLSTM(num_classes=3)
    elif model_name in ("lstm_attn", "attn"):
        ckpt_path = os.path.join(models_dir, "lstm_attn.pt")
        scaler_path = os.path.join(models_dir, "scaler_lstm_attn.joblib")
        net = PushupAttnLSTM(num_classes=3)
    elif model_name == "tcn":
        ckpt_path = os.path.join(models_dir, "tcn.pt")
        scaler_path = os.path.join(models_dir, "scaler_tcn.joblib")
        net = PushupTCN(num_classes=3)
    else:
        raise ValueError(f"Unknown torch model: {model_name}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    scaler = joblib.load(scaler_path)

    state = torch.load(ckpt_path, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    net.load_state_dict(state, strict=True)
    net.to(dev).eval()

    buf = deque(maxlen=int(seq_len))

    def predict_frame(feat3):
        x = np.asarray(feat3, dtype=np.float32).reshape(1, -1)
        x_s = scaler.transform(x).reshape(-1).astype(np.float32)
        buf.append(x_s)
        if len(buf) < seq_len:
            return UNCERTAIN
        seq = np.stack(list(buf), axis=0)                  # (T,3)
        xt = torch.from_numpy(seq).unsqueeze(0).to(dev)    # (1,T,3)
        with torch.no_grad():
            logits = net(xt)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        cls = int(np.argmax(probs))
        if cls == 0:
            return DOWN
        if cls == 1:
            return UP
        return UNCERTAIN

    return predict_frame


# =========================
# Video feature extraction
# =========================
def extract_video(video_path, cfg):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=cfg["model_complexity"],
        min_detection_confidence=cfg["det_conf"],
        min_tracking_confidence=cfg["track_conf"],
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    feats_list = []
    gt_list = []

    diag = {"frames_read": 0, "no_landmarks": 0, "bad_vis": 0, "ok_frames": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        diag["frames_read"] += 1
        if cfg["max_frames"] > 0 and diag["frames_read"] > cfg["max_frames"]:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            diag["no_landmarks"] += 1
            continue

        lm = results.pose_landmarks.landmark
        if not valid_pose(lm, cfg["vis_thresh"], cfg["min_ok"]):
            diag["bad_vis"] += 1
            continue

        feats = extract_features_from_landmarks(lm, frame.shape)
        gt = label_rule(lm, feats)

        feats_list.append(feats)
        gt_list.append(gt)
        diag["ok_frames"] += 1

    cap.release()
    return feats_list, gt_list, diag


# =========================
# Main evaluation per model
# =========================
def eval_one_model(model_name, feats_list, gt_list, cfg):
    model_name = model_name.lower()

    if model_name in ("logistic", "mlp"):
        mode = "frame"
        predict = load_sklearn(cfg["models_dir"], model_name, thresh=cfg["thresh"])
    elif model_name in ("lstm", "lstm_attn", "attn", "tcn"):
        mode = "window"
        predict = load_torch_seq(cfg["models_dir"], model_name, device=cfg["device"], seq_len=cfg["seq_len"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    y_true = np.asarray(gt_list, dtype=int)
    y_pred = np.asarray([predict(f) for f in feats_list], dtype=int)

    # classification labels
    if cfg["include_uncertain"]:
        labels = [DOWN, UP, UNCERTAIN]
        names = ["DOWN(0)", "UP(1)", "UNCERTAIN(-1)"]
        yt, yp = y_true, y_pred
    else:
        # strict 2-class (filter uncertain in BOTH true/pred)
        mask = np.isin(y_true, [DOWN, UP]) & np.isin(y_pred, [DOWN, UP])
        yt, yp = y_true[mask], y_pred[mask]
        labels = [DOWN, UP]
        names = ["DOWN(0)", "UP(1)"]

    cm = confusion_matrix(yt, yp, labels=labels)
    report = classification_report(yt, yp, labels=labels, target_names=names, digits=4)

    # counting (always 2-class reps, ignore UNCERTAIN by default)
    count_true = count_reps(y_true, min_consistent=cfg["count_min_consistent"],
                           transition=cfg["count_transition"], ignore_uncertain=not cfg["count_include_uncertain"])
    count_pred = count_reps(y_pred, min_consistent=cfg["count_min_consistent"],
                           transition=cfg["count_transition"], ignore_uncertain=not cfg["count_include_uncertain"])
    abs_err = abs(int(count_true) - int(count_pred))

    return {
        "model": model_name,
        "mode": mode,
        "n_total": int(len(y_true)),
        "n_eval": int(len(yt)),
        "confusion_matrix": cm,
        "report": report,
        "count_true": int(count_true),
        "count_pred": int(count_pred),
        "count_abs_error": int(abs_err),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--models_dir", default="../models")
    p.add_argument("--models", nargs="+", default=["logistic", "mlp", "lstm", "lstm_attn", "tcn"])
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--thresh", type=float, default=0.75)
    p.add_argument("--device", default="cpu")
    p.add_argument("--include_uncertain", action="store_true")

    # mediapipe
    p.add_argument("--vis_thresh", type=float, default=0.25)
    p.add_argument("--min_ok", type=int, default=4)
    p.add_argument("--model_complexity", type=int, default=2)
    p.add_argument("--det_conf", type=float, default=0.3)
    p.add_argument("--track_conf", type=float, default=0.3)

    # limit
    p.add_argument("--max_frames", type=int, default=0)

    # counting config
    p.add_argument("--count_transition", default="down2up", choices=["down2up", "up2down"])
    p.add_argument("--count_min_consistent", type=int, default=4)
    p.add_argument("--count_include_uncertain", action="store_true",
                   help="If set, UNCERTAIN participates in debounce stability (not recommended).")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = vars(args)

    print("=" * 80)
    print("Config:")
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")
    print("=" * 80)

    feats_list, gt_list, diag = extract_video(args.video, cfg)
    print("Diagnostics:", diag)
    print(f"Collected frames: {len(feats_list)}")
    if len(feats_list) == 0:
        print("No usable frames. Try relaxing vis_thresh/min_ok or mediapipe conf settings.")
        return

    for m in args.models:
        r = eval_one_model(m, feats_list, gt_list, cfg)
        print("=" * 80)
        print(f"Model: {r['model']} | Mode: {r['mode']} | total={r['n_total']} eval={r['n_eval']}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(r["confusion_matrix"])
        print(r["report"])
        print(f"Count: true={r['count_true']} pred={r['count_pred']} abs_error={r['count_abs_error']} "
              f"(transition={cfg['count_transition']} min_consistent={cfg['count_min_consistent']} "
              f"ignore_uncertain={not cfg['count_include_uncertain']})")
        print("=" * 80)


if __name__ == "__main__":
    main()
