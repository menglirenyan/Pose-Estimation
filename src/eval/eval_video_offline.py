# eval_video_offline_batch.py
# 一次运行：抽特征 + weak GT，然后评估所有模型，并写出 json/txt 日志

#运行
#python eval_video_offline.py --video your_video.mp4

#临时只评估两个模型
#python eval_video_offline.py --video your_video.mp4 --models mlp lstm_attn

#把 -1 三分类也纳入评估
#python eval_video_offline.py --video your_video.mp4 --include_uncertain


import os
import json
import time
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
# Default config (edit here)
# =========================
DEFAULT_CFG = {
    "models_dir": "../models",
    "models": ["logistic", "mlp", "lstm", "lstm_attn"],  # default: evaluate all
    "seq_len": 16,
    "thresh": 0.75,          # sklearn confidence threshold
    "device": "cpu",

    # pose filter
    "vis_thresh": 0.25,
    "min_ok": 4,

    # mediapipe knobs (side-view friendly)
    "model_complexity": 2,
    "det_conf": 0.3,
    "track_conf": 0.3,

    # evaluation
    "include_uncertain": False,
    "max_frames": 0,
    "progress_every": 50,

    # output
    "out_dir": "eval_outputs",
}

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
    shoulder_hip_dist = (hip_y - shoulder_y) / h  # normalized

    return np.array([left_elbow_angle, right_elbow_angle, shoulder_hip_dist], dtype=np.float32)

def valid_pose_side_friendly(lm, vis_thresh=0.25, min_ok=4):
    need = [11, 12, 13, 14, 15, 16, 23, 24]
    ok = sum(lm[i].visibility >= vis_thresh for i in need)
    return ok >= min_ok

def label_rule(lm, feats, horiz_thresh=0.35, zdiff_thresh=0.2):
    left_elbow_angle, right_elbow_angle, shoulder_hip_dist = feats.tolist()
    is_horizontal = shoulder_hip_dist < horiz_thresh

    z_diff = abs(lm[11].z - lm[12].z)
    if z_diff > zdiff_thresh:
        # side view
        if lm[11].z > lm[12].z:
            view_type = 1
            main_elbow_angle = left_elbow_angle
        else:
            view_type = 2
            main_elbow_angle = right_elbow_angle
    else:
        # front view
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
# Model loaders
# =========================
def load_sklearn_predictor(models_dir, model_name, thresh=0.75):
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
        raise ValueError(f"Sklearn model classes_ must include 0 and 1, got: {classes_}")
    idx_down = classes_.index(0)
    idx_up = classes_.index(1)

    def predict_frame(feat_3):
        x = scaler.transform(np.asarray(feat_3, dtype=np.float32).reshape(1, -1))
        proba = model.predict_proba(x)[0]
        p_down = float(proba[idx_down])
        p_up = float(proba[idx_up])
        if p_up >= thresh and p_up > p_down:
            return UP
        if p_down >= thresh and p_down > p_up:
            return DOWN
        return UNCERTAIN

    return predict_frame

#读取模型参数
def collect_model_params(models_dir, model_name):
    """
    Return a JSON-serializable dict of model parameters / metadata.
    For sklearn: uses get_params().
    For torch: reads checkpoint dict with 'hparams' (requires training script save).
    """
    model_name = model_name.lower()

    if model_name in ("logistic", "mlp"):
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(models_dir, f"scaler_{model_name}.joblib")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        meta = {
            "model_path": os.path.abspath(model_path),
            "scaler_path": os.path.abspath(scaler_path),
            "sklearn_estimator": type(model).__name__,
            "sklearn_params": model.get_params(deep=True),
            "classes_": getattr(model, "classes_", None).tolist() if hasattr(model, "classes_") else None,
            "scaler_mean_": getattr(scaler, "mean_", None).tolist() if hasattr(scaler, "mean_") else None,
            "scaler_scale_": getattr(scaler, "scale_", None).tolist() if hasattr(scaler, "scale_") else None,
        }
        return meta

    if model_name in ("lstm", "lstm_attn", "attn"):
        if model_name == "lstm":
            ckpt_path = os.path.join(models_dir, "lstm.pt")
            scaler_path = os.path.join(models_dir, "scaler_lstm.joblib")
        else:
            ckpt_path = os.path.join(models_dir, "lstm_attn.pt")
            scaler_path = os.path.join(models_dir, "scaler_lstm_attn.joblib")

        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # ckpt may be state_dict only (old) or {"state_dict":..., "hparams":...} (new)
        if isinstance(ckpt, dict) and ("hparams" in ckpt or "state_dict" in ckpt):
            hparams = ckpt.get("hparams", {})
        else:
            hparams = {}  # cannot recover

        scaler = joblib.load(scaler_path)

        meta = {
            "ckpt_path": os.path.abspath(ckpt_path),
            "scaler_path": os.path.abspath(scaler_path),
            "hparams": hparams,
            "scaler_mean_": getattr(scaler, "mean_", None).tolist() if hasattr(scaler, "mean_") else None,
            "scaler_scale_": getattr(scaler, "scale_", None).tolist() if hasattr(scaler, "scale_") else None,
            "note": "If hparams is empty, your checkpoint was saved as state_dict only; update training script to save hparams.",
        }
        return meta

    return {"note": f"Unknown model_name={model_name}"}


def load_torch_predictor(models_dir, model_name, device="cpu"):
    import torch
    import torch.nn as nn

    class TemporalAttention(nn.Module):
        def __init__(self, hidden_dim: int, attn_dim: int = 64):
            super().__init__()
            self.proj = nn.Linear(hidden_dim, attn_dim)
            self.v = nn.Linear(attn_dim, 1, bias=False)

        def forward(self, out):  # (B,T,H)
            score = self.v(torch.tanh(self.proj(out))).squeeze(-1)  # (B,T)
            alpha = torch.softmax(score, dim=1)                     # (B,T)
            context = torch.sum(out * alpha.unsqueeze(-1), dim=1)   # (B,H)
            return context, alpha

    class PushupLSTM(nn.Module):
        def __init__(self, input_size=3, hidden_size=64, num_layers=1,
                     bidirectional=False, dropout=0.1, num_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0
            )
            hidden_dim = hidden_size * (2 if bidirectional else 1)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.fc(last)

    class PushupAttnLSTM(nn.Module):
        def __init__(self, input_size=3, hidden_size=64, num_layers=1,
                     bidirectional=False, dropout=0.1, num_classes=3, attn_dim=64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.hidden_dim = hidden_size * (2 if bidirectional else 1)
            self.attn = TemporalAttention(self.hidden_dim, attn_dim=attn_dim)
            self.fc = nn.Linear(self.hidden_dim, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            context, _ = self.attn(out)
            return self.fc(context)

    if model_name == "lstm":
        ckpt_path = os.path.join(models_dir, "lstm.pt")
        scaler_path = os.path.join(models_dir, "scaler_lstm.joblib")
    elif model_name in ("lstm_attn", "attn"):
        ckpt_path = os.path.join(models_dir, "lstm_attn.pt")
        scaler_path = os.path.join(models_dir, "scaler_lstm_attn.joblib")
    else:
        raise ValueError(f"Unknown torch model name: {model_name}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing torch checkpoint: {ckpt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    scaler = joblib.load(scaler_path)

    import torch
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        hparams = ckpt.get("hparams", {})
    else:
        state_dict = ckpt
        hparams = {}

    input_size = int(hparams.get("input_size", 3))
    hidden_size = int(hparams.get("hidden_size", 64))
    num_layers = int(hparams.get("num_layers", 1))
    bidirectional = bool(hparams.get("bidirectional", False))
    dropout = float(hparams.get("dropout", 0.1))
    num_classes = int(hparams.get("num_classes", 3))
    attn_dim = int(hparams.get("attn_dim", 64))

    if model_name == "lstm":
        net = PushupLSTM(input_size, hidden_size, num_layers, bidirectional, dropout, num_classes)
    else:
        net = PushupAttnLSTM(input_size, hidden_size, num_layers, bidirectional, dropout, num_classes, attn_dim)

    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()

    def predict_window(seq_Tx3):
        x = np.asarray(seq_Tx3, dtype=np.float32)
        x = scaler.transform(x)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = net(xt)
            pred = int(torch.argmax(logits, dim=1).item())
        return pred

    return predict_window

# =========================
# Pass 1: Extract features + weak GT once
# =========================
def extract_video_features(video_path, cfg):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

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
    frame_count = 0

    n_no_landmarks = 0
    n_bad_vis = 0
    n_ok = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if cfg["max_frames"] > 0 and frame_count > cfg["max_frames"]:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            n_no_landmarks += 1
            continue

        lm = results.pose_landmarks.landmark
        if not valid_pose_side_friendly(lm, vis_thresh=cfg["vis_thresh"], min_ok=cfg["min_ok"]):
            n_bad_vis += 1
            continue

        n_ok += 1
        feats = extract_features_from_landmarks(lm, frame.shape)
        gt = label_rule(lm, feats)

        feats_list.append(feats)
        gt_list.append(int(gt))

        if cfg["progress_every"] > 0 and (frame_count % cfg["progress_every"] == 0):
            print(f"[INFO] frames={frame_count} ok={n_ok} no_lm={n_no_landmarks} bad_vis={n_bad_vis} kept={len(feats_list)}",
                  flush=True)

    cap.release()

    diag = {
        "frames_read": frame_count,
        "no_landmarks": n_no_landmarks,
        "bad_vis": n_bad_vis,
        "ok_frames": n_ok,
        "kept_frames": len(feats_list),
    }
    return feats_list, gt_list, diag

# =========================
# Pass 2: Evaluate one model
# =========================
def evaluate_one_model(model_name, feats_list, gt_list, cfg):
    model_name = model_name.lower()

    # pick predictor + mode automatically
    if model_name in ("logistic", "mlp"):
        mode = "frame"
        pred_fn_frame = load_sklearn_predictor(cfg["models_dir"], model_name, thresh=cfg["thresh"])
        y_true = np.array(gt_list, dtype=int)
        y_pred = np.array([pred_fn_frame(f) for f in feats_list], dtype=int)

    elif model_name in ("lstm", "lstm_attn", "attn"):
        mode = "window"
        pred_fn_window = load_torch_predictor(cfg["models_dir"], model_name, device=cfg["device"])

        seq_len = int(cfg["seq_len"])
        mid = seq_len // 2
        buf_f = deque(maxlen=seq_len)
        buf_g = deque(maxlen=seq_len)

        y_true, y_pred = [], []
        for f, g in zip(feats_list, gt_list):
            buf_f.append(f)
            buf_g.append(g)
            if len(buf_f) == seq_len:
                seq = np.stack(list(buf_f), axis=0)
                gt_mid = int(list(buf_g)[mid])
                pred = int(pred_fn_window(seq))
                y_true.append(gt_mid)
                y_pred.append(pred)

        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # filtering / labels
    if cfg["include_uncertain"]:
        labels = [DOWN, UP, UNCERTAIN]
        target_names = ["DOWN(0)", "UP(1)", "UNCERTAIN(-1)"]
        yt, yp = y_true, y_pred
    else:
        mask = np.isin(y_true, [DOWN, UP]) & np.isin(y_pred, [DOWN, UP])
        yt, yp = y_true[mask], y_pred[mask]
        labels = [DOWN, UP]
        target_names = ["DOWN(0)", "UP(1)"]

    if len(yt) == 0:
        meta = collect_model_params(cfg["models_dir"], model_name)
        return {
            "model": model_name,
            "mode": mode,
            "n_total": int(len(y_true)),
            "n_eval": 0,
            "confusion_matrix": None,
            "report": None,
            "model_params": meta,
            "hint": "All samples filtered out. Try include_uncertain=True or reduce thresh.",
        }

    cm = confusion_matrix(yt, yp, labels=labels)
    report = classification_report(yt, yp, labels=labels, target_names=target_names, digits=4)

    return {
        "model": model_name,
        "mode": mode,
        "n_total": int(len(y_true)),
        "n_eval": int(len(yt)),
        "labels": labels,
        "target_names": target_names,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }

# =========================
# Save outputs
# =========================
def save_outputs(video_path, cfg, diag, per_model_results):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(video_path))[0]
    run_name = f"{base}__{ts}"

    out_json = os.path.join(cfg["out_dir"], f"{run_name}.json")
    out_txt = os.path.join(cfg["out_dir"], f"{run_name}.txt")

    payload = {
        "video": os.path.abspath(video_path),
        "timestamp": ts,
        "config": cfg,
        "diagnostics": diag,
        "results": per_model_results,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # human-readable
    lines = []
    lines.append("=" * 80)
    lines.append(f"Video: {payload['video']}")
    lines.append(f"Time : {ts}")
    lines.append("-" * 80)
    lines.append("Config:")
    for k, v in cfg.items():
        lines.append(f"  {k}: {v}")
    lines.append("-" * 80)
    lines.append("Diagnostics:")
    for k, v in diag.items():
        lines.append(f"  {k}: {v}")
    lines.append("=" * 80)

    for r in per_model_results:
        lines.append(f"Model: {r['model']} | Mode: {r['mode']} | total={r['n_total']} eval={r['n_eval']}")
        if r["confusion_matrix"] is None:
            lines.append(f"  [NO REPORT] {r.get('hint','')}")
        else:
            lines.append("Confusion matrix (rows=true, cols=pred):")
            lines.append(str(np.array(r["confusion_matrix"], dtype=int)))
            lines.append("Report:")
            lines.append(r["report"])
        lines.append("-" * 80)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_json, out_txt

# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    # optional overrides (if you don’t pass, DEFAULT_CFG is used)
    p.add_argument("--models_dir", default=None)
    p.add_argument("--models", nargs="*", default=None)  # e.g. --models logistic mlp
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--thresh", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--vis_thresh", type=float, default=None)
    p.add_argument("--min_ok", type=int, default=None)
    p.add_argument("--include_uncertain", action="store_true")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()

def merge_cfg(args):
    cfg = dict(DEFAULT_CFG)

    # override by args if provided
    if args.models_dir is not None: cfg["models_dir"] = args.models_dir
    if args.models is not None and len(args.models) > 0: cfg["models"] = args.models
    if args.seq_len is not None: cfg["seq_len"] = args.seq_len
    if args.thresh is not None: cfg["thresh"] = args.thresh
    if args.device is not None: cfg["device"] = args.device
    if args.vis_thresh is not None: cfg["vis_thresh"] = args.vis_thresh
    if args.min_ok is not None: cfg["min_ok"] = args.min_ok
    if args.max_frames is not None: cfg["max_frames"] = args.max_frames
    if args.out_dir is not None: cfg["out_dir"] = args.out_dir

    # include_uncertain only from flag (default False)
    cfg["include_uncertain"] = bool(args.include_uncertain)

    return cfg

def main():
    args = parse_args()
    cfg = merge_cfg(args)

    print("[INFO] Using config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    feats_list, gt_list, diag = extract_video_features(args.video, cfg)
    if len(feats_list) == 0:
        print("=" * 80)
        print("No usable frames collected.")
        print("Diagnostics:", diag)
        print("Hint: Try increasing model_complexity, lowering det/track_conf, or relaxing vis_thresh/min_ok.")
        print("=" * 80)
        return

    per_model_results = []
    for m in cfg["models"]:
        try:
            r = evaluate_one_model(m, feats_list, gt_list, cfg)
        except Exception as e:
            r = {
                "model": m,
                "mode": "N/A",
                "n_total": 0,
                "n_eval": 0,
                "confusion_matrix": None,
                "report": None,
                "hint": f"{type(e).__name__}: {e}",
            }
        per_model_results.append(r)

    out_json, out_txt = save_outputs(args.video, cfg, diag, per_model_results)

    print("=" * 80)
    print(f"Saved JSON: {out_json}")
    print(f"Saved TXT : {out_txt}")
    print("=" * 80)

if __name__ == "__main__":
    main()
