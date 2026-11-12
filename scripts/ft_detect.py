#!/usr/bin/env python3
"""学習済み RF-DETR で全フレームを検出し，左右反転 TTA で枠を融合して detections npz に保存する（rfdetr-allclip venv）．
元画像と左右反転画像の検出を，IoU が閾値以上の組で信頼度重み付き平均（WBF 風）にする．片方にしかない枠は信頼度を半分にして残す．"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRLarge, RFDETRMedium

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--resolution", type=int, default=576)
ap.add_argument("--model", choices=["medium", "large"], default="medium")
ap.add_argument("--confidence", type=float, default=0.1)
ap.add_argument("--fuse-iou", type=float, default=0.55)
ap.add_argument("--no-tta", action="store_true")
a = ap.parse_args()


def iou(b, B):
    x1 = np.maximum(b[0], B[:, 0]); y1 = np.maximum(b[1], B[:, 1]); x2 = np.minimum(b[2], B[:, 2]); y2 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def fuse(b1, c1, b2, c2, thr):
    """b1/b2: (N,4) xyxy，c: (N,)．c1 を基準に greedy で b2 と対応づける．"""
    out_b, out_c, used = [], [], np.zeros(len(b2), bool)
    for i in np.argsort(-c1):
        j = -1
        if len(b2):
            v = iou(b1[i], b2); v[used] = 0
            if v.max() >= thr: j = int(v.argmax())
        if j >= 0:
            used[j] = True; w = np.array([c1[i], c2[j]])
            out_b.append((b1[i] * w[0] + b2[j] * w[1]) / w.sum()); out_c.append(w.mean())
        else:
            out_b.append(b1[i]); out_c.append(c1[i] / 2)
    for j in np.where(~used)[0]:
        out_b.append(b2[j]); out_c.append(c2[j] / 2)
    return np.array(out_b).reshape(-1, 4), np.array(out_c)


model = {"medium": RFDETRMedium, "large": RFDETRLarge}[a.model](pretrain_weights=str(a.checkpoint), resolution=a.resolution)
cap = cv2.VideoCapture(str(a.video))
rows, i = [], 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    d = model.predict(rgb, threshold=a.confidence)
    b, c = np.asarray(d.xyxy, np.float32).reshape(-1, 4), np.asarray(d.confidence, np.float32)
    if not a.no_tta:
        W = rgb.shape[1]
        df = model.predict(np.ascontiguousarray(rgb[:, ::-1]), threshold=a.confidence)
        bf = np.asarray(df.xyxy, np.float32).reshape(-1, 4).copy()
        bf[:, [0, 2]] = W - bf[:, [2, 0]]
        b, c = fuse(b, c, bf, np.asarray(df.confidence, np.float32), a.fuse_iou)
    for xyxy, conf in zip(b, c):
        rows.append([i, *xyxy, conf])
    i += 1
    if i % 1000 == 0:
        print(i, flush=True)
r = np.asarray(rows, np.float32).reshape(-1, 6)
np.savez_compressed(a.out, frame=r[:, 0].astype(np.int32), xyxy=r[:, 1:5], confidence=r[:, 5])
print("frames", i, "detections", len(r))
