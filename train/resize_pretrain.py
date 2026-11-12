#!/usr/bin/env python3
"""RF-DETR の事前学習重みを別解像度向けに作り直す（位置埋め込みを bicubic で補間）．

  resize_pretrain.py --src rf-detr-medium.pth --resolution 768 --patch 16 --out rf-detr-medium-768.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, required=True)
ap.add_argument("--resolution", type=int, required=True)
ap.add_argument("--patch", type=int, default=16)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model"] if "model" in ck else ck
g = a.resolution // a.patch
n = 0
for k, v in sd.items():
    if k.endswith("embeddings.position_embeddings") and v.ndim == 3:
        cls, grid = v[:, :1], v[:, 1:]
        s = int(grid.shape[1] ** 0.5)
        grid = grid.reshape(1, s, s, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid.float(), size=(g, g), mode="bicubic", align_corners=False).to(v.dtype)
        sd[k] = torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, g * g, -1)], 1)
        print(k, tuple(v.shape), "->", tuple(sd[k].shape)); n += 1
assert n, "position_embeddings が見つからない"
torch.save(ck, a.out)
