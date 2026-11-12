#!/usr/bin/env python3
"""チャレンジの TI-HOTA を算出する．公式 Colab チュートリアル（data/ref/colab/tutorial.ipynb の §8-9）の
compute_ti_hota_alpha / compute_ti_hota / compute_location_diagnostics と同じ計算（行列の作り方だけベクトル化）．

  python eval_ti_hota.py --gt <GT> --pred <predictions.csv> [--start-frame N] [--end-frame M] [--out summary.json]

GT は 2 形式を受ける: ヘッダ付きの Challenge 形式 csv（frame_id,court_x,court_y,team_id,jersey_number），
または training の court position txt（ヘッダなし frame,court_x,court_y,team,jersey）．
予測は Challenge 形式 csv．frame_id は両者とも 1 始まり．
--start-frame / --end-frame は GT と予測の両方を同じ範囲（両端を含む）に切る．
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

TAU_CM = 100.0
COLS = ["frame_id", "court_x", "court_y", "team_id", "jersey_number"]


def read_table(path: Path) -> dict[int, list[tuple[float, float, str, int]]]:
    """frame_id -> [(x, y, team, jersey)]"""
    with path.open() as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0] == "frame_id":
        if rows[0] != COLS:
            raise ValueError("columns must be: " + ", ".join(COLS))
        rows = rows[1:]
    out: dict[int, list] = defaultdict(list)
    for r in rows:
        out[int(r[0])].append((float(r[1]), float(r[2]), str(r[3]), int(float(r[4]))))
    return out


def ident(team: str, jersey: int) -> str:
    return team + str(jersey)  # Colab と同じ: team_id + jersey_number が選手の同一性


def build_frames(gt: dict, pred: dict, lo: int | None, hi: int | None) -> list[dict]:
    ids = sorted(set(gt) | set(pred))
    ids = [i for i in ids if (lo is None or i >= lo) and (hi is None or i <= hi)]
    f = []
    for i in ids:
        f.append({"gt": [((x, y), ident(t, j)) for x, y, t, j in gt.get(i, [])],
                  "pred": [((x, y), ident(t, j)) for x, y, t, j in pred.get(i, [])]})
    return f


def similarity(frame: dict, tau: float) -> np.ndarray:
    g, p = frame["gt"], frame["pred"]
    if not g or not p:
        return np.zeros((len(g), len(p)))
    gp = np.array([x[0] for x in g]); pp = np.array([x[0] for x in p])
    d2 = ((gp[:, None, :] - pp[None, :, :]) ** 2).sum(-1)
    loc = np.exp(math.log(0.05) * d2 / tau ** 2)  # 距離 0 で 1，tau で 0.05
    same = np.array([[a[1] == b[1] for b in p] for a in g], float)  # IdSim: チームと背番号が一致なら 1
    return loc * same


def ti_hota_alpha(frames: list[dict], sims: list[np.ndarray], alpha: float):
    tp = fp = fn = 0
    assoc: dict[str, dict] = {}
    fp_assoc: dict[str, int] = {}
    for fr, s in zip(frames, sims):
        g, p = fr["gt"], fr["pred"]
        match = {}
        if g and p:
            cost = 1.0 - s
            cost[s < alpha] = 1e6
            for i, j in zip(*linear_sum_assignment(cost)):
                if s[i, j] >= alpha:
                    match[i] = j
        t = len(match)
        tp += t; fp += len(p) - t; fn += len(g) - t
        for i, (_, idn) in enumerate(g):
            a = assoc.setdefault(idn, {"TPA": 0, "FNA": 0})
            a["TPA" if i in match else "FNA"] += 1
        used = set(match.values())
        for j, (_, idn) in enumerate(p):
            if j not in used:
                fp_assoc[idn] = fp_assoc.get(idn, 0) + 1
    det_a = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    parts = []
    for idn, c in assoc.items():
        den = c["TPA"] + c["FNA"] + fp_assoc.get(idn, 0)
        if den > 0:
            parts.append(c["TPA"] / den)
    ass_a = sum(parts) / len(parts) if parts else 0.0
    return math.sqrt(det_a * ass_a), det_a, ass_a, tp, fp, fn


def compute_ti_hota(frames: list[dict], tau: float = TAU_CM) -> dict:
    sims = [similarity(f, tau) for f in frames]
    alphas = np.arange(0.05, 1.0, 0.05)
    res = [ti_hota_alpha(frames, sims, a) for a in alphas]
    m = lambda k: float(np.mean([r[k] for r in res]))
    return {"TI-HOTA": m(0), "TI-DetA": m(1), "TI-AssA": m(2), "TI-TP": m(3), "TI-FP": m(4), "TI-FN": m(5),
            "TI-HOTA_alphas": {f"{a:.2f}": r[0] for a, r in zip(alphas, res)}}


def location_diagnostics(frames: list[dict], max_distance_cm: float = 100.0) -> dict:
    """位置だけで対応づけ，距離 max_distance_cm 以内の組についてチーム・背番号の正解率と位置誤差の中央値を出す．"""
    errs, team_ok, jersey_ok = [], [], []
    for f in frames:
        g, p = f["gt"], f["pred"]
        if not g or not p:
            continue
        gp = np.array([x[0] for x in g]); pp = np.array([x[0] for x in p])
        d = np.linalg.norm(gp[:, None, :] - pp[None, :, :], axis=2)
        cost = d.copy(); cost[d > max_distance_cm] = 1e6
        for i, j in zip(*linear_sum_assignment(cost)):
            if d[i, j] > max_distance_cm:
                continue
            errs.append(d[i, j])
            team_ok.append(g[i][1][0] == p[j][1][0])
            jersey_ok.append(g[i][1][1:] == p[j][1][1:])
    if not errs:
        return {"matched_pairs": 0}
    return {"matched_pairs": len(errs), "median_position_error_cm": float(np.median(errs)),
            "team_accuracy": float(np.mean(team_ok)), "jersey_accuracy": float(np.mean(jersey_ok))}


def evaluate(gt_path: Path, pred_path: Path, lo: int | None = None, hi: int | None = None) -> dict:
    frames = build_frames(read_table(gt_path), read_table(pred_path), lo, hi)
    out = compute_ti_hota(frames)
    out["frames"] = len(frames)
    out["diagnostics"] = location_diagnostics(frames)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--start-frame", type=int)
    ap.add_argument("--end-frame", type=int)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    r = evaluate(a.gt, a.pred, a.start_frame, a.end_frame)
    if a.out:
        a.out.write_text(json.dumps(r, indent=1))
    d = r["diagnostics"]
    print(f"frames {r['frames']}  TI-HOTA {r['TI-HOTA']:.4f}  DetA {r['TI-DetA']:.4f}  AssA {r['TI-AssA']:.4f}")
    print(f"  TP {r['TI-TP']:.0f} FP {r['TI-FP']:.0f} FN {r['TI-FN']:.0f}  (α 平均)")
    if d.get("matched_pairs"):
        print(f"  位置誤差の中央値 {d['median_position_error_cm']:.1f} cm  チーム {d['team_accuracy']:.4f}  背番号 {d['jersey_accuracy']:.4f}  対 {d['matched_pairs']}")


if __name__ == "__main__":
    main()
