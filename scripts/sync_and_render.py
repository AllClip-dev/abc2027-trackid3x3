#!/usr/bin/env python3
"""Astra の身元（チーム・背番号）を BoT-SORT の track に同期し，身元つきで動画に描く（Mac で動く）．

sync_astra.py の merge_batches（バッチ間の person_id 連結）と match_and_split（track への投票，
身元が変わる所での分割，同時 2 重取りの解消）をそのまま使う．コート投影・ボール状態は使わない（3x3 課題に不要）．
sync_astra.py は import 時に roboflow `sports` を要求するが，ここで使う関数は使わないので空のモジュールで置き換える．

Astra が身元を与えた tracklet だけを描く（ベンチ・記録係・黒シャツなど Astra が選ばなかった人は描かない）．
ラベルは「チーム + 背番号」（読めなければ ?）．

BoT-SORT の欠損は身元（人）ごとに補間する:
  1. Astra が答えたフレーム（2 fps）で，その人に BoT-SORT の枠がなければ Astra の枠を足場に加える（細い実線，ラベル末尾 "a"）．
  2. 足場どうしの間隔が --max-fill フレーム以下の穴を直線補間する．補間位置と IoU が --fill-iou 以上で，
     どの身元にも使われていない BoT-SORT の枠があれば採用（実線，末尾 "~"），なければ補間枠そのもの（破線，末尾 "~"）．
  最初の足場より前と最後の足場より後は補間しない．

  python sync_and_render.py --video V.MP4 --tracks tracks.json --astra-results results.jsonl --clip IMG_0114 \
      --max-batches 6 --out-dir OUT
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

for name in ("sports", "sports.basketball"):
    sys.modules[name] = types.SimpleNamespace(MeasurementUnit=None, CourtConfiguration=None, League=None,
                                              draw_court=None, draw_points_on_court=None)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_astra import load_astra, match_and_split, merge_batches  # noqa: E402

TEAM_BGR = {"A": (0, 215, 255), "B": (80, 200, 60)}  # 黄，緑（IMG_0114 のビブス色）


COURT_W, COURT_H = 1505, 1105  # 屋外 3x3 コート [cm]．ミニマップは 1 cm = 1 px（公式チュートリアルと同じ）


def court_point(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """画像上の点を court_registration の ホモグラフィでコート座標 [cm] へ．"""
    v = H @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def draw_minimap(bg: np.ndarray, pts: list, scale: float) -> np.ndarray:
    """公式の create_minimap と同じく x を左右反転して描く（x -> 1505 - x，y はそのまま）．"""
    img = cv2.resize(bg, (int(COURT_W * scale), int(COURT_H * scale)))
    for cx, cy, text, col in pts:
        x, y = int((COURT_W - cx) * scale), int(cy * scale)
        cv2.circle(img, (x, y), max(int(30 * scale), 3), col, -1)
        cv2.putText(img, text, (x + int(10 * scale), y - int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    4 * scale, col, max(int(8 * scale), 1), cv2.LINE_AA)
    return img


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dashed_rect(img, p1, p2, color, thickness=2, dash=8):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def smooth_rows(rows: list, window: int, sigma: float = 0.0) -> list:
    """身元（team, jersey）ごとに，連続したフレームの区間で足元座標を中央値フィルタし，sigma > 0 なら
    続けてガウスフィルタ（標準偏差 sigma フレーム）をかける（枠は変えない）．"""
    from scipy.ndimage import gaussian_filter1d, median_filter
    series: dict = defaultdict(list)
    for fr, x, y, team, jersey in rows:
        series[(team, jersey)].append((fr, x, y))
    out = []
    for (team, jersey), pts in series.items():
        pts.sort()
        frames = np.array([p[0] for p in pts])
        xy = np.array([[p[1], p[2]] for p in pts])
        for seg in np.split(np.arange(len(pts)), np.where(np.diff(frames) > 1)[0] + 1):
            sm = xy[seg]
            if len(seg) > 2:
                if window > 1:
                    sm = median_filter(sm, size=(window, 1), mode="nearest")
                if sigma > 0:
                    sm = gaussian_filter1d(sm, sigma, axis=0, mode="nearest")
            out += [(int(frames[i]), round(float(sm[k, 0]), 2), round(float(sm[k, 1]), 2), team, jersey) for k, i in enumerate(seg)]
    return sorted(out, key=lambda r: (r[0], r[3], r[4]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--astra-results", type=Path, required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--min-iou", type=float, default=0.4)
    ap.add_argument("--team-bgr", nargs=2, action="append", metavar=("TEAM", "B,G,R"),
                    help="チームの描画色（既定 A=黄 B=緑）")
    ap.add_argument("--court-registration", type=Path, help="court_registration.csv（画像座標とコート座標[cm]の対応点）．指定でミニマップを描く")
    ap.add_argument("--court-image", type=Path, help="コート背景画像（公式チュートリアルの Outdoor.png）")
    ap.add_argument("--max-fill", type=int, default=45, help="補間する穴の最大長（フレーム）")
    ap.add_argument("--fill-iou", type=float, default=0.3)
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--sam3-boxes", type=Path,
                    help="sam3_propagate.py の出力 json（{身元ラベル: {フレーム: [x1,y1,x2,y2]}}）．穴の中でこの枠があるフレームは，補間の代わりにこの枠で埋める")
    ap.add_argument("--fill-over-astra-absent", action="store_true",
                    help="Astra が選手を返さなかったサンプルをまたぐ穴も埋める（既定は埋めず，その区間は 6 人未満のままにする）")
    ap.add_argument("--identity", choices=["merge", "viterbi"], default="merge",
                    help="merge: 色ありプロンプト（A/B 名と person_id の連結）．viterbi: 色なしプロンプト（T1/T2，astra_identity.py）")
    ap.add_argument("--anchor-only", action="store_true",
                    help="トラックへの投票で身元を決めない．Astra の枠だけを足場に補間し，BoT-SORT の枠は補間位置に最も合うものを"
                         "フレームごとの Hungarian で選ぶだけにする（画像上で重なる 2 人の入れ替わり対策）")
    ap.add_argument("--snap-cm", type=float, default=30.0,
                    help="--anchor-only で BoT-SORT の枠を採用する条件: 補間枠との足元のコート上の距離 [cm] の上限（--court-registration が要る）")
    ap.add_argument("--verify-cm", type=float, default=0.0,
                    help="track の枠を検証する: Astra の足場（2 fps）とその補間経路から足元がコート上でこの距離 [cm] より離れた枠は，"
                         "補間位置の枠に差し替える（画像上で重なる 2 人の入れ替わり対策）．0 で無効．--court-registration が要る")
    ap.add_argument("--cut-metric", choices=["iou", "jump"], default="iou",
                    help="身元の切り替え位置の選び方: iou=前後フレームの枠の IoU が最小，jump=枠の辺の動き（px）が最大")
    ap.add_argument("--ambig-margin", type=float, default=0.0,
                    help="Astra の枠とトラックの対応づけが曖昧（2 番目の候補も --min-iou 以上で，IoU の差がこの値未満）なとき，"
                         "前後のサンプルで同じ身元に対応づいたトラックを優先する（画像上で重なる 2 人の取り違え対策）．0 で無効")
    ap.add_argument("--ambig-mode", choices=["continuity", "foot"], default="continuity",
                    help="曖昧な対応づけの決め方: continuity=前後の票と同じ身元のトラック，foot=足元（枠の下辺中央）が近いトラック")
    ap.add_argument("--override-boxes", type=Path,
                    help="occlusion_fix.py boxes の出力 json（{身元ラベル: {フレーム: [x1,y1,x2,y2]}}）．そのフレームの身元の枠をこれで置き換える")
    ap.add_argument("--no-video", action="store_true", help="identified.mp4 / minimap.mp4 を書かない（predictions.csv と summary.json だけ）")
    ap.add_argument("--smooth-median", type=int, default=9,
                    help="predictions.csv の足元座標を，身元ごとに連続したフレーム区間で中央値フィルタする窓幅（フレーム）．0 で無効")
    ap.add_argument("--smooth-gauss", type=float, default=0.0,
                    help="中央値フィルタの後にかけるガウスフィルタの標準偏差（フレーム）．0 で無効")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    colors = dict(TEAM_BGR)
    for team, bgr in args.team_bgr or []:
        colors[team] = tuple(int(v) for v in bgr.split(","))

    tracks = json.loads(args.tracks.read_text())
    W, H = tracks["width"], tracks["height"]
    records = load_astra(args.astra_results, args.clip)[:args.max_batches]
    # Astra は 15 フレームに 1 枚なので最後のサンプルは動画の末尾より手前．評価は全フレームなので，枠は動画の最後まで描く
    last = max(records[-1]["frames"][-1]["frame"], tracks["frames"][-1]["frame"])
    if args.identity == "viterbi":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from astra_identity import build_identities  # noqa: E402
        samples, people, stats = build_identities(records, W, H, args.court_registration, args.clip)
    else:
        samples, people, stats, _, _ = merge_batches(records, W, H, one_per_batch=True, link_leftovers=True,
                                                     link_jerseys=False)
    boxes: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for f in tracks["frames"]:
        if f["frame"] > last:
            break
        for p in f["people"]:
            boxes[p["track_id"]][f["frame"]] = np.array(p["xyxy"], np.float32)
    res = match_and_split(samples, boxes, args.min_iou, cut_metric=args.cut_metric, ambig_margin=args.ambig_margin,
                          ambig_mode=args.ambig_mode)

    label = {gid: f"{p['team']}{p['jersey'] or '?'}" for gid, p in people.items()}
    # 身元ごとの枠: pid -> {frame: (xyxy, kind)}  kind: track | astra | fill_track | fill_interp
    ident: dict[int, dict[int, tuple]] = defaultdict(dict)
    claimed: dict[int, set] = defaultdict(set)  # frame -> 身元に使われた track_id
    for tl in ([] if args.anchor_only else res["tracklets"]):
        if tl["pid"] is None:
            continue
        for t, xyxy in boxes[tl["source_track"]].items():
            if tl["start"] <= t <= tl["end"]:
                ident[tl["pid"]][t] = (xyxy, "track")
                claimed[t].add(tl["source_track"])
    n_astra = n_fill_track = n_fill_interp = n_verified = n_fill_sam3 = 0
    sam3_boxes: dict[str, dict[int, list]] = {}
    if args.sam3_boxes:
        sam3_boxes = {lab: {int(t): b for t, b in v.items()} for lab, v in json.loads(args.sam3_boxes.read_text()).items()}
    if args.verify_cm and args.court_registration and not args.anchor_only:
        reg = list(csv.DictReader(args.court_registration.open()))
        Hv, _ = cv2.findHomography(
            np.array([[float(r["image_x"]), float(r["image_y"])] for r in reg], np.float32),
            np.array([[float(r["court_x"]), float(r["court_y"])] for r in reg], np.float32))
        foot = lambda b: np.array(court_point(Hv, (b[0] + b[2]) / 2, b[3]))  # noqa: E731
        anchors: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        for t, dets in samples:
            for gid, xyxy in dets:
                anchors[gid][t] = np.array(xyxy, np.float32)
        for gid, at in anchors.items():
            ts = sorted(at)
            for t in ts:  # 足場のフレーム: track の枠が Astra の枠から離れていれば Astra の枠
                cur = ident[gid].get(t)
                if cur is not None and cur[1] == "track" and np.linalg.norm(foot(cur[0]) - foot(at[t])) > args.verify_cm:
                    ident[gid][t] = (at[t], "astra"); n_verified += 1
            for a, b in zip(ts, ts[1:]):  # 足場の間: 補間経路から離れていれば補間位置の枠
                if not 1 < b - a <= args.max_fill + 1:
                    continue
                for t in range(a + 1, b):
                    cur = ident[gid].get(t)
                    if cur is None or cur[1] != "track":
                        continue
                    w = (t - a) / (b - a)
                    box = (1 - w) * at[a] + w * at[b]
                    if np.linalg.norm(foot(cur[0]) - foot(box)) > args.verify_cm:
                        ident[gid][t] = (box, "fill_interp"); n_verified += 1
    if not args.no_fill:
        for t, dets in samples:  # 1. Astra の枠を足場に
            for gid, xyxy in dets:
                if t not in ident[gid]:
                    ident[gid][t] = (np.array(xyxy, np.float32), "astra")
                    n_astra += 1
        if args.anchor_only:  # 2'. 足場の間を補間し，各フレームで補間枠と BoT-SORT の枠を Hungarian で対応づける
            interp: dict[int, dict] = defaultdict(dict)
            for pid, frames_of in ident.items():
                known = sorted(frames_of)
                for a, b in zip(known, known[1:]):
                    if 1 < b - a <= args.max_fill + 1:
                        for t in range(a + 1, b):
                            w = (t - a) / (b - a)
                            interp[t][pid] = (1 - w) * frames_of[a][0] + w * frames_of[b][0]
            H_snap = None
            if args.court_registration:
                reg = list(csv.DictReader(args.court_registration.open()))
                H_snap, _ = cv2.findHomography(
                    np.array([[float(r["image_x"]), float(r["image_y"])] for r in reg], np.float32),
                    np.array([[float(r["court_x"]), float(r["court_y"])] for r in reg], np.float32))

            def foot_cm(b):
                return np.array(court_point(H_snap, (b[0] + b[2]) / 2, b[3]))

            for t, want in interp.items():
                pids = list(want)
                tids = [tid for tid, bs in boxes.items() if t in bs]
                adopted = {}
                if tids:
                    m = np.array([[box_iou(want[p], boxes[tid][t]) for tid in tids] for p in pids])
                    if H_snap is not None:  # コート上で離れすぎた枠は候補から外す（画像上で重なる別人の枠を取らない）
                        for i, p in enumerate(pids):
                            fp = foot_cm(want[p])
                            for j, tid in enumerate(tids):
                                if np.linalg.norm(foot_cm(boxes[tid][t]) - fp) > args.snap_cm:
                                    m[i, j] = 0.0
                    for i, j in zip(*linear_sum_assignment(-m)):
                        if m[i, j] >= args.fill_iou:
                            adopted[pids[i]] = tids[j]
                for p in pids:
                    if p in adopted:
                        ident[p][t] = (boxes[adopted[p]][t], "fill_track"); n_fill_track += 1
                    else:
                        ident[p][t] = (want[p], "fill_interp"); n_fill_interp += 1
        sample_frames = sorted(t for t, _ in samples)
        for pid, frames_of in ([] if args.anchor_only else ident.items()):  # 2. 足場どうしの穴を埋める
            known = sorted(frames_of)
            for a, b in zip(known, known[1:]):
                if not 1 < b - a <= args.max_fill + 1:
                    continue
                skip = not args.fill_over_astra_absent and any(a < s < b for s in sample_frames)
                # 穴の中に Astra のサンプルがある = Astra はその選手を返していない（写っていない）ので，通常は埋めない
                for t in range(a + 1, b):
                    sb = sam3_boxes.get(label[pid], {}).get(t)
                    if sb is not None:
                        frames_of[t] = (np.array(sb, np.float32), "fill_sam3")
                        n_fill_sam3 += 1
                        continue
                    if skip:
                        continue
                    w = (t - a) / (b - a)
                    box = (1 - w) * frames_of[a][0] + w * frames_of[b][0]
                    best, best_iou = None, args.fill_iou
                    for tid, bs in boxes.items():
                        if t in bs and tid not in claimed[t]:
                            iou = box_iou(box, bs[t])
                            if iou >= best_iou:
                                best, best_iou = tid, iou
                    if best is not None:
                        frames_of[t] = (boxes[best][t], "fill_track")
                        claimed[t].add(best)
                        n_fill_track += 1
                    else:
                        frames_of[t] = (box, "fill_interp")
                        n_fill_interp += 1
    n_override = 0
    if args.override_boxes:
        gid_of_label = {lab: g for g, lab in label.items()}
        for lab, by_t in json.loads(args.override_boxes.read_text()).items():
            for t, b in by_t.items():
                if int(t) in ident[gid_of_label[lab]]:
                    ident[gid_of_label[lab]][int(t)] = (np.array(b, np.float32), "astra_fix")
                    n_override += 1
    per_frame: dict[int, list] = defaultdict(list)  # frame -> [(gid, xyxy, kind)]
    for pid, frames_of in ident.items():
        for t, (xyxy, kind) in frames_of.items():
            per_frame[t].append((pid, xyxy, kind))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ident_boxes.json").write_text(json.dumps(  # 身元ごとの枠（SAM3 の起点・評価用）: {フレーム(0 始まり): {ラベル: [x1,y1,x2,y2,kind]}}
        {str(t): {label[g]: [*map(float, xy), kind] for g, xy, kind in v} for t, v in per_frame.items()}))
    homography = court_bg = None
    if args.court_registration:
        reg = list(csv.DictReader(args.court_registration.open()))
        homography, _ = cv2.findHomography(
            np.array([[float(r["image_x"]), float(r["image_y"])] for r in reg], np.float32),
            np.array([[float(r["court_x"]), float(r["court_y"])] for r in reg], np.float32))
        court_bg = cv2.resize(cv2.imread(str(args.court_image)), (COURT_W, COURT_H))
    pred_rows = []  # チャレンジ提出形式: frame_id（1 始まり）, court_x, court_y, team_id, jersey_number
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = None if args.no_video else cv2.VideoWriter(str(args.out_dir / "identified.mp4"),
                                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    mini_out = None
    if homography is not None and not args.no_video:
        mini_out = cv2.VideoWriter(str(args.out_dir / "minimap.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                   (COURT_W // 2, COURT_H // 2))
    n_ident = []
    for t in range(last + 1):
        if args.no_video:
            frame = None
        else:
            ok, frame = cap.read()
            if not ok:
                break
        items = per_frame.get(t, [])
        n_ident.append(len(items))
        for gid, (x1, y1, x2, y2), kind in ([] if frame is None else items):
            team = people[gid]["team"]
            col = colors.get(team, (200, 200, 200))
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            if kind == "fill_interp":
                dashed_rect(frame, p1, p2, col)
            else:
                cv2.rectangle(frame, p1, p2, col, 1 if kind == "astra" else 2)
            text = label[gid] + {"track": "", "astra": " a", "fill_track": " ~", "fill_interp": " ~", "fill_sam3": " s", "astra_fix": " f"}[kind]
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), col, -1)
            cv2.putText(frame, text, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if homography is not None:
            pts = []
            for gid, (x1, y1, x2, y2), kind in items:
                cx, cy = court_point(homography, (x1 + x2) / 2, y2)  # 足元（枠の下辺中央）
                team, jersey = people[gid]["team"], people[gid]["jersey"]
                pts.append((cx, cy, label[gid], colors.get(team, (200, 200, 200))))
                if jersey is not None:
                    pred_rows.append((t + 1, round(cx, 2), round(cy, 2), team, jersey))
            if frame is None:
                continue
            mini = draw_minimap(court_bg, pts, 0.5)
            mini_out.write(mini)
            ow = W // 4  # 動画の右上に重ねる（公式の overlay と同様，右上・不透明度 0.9）
            small = cv2.resize(mini, (ow, int(ow * COURT_H / COURT_W)))
            roi = frame[8:8 + small.shape[0], W - ow - 8:W - 8]
            frame[8:8 + small.shape[0], W - ow - 8:W - 8] = cv2.addWeighted(roi, 0.1, small, 0.9, 0)
        if frame is None:
            continue
        cv2.putText(frame, f"frame {t}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
    if out is not None:
        out.release()
    if mini_out is not None:
        mini_out.release()
    if homography is not None:
        if args.smooth_median > 1 or args.smooth_gauss > 0:
            pred_rows = smooth_rows(pred_rows, args.smooth_median, args.smooth_gauss)
        with (args.out_dir / "predictions.csv").open("w") as fh:
            fh.write("frame_id,court_x,court_y,team_id,jersey_number\n")
            for row in pred_rows:
                fh.write(",".join(str(v) for v in row) + "\n")
        inside = sum(1 for r in pred_rows if 0 <= r[1] <= COURT_W and 0 <= r[2] <= COURT_H)

    summary = {
        "batches": len(records), "frames_rendered": len(n_ident), "astra_people": len(people),
        "people": {str(g): {"team": p["team"], "jersey": p["jersey"], "reads": p["jersey_reads"]}
                   for g, p in people.items()},
        "link_stats": stats["links"], "identity_report": {k: v for k, v in stats.items() if k != "links"},
        "tracklets": len(res["tracklets"]), "identified_tracklets": sum(1 for tl in res["tracklets"] if tl["pid"] is not None),
        "splits": len(res["splits"]), "conflicts": len(res["conflicts"]), "ambig_margin": args.ambig_margin, "ambig_mode": args.ambig_mode,
        "rematched": res["rematched"],
        "merges": {str(k): v for k, v in res["merges"].items()},
        "verified_replaced": n_verified, "override_boxes": n_override,
        "fill": {"astra_anchor_boxes": n_astra, "filled_with_track_box": n_fill_track,
                 "filled_with_interpolated_box": n_fill_interp, "filled_with_sam3_box": n_fill_sam3, "max_fill": args.max_fill},
        "minimap": None if homography is None else {"prediction_rows": len(pred_rows), "inside_court": inside,
                                                      "homography": homography.round(6).tolist()},
        "identified_per_frame": dict(Counter(n_ident)),
        "mean_match_iou": round(float(np.mean(res["match_ious"])), 3) if res["match_ious"] else None,
        "unmatched_astra_boxes": res["unmatched_astra"],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
