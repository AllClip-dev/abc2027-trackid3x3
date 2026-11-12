#!/usr/bin/env python3
"""predictions.csv（ヘッダ付き）→ 提出形式（ヘッダなし・カンマ区切りの <動画名>.txt）に変換し，形式を検証する．

  python make_submission.py --pred predictions.csv --video IMG_0114 --out-dir submission/

提出形式（2026-10-19 の主催者連絡）: frame_id,court_x,court_y,team_id,jersey_number
frame_id は 1 始まり，座標は cm，team_id は A/B，トラック ID の列なし．
--num-frames を渡すと，行のないフレームを数え，直前のフレームの行をそのまま写して埋める（全フレームが評価されるため）．
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

COLS = ["frame_id", "court_x", "court_y", "team_id", "jersey_number"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--video", required=True, help="例 IMG_0114（出力名は <video>.txt）")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-frames", type=int, help="動画の総フレーム数（指定すると 1..N の欠けを数える）")
    a = ap.parse_args()

    with a.pred.open() as f:
        rows = list(csv.reader(f))
    if rows and rows[0] == COLS:
        rows = rows[1:]
    errors, per_frame, seen = [], Counter(), set()
    out = []
    for i, r in enumerate(rows, 1):
        if len(r) != 5:
            errors.append(f"行 {i}: 列数 {len(r)}")
            continue
        try:
            fid, x, y, jersey = int(r[0]), float(r[1]), float(r[2]), int(float(r[4]))
        except ValueError:
            errors.append(f"行 {i}: 数値に変換できない {r}")
            continue
        if fid < 1:
            errors.append(f"行 {i}: frame_id が 1 未満")
        if r[3] not in ("A", "B"):
            errors.append(f"行 {i}: team_id が A/B でない {r[3]!r}")
        if (fid, r[3], jersey) in seen:
            errors.append(f"行 {i}: 同じフレームに同じ選手が重複 {(fid, r[3], jersey)}")
        seen.add((fid, r[3], jersey))
        per_frame[fid] += 1
        out.append(f"{fid},{x:.2f},{y:.2f},{r[3]},{jersey}")
    if errors:
        print("検証エラー:", *errors[:20], sep="\n  ", file=sys.stderr)
        sys.exit(1)

    padded = 0
    if a.num_frames:
        by_frame: dict[int, list[str]] = {}
        for l in out:
            by_frame.setdefault(int(l.split(",")[0]), []).append(l)
        out, last = [], None
        for fid in range(1, a.num_frames + 1):
            if fid in by_frame:
                last = by_frame[fid]
            elif last is not None:
                by_frame[fid] = [f"{fid}," + l.split(",", 1)[1] for l in last]
                padded += 1
            else:
                continue
            out += by_frame[fid]
        per_frame = Counter({fid: len(v) for fid, v in by_frame.items()})

    a.out_dir.mkdir(parents=True, exist_ok=True)
    dst = a.out_dir / f"{a.video}.txt"
    dst.write_text("\n".join(out) + "\n")
    frames = sorted(per_frame)
    xs = [float(l.split(",")[1]) for l in out]
    ys = [float(l.split(",")[2]) for l in out]
    print(f"{dst}: {len(out)} 行, frame {frames[0]}..{frames[-1]}, x {min(xs):.0f}..{max(xs):.0f} cm, y {min(ys):.0f}..{max(ys):.0f} cm")
    print("  1 フレームあたりの行数:", dict(sorted(Counter(per_frame.values()).items())))
    if a.num_frames:
        missing = [i for i in range(1, a.num_frames + 1) if i not in per_frame]
        print(f"  直前のフレームで埋めた: {padded}, まだ行のないフレーム: {len(missing)} / {a.num_frames}")


if __name__ == "__main__":
    main()
