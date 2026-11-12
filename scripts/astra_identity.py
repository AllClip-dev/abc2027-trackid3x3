#!/usr/bin/env python3
"""色なしプロンプトの Astra 結果（チームは T1/T2，person_id は 1 バッチ内だけ有効）から，動画全体で一貫した
身元（チーム A/B + 背番号）を作る．merge_batches の代わりに使い，samples / people を同じ形で返す．

1. T1/T2 → A/B（公式規則）:
   最初のバッチの最初のフレームで，コート座標（court_registration の homography で足元を変換）が
   「コート端線と平行な外側ラインの中点（x=中央, y=1105）」にいちばん近い選手を初期オフェンス = A とし，
   その選手にいちばん近い選手（他チーム）を初期ディフェンス = B とする．以降のバッチは，重なりフレーム
   （前バッチの最後 = 次バッチの最初の同じ 1 枚）の箱の IoU で前バッチの A/B を引き継ぐ．背番号の名簿では決めない．
2. 身元 = 名簿の背番号（チームごとに 3 人）．サンプルフレームごとに「箱 → 身元」の割り当てを Viterbi で決める:
   - 読み（Astra が各フレームで読んだ背番号）と食い違う割り当てには罰則（READ_PENALTY）
   - 前サンプルからの足元の移動距離が大きい割り当てには罰則（連続性）
   Astra の person_id は使わない（バッチ内でも入れ替わる: IMG_0115 の frame 7200-7215 の A4/A12 など）．
"""
from __future__ import annotations

import csv
import itertools
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

COURT_W, COURT_H = 1505.0, 1105.0
READ_PENALTY = float(os.environ.get("ASTRA_READ_PENALTY", "8.0"))  # 読みと違う身元を割り当てる罰則（論文の要素ごとの効果用に環境変数で変えられる）
MOVE_SCALE_PX = 25.0  # この距離（px）の移動 = 罰則 1
MOVE_CAP_PX = 250.0
# 前サンプルにいなかった（欠けていた）身元が戻ってくるとき，最後に見えた位置からの移動にも罰則をかける（位置の記憶）．
# 罰則 = 重み × max(0, 距離 − 許容 × 欠けたサンプル数) / MOVE_SCALE_PX．重み 0 で無効（従来どおり）．
MEMORY_W = float(os.environ.get("ASTRA_MEMORY_W", "1.0"))
MEMORY_ALLOW_PX = float(os.environ.get("ASTRA_MEMORY_ALLOW_PX", "40.0"))  # 欠けた 1 サンプル（15 フレーム）あたりに動いてよい距離
SAMPLE_STEP = 15

ROSTERS = {"IMG_0104": {"A": ["4", "10", "12"], "B": ["6", "9", "11"]},
           "IMG_0110": {"A": ["4", "10", "12"], "B": ["6", "9", "11"]},  # 2 本目の評価用（論文の追加実験）
           "IMG_0114": {"A": ["6", "9", "10"], "B": ["6", "9", "11"]},
           "IMG_0115": {"A": ["4", "10", "12"], "B": ["1", "3", "5"]}}


def load_homography(path: Path) -> np.ndarray:
    reg = list(csv.DictReader(path.open()))
    h, _ = cv2.findHomography(np.array([[float(r["image_x"]), float(r["image_y"])] for r in reg], np.float32),
                              np.array([[float(r["court_x"]), float(r["court_y"])] for r in reg], np.float32))
    return h


def to_court(h: np.ndarray, xyxy) -> np.ndarray:
    p = np.array([[[(xyxy[0] + xyxy[2]) / 2, xyxy[3]]]], np.float32)  # 足元（枠の下辺中央）
    return cv2.perspectiveTransform(p, h)[0, 0]


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def frame_dets(rec: dict, W: int, H: int) -> list[tuple[int, list[dict]]]:
    """[(絶対フレーム, [{team: T1/T2, read, xyxy(px)}])]"""
    team = {p["person_id"]: p["team"] for p in rec["parsed"]["people"]}
    out = []
    for s, fr in zip(rec["frames"], rec["parsed"]["frames"]):
        dets = []
        for d in fr["detections"]:
            b = d["bbox_1000"]
            dets.append({"team": team[d["person_id"]], "read": None if d.get("jersey") is None else str(d["jersey"]),
                         "xyxy": np.array([b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000], float)})
        out.append((s["frame"], dets))
    return out


def initial_ab(dets: list[dict], h: np.ndarray) -> tuple[dict, dict]:
    """公式規則: 上側ライン中点に最も近い選手のチーム = A，その選手に最も近い選手 = 初期ディフェンス（B のはず）．"""
    pts = [to_court(h, d["xyxy"]) for d in dets]
    top = np.array([COURT_W / 2, COURT_H])
    off = int(np.argmin([np.linalg.norm(p - top) for p in pts]))
    others = [i for i in range(len(dets)) if i != off]
    de = min(others, key=lambda i: np.linalg.norm(pts[i] - pts[off]))
    ta = dets[off]["team"]
    tb = "T2" if ta == "T1" else "T1"
    info = {"offense_team": ta, "offense_court_xy": [round(float(v), 1) for v in pts[off]],
            "defender_team": dets[de]["team"], "defender_is_opposite": dets[de]["team"] == tb}
    return {ta: "A", tb: "B"}, info


def team_maps(records: list[dict], W: int, H: int, h: np.ndarray) -> tuple[list[dict], dict]:
    """バッチごとの {T1/T2: A/B}．最初は公式規則，以降は重なりフレームの箱の一致で引き継ぐ．"""
    per = [frame_dets(r, W, H) for r in records]
    maps, report = [], {"batches_without_overlap_votes": []}
    m0, info = initial_ab(per[0][0][1], h)
    maps.append(m0)
    report["initial_rule"] = info
    for k in range(1, len(records)):
        prev_t, prev_dets = per[k - 1][-1]
        cur_t, cur_dets = per[k][0]
        vote = Counter()
        if prev_t == cur_t:
            for d in cur_dets:
                best = max(prev_dets, key=lambda p: iou(d["xyxy"], p["xyxy"]), default=None)
                if best is not None and iou(d["xyxy"], best["xyxy"]) > 0.5:
                    vote[(d["team"], maps[k - 1][best["team"]])] += 1
        s = vote[("T1", "A")] + vote[("T2", "B")] - vote[("T1", "B")] - vote[("T2", "A")]
        if s == 0:
            report["batches_without_overlap_votes"].append(records[k]["clip"])
            maps.append(dict(maps[k - 1]))
        else:
            maps.append({"T1": "A", "T2": "B"} if s > 0 else {"T1": "B", "T2": "A"})
    report["T1_flips"] = sum(1 for a, b in zip(maps, maps[1:]) if a["T1"] != b["T1"])
    return maps, report


def viterbi(frames: list[tuple[int, list[dict]]], roster: list[str]) -> list[tuple]:
    """frames: [(t, [{read, xyxy}])]（1 チームぶん）．各フレームの割り当て（箱ごとの roster インデックス）を返す．"""
    n_id = len(roster)
    cand = []
    for _, dets in frames:
        dets = dets[:n_id]
        cand.append(list(itertools.permutations(range(n_id), len(dets))))

    def emit(dets, assign) -> float:
        c = 0.0
        for d, a in zip(dets, assign):
            if d["read"] in roster and roster[a] != d["read"]:
                c += READ_PENALTY
        return c

    def foot(d):
        return np.array([(d["xyxy"][0] + d["xyxy"][2]) / 2, d["xyxy"][3]])

    def trans(dp, a_prev, dc, a_cur, mem, t_cur) -> float:
        pos = {a: foot(d) for d, a in zip(dp, a_prev)}
        c = 0.0
        for d, a in zip(dc, a_cur):
            if a in pos:
                c += min(float(np.linalg.norm(foot(d) - pos[a])), MOVE_CAP_PX) / MOVE_SCALE_PX
            elif MEMORY_W > 0 and a in mem:  # 欠けていた身元: 最後に見えた位置からの移動
                p, t_seen = mem[a]
                gap = max(1.0, (t_cur - t_seen) / SAMPLE_STEP)
                far = max(0.0, float(np.linalg.norm(foot(d) - p)) - MEMORY_ALLOW_PX * gap)
                c += MEMORY_W * min(far, MOVE_CAP_PX) / MOVE_SCALE_PX
        return c

    def remember(mem, dets, assign, t) -> dict:
        m = dict(mem)
        for d, a in zip(dets, assign):
            m[a] = (foot(d), t)
        return m

    cost = [np.array([emit(frames[0][1][:n_id], a) for a in cand[0]])]
    mems = [[remember({}, frames[0][1][:n_id], a, frames[0][0]) for a in cand[0]]]  # 状態ごと，その状態へ至る最良経路の「最後に見えた位置」
    back = []
    for i in range(1, len(frames)):
        dp, dc = frames[i - 1][1][:n_id], frames[i][1][:n_id]
        t_cur = frames[i][0]
        cur = np.zeros(len(cand[i]))
        bk = np.zeros(len(cand[i]), int)
        mem_i = []
        for j, a in enumerate(cand[i]):
            tot = [cost[-1][k] + trans(dp, ap, dc, a, mems[-1][k], t_cur) for k, ap in enumerate(cand[i - 1])]
            bk[j] = int(np.argmin(tot))
            cur[j] = tot[bk[j]] + emit(dc, a)
            mem_i.append(remember(mems[-1][bk[j]], dc, a, t_cur))
        cost.append(cur)
        mems.append(mem_i)
        back.append(bk)
    j = int(np.argmin(cost[-1]))
    path = [j]
    for bk in reversed(back):
        j = int(bk[j])
        path.append(j)
    path.reverse()
    return [cand[i][j] for i, j in enumerate(path)]


def build_identities(records: list[dict], W: int, H: int, court_registration: Path, video: str):
    """merge_batches と同じ形で (samples, people, stats) を返す．gid: A の名簿順に 1-3，B の名簿順に 4-6．"""
    roster = ROSTERS[video]
    h = load_homography(court_registration)
    maps, report = team_maps(records, W, H, h)
    # フレームごとに A/B の検出を集める（重なりフレームは検出が多いほう，同数なら後のバッチ）
    by_frame: dict[int, tuple[int, list[dict]]] = {}
    for k, rec in enumerate(records):
        for t, dets in frame_dets(rec, W, H):
            for d in dets:
                d["ab"] = maps[k][d["team"]]
            if t not in by_frame or len(dets) >= len(by_frame[t][1]):
                by_frame[t] = (k, dets)
    ts = sorted(by_frame)
    gid_of = {("A", i): 1 + i for i in range(3)} | {("B", i): 4 + i for i in range(3)}
    assign_of: dict[tuple[str, int], tuple] = {}
    reads = {g: Counter() for g in gid_of.values()}
    samples_d: dict[int, list] = {t: [] for t in ts}
    for ab in ("A", "B"):
        frames = [(t, [d for d in by_frame[t][1] if d["ab"] == ab]) for t in ts]
        frames = [(t, ds) for t, ds in frames if ds]
        path = viterbi(frames, roster[ab])
        for (t, ds), assign in zip(frames, path):
            for d, a in zip(ds[:3], assign):
                g = gid_of[(ab, a)]
                samples_d[t].append((g, d["xyxy"]))
                if d["read"] is not None:
                    reads[g][d["read"]] += 1
    samples = [(t, samples_d[t]) for t in ts]
    people = {}
    for (ab, i), g in gid_of.items():
        people[g] = {"person_id": g, "team": ab, "jersey": roster[ab][i], "jersey_reads": dict(reads[g].most_common()),
                     "batch_person_ids": []}
    stats = {"links": {"identity_mode": "viterbi"}, **report}
    return samples, people, stats
