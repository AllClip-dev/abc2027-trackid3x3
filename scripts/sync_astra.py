#!/usr/bin/env python3
"""Stage 3b of basketball_demo: sync Astra identities onto BoT-SORT tracks, then render.

Inputs (per clip job dir): tracks.json (run_detect_track.py), court_v14/court_xy.npz
(run_court_projection.py), and the clip's record in the Astra results.jsonl (run_astra_clips.py).

1. On every Astra-sampled frame, Hungarian-match Astra boxes to BoT-SORT boxes (IoU >= --min-iou).
2. Each track collects (frame, Astra person_id) votes.
3. A track whose votes change person is an ID switch: it is split at the frame, between the two
   disagreeing votes, where its box moves most (lowest IoU with the previous frame's box).
4. Tracklets voted to the same person are one identity (merge). Two tracklets of one person that
   overlap in time are a conflict: the one with more votes keeps the identity.
5. Identity = (team, jersey) when Astra read a jersey, else clip-local person. Names from rosters.json;
   a jersey missing from the roster is kept but flagged.
6. Ball-holder evidence per frame (identity): Astra's "ball_holder" on its sampled frames > RF-DETR's
   player-in-possession class (Astra null = no evidence from Astra).
7. Frames without evidence between holder A and holder B: shot (RF-DETR shot class of A / ball-in-basket)
   -> no holder, ball hidden until B; pass (A != B) -> no holder, ball court position interpolated from A
   to B; A == B -> A keeps it. The ball's court position is otherwise its holder's.

Writes <job>/identity/identities.json, ball_holder.json, court_identities.npz, identified.mp4, summary.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from sports import MeasurementUnit
from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court

HERE = Path(__file__).resolve().parent
HOLDER_COLOR = sv.Color.from_hex("#FACC15")  # ball holder (box, label, court point)
BALL_COLOR = sv.Color.from_hex("#FF8C00")
SHOT_COLOR = sv.Color.from_hex("#EF4444")  # shooter, ball and arrow while a shot is in the air
RIM_FROM_BASELINE_FT = 1.575 / 0.3048  # FIBA
STATE_LABELS = {"held": "保持", "pass": "パス", "shot": "シュート", "loose": "ルーズボール", "unknown": "不明"}
STATE_COLORS = {"held": "#FACC15", "pass": "#FF8C00", "shot": "#EF4444", "loose": "#A855F7", "unknown": "#6B7280"}
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_patches: dict = {}


def state_patch(text: str, hex_color: str) -> np.ndarray:
    """The top-left ball-state label (Japanese needs a CJK font, so it is drawn with PIL once per text)."""
    key = (text, hex_color)
    if key not in _patches:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(FONT_PATH, 44)
        x0, y0, x1, y1 = font.getbbox(text)
        w, h = x1 - x0 + 36, y1 - y0 + 24
        img = Image.new("RGB", (w, h), hex_color)
        r, g, b = sv.Color.from_hex(hex_color).as_rgb()
        fg = (17, 17, 17) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)
        ImageDraw.Draw(img).text((18 - x0, 12 - y0), text, font=font, fill=fg)
        _patches[key] = np.array(img)[:, :, ::-1].copy()
    return _patches[key]


def text_color(c: sv.Color) -> tuple[int, int, int]:
    """Dark text on light fills (white team, holder yellow), white text otherwise (BGR)."""
    r, g, b = c.as_rgb()
    return (17, 17, 17) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def load_astra(results: Path, clip: str) -> list[dict]:
    """All successful records of this clip: one ("<clip>") or several batches ("<clip>_bNN"),
    in batch order. The last successful record of each batch wins."""
    recs = {}
    for line in results.read_text().splitlines():
        r = json.loads(line)
        name = r["clip"]
        if (name == clip or (name.startswith(clip + "_b") and name[len(clip) + 2:].isdigit())) \
                and r.get("returncode") == 0 and r.get("parsed") and not r.get("schema_error"):
            recs[name] = r
    if not recs:
        raise SystemExit(f"no successful Astra record for {clip} in {results}")
    return [recs[k] for k in sorted(recs)]


def merge_batches(records: list[dict], W: int, H: int, min_iou: float = 0.3, force_iou: float = 0.7,
                  one_per_batch: bool = False, link_leftovers: bool = False, link_jerseys: bool = True
                  ) -> tuple[list, dict, dict]:
    """Astra person_ids are only consistent inside one call. Link them across calls by
    (a) box IoU on the frame shared by consecutive batches (1-frame overlap), same team, and
    (b) the same (team, jersey) when both are read. Two different known jerseys are not linked unless
    the overlap boxes nearly coincide (IoU >= force_iou): Astra reads a number differently from one call
    to the next (6 vs 8 seen on the 1-minute AllClip clip), while the same box at the same frame is
    the same person. Split jersey reads are resolved by majority, a tie is shown as "6/8".
    one_per_batch: never link two groups that both contain a person of the same batch (two people of one
    call are different people). Off by default here; sync_astra.py main turns it on since 2026-10-16 (the Opus
    run merged two Blue players through one misread number), and the 4v4 runs need it:
    without it, chains of overlap and jersey links merged the four Green players of 30 batches into one.
    link_leftovers (with one_per_batch): at a batch boundary, if exactly one group of a team ends in the
    previous batch and exactly one group of that team starts in the next, they are the same person (the
    4v4 teams keep 4 players per call; seen when two players tangle on the overlap frame and their jersey
    reads disagree).
    link_jerseys: link groups with the same (team, jersey) across non-adjacent batches (step b). The 4v4 runs
    turn it off: a misread number linked two different players into one group with holes in it.
    Returns samples [(frame, [(gid, xyxy_px, jersey_read)])], people {gid: {person_id, team, jersey}}, stats."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # jersey reads: per detection ("jersey" on each box, the AllClip prompt since 2026-09-28) or, for older
    # prompts, one read per person per call. Each batch person's number = majority of its frame reads.
    reads = defaultdict(Counter)  # (batch, person_id) -> Counter of per-frame jersey reads
    for bi, r in enumerate(records):
        for fr in r["parsed"]["frames"]:
            for d in fr["detections"]:
                if d.get("jersey") is not None:
                    reads[(bi, d["person_id"])][str(d["jersey"])] += 1

    def majority(counter: Counter) -> set:
        if not counter:
            return set()
        top = max(counter.values())
        return {j for j, c in counter.items() if c == top}

    info = {}
    for bi, r in enumerate(records):
        for p in r["parsed"]["people"]:
            key = (bi, p["person_id"])
            parent[key] = key
            js = majority(reads[key]) if reads[key] else ({p.get("jersey")} - {None})
            info[key] = {"team": p["team"], "jerseys": {str(j) for j in js}, "batches": {bi}}

    def union(a, b, force: bool = False) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        if info[ra]["team"] != info[rb]["team"]:
            return False
        if one_per_batch and info[ra]["batches"] & info[rb]["batches"]:
            return False
        ja, jb = info[ra]["jerseys"], info[rb]["jerseys"]
        if ja and jb and not (ja & jb) and not force:
            return False
        parent[rb] = ra
        info[ra]["jerseys"] |= jb
        info[ra]["batches"] |= info[rb]["batches"]
        return True

    def boxes_px(det):
        return [v * s / 1000 for v, s in zip(det["bbox_1000"], (W, H, W, H))]

    links = {"overlap": 0, "overlap_forced_over_jersey": 0, "jersey": 0, "refused": 0, "leftover": 0}
    for bi in range(1, len(records)):
        prev, cur = records[bi - 1], records[bi]
        prev_frames = {s_["frame"]: fr for s_, fr in zip(prev["frames"], prev["parsed"]["frames"])}
        for s_, fr in zip(cur["frames"], cur["parsed"]["frames"]):
            if s_["frame"] not in prev_frames:
                continue
            pf = prev_frames[s_["frame"]]
            a = np.array([boxes_px(d) for d in pf["detections"]]).reshape(-1, 4)
            b = np.array([boxes_px(d) for d in fr["detections"]]).reshape(-1, 4)
            iou = iou_matrix(a, b)
            if not iou.size:
                continue
            for r_, c_ in zip(*linear_sum_assignment(-iou)):
                if iou[r_, c_] >= min_iou:
                    a_, b_ = (bi - 1, pf["detections"][r_]["person_id"]), (bi, fr["detections"][c_]["person_id"])
                    if union(a_, b_):
                        links["overlap"] += 1
                    elif iou[r_, c_] >= force_iou and union(a_, b_, force=True):
                        links["overlap_forced_over_jersey"] += 1
                    else:
                        links["refused"] += 1
    by_jersey = defaultdict(list)
    for key, v in (info.items() if link_jerseys else ()):
        for j in v["jerseys"]:
            if v["team"] != "referee":
                by_jersey[(v["team"], j)].append(key)
    for keys in by_jersey.values():
        for k in keys[1:]:
            if find(k) != find(keys[0]):
                links["jersey" if union(keys[0], k) else "refused"] += 1

    if link_leftovers:
        for bi in range(1, len(records)):
            for team in {v["team"] for v in info.values()} - {"referee"}:
                groups = {find(k) for k in parent if info[find(k)]["team"] == team}
                ending = [g for g in groups if bi - 1 in info[g]["batches"] and bi not in info[g]["batches"]]
                starting = [g for g in groups if bi in info[g]["batches"] and bi - 1 not in info[g]["batches"]]
                if len(ending) == 1 and len(starting) == 1 and union(ending[0], starting[0], force=True):
                    links["leftover"] += 1

    roots = sorted({find(k) for k in parent})
    gid = {r: i + 1 for i, r in enumerate(roots)}
    people = {}
    for r in roots:
        members = [k for k in parent if find(k) == r]
        counts = Counter()
        for m in members:
            counts.update(reads[m])
        if not counts:  # older prompts: one read per person per call
            counts = Counter(str(p["jersey"]) for bi, rec in enumerate(records) for p in rec["parsed"]["people"]
                             if (bi, p["person_id"]) in members and p.get("jersey") is not None)
        jersey = None
        if counts and info[r]["team"] != "referee":
            top = max(counts.values())
            jersey = "/".join(sorted((j for j, c in counts.items() if c == top), key=int))
        people[gid[r]] = {"person_id": gid[r], "team": info[r]["team"], "jersey": jersey,
                          "jersey_reads": dict(counts.most_common()), "batch_person_ids": [list(m) for m in members]}
    # two groups of one team with the same number: only the group with more reads of it keeps the number
    # (with one_per_batch they are different people; the Opus 1-minute run read #6 on a second Blue player
    # in 2 frames, against 12 reads on the real #6, and the shared key would have merged them downstream)
    shared_jersey_dropped = 0
    by_number = defaultdict(list)
    for p in people.values():
        if p["jersey"] is not None:
            by_number[(p["team"], p["jersey"])].append(p)
    for group in by_number.values():
        group.sort(key=lambda p: -sum(p["jersey_reads"].get(j, 0) for j in p["jersey"].split("/")))
        for p in group[1:]:
            p["jersey"] = None
            shared_jersey_dropped += 1
    samples = []
    holders: dict[int, int | None] = {}  # frame -> global id of Astra's ball holder (None = no / unsure)
    states: dict[int, str | None] = {}  # frame -> Astra's ball_state (held / pass / shot / loose / None)
    holder_disagree = state_disagree = 0
    for bi, r in enumerate(records):
        for s_, fr in zip(r["frames"], r["parsed"]["frames"]):
            t = s_["frame"]
            samples.append((t, [(gid[find((bi, d["person_id"]))], boxes_px(d), d.get("jersey"))
                                for d in fr["detections"]]))
            if "ball_state" in fr:
                st = fr["ball_state"]
                if t in states and states[t] != st:  # overlap frame answered differently by two calls
                    state_disagree += 1
                    st = None
                states[t] = st
            if "ball_holder" not in fr:
                continue
            h = fr["ball_holder"]
            h = gid[find((bi, h))] if h is not None and (bi, h) in parent else None
            if h is not None and people[h]["team"] == "referee":
                h = None
            if t in holders and holders[t] != h:  # overlap frame answered differently by two calls
                holder_disagree += 1
                h = None
            holders[t] = h
    stats = {"batches": len(records), "batch_people": sum(len(r["parsed"]["people"]) for r in records),
             "global_people": len(people), "links": links, "shared_jersey_dropped": shared_jersey_dropped,
             "holder_answers": {"frames": len(holders), "player": sum(1 for v in holders.values() if v is not None),
                                "null": sum(1 for v in holders.values() if v is None),
                                "overlap_disagreements": holder_disagree},
             "state_answers": {**{k: sum(1 for v in states.values() if v == k) for k in ("held", "pass", "shot", "loose")},
                               "null": sum(1 for v in states.values() if v is None),
                               "overlap_disagreements": state_disagree}}
    return samples, people, stats, holders, states


def match_and_split(samples: list, boxes: dict[int, dict[int, np.ndarray]], min_iou: float,
                    numbers: dict[int, str | None] | None = None, min_reads: int = 2, cut_metric: str = "iou",
                    ambig_margin: float = 0.0, ambig_mode: str = "continuity") -> dict:
    """Steps 1-4: vote each track by the Astra people its boxes match, split tracks where the vote changes,
    resolve two tracklets claiming one person at once. Returns tracklets (source_track, start, end, pid,
    votes), splits, conflicts, merges and match statistics.
    numbers ({pid: jersey}, optional): verify each run of one track's votes by the jersey reads on its own
    boxes. A VLM can swap the person_ids of two players inside one call (Opus, AllClip 1 minute: two Blue
    players that came close, ~450 frames), which shows up as a run voting for A while its boxes read B's
    number. Such a run (>= min_reads reads of B's number, more than of any other, A's number differs) is
    relabelled B, and runs voting B on other tracks over the same frames without reads of B's number are
    relabelled A (the other half of the swap).
    ambig_margin (> 0 enables): a match is ambiguous when another track (or another Astra box) also reaches
    min_iou and comes within ambig_margin of the matched IoU. On a sample with ambiguous matches, the
    ambiguous Astra boxes prefer the track whose nearest unambiguous votes (before / after, on that track)
    are the same person: +1 per side added to the IoU before the Hungarian is solved again. Two overlapping
    players can swap their Astra-to-track matches by IoU alone (IMG_0115 frame 2610: the summed IoU of the
    wrong pairs beat the right pairs by 0.005). Clear matches still follow IoU, so a real tracker switch
    still shows up as a vote change.
    ambig_mode="foot": instead of the votes, an ambiguous Astra box takes the track (among those within
    ambig_margin of its best IoU) whose foot point (bottom centre) is nearest to its own. The continuity rule
    also overrides a real short tracker swap that a sample happens to catch (IMG_0115 frame 11400: tracks
    14 and 27 swap two overlapping players for ~8 frames); the feet tell both cases apart (2610: 7 vs 16 px
    for the right track, 11400: 8 vs 11 px for the swapped track)."""
    def foot(b):
        return np.array([(b[0] + b[2]) / 2, b[3]])

    # 1. per sample: Hungarian on IoU, then mark ambiguous matches
    matches = []  # (t, a_ids, a_reads, t_ids, iou, rows, cols, ambiguous rows, a_xyxy, t_xyxy)
    for t, dets in samples:
        a_ids = [d[0] for d in dets]
        a_xyxy = np.array([d[1] for d in dets]).reshape(-1, 4)
        a_reads = [d[2] if len(d) > 2 else None for d in dets]
        t_ids = [tid for tid in boxes if t in boxes[tid]]
        t_xyxy = np.array([boxes[tid][t] for tid in t_ids]).reshape(-1, 4)
        iou = iou_matrix(a_xyxy, t_xyxy)
        rows, cols = linear_sum_assignment(-iou) if iou.size else ([], [])
        amb = set()
        if ambig_margin > 0:
            for r, c in zip(rows, cols):
                if iou[r, c] < min_iou:
                    continue
                rival = max([iou[r, k] for k in range(iou.shape[1]) if k != c] +
                            [iou[k, c] for k in range(iou.shape[0]) if k != r] + [0.0])
                if rival >= min_iou and iou[r, c] - rival < ambig_margin:
                    amb.add(int(r))
        matches.append([t, a_ids, a_reads, t_ids, iou, rows, cols, amb, a_xyxy, t_xyxy])
    rematched = []
    if ambig_margin > 0:
        sure: dict[int, list[tuple[int, int]]] = defaultdict(list)  # track -> [(frame, pid)] of clear matches
        for t, a_ids, _, t_ids, iou, rows, cols, amb, _, _ in matches:
            for r, c in zip(rows, cols):
                if iou[r, c] >= min_iou and r not in amb:
                    sure[t_ids[c]].append((t, a_ids[r]))
        for m in matches:
            t, a_ids, _, t_ids, iou, rows, cols, amb, a_xyxy, t_xyxy = m
            if not amb:
                continue
            score = iou.copy()
            for r in amb:
                best = iou[r].max()
                for c, tid in enumerate(t_ids):
                    if iou[r, c] < min_iou:
                        continue
                    if ambig_mode == "foot":
                        if iou[r, c] >= best - ambig_margin:  # above any IoU; the nearer foot scores higher
                            score[r, c] = 2.0 + 1.0 / (1.0 + float(np.linalg.norm(foot(a_xyxy[r]) - foot(t_xyxy[c]))) / 10.0)
                        continue
                    hist = sure.get(tid, [])
                    prev = next((p for tt, p in reversed(hist) if tt < t), None)
                    nxt = next((p for tt, p in hist if tt > t), None)
                    score[r, c] += (prev == a_ids[r]) + (nxt == a_ids[r])
            new_rows, new_cols = linear_sum_assignment(-score)
            old = {int(r): int(c) for r, c in zip(rows, cols)}
            for r, c in zip(new_rows, new_cols):
                if old.get(int(r)) != int(c) and iou[r, c] >= min_iou:
                    rematched.append({"frame": int(t), "pid": a_ids[r],
                                      "from_track": t_ids[old[int(r)]] if int(r) in old else None,
                                      "to_track": t_ids[c]})
            m[5], m[6] = new_rows, new_cols

    # 2. votes
    votes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    vote_reads: dict[tuple[int, int], str] = {}  # (track, frame) -> jersey read on the matched VLM box
    match_ious, unmatched_astra = [], 0
    for t, a_ids, a_reads, t_ids, iou, rows, cols, *_ in matches:
        matched = 0
        for r, c in zip(rows, cols):
            if iou[r, c] >= min_iou:
                votes[t_ids[c]].append((t, a_ids[r]))
                if a_reads[r] is not None:
                    vote_reads[(t_ids[c], t)] = str(a_reads[r])
                match_ious.append(float(iou[r, c]))
                matched += 1
        unmatched_astra += len(a_ids) - matched

    # 3. split tracks at vote changes -> tracklets
    runs_of: dict[int, list[dict]] = {}
    for tid in boxes:
        runs = []  # consecutive equal votes
        for t, pid in sorted(votes.get(tid, [])):
            if runs and runs[-1]["pid"] == pid:
                runs[-1]["votes"].append(t)
            else:
                runs.append({"pid": pid, "votes": [t]})
        runs_of[tid] = runs

    # 3a. jersey check of each run (see docstring)
    relabels = []
    owner = {}
    for pid, n in (numbers or {}).items():
        if n is not None and "/" not in str(n):
            owner[str(n)] = pid
    for tid, runs in runs_of.items():
        for run in runs:
            reads = Counter(vote_reads[(tid, t)] for t in run["votes"] if (tid, t) in vote_reads)
            if not reads:
                continue
            n, c = reads.most_common(1)[0]
            a, b = run["pid"], owner.get(n)
            if c < min_reads or c <= sum(reads.values()) - c or b is None or b == a or numbers.get(a) == n:
                continue
            lo, hi = run["votes"][0], run["votes"][-1]
            run["pid"] = b
            partners = []
            for tid2, runs2 in runs_of.items():
                for r2 in runs2:
                    if tid2 == tid or r2["pid"] != b or r2["votes"][-1] < lo or r2["votes"][0] > hi:
                        continue
                    if any(vote_reads.get((tid2, t)) == n for t in r2["votes"]):
                        continue
                    r2["pid"] = a
                    partners.append({"source_track": tid2, "frames": [r2["votes"][0], r2["votes"][-1]]})
            relabels.append({"source_track": tid, "frames": [lo, hi], "from_pid": a, "to_pid": b,
                             "reads": dict(reads), "partners": partners})
    for tid, runs in runs_of.items():  # relabelled runs may now repeat their neighbour's pid
        merged = []
        for run in runs:
            if merged and merged[-1]["pid"] == run["pid"]:
                merged[-1]["votes"] += run["votes"]
            else:
                merged.append(run)
        runs_of[tid] = merged

    tracklets = []  # dict(source_track, start, end, pid|None, votes)
    splits = []
    for tid, frames_of in boxes.items():
        fs = sorted(frames_of)
        runs = runs_of[tid]
        if not runs:
            tracklets.append({"source_track": tid, "start": fs[0], "end": fs[-1], "pid": None, "votes": []})
            continue
        bounds = [fs[0]]
        for prev, nxt in zip(runs, runs[1:]):
            a, b = prev["votes"][-1], nxt["votes"][0]
            cand = [t for t in fs if a < t <= b]
            if cand:
                # a track that was lost and re-linked jumps to another player at the frame after the gap: cut there first
                if cut_metric == "jump":
                    # overlapping players keep a high IoU across an ID switch; the box edges still jump (px)
                    cut = max(cand, key=lambda t: float(np.abs(frames_of[t] - frames_of[t - 1]).max())
                              if t - 1 in frames_of else 1e9)
                else:
                    cut = min(cand, key=lambda t: iou_matrix(frames_of[t - 1][None], frames_of[t][None])[0, 0]
                              if t - 1 in frames_of else -1.0)
            else:
                cut = b
            bounds.append(cut)
            splits.append({"source_track": tid, "cut_frame": int(cut), "from_pid": prev["pid"], "to_pid": nxt["pid"]})
        bounds.append(fs[-1] + 1)
        for run, s, e in zip(runs, bounds, bounds[1:]):
            seg = [t for t in fs if s <= t < e]
            if seg:
                tracklets.append({"source_track": tid, "start": seg[0], "end": seg[-1], "pid": run["pid"],
                                  "votes": run["votes"]})

    # 4. conflicts: same pid on two tracklets at once. In a swap the two tracks are cut a few
    # frames apart, so the tracklet with fewer votes loses the identity only on the overlapping
    # frames (they become an unidentified piece), not on its whole span.
    conflicts = []
    by_pid = defaultdict(list)
    for tl in tracklets:
        if tl["pid"] is not None:
            by_pid[tl["pid"]].append(tl)
    trimmed = []
    for pid, tls in by_pid.items():
        tls.sort(key=lambda tl: -len(tl["votes"]))
        kept = []
        for tl in tls:
            frames_of = boxes[tl["source_track"]]
            fs = [t for t in sorted(frames_of) if tl["start"] <= t <= tl["end"]]
            taken = {t for t in fs if any(k["start"] <= t <= k["end"] for k in kept)}
            if not taken:
                kept.append(tl)
                continue
            conflicts.append({"pid": pid, "source_track": tl["source_track"],
                              "frames": [min(taken), max(taken)], "n_frames": len(taken)})
            # contiguous pieces: free pieces keep pid only if they hold a vote, taken pieces lose it
            pieces, cur = [], []
            for t in fs:
                if cur and ((t in taken) != (cur[-1] in taken)):
                    pieces.append(cur)
                    cur = []
                cur.append(t)
            pieces.append(cur)
            for piece in pieces:
                piece_votes = [t for t in tl["votes"] if piece[0] <= t <= piece[-1]]
                keep = piece[0] not in taken and (piece_votes or len(pieces) == 1)
                trimmed.append({"source_track": tl["source_track"], "start": piece[0], "end": piece[-1],
                                "pid": pid if keep else None, "votes": piece_votes})
            tl["drop"] = True
            kept.extend(p for p in trimmed[-len(pieces):] if p["pid"] == pid)
    tracklets = [tl for tl in tracklets if not tl.get("drop")] + trimmed
    tracklets.sort(key=lambda tl: (tl["source_track"], tl["start"]))
    tracks_of_pid = defaultdict(set)
    for tl in tracklets:
        if tl["pid"] is not None:
            tracks_of_pid[tl["pid"]].add(tl["source_track"])
    merges = {pid: sorted(ts) for pid, ts in tracks_of_pid.items() if len(ts) > 1}

    return {"tracklets": tracklets, "splits": splits, "conflicts": conflicts, "merges": merges,
            "match_ious": match_ious, "unmatched_astra": unmatched_astra, "relabels": relabels,
            "rematched": rematched}


SHOT_LOOKBACK = 10  # frames (resolve_ball)


def resolve_ball(evidence: list, evidence_src: list, astra_states: dict, shot_frames_of: dict,
                 ident_xy: np.ndarray, kcol: dict, T: int) -> dict:
    """Step 7 (see the comment in main): per-frame ball state, holder, shooter and the ball's court position
    from the per-frame holder evidence (step 6) and Astra's ball_state answers."""
    def position(ident, t, search=15):
        """identity's court position at t, or the nearest frame within +-search where it is known"""
        if ident not in kcol:
            return None
        for d in range(search + 1):
            for tt in (t - d, t + d):
                if 0 <= tt < T and np.all(np.isfinite(ident_xy[tt, kcol[ident]])):
                    return ident_xy[tt, kcol[ident]]
        return None

    state = ["unknown"] * T
    holder_by_frame = [None] * T
    shooter = [None] * T  # the shooter while state == shot
    shot_origin = [None] * T  # the shooter's court position at release (for the arrow to the basket)
    holder_src = ["none"] * T
    ball_xy = np.full((T, 2), np.nan, np.float32)
    for t in range(T):
        if evidence[t] is not None:
            state[t], holder_by_frame[t], holder_src[t] = "held", evidence[t], evidence_src[t]
    ev_frames = [t for t in range(T) if evidence[t] is not None]
    bounds = ([(-1, ev_frames[0])] if ev_frames and ev_frames[0] > 0 else []) \
        + list(zip(ev_frames, ev_frames[1:])) \
        + ([(ev_frames[-1], T)] if ev_frames and ev_frames[-1] < T - 1 else [])
    if not ev_frames:
        bounds = [(-1, T)]
    gaps = []
    for t1, t2 in bounds:
        if t2 - t1 <= 1:
            continue
        a = evidence[t1] if t1 >= 0 else None
        b = evidence[t2] if t2 < T else None
        gap = list(range(t1 + 1, t2))
        shot_signal = a is not None and any(t in shot_frames_of.get(a, ()) for t in range(t1 - SHOT_LOOKBACK, t2))
        default = ("unknown" if a is None else "shot" if shot_signal else
                   "held" if b is None or a == b else "pass")
        answers = [(t, astra_states[t]) for t in gap if astra_states.get(t) in ("pass", "shot", "loose", "held")]
        cur = default
        first_informative = next(((t, st) for t, st in answers if st in ("pass", "shot")), None)
        if answers and first_informative and first_informative[0] == answers[0][0]:
            cur = first_informative[1]  # the release is at the start of the gap
        ans = dict(answers)
        seq = []
        for t in gap:
            st = ans.get(t)
            if st in ("pass", "shot", "loose"):
                cur = st
            elif st == "held" and a is not None and (b is None or a == b):
                cur = "held"
            seq.append(cur)
        release = position(a, t1) if a is not None and "shot" in seq else None
        for t, st in zip(gap, seq):
            state[t] = st
            holder_src[t] = "inherited" if st == "held" else st
            if st == "held":
                holder_by_frame[t] = a
            if st == "shot":
                shooter[t], shot_origin[t] = a, release
        if any(st == "pass" for st in seq) and a is not None and b is not None and a != b:
            pa, pb = position(a, t1), position(b, t2)
            if pa is not None and pb is not None:
                for t, st in zip(gap, seq):
                    if st == "pass":
                        w = (t - t1) / (t2 - t1)
                        ball_xy[t] = (1 - w) * pa + w * pb
        gaps.append({"from": a, "to": b, "start": gap[0], "end": gap[-1], "default": default,
                     "astra": [[t, st] for t, st in answers], "states": dict(Counter(seq))})
    for t in range(T):
        if state[t] == "held" and holder_by_frame[t] in kcol:
            ball_xy[t] = ident_xy[t, kcol[holder_by_frame[t]]]
    ball_mode = ["held" if state[t] == "held" else "interpolated" if np.all(np.isfinite(ball_xy[t])) else "hidden"
                 for t in range(T)]
    return {"state": state, "holder_by_frame": holder_by_frame, "shooter": shooter, "shot_origin": shot_origin,
            "holder_src": holder_src, "ball_xy": ball_xy, "gaps": gaps, "ball_mode": ball_mode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--astra-results", type=Path, required=True)
    parser.add_argument("--rosters", type=Path, default=HERE / "rosters.json")
    parser.add_argument("--court-dir", default="court_v14")
    parser.add_argument("--out-name", default="identity",
                        help="output directory under the job dir (e.g. identity_opus for a second VLM's results)")
    parser.add_argument("--allow-same-batch-links", action="store_true",
                        help="old behaviour: allow linking two groups that hold two people of one call (before "
                             "2026-10-16; a misread number then merged two Blue players of the Opus run)")
    parser.add_argument("--no-jersey-check", action="store_true",
                        help="do not relabel track runs whose own jersey reads name another person (before 2026-10-16)")
    parser.add_argument("--min-iou", type=float, default=0.4)
    parser.add_argument("--league", default="NBA", choices=["NBA", "FIBA"])
    parser.add_argument("--video", type=Path, default=None, help="frames to draw on (default: tracks.json video)")
    parser.add_argument("--court-keypoints", action="store_true",
                        help="AllClip: draw the court keypoints (33 vertices) and lines from <job>/court_calibration.json")
    parser.add_argument("--hide-unidentified", action="store_true",
                        help="do not draw tracks without an Astra identity (AllClip: bench / off-court people)")
    # detections are kept from confidence 0.1 for the tracker; possession / shot evidence needs the
    # notebook's detection threshold (at 0.1, ball-in-basket fired on 1,730 of 1,799 frames)
    parser.add_argument("--evidence-confidence", type=float, default=0.4)
    parser.add_argument("--min-possession-run", type=int, default=3,
                        help="RF-DETR possession evidence must last this many consecutive frames on one person")
    parser.add_argument("--smooth-window", type=int, default=31,
                        help="Savitzky-Golay window (frames, odd) on each identity's court trajectory; 0 = off")
    parser.add_argument("--mirror-x", action="store_true",
                        help="draw the court map with x -> L - x (AllClip: court x = L is the camera's left)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    job = args.job_dir
    out = job / args.out_name
    out.mkdir(exist_ok=True)
    tracks = json.loads((job / "tracks.json").read_text())
    clip = Path(tracks["video"]).stem
    W, H, T = tracks["width"], tracks["height"], len(tracks["frames"])
    records = load_astra(args.astra_results, clip)
    samples, people, batch_stats, astra_holders, astra_states = merge_batches(
        records, W, H, one_per_batch=not args.allow_same_batch_links)
    rosters = json.loads(args.rosters.read_text())

    # per-track box per frame
    boxes: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for f in tracks["frames"]:
        for p in f["people"]:
            boxes[p["track_id"]][f["frame"]] = np.array(p["xyxy"])

    numbers = None if args.no_jersey_check else {pid: p["jersey"] for pid, p in people.items()}
    ms = match_and_split(samples, boxes, args.min_iou, numbers)
    tracklets, splits, conflicts, merges = ms["tracklets"], ms["splits"], ms["conflicts"], ms["merges"]
    match_ious, unmatched_astra = ms["match_ious"], ms["unmatched_astra"]

    # 5. identities
    identities = {}
    for pid, p in people.items():
        team, jersey = p["team"], p.get("jersey")
        if team == "referee":
            key, name, in_roster = f"referee:{pid}", None, None
        elif jersey is not None:
            key = f"{team}#{jersey}"
            name = rosters["teams"][team]["players"].get(str(jersey))
            in_roster = name is not None
        else:
            key, name, in_roster = f"{team}:{clip}:{pid}", None, None
        identities[pid] = {"identity": key, "team": team, "jersey": jersey, "name": name, "in_roster": in_roster}
    for i, tl in enumerate(tracklets):
        tl["tracklet_id"] = i
        tl.update(identities.get(tl["pid"], {"identity": None, "team": None, "jersey": None, "name": None,
                                             "in_roster": None}))

    # per-frame lookup (source_track, frame) -> tracklet
    lookup = {}
    for tl in tracklets:
        for t in range(tl["start"], tl["end"] + 1):
            lookup[(tl["source_track"], t)] = tl

    # 6. ball-holder evidence per frame, at identity level (tracks fragment; identities do not):
    #    Astra's answer on its sampled frames (1 fps) > RF-DETR's player-in-possession class (every frame,
    #    highest confidence if several). Astra null = "no one / unsure" = no evidence from Astra. Frames
    #    without evidence are resolved in step 7. No ball-in-box geometry: a player can hold the ball
    #    outside his box (raised over the head).
    evidence, evidence_src = [], []  # per frame: identity with direct evidence of possession, or None
    check = Counter()
    for f in tracks["frames"]:
        t = f["frame"]
        poss = [(p["confidence"], lookup[(p["track_id"], t)]["identity"]) for p in f["people"]
                if p["class"] == "player-in-possession" and p["confidence"] >= args.evidence_confidence
                and (p["track_id"], t) in lookup
                and lookup[(p["track_id"], t)]["identity"] and lookup[(p["track_id"], t)]["team"] != "referee"]
        rf = max(poss)[1] if poss else None
        astra = identities[astra_holders[t]]["identity"] if astra_holders.get(t) is not None else None
        if t in astra_holders:
            check["astra_player" if astra else "astra_null"] += 1
            if astra and rf:
                check["astra_and_rfdetr_agree" if astra == rf else "astra_overrides_rfdetr"] += 1
            if not astra and rf:
                check["astra_null_rfdetr_player"] += 1
        if astra:
            evidence.append(astra)
            evidence_src.append("astra")
        elif rf:
            evidence.append(rf)
            evidence_src.append("rfdetr_possession")
        else:
            evidence.append(None)
            evidence_src.append(None)
    # RF-DETR's possession class flickers between two players standing close (1-2 frame "passes" back and
    # forth, 2026-09-28 run): a run of RF-DETR evidence shorter than --min-possession-run frames is not
    # evidence. Astra's answers are always kept.
    dropped_runs = 0
    t = 0
    while t < len(evidence):
        if evidence_src[t] != "rfdetr_possession":
            t += 1
            continue
        u = t
        while u + 1 < len(evidence) and evidence_src[u + 1] == "rfdetr_possession" and evidence[u + 1] == evidence[t]:
            u += 1
        if u - t + 1 < args.min_possession_run:
            for k in range(t, u + 1):
                evidence[k], evidence_src[k] = None, None
            dropped_runs += 1
        t = u + 1
    check["rfdetr_runs_dropped_short"] = dropped_runs

    # shot signals (RF-DETR): the shooter's own shot classes, and ball-in-basket detections
    shot_frames_of = defaultdict(set)  # identity -> frames where its box is a shooting class
    for f in tracks["frames"]:
        for p in f["people"]:
            tl = lookup.get((p["track_id"], f["frame"]))
            if tl and tl["identity"] and p["class"] in ("player-jump-shot", "player-layup-dunk") \
                    and p["confidence"] >= args.evidence_confidence:
                shot_frames_of[tl["identity"]].add(f["frame"])
    # court trajectories per identity
    court = np.load(job / args.court_dir / "court_xy.npz")
    col = {int(t): i for i, t in enumerate(court["track_ids"])}
    keys = sorted({tl["identity"] for tl in tracklets if tl["identity"]})
    kcol = {k: i for i, k in enumerate(keys)}
    ident_xy = np.full((T, len(keys), 2), np.nan, np.float32)
    for (tid, t), tl in lookup.items():
        if tl["identity"] and tid in col:
            ident_xy[t, kcol[tl["identity"]]] = court["cleaned_xy"][t, col[tid]]
    # light smoothing of each identity's court trajectory (Savitzky-Golay over every visible stretch that
    # is long enough; shorter stretches are left as they are). The ball follows the smoothed holder.
    if args.smooth_window >= 5:
        from scipy.signal import savgol_filter
        win = args.smooth_window | 1
        for k in range(len(keys)):
            ok = np.isfinite(ident_xy[:, k]).all(axis=1)
            t = 0
            while t < T:
                if not ok[t]:
                    t += 1
                    continue
                u = t
                while u + 1 < T and ok[u + 1]:
                    u += 1
                if u - t + 1 >= win:
                    ident_xy[t:u + 1, k] = savgol_filter(ident_xy[t:u + 1, k], win, 2, axis=0)
                t = u + 1
    # 7. ball state per frame: held / pass / shot / loose / unknown (the definitions given to Astra):
    #   held  a player has the ball (holding, overhead, dribbling)       -> ball at the holder's feet
    #   pass  passed, the ball has left the passer's hands, until caught  -> ball interpolated passer -> receiver
    #   shot  shot, the ball has left the shooter's hands, until it goes in / is touched / comes off the rim
    #   loose none of the above, no team controls the ball              -> ball hidden (shot / loose too:
    #                                                                     its court position in the air is unknown)
    # Frames with holder evidence (step 6) are held. A gap without evidence between holder A (t1) and holder
    # B (t2) starts in a default state -- shot if A shows an RF-DETR shooting class within SHOT_LOOKBACK
    # frames before or inside the gap, else pass if A != B, else held by A -- and then follows Astra's
    # ball_state answers inside the gap (1 fps): each answer holds from its frame on; the first answer, if
    # it is pass or shot, also covers the gap from its start (the release is the start of the gap). An
    # Astra "held" without a usable holder keeps the current state. Before the first evidence: unknown;
    # after the last: held by the last holder unless Astra says otherwise.
    # (RF-DETR ball-in-basket is not used: on this footage it fires on players' shoes.)
    rb = resolve_ball(evidence, evidence_src, astra_states, shot_frames_of, ident_xy, kcol, T)
    state, holder_by_frame, shooter, shot_origin = rb["state"], rb["holder_by_frame"], rb["shooter"], rb["shot_origin"]
    holder_src, ball_xy, gaps, ball_mode = rb["holder_src"], rb["ball_xy"], rb["gaps"], rb["ball_mode"]
    src_count = Counter(holder_src)
    # agreement of the final state with Astra's own answers (frames where Astra answered a state)
    state_check = Counter()
    for t, st in astra_states.items():
        if st in ("held", "pass", "shot", "loose") and t < T:
            state_check["agree" if state[t] == st else f"astra_{st}_final_{state[t]}"] += 1
    (out / "ball_holder.json").write_text(json.dumps(
        {"frames": [{"frame": t, "state": state[t], "holder": holder_by_frame[t], "shooter": shooter[t],
                     "source": holder_src[t], "ball": ball_mode[t],
                     "ball_court_xy": None if not np.all(np.isfinite(ball_xy[t])) else ball_xy[t].round(2).tolist()}
                    for t in range(T)],
         "gaps": gaps}, ensure_ascii=False))
    np.savez_compressed(out / "court_identities.npz", identities=np.array(keys), xy=ident_xy, ball_xy=ball_xy,
                        ball_holder=np.array([h or "" for h in holder_by_frame]), ball_state=np.array(state),
                        ball_holder_source=np.array(holder_src), ball_mode=np.array(ball_mode))

    n_players = sum(1 for p in people.values() if p["team"] != "referee")
    summary = {
        "clip": clip,
        "astra": {"people": len(people), "players": n_players,
                  "jerseys_read": sum(1 for p in people.values() if p["team"] != "referee" and p.get("jersey")),
                  "jerseys_in_roster": sum(1 for i in identities.values() if i["in_roster"]),
                  "jerseys_not_in_roster": [i["identity"] for i in identities.values() if i["in_roster"] is False],
                  "sampled_frames": len({t for t, _ in samples}),
                  "elapsed_s": round(sum(r["elapsed_s"] for r in records), 1), **batch_stats},
        "matching": {"matched": len(match_ious), "unmatched_astra_boxes": unmatched_astra,
                     "iou_median": round(float(np.median(match_ious)), 3) if match_ious else None,
                     "iou_min": round(float(min(match_ious)), 3) if match_ious else None},
        "botsort_tracks": len(boxes),
        "tracklets": len(tracklets),
        "identities": len(keys),
        "splits": splits, "jersey_relabels": ms["relabels"],
        "merges": {str(pid): ts for pid, ts in merges.items()},
        "conflicts": conflicts,
        "unidentified_tracklets": [{"source_track": tl["source_track"], "start": tl["start"], "end": tl["end"]}
                                   for tl in tracklets if tl["identity"] is None],
        "ball_holder": {"frames_by_source": dict(src_count), "on_astra_frames": dict(check),
                        "state_frames": dict(Counter(state)),
                        "state_vs_astra": dict(state_check),
                        "gaps": len(gaps),
                        "ball_frames": dict(Counter(ball_mode))},
    }
    summary["seconds"] = {"sync": round(time.perf_counter() - t_start, 2)}
    (out / "identities.json").write_text(json.dumps({"summary": summary, "people": list(people.values()),
                                                     "tracklets": tracklets}, indent=1, ensure_ascii=False))
    t_render0 = time.perf_counter()

    # render: broadcast with team-coloured boxes + court map with jersey labels
    def color_of(tl) -> sv.Color:
        if tl is None or tl["team"] is None:
            return sv.Color.from_hex("#FFFFFF")
        if tl["team"] == "referee":
            return sv.Color.from_hex(rosters["referee_color"])
        return sv.Color.from_hex(rosters["teams"][tl["team"]]["color"])

    def label_of(tl) -> str:
        if tl is None or tl["identity"] is None:
            return "?"
        if tl["team"] == "referee":
            return "REF"
        if tl["jersey"] is None:
            return "#?"
        has_roster = bool(rosters["teams"].get(tl["team"], {}).get("players"))
        return f"#{tl['jersey']} {tl['name'] or '(not in roster)'}" if has_roster else f"#{tl['jersey']}"

    config = CourtConfiguration(league=League[args.league], measurement_unit=MeasurementUnit.FEET)
    Lc = config.court_length
    Wc = config.court_width
    identities_by_key = {v["identity"]: v for v in identities.values()}
    disp = (lambda xy: np.c_[Lc - xy[..., 0], xy[..., 1]].reshape(xy.shape)) if args.mirror_x else (lambda xy: xy)  # noqa: E731
    court0 = draw_court(config=config)
    ch, cw = court0.shape[:2]
    vh = int(round(H * cw / W))
    scale = cw / W
    kp_px = line_px = None
    if args.court_keypoints:
        from allclip_court_fit import analytic_template
        from fisheye_geometry import CourtMapper, Geometry
        cal = json.loads((job / "court_calibration.json").read_text())
        mapper = CourtMapper(Geometry.load(job / "geometry.json"), np.array(cal["h_gnomonic_to_court"]))
        line_px = (mapper.court_to_panorama(analytic_template(*cal["court_size_ft"], 0.1)) * scale).astype(int)
        kp_px = mapper.court_to_panorama(np.array(config.vertices)) * scale
    with sv.VideoSink(str(out / "identified.mp4"), sv.VideoInfo(width=cw, height=vh + ch, fps=tracks["fps"])) as sink:
        for f, frame in zip(tracks["frames"], sv.get_video_frames_generator(str(args.video or tracks["video"]))):
            t = f["frame"]
            top = cv2.resize(frame, (cw, vh))
            if kp_px is not None:
                ok = (line_px[:, 0] >= 0) & (line_px[:, 0] < cw) & (line_px[:, 1] >= 0) & (line_px[:, 1] < vh)
                top[line_px[ok, 1], line_px[ok, 0]] = (255, 0, 255)
                for i, (x, y) in enumerate(kp_px):
                    if 0 <= x < cw and 0 <= y < vh:
                        cv2.circle(top, (int(x), int(y)), 5, (147, 20, 255), -1)
                        cv2.circle(top, (int(x), int(y)), 5, (255, 255, 255), 1)
                        cv2.putText(top, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cmap = court0.copy()
            holder = holder_by_frame[t]  # identity (step 6)
            if np.all(np.isfinite(ball_xy[t])) and ball_mode[t] == "held":  # ring behind the holder's point
                cmap = draw_points_on_court(config=config, xy=disp(ball_xy[t][None]),
                                            fill_color=BALL_COLOR, size=46, court=cmap)
            for p in f["people"]:
                tl = lookup.get((p["track_id"], t))
                if args.hide_unidentified and (tl is None or tl["identity"] is None):
                    continue
                is_holder = holder is not None and tl is not None and tl["identity"] == holder
                is_shooter = state[t] == "shot" and tl is not None and tl["identity"] == shooter[t]
                c = SHOT_COLOR if is_shooter else HOLDER_COLOR if is_holder else color_of(tl)
                x1, y1, x2, y2 = (int(v * scale) for v in p["xyxy"])
                cv2.rectangle(top, (x1, y1), (x2, y2), c.as_bgr(), 4 if (is_holder or is_shooter) else 2)
                text = ("SHOT " if is_shooter else "BALL " if is_holder else "") + label_of(tl)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(top, (x1, y2), (x1 + tw + 6, y2 + th + 8), c.as_bgr(), -1)
                cv2.putText(top, text, (x1 + 3, y2 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color(c), 2)
                xy = court["cleaned_xy"][t, col[p["track_id"]]]
                if tl is not None and tl["identity"] in kcol:  # the smoothed identity trajectory
                    xy = ident_xy[t, kcol[tl["identity"]]]
                if np.all(np.isfinite(xy)) and np.all(np.isfinite(court["raw_xy"][t, col[p["track_id"]]])):
                    short = "R" if tl and tl["team"] == "referee" else (str(tl["jersey"]) if tl and tl["jersey"] else "?")
                    cmap = draw_points_on_court(config=config, xy=disp(xy[None]), labels=[short], fill_color=c,
                                                text_color=sv.Color.from_hex("#111111") if sum(c.as_rgb()) > 500 else sv.Color.WHITE,
                                                court=cmap)
            if np.all(np.isfinite(ball_xy[t])) and ball_mode[t] == "interpolated":  # pass in flight
                cmap = draw_points_on_court(config=config, xy=disp(ball_xy[t][None]), fill_color=BALL_COLOR,
                                            edge_color=sv.Color.WHITE, size=18, court=cmap)
            if state[t] == "shot" and shot_origin[t] is not None:  # arrow from the release point to the basket
                o = shot_origin[t]
                rims = [np.array([rim_x, Wc / 2]) for rim_x in (RIM_FROM_BASELINE_FT, Lc - RIM_FROM_BASELINE_FT)]
                rim = min(rims, key=lambda r: np.linalg.norm(r - o))
                p0, p1 = (disp(np.array([o, rim])) * 20 + 50).astype(int)  # draw_court scale / padding
                cv2.arrowedLine(cmap, tuple(int(v) for v in p0), tuple(int(v) for v in p1), SHOT_COLOR.as_bgr(), 6,
                                tipLength=0.08)
            ball_c = SHOT_COLOR if state[t] == "shot" else BALL_COLOR
            if f["ball"]:
                bx1, by1, bx2, by2 = (int(v * scale) for v in f["ball"]["xyxy"])
                pad = 3
                cv2.rectangle(top, (bx1 - pad, by1 - pad), (bx2 + pad, by2 + pad), ball_c.as_bgr(), 2)
                cv2.putText(top, "ball", (bx1 - pad, by1 - pad - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ball_c.as_bgr(), 1)
            frame_out = np.vstack([top, cmap])
            who = holder if state[t] == "held" else shooter[t] if state[t] == "shot" else None
            who_tl = {"identity": who, **identities_by_key.get(who, {})} if who else None
            text = STATE_LABELS[state[t]] + (f"  {label_of(who_tl)}" if who_tl and who_tl.get("team") else "")
            patch = state_patch(text, STATE_COLORS[state[t]])
            ph, pw = patch.shape[:2]
            frame_out[16:16 + ph, 16:16 + pw] = patch
            sink.write_frame(frame_out)
    summary["seconds"]["render"] = round(time.perf_counter() - t_render0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
