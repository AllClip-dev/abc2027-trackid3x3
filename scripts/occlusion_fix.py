#!/usr/bin/env python3
"""重なり（他の選手の枠と IoU >= --occl-iou）の最中に枠の高さが急に変わった区間を，Astra に枠を描き直してもらって直す．

  segments: sync_and_render.py の ident_boxes.json から区間を探す（合図 = 重なり中に高さが前フレームから
            --dh 以上変わる．区間 = 合図から，高さが合図直前の高さの ±dh/2 に戻るまで（最大 --max-len フレーム））．
            同じ身元で重なる区間はまとめる．区間ごとに，関係する選手（対象 + 重なっている選手）の枠を全フレームで
            囲む範囲を切り抜いて拡大し，参照フレーム（区間の直前，対象に赤枠）+ --step フレームおきの画像を
            <out>/<区間名>/ に書き，manifest.json を作る．
  run:      区間ごとに 1 回 Astra（codex exec）を呼び，各画像での対象選手の全身の枠と足元の点を返してもらう．
            --root は AGENTS.md を含むディレクトリの外に置く（codex がそれも読むため）．
  boxes:    Astra の答えを動画の画素座標に戻し，画像の間は線形補間し，トラッカーの枠と足元がコート上で
            --min-move-cm 以上離れたフレームだけ置き換える枠を JSON（{ラベル: {フレーム: [x1,y1,x2,y2]}}）に書く．
            sync_and_render.py --override-boxes に渡す．

  python occlusion_fix.py segments --ident-boxes ident_boxes.json --video v.mp4 --out work/occl_IMG_0104
  python occlusion_fix.py run --root work/occl_IMG_0104 --prompt-file prompts/astra_occlusion_fix.txt
  python occlusion_fix.py boxes --root ... --ident-boxes ident_boxes.json --court-registration reg.csv --out fix.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

QUOTA_MARKERS = ("out of credits", "usage limit", "401 Unauthorized", "sign in again")


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def load_ident(path: Path) -> dict[int, dict[str, list]]:
    return {int(t): v for t, v in json.loads(path.read_text()).items()}


def find_segments(ident: dict, occl_iou: float, dh: float, max_len: int, gap: int,
                  trigger: str = "height") -> list[dict]:
    """trigger: height = 高さの変化率 >= dh, 終了 = 高さが合図直前の ±dh/2 に戻る．
    bottom = 下辺の変化 >= dh * 直前の高さ, 終了 = 下辺が直前の動き（5 フレームの平均速度）の延長から ±dh/2 * 高さ に戻る，または重なりが解ける（IoU < 0.2）．
    bottom_still_top = bottom に加えて，上辺の変化 < dh/2 * 高さ（ジャンプ = 上辺も下辺も動く，を除く）．"""
    raw = []
    for t in sorted(ident):
        for lab, v in ident[t].items():
            prev = ident.get(t - 1, {}).get(lab)
            if prev is None:
                continue
            others = [(l2, iou(v, w)) for l2, w in ident[t].items() if l2 != lab]
            if not others or max(o[1] for o in others) < occl_iou:
                continue
            h0 = prev[3] - prev[1]
            if trigger == "height":
                if abs((v[3] - v[1]) / h0 - 1) < dh:
                    continue
            else:
                if abs(v[3] - prev[3]) < dh * h0:
                    continue
                if trigger == "bottom_still_top" and abs(v[1] - prev[1]) >= dh / 2 * h0:
                    continue
                back = ident.get(t - 6, {}).get(lab)
                vel = (prev[3] - back[3]) / 5 if back is not None else 0.0
            end = t
            for k in range(t, t + max_len + 1):
                w = ident.get(k, {}).get(lab)
                if w is None:
                    break
                end = k
                if k > t:
                    if trigger == "height":
                        if abs((w[3] - w[1]) / h0 - 1) < dh / 2:
                            break
                    elif abs(w[3] - (prev[3] + vel * (k - t + 1))) < dh / 2 * h0:
                        break
                    elif max([iou(w, x) for l2, x in ident[k].items() if l2 != lab] + [0.0]) < 0.2:
                        break  # 重なりが解けた
            raw.append({"label": lab, "start": t, "end": end})
    segs = []
    for s in sorted(raw, key=lambda s: (s["label"], s["start"])):
        if segs and segs[-1]["label"] == s["label"] and s["start"] <= segs[-1]["end"] + gap:
            segs[-1]["end"] = max(segs[-1]["end"], s["end"])
        else:
            segs.append(dict(s))
    return segs


def cmd_segments(a) -> None:
    ident = load_ident(a.ident_boxes)
    segs = find_segments(ident, a.occl_iou, a.dh, a.max_len, a.gap, a.trigger)
    a.out.mkdir(parents=True, exist_ok=True)
    reuse = {}  # 既存の区間（Astra 回答済み）が同じ身元で新しい区間を丸ごと含むなら，Astra を呼ばずにその答えを使う
    for root in a.reuse_root or []:
        for mp in sorted(root.glob("*/manifest.json")):
            m = json.loads(mp.read_text())
            src = mp.parent
            if m.get("reuse"):  # 再利用の区間は，答えを持つ元の区間をたどる（範囲は再利用側のもの）
                src = Path(m["reuse"])
            if (src / "manifest.json").exists() and "frames" in json.loads((src / "manifest.json").read_text()):
                reuse.setdefault(m["label"], []).append({**m, "src": str(src.resolve())})
    cap = cv2.VideoCapture(str(a.video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{len(segs)} segments, {sum(s['end'] - s['start'] + 1 for s in segs)} frames")
    for s in segs:
        lab, t0, t1 = s["label"], s["start"], s["end"]
        cover = [m for m in reuse.get(lab, []) if m["start"] <= t0 and m["end"] >= t1]
        if cover:
            name = f"{lab}_{t0:06d}_{t1:06d}"
            (a.out / name).mkdir(exist_ok=True)
            (a.out / name / "manifest.json").write_text(json.dumps({
                "clip": name, "label": lab, "start": t0, "end": t1,
                "reuse": cover[0]["src"]}, indent=1))
            continue
        ref = t0 - 1  # 参照: 区間の直前で，対象がいちばん重なっていないフレーム（最大 30 フレーム前まで）
        best = None
        for k in range(t0 - 1, max(-1, t0 - 31), -1):
            v = ident.get(k, {}).get(lab)
            if v is None:
                break
            ov = max([iou(v, w) for l2, w in ident[k].items() if l2 != lab] + [0.0])
            if best is None or ov < best[0]:
                best = (ov, k)
            if ov < 0.2:
                break
        if best:
            ref = best[1]
        frames = [ref] + list(range(t0, t1 + 1, a.step))
        if frames[-1] != t1:
            frames.append(t1)
        # 関係する選手: 区間中に対象と IoU > 0.1 で重なる選手
        involved = {lab}
        for k in range(t0, t1 + 1):
            v = ident[k][lab]
            involved |= {l2 for l2, w in ident[k].items() if l2 != lab and iou(v, w) > 0.1}
        boxes = [ident[k][l][:4] for k in frames for l in involved if l in ident.get(k, {})]
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * (1 + a.margin)
        cx1, cy1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        cx2, cy2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        scale = a.crop_size / max(cx2 - cx1, cy2 - cy1)
        name = f"{lab}_{t0:06d}_{t1:06d}"
        d = a.out / name
        d.mkdir(exist_ok=True)
        files = []
        for i, k in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = cap.read()
            if not ok:
                raise SystemExit(f"cannot read frame {k}")
            crop = cv2.resize(img[cy1:cy2, cx1:cx2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if i == 0:
                v = ident[k][lab]
                p1 = (int((v[0] - cx1) * scale), int((v[1] - cy1) * scale))
                p2 = (int((v[2] - cx1) * scale), int((v[3] - cy1) * scale))
                cv2.rectangle(crop, p1, p2, (0, 0, 255), 2)
            fn = f"{i:02d}_f{k:06d}.jpg"
            cv2.imwrite(str(d / fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            files.append({"file_name": fn, "frame": k, "reference": i == 0})
        (d / "manifest.json").write_text(json.dumps({
            "clip": name, "label": lab, "start": t0, "end": t1, "involved": sorted(involved),
            "crop_xyxy": [cx1, cy1, cx2, cy2], "scale": scale, "frames": files}, indent=1))
    (a.out / "segments.json").write_text(json.dumps(segs, indent=1))


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in answer")
    return json.loads(m.group(0))


def cmd_run(a) -> None:
    root = a.root.resolve()
    os.chdir(root)
    template = a.prompt_file.read_text()
    results = root / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            if r.get("returncode") == 0 and r.get("parsed"):
                done.add(r["clip"])
    for mp in sorted(root.glob("*/manifest.json")):
        m = json.loads(mp.read_text())
        if m["clip"] in done or m.get("reuse"):
            continue
        n = len(m["frames"]) - 1
        prompt = template.replace("{N}", str(n)).replace("{JERSEY}", m["label"][1:]).replace(
            "{TEAM}", m["label"][0])
        out_txt = root / f".{m['clip']}.answer.txt"
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-m", a.model,
               "-c", f"model_reasoning_effort={a.reasoning_effort}"]
        for f in m["frames"]:
            cmd += ["-i", str(mp.parent / f["file_name"])]
        cmd += ["-o", str(out_txt), "-"]
        started = time.time()
        proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True)
        rec = {"clip": m["clip"], "model": a.model, "reasoning_effort": a.reasoning_effort,
               "elapsed_s": round(time.time() - started, 1), "returncode": proc.returncode}
        if proc.returncode != 0:
            rec["error"] = proc.stderr[-2000:]
        else:
            raw = out_txt.read_text() if out_txt.exists() else ""
            rec["raw_answer"] = raw
            try:
                rec["parsed"] = extract_json(raw)
                if len(rec["parsed"].get("frames", [])) != n:
                    rec["schema_error"] = f"expected {n} frames"
            except Exception as exc:  # noqa: BLE001
                rec["parsed"], rec["parse_error"] = None, str(exc)
        print(f"{m['clip']}: rc={proc.returncode} {rec['elapsed_s']}s {rec.get('schema_error') or rec.get('parse_error') or ''}", flush=True)
        with results.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if proc.returncode != 0 and any(q in proc.stderr for q in QUOTA_MARKERS):
            raise SystemExit(75)


def cmd_boxes(a) -> None:
    ident = load_ident(a.ident_boxes)
    reg = list(csv.DictReader(a.court_registration.open()))
    Hm, _ = cv2.findHomography(np.array([[float(r["image_x"]), float(r["image_y"])] for r in reg], np.float32),
                               np.array([[float(r["court_x"]), float(r["court_y"])] for r in reg], np.float32))

    def court(b):
        return cv2.perspectiveTransform(np.array([[[(b[0] + b[2]) / 2, b[3]]]], np.float32), Hm)[0, 0]

    recs = {}
    reuse_recs: dict = {}
    res = a.root / "results.jsonl"
    for line in (res.read_text().splitlines() if res.exists() else []):
        r = json.loads(line)
        if r.get("returncode") == 0 and r.get("parsed") and not r.get("schema_error"):
            recs[r["clip"]] = r
    out: dict[str, dict[str, list]] = {}
    stats = {"segments": 0, "reused": 0, "answered": 0, "keyframes": 0, "invisible": 0, "replaced": 0, "kept": 0}
    for mp in sorted(a.root.glob("*/manifest.json")):
        m = json.loads(mp.read_text())
        stats["segments"] += 1
        span = (m["start"], m["end"])
        if m.get("reuse"):  # 既存の区間の答えを使い，置き換えはこの区間の範囲だけ
            rd = Path(m["reuse"])
            m = json.loads((rd / "manifest.json").read_text())
            if rd.parent not in reuse_recs:
                reuse_recs[rd.parent] = {}
                for line in (rd.parent / "results.jsonl").read_text().splitlines():
                    r = json.loads(line)
                    if r.get("returncode") == 0 and r.get("parsed") and not r.get("schema_error"):
                        reuse_recs[rd.parent][r["clip"]] = r
            r = reuse_recs[rd.parent].get(m["clip"])
            stats["reused"] += 1
        else:
            r = recs.get(m["clip"])
        if r is None:
            continue
        stats["answered"] += 1
        cx1, cy1, _, _ = m["crop_xyxy"]
        side_w = (m["crop_xyxy"][2] - cx1) * m["scale"]
        side_h = (m["crop_xyxy"][3] - cy1) * m["scale"]
        keys = {}
        for f, ans in zip(m["frames"][1:], r["parsed"]["frames"]):
            stats["keyframes"] += 1
            b = ans.get("bbox_1000")
            if not ans.get("visible", True) or not b or len(b) != 4:
                stats["invisible"] += 1
                continue
            # 1000 分率（切り抜き画像）→ 動画の画素
            box = [cx1 + b[0] / 1000 * side_w / m["scale"], cy1 + b[1] / 1000 * side_h / m["scale"],
                   cx1 + b[2] / 1000 * side_w / m["scale"], cy1 + b[3] / 1000 * side_h / m["scale"]]
            ft = ans.get("foot_1000")
            if a.use_foot and ft and len(ft) == 2:  # 足元の点を枠の下辺中央に合わせる（幅と高さは枠のまま）
                fx, fy = cx1 + ft[0] / 1000 * side_w / m["scale"], cy1 + ft[1] / 1000 * side_h / m["scale"]
                w, h = box[2] - box[0], box[3] - box[1]
                box = [fx - w / 2, fy - h, fx + w / 2, fy]
            keys[f["frame"]] = np.array(box)
        ks = sorted(keys)
        lab = m["label"]
        for t in range(span[0], span[1] + 1):
            if t in keys:
                box = keys[t]
            else:
                lo = max([k for k in ks if k < t], default=None)
                hi = min([k for k in ks if k > t], default=None)
                if lo is None or hi is None:
                    continue
                w = (t - lo) / (hi - lo)
                box = (1 - w) * keys[lo] + w * keys[hi]
            cur = ident.get(t, {}).get(lab)
            if cur is None:
                continue
            if float(np.linalg.norm(court(box) - court(cur[:4]))) >= a.min_move_cm:
                out.setdefault(lab, {})[str(t)] = [round(float(v), 1) for v in box]
                stats["replaced"] += 1
            else:
                stats["kept"] += 1
    a.out.write_text(json.dumps(out))
    print(json.dumps(stats))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segments")
    s.add_argument("--ident-boxes", type=Path, required=True)
    s.add_argument("--video", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--occl-iou", type=float, default=0.5)
    s.add_argument("--dh", type=float, default=0.1, help="合図: 前フレームからの高さの変化率")
    s.add_argument("--max-len", type=int, default=45)
    s.add_argument("--trigger", choices=["height", "bottom", "bottom_still_top"], default="height",
                   help="合図: 高さの変化 / 下辺の変化 / 下辺の変化かつ上辺はほぼ動かない")
    s.add_argument("--reuse-root", type=Path, nargs="+", help="Astra 回答済みの segments の出力（含まれる区間は再利用．複数可）")
    s.add_argument("--gap", type=int, default=5, help="同じ身元の区間がこのフレーム数以内なら 1 つにまとめる")
    s.add_argument("--step", type=int, default=5, help="Astra に見せる画像の間隔（フレーム）")
    s.add_argument("--margin", type=float, default=0.6, help="切り抜き範囲を関係する選手の枠の外側へ広げる割合")
    s.add_argument("--crop-size", type=int, default=768, help="切り抜きを拡大したときの長辺（px）")
    r = sub.add_parser("run")
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--prompt-file", type=Path, required=True)
    r.add_argument("--model", default="gpt-6-astra")
    r.add_argument("--reasoning-effort", default="medium")
    b = sub.add_parser("boxes")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--ident-boxes", type=Path, required=True)
    b.add_argument("--court-registration", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--min-move-cm", type=float, default=30.0)
    b.add_argument("--use-foot", action="store_true", help="枠の下辺中央を Astra の足元の点に合わせる")
    a = ap.parse_args()
    {"segments": cmd_segments, "run": cmd_run, "boxes": cmd_boxes}[a.cmd](a)


if __name__ == "__main__":
    main()
