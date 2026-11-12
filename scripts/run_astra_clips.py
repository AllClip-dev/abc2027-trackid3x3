#!/usr/bin/env python3
"""Stage 3 of basketball_demo: one Astra call per clip via `codex exec` (runs on the Mac).

For every <frames-root>/<clip>/manifest.json (from extract_astra_frames.py) not yet done, sends
the clip's frames with prompts/astra_reid_team_jersey.txt and appends one record to
<frames-root>/results.jsonl (+ token usage to usage.jsonl). Resumable: a clip with a successful,
schema-valid record is skipped.

Codex runs with cwd = --frames-root; keep that directory outside any tree with an AGENTS.md, which
Codex would also read. Stops with exit 75 on out-of-credits / usage-limit / 401.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUOTA_MARKERS = ("out of credits", "usage limit", "401 Unauthorized", "sign in again")
EXIT_QUOTA = 75
# defaults = sample clips; TrackID3x3 runs pass --team-a/--team-b (see main)
TEAMS = {
    "TEAM_A": "Boston Celtics", "TEAM_A_COLOR": "white jerseys with green numbers",
    "TEAM_B": "New York Knicks", "TEAM_B_COLOR": "blue jerseys with orange and white numbers",
}
VALID_TEAMS = {TEAMS["TEAM_A"], TEAMS["TEAM_B"], "referee"}


def render_prompt(template: str, n: int, interval_s: float = 1.0) -> str:
    text = template.replace("{N_LAST}", str(n - 1)).replace("{N}", str(n))
    text = text.replace("{INTERVAL}", f"{interval_s:g}")
    for key, value in TEAMS.items():
        text = text.replace("{" + key + "}", value)
    return text


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip()).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def schema_error(parsed: dict, n: int, require_holder: bool = False, require_det_jersey: bool = False,
                 require_state: bool = False, require_ball_box: bool = False) -> str | None:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("people"), list) \
            or not isinstance(parsed.get("frames"), list):
        return "missing 'people' or 'frames' list"
    ids = set()
    for person in parsed["people"]:
        if not isinstance(person.get("person_id"), int):
            return f"bad person_id {person.get('person_id')!r}"
        if person.get("team") not in VALID_TEAMS:
            return f"person {person['person_id']}: bad team {person.get('team')!r}"
        if person.get("jersey") is not None and not re.fullmatch(r"\d{1,2}", str(person["jersey"])):
            return f"person {person['person_id']}: bad jersey {person.get('jersey')!r}"
        ids.add(person["person_id"])
    if len(parsed["frames"]) != n:
        return f"expected {n} frames, got {len(parsed['frames'])}"
    for frame in parsed["frames"]:
        if require_state:
            if frame.get("ball_state", "missing") not in ("held", "pass", "shot", "loose", None):
                return f"frame {frame.get('frame')}: bad ball_state {frame.get('ball_state', 'missing')!r}"
        if require_ball_box:  # the 4v4 ball prompt: the match ball's box or null on every frame
            if "ball_bbox_1000" not in frame:
                return f"frame {frame.get('frame')}: missing ball_bbox_1000"
            bb = frame["ball_bbox_1000"]
            if bb is not None and not (isinstance(bb, list) and len(bb) == 4
                                       and all(isinstance(v, (int, float)) for v in bb)):
                return f"frame {frame.get('frame')}: bad ball_bbox_1000 {bb!r}"
        if require_holder:
            if "ball_holder" not in frame:
                return f"frame {frame.get('frame')}: missing ball_holder"
            if frame["ball_holder"] is not None and frame["ball_holder"] not in ids:
                return f"frame {frame.get('frame')}: ball_holder {frame['ball_holder']!r} is not a person_id"
        seen = set()
        for det in frame.get("detections", []):
            pid = det.get("person_id")
            if pid not in ids:
                return f"frame {frame.get('frame')}: unknown person_id {pid!r}"
            if pid in seen:
                return f"frame {frame.get('frame')}: person_id {pid} twice"
            seen.add(pid)
            bbox = det.get("bbox_1000")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                return f"frame {frame.get('frame')}: bad bbox_1000 {bbox!r}"
            if require_det_jersey:
                if "jersey" not in det:
                    return f"frame {frame.get('frame')}: person {pid} has no per-frame jersey"
                if det["jersey"] is not None and not re.fullmatch(r"\d{1,2}", str(det["jersey"])):
                    return f"frame {frame.get('frame')}: person {pid}: bad jersey {det['jersey']!r}"
    return None


def done_clips(results: Path, prompt_sha: str) -> set[str]:
    """Clips already answered successfully WITH THIS PROMPT (a changed prompt reruns everything)."""
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            if r.get("returncode") == 0 and r.get("parsed") and not r.get("schema_error") \
                    and r.get("prompt_sha") == prompt_sha:
                done.add(r["clip"])
    return done


def rollout_cwd(path: Path) -> str | None:
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("type") == "session_meta":
                return d.get("payload", {}).get("cwd")
    return None


def newest_rollout_since(root: Path, since: float, cwd: Path) -> Path | None:
    """The rollout of OUR call: modified after `since` AND started in our cwd. Another codex session
    running at the same time (e.g. the annotation runner) must not be picked up."""
    cands = [p for p in root.rglob("rollout-*.jsonl") if p.stat().st_mtime >= since - 1] if root.is_dir() else []
    cands = [p for p in cands if rollout_cwd(p) == str(cwd)]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def usage_total(rollout: Path) -> dict:
    last: dict = {}
    for line in rollout.read_text().splitlines():
        m = re.search(r'"usage":\{([^}]*)\}', line)
        if m:
            last = {k: int(v) for k, v in re.findall(r'"(\w+)":(\d+)', m.group(1))}
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=HERE / "prompts" / "astra_reid_team_jersey.txt")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--clips", nargs="*", help="clip stems to run (default: all)")
    parser.add_argument("--team-a", nargs=2, metavar=("NAME", "COLOUR"), help='e.g. White "white jerseys"')
    parser.add_argument("--team-b", nargs=2, metavar=("NAME", "COLOUR"))
    args = parser.parse_args()
    if args.team_a and args.team_b:
        TEAMS.update(TEAM_A=args.team_a[0], TEAM_A_COLOR=args.team_a[1], TEAM_B=args.team_b[0], TEAM_B_COLOR=args.team_b[1])
        VALID_TEAMS.clear()
        VALID_TEAMS.update({args.team_a[0], args.team_b[0], "referee"})

    root = args.frames_root.resolve()
    os.chdir(root)
    template = args.prompt_file.read_text()
    prompt_sha = hashlib.sha256((template + json.dumps(TEAMS, sort_keys=True)).encode()).hexdigest()[:16]
    require_holder = '"ball_holder"' in template
    require_det_jersey = '"person_id": 1, "jersey"' in template  # per-frame jersey reads (TrackID3x3 prompt)
    require_state = '"ball_state"' in template
    require_ball_box = '"ball_bbox_1000"' in template
    results, usage_log = root / "results.jsonl", root / "usage.jsonl"
    done = done_clips(results, prompt_sha)
    manifests = sorted(root.glob("*/manifest.json"))
    if args.clips:
        manifests = [m for m in manifests if m.parent.name in args.clips]

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        clip, frames = manifest["clip"], manifest["frames"]
        if clip in done:
            print(f"{clip}: done, skipping", flush=True)
            continue
        out_txt = root / f".{clip}.answer.txt"
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
               "-m", args.model, "-c", f"model_reasoning_effort={args.reasoning_effort}"]
        for f in frames:
            cmd += ["-i", str(manifest_path.parent / f["file_name"])]
        cmd += ["-o", str(out_txt), "-"]
        print(f"{clip}: {len(frames)} frames ...", flush=True)
        started = time.time()
        # sampling interval from the manifest (1 fps -> "1", 2 fps -> "0.5")
        interval_s = round((frames[1]["frame"] - frames[0]["frame"]) / manifest["fps"], 2) if len(frames) > 1 else 1.0
        proc = subprocess.run(cmd, input=render_prompt(template, len(frames), interval_s), text=True,
                              capture_output=True)
        record = {"clip": clip, "model": args.model, "reasoning_effort": args.reasoning_effort,
                  "prompt_file": str(args.prompt_file), "prompt_sha": prompt_sha,
                  "frames": frames, "elapsed_s": round(time.time() - started, 1), "returncode": proc.returncode}
        if proc.returncode != 0:
            record["error"] = proc.stderr[-2000:]
            print(f"{clip}: FAILED rc={proc.returncode}: {proc.stderr[-400:]}", flush=True)
        else:
            raw = out_txt.read_text() if out_txt.exists() else ""
            record["raw_answer"] = raw
            try:
                record["parsed"] = extract_json(raw)
                record["schema_error"] = schema_error(record["parsed"], len(frames), require_holder, require_det_jersey,
                                                      require_state, require_ball_box)
            except Exception as exc:  # noqa: BLE001
                record["parsed"], record["parse_error"] = None, str(exc)
            print(f"{clip}: {record['elapsed_s']}s, schema_error={record.get('schema_error')}, "
                  f"parse_error={record.get('parse_error')}", flush=True)
        with results.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        rollout = newest_rollout_since(Path.home() / ".codex" / "sessions", started, root)
        with usage_log.open("a") as fh:
            fh.write(json.dumps({"clip": clip, "rollout": str(rollout) if rollout else None,
                                 "usage": usage_total(rollout) if rollout else {}}) + "\n")
        if proc.returncode != 0 and any(m in proc.stderr for m in QUOTA_MARKERS):
            print(f"quota/auth stop (exit {EXIT_QUOTA})", flush=True)
            sys.exit(EXIT_QUOTA)


if __name__ == "__main__":
    main()
