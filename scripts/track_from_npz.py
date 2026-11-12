#!/usr/bin/env python3
"""detections_ft.npz を BoT-SORT（--no-cmc 相当）に流して tracks.json を作る（basketball-demo venv）．
run_detect_track.py と同じ tracks.json 形式（people のみ，class は player，ball は無し）．"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--detections", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

info = sv.VideoInfo.from_video_path(str(a.video))
d = np.load(a.detections)
tracker = BoTSORTTracker(frame_rate=info.fps, enable_cmc=False)
frames_out = []
for index, frame in enumerate(sv.get_video_frames_generator(str(a.video))):
    m = d["frame"] == index
    xyxy = d["xyxy"][m]
    dets = sv.Detections(xyxy=xyxy.astype(np.float32), confidence=d["confidence"][m],
                         class_id=np.zeros(len(xyxy), int), data={"class_name": np.array(["player"] * len(xyxy))})
    t = tracker.update(dets, frame=frame)
    t = t[t.tracker_id >= 0]
    frames_out.append({"frame": index, "people": [
        {"track_id": int(tid), "class": "player", "confidence": round(float(c), 4), "xyxy": [round(float(v), 1) for v in b]}
        for b, tid, c in zip(t.xyxy, t.tracker_id, t.confidence)], "ball": None})
a.out.write_text(json.dumps({"video": str(a.video), "fps": info.fps, "width": info.width, "height": info.height,
                             "frames": frames_out}, indent=1))
print("frames", len(frames_out))
