#!/usr/bin/env python3
"""Sample frames for Astra (stage 3): every --every-n-frames frames of each clip, as JPEG.

Writes <out-root>/<clip_stem>/fNNNNN.jpg and manifest.json ({"clip", "video", "fps", "width",
"height", "frames": [{"frame": idx, "file_name": ...}]}). A clip gets one Astra call, so at
most --max-frames frames (the prompt batches 16 frames, as skalskip does).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import supervision as sv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--every-n-frames", type=int, default=30)  # 1 fps at 30 fps
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="split into <stem>_bNN batches of this many frames (one Astra call each) instead of "
                             "stopping at --max-frames; consecutive batches share --batch-overlap frames")
    parser.add_argument("--batch-overlap", type=int, default=1)
    args = parser.parse_args()

    for video in args.videos:
        info = sv.VideoInfo.from_video_path(str(video))
        limit = None if args.batch_size else args.max_frames
        pool = args.out_root / video.stem / "_all"
        pool.mkdir(parents=True, exist_ok=True)
        frames = []
        for idx, frame in enumerate(sv.get_video_frames_generator(str(video))):
            if idx % args.every_n_frames or (limit is not None and len(frames) >= limit):
                continue
            name = f"f{idx:05d}.jpg"
            cv2.imwrite(str(pool / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append({"frame": idx, "file_name": name})
        if args.batch_size:
            step = args.batch_size - args.batch_overlap
            groups = [frames[i:i + args.batch_size] for i in range(0, max(len(frames) - args.batch_overlap, 1), step)]
            names = [f"{video.stem}_b{k:02d}" for k in range(len(groups))]
        else:
            groups, names = [frames], [video.stem]
        for name, group in zip(names, groups):
            out = args.out_root / name
            out.mkdir(parents=True, exist_ok=True)
            for f in group:
                target = out / f["file_name"]
                if not target.exists():
                    target.hardlink_to(pool / f["file_name"])
            (out / "manifest.json").write_text(json.dumps(
                {"clip": name, "video": str(video), "fps": info.fps, "width": info.width,
                 "height": info.height, "frames": group}, indent=1))
            print(name, len(group), group[0]["frame"], group[-1]["frame"])


if __name__ == "__main__":
    main()
