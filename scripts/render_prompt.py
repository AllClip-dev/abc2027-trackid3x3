#!/usr/bin/env python3
"""動画ごとの名簿（背番号 2 組）を色なしの 3x3 プロンプトに埋め込み，prompts/rendered_nocolor/<video>.txt に書く．

チームは T1/T2 の任意ラベルで答えさせ，ビブス色は渡さない．
{N}, {N_LAST}, {INTERVAL} は run_astra_clips.py が埋めるので残す．
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
# games_roster.csv の "team: jersey"
VIDEOS = {
    "IMG_0104": {"A": "4, 10, 12", "B": "6, 9, 11"},  # train の評価用
    "IMG_0110": {"A": "4, 10, 12", "B": "6, 9, 11"},  # 2 本目の評価用（論文の追加実験）
    "IMG_0114": {"A": "6, 9, 10", "B": "6, 9, 11"},
    "IMG_0115": {"A": "4, 10, 12", "B": "1, 3, 5"},
}

template = (HERE / "prompts" / "astra_reid_team_jersey_3x3_nocolor.txt").read_text()
out = HERE / "prompts" / "rendered_nocolor"
out.mkdir(exist_ok=True)
for video, teams in VIDEOS.items():
    text = template.replace("{NUMBERS_1}", teams["A"]).replace("{NUMBERS_2}", teams["B"])
    (out / f"{video}.txt").write_text(text)
    print(video, "->", out / f"{video}.txt")
