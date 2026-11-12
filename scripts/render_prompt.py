#!/usr/bin/env python3
"""動画ごとの名簿とビブス色を 3x3 プロンプトに埋め込み，prompts/rendered/<video>.txt に書く．

{N}, {N_LAST}, {INTERVAL}, {TEAM_A}, {TEAM_B}, {TEAM_A_COLOR}, {TEAM_B_COLOR} は run_astra_clips.py が埋めるので残す．
ビブス色は training の実フレームと final のフレームを目視して決めた（下の VIDEOS）．
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
# games_roster.csv の "team: jersey" と，フレームで確認したビブス色
VIDEOS = {
    "IMG_0104": {"A": ("", "4, 10, 12"), "B": ("", "6, 9, 11")},  # train の評価用．色なし版だけ使う（色は未確認）
    "IMG_0110": {"A": ("", "4, 10, 12"), "B": ("", "6, 9, 11")},  # 2 本目の評価用（論文の追加実験）．色なし版だけ使う
    "IMG_0114": {"A": ("yellow bibs", "6, 9, 10"), "B": ("green bibs", "6, 9, 11")},
    "IMG_0115": {"A": ("purple bibs", "4, 10, 12"), "B": ("pink bibs", "1, 3, 5")},
}

template = (HERE / "prompts" / "astra_reid_team_jersey_3x3.txt").read_text()
out = HERE / "prompts" / "rendered"
out.mkdir(exist_ok=True)
for video, teams in VIDEOS.items():
    text = template.replace("{TEAM_A_NUMBERS}", teams["A"][1]).replace("{TEAM_B_NUMBERS}", teams["B"][1])
    (out / f"{video}.txt").write_text(text)
    print(video, "->", out / f"{video}.txt")

# 色指定なし版（チームは T1/T2 の任意ラベル，名簿は 2 組の番号だけ渡す）
nc_template = (HERE / "prompts" / "astra_reid_team_jersey_3x3_nocolor.txt").read_text()
nc_out = HERE / "prompts" / "rendered_nocolor"
nc_out.mkdir(exist_ok=True)
for video, teams in VIDEOS.items():
    text = nc_template.replace("{NUMBERS_1}", teams["A"][1]).replace("{NUMBERS_2}", teams["B"][1])
    (nc_out / f"{video}.txt").write_text(text)
    print(video, "(nocolor) ->", nc_out / f"{video}.txt")
