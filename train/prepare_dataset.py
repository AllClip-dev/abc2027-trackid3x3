#!/usr/bin/env python3
"""TrackID3x3 の training 動画 8 本から，RF-DETR 用の COCO データセット（player 1 クラス）を作る（Mac で実行）．

* 分割は動画単位（隣り合うフレームはほぼ同一なので，フレーム単位で分けると漏れる）:
  test = IMG_0104（公式 Colab のチュートリアル区間 = IMG_0104 のフレーム 116–1233 を含むので，後で比較できるように）
  valid = IMG_0110，train = IMG_0105, 0106, 0107, 0108, 0109, 0111（4 色のビブスがすべて train に入る）．
* アノテーションのあるフレームだけを使う（無い所を空画像にすると「選手なし」と教えてしまう）．
  bbox は 1 始まりのフレーム番号 + 左上基準の x, y, w, h（3840×2160）．
* 画像は final challenge と同じ 1280×720 に縮小して（INTER_AREA）保存し，bbox は 1/3 にして画像内に切り詰める．
* --every-n フレームおき（既定 15 = 2 fps）．ベンチ・記録係・黒シャツ・審判は元々ラベルが無く，背景になる．

  python prepare_dataset.py --data-root ../data --out ../data/rfdetr_dataset
出力: <out>/{train,valid,test}/*.jpg と _annotations.coco.json（rfdetr 1.6 の COCO 形式，category id 1 = player），split.json．
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

SPLITS = {"test": ["IMG_0104"], "valid": ["IMG_0110"],
          "train": ["IMG_0105", "IMG_0106", "IMG_0107", "IMG_0108", "IMG_0109", "IMG_0111"]}
W, H = 1280, 720


def process(args: tuple) -> dict:
    video, split, data_root, out_dir, every_n = args
    ann_path = data_root / "Training/full player annotations/Image-space bounding boxes" / f"{video}.txt"
    boxes = defaultdict(list)  # 1 始まりのフレーム番号 -> [[x, y, w, h]]
    for line in ann_path.read_text().splitlines():
        p = line.split(",")
        boxes[int(p[0])].append([float(v) for v in p[2:6]])
    cap = cv2.VideoCapture(str(data_root / "Training/vidoes" / f"{video}.MOV"))
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = W / sw, H / sh
    images, anns, dropped = [], [], 0
    f = 0  # 0 始まりの動画フレーム番号
    while True:
        if f % every_n or (f + 1) not in boxes:
            if not cap.grab():
                break
            f += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        name = f"{video}_f{f + 1:06d}.jpg"
        cv2.imwrite(str(out_dir / split / name), cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        images.append({"file_name": name, "width": W, "height": H, "video": video, "frame": f + 1})
        for x, y, w, h in boxes[f + 1]:
            x1, y1, x2, y2 = max(0.0, x * sx), max(0.0, y * sy), min(W, (x + w) * sx), min(H, (y + h) * sy)
            if x2 - x1 < 2 or y2 - y1 < 2:
                dropped += 1
                continue
            anns.append({"file_name": name, "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]})
        f += 1
    return {"video": video, "split": split, "src_size": [sw, sh], "images": images, "anns": anns, "dropped": dropped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every-n", type=int, default=15, help="train の間引き（既定 15 = 2 fps，2 = 15 fps）")
    ap.add_argument("--every-n-eval", type=int, default=15, help="valid / test の間引き．以前の結果と比べられるよう 15 のまま")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-all", action="store_true",
                    help="8 本すべてを train に入れる（提出用．valid / test は従来どおりだが train と重なるので評価には使えない）")
    ap.add_argument("--videos", nargs="+", help="この動画だけ処理する（既存のデータセットに足す分だけ作るとき）")
    args = ap.parse_args()
    global SPLITS
    if args.train_all:
        SPLITS = {**SPLITS, "train": sorted({v for vs in SPLITS.values() for v in vs})}
    for s in SPLITS:
        (args.out / s).mkdir(parents=True, exist_ok=True)
    jobs = [(v, s, args.data_root.resolve(), args.out.resolve(), args.every_n if s == "train" else args.every_n_eval) for s, vs in SPLITS.items() for v in vs
            if not args.videos or v in args.videos]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(process, jobs))
    coco = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "player", "supercategory": "player"}]}
            for s in SPLITS}
    summary = {}
    for r in results:
        c = coco[r["split"]]
        ids = {}
        for im in r["images"]:
            ids[im["file_name"]] = len(c["images"]) + 1
            c["images"].append({"id": ids[im["file_name"]], **im})
        for a in r["anns"]:
            c["annotations"].append({"id": len(c["annotations"]) + 1, "image_id": ids[a["file_name"]], "category_id": 1,
                                     "bbox": a["bbox"], "area": round(a["bbox"][2] * a["bbox"][3], 2), "iscrowd": 0})
        summary[r["video"]] = {"split": r["split"], "src_size": r["src_size"], "images": len(r["images"]),
                               "boxes": len(r["anns"]), "dropped": r["dropped"]}
    for s, c in coco.items():
        (args.out / s / "_annotations.coco.json").write_text(json.dumps(c))
    (args.out / "split.json").write_text(json.dumps({"splits": SPLITS, "every_n": args.every_n, "every_n_eval": args.every_n_eval, "size": [W, H],
                                                      "per_video": summary}, indent=1))
    print(json.dumps({s: {"images": len(c["images"]), "boxes": len(c["annotations"])} for s, c in coco.items()}))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
