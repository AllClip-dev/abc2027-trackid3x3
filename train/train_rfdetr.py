#!/usr/bin/env python3
"""RF-DETR medium を TrackID3x3 の training（player 1 クラス）で fine-tune する．

事前学習重みは COCO の rf-detr-medium.pth を明示する（クラス数が違うので検出ヘッドは作り直される）．
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfdetr import RFDETRLarge, RFDETRMedium


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pretrain", type=Path, required=True, help="COCO の rf-detr-medium.pth（768 では resize_pretrain.py で補間したもの）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4)
    ap.add_argument("--resolution", type=int, default=576)
    ap.add_argument("--checkpoint-interval", type=int, default=10, help="1 個 約 0.5 GB．ディスクが厳しいので間引く")
    ap.add_argument("--lr-scheduler", choices=["step", "cosine"], default="step")
    ap.add_argument("--lr-min-factor", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--early-stopping-patience", type=int, default=6)
    ap.add_argument("--model", choices=["medium", "large"], default="medium")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true", help="W&B に送る")
    ap.add_argument("--project", default="abc2027-trackid3x3")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / "train_config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1), flush=True)

    model = {"medium": RFDETRMedium, "large": RFDETRLarge}[args.model](pretrain_weights=str(args.pretrain), resolution=args.resolution)
    model.train(
        dataset_dir=str(args.dataset_dir), output_dir=str(args.output_dir), class_names=["player"],
        epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
        lr_scheduler=args.lr_scheduler, lr_min_factor=args.lr_min_factor,
        lr=args.lr, lr_encoder=args.lr_encoder, resolution=args.resolution,
        checkpoint_interval=args.checkpoint_interval, eval_interval=1, num_workers=args.num_workers,
        early_stopping=True, early_stopping_patience=args.early_stopping_patience, run_test=True,
        seed=args.seed, progress_bar="tqdm",
        wandb=args.wandb, project=args.project, run=args.run_name,
    )


if __name__ == "__main__":
    main()
