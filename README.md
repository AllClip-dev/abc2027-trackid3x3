# ABC 2027 TrackID3x3 Challenge — AllClip

Code for our final submission to the [TrackID3x3 Challenge](https://sites.google.com/view/abc2027-trackid3x3-challenge/the-challenge) (ABC 2027): positions and identities (team, jersey number) of the six players in every frame of a 3x3 basketball video.

```
video ─ RF-DETR ─ BoT-SORT ──────────────────────────────┐
  └─ VLM every 15 frames ─ roster-constrained Viterbi ─ identity assignment ─ occlusion box re-drawing (VLM) ─ court projection
```

The VLM is GPT-6 Astra, called with `codex exec`. Every call is automatic, and no frame is annotated or corrected by hand.

## Contents

| Path | What it does |
|---|---|
| `train/` | Detector training: `prepare_dataset.py`, `resize_pretrain.py`, `train_rfdetr.py` |
| `scripts/ft_detect.py`, `track_from_npz.py` | Detection and BoT-SORT tracking |
| `scripts/extract_astra_frames.py`, `run_astra_clips.py`, `resume_astra.sh` | Frame sampling and VLM calls |
| `scripts/astra_identity.py`, `sync_astra.py`, `sync_and_render.py` | Roster Viterbi, assignment to tracks, court projection |
| `scripts/occlusion_fix.py` | Box re-drawing in occlusion segments |
| `scripts/make_submission.py`, `eval_ti_hota.py` | Submission files, TI-HOTA |
| `prompts/` | VLM prompts (`rendered_nocolor/` holds the roster of each video) |

Weights: `trackid3x3_rfdetr_medium768_all8.pth` in the [Releases](../../releases) page. This is RF-DETR Medium fine-tuned on all eight training videos (15 fps, 768 px, 30 epochs).

## Setup

```bash
pip install -r requirements.txt   # rfdetr, trackers, supervision, opencv, scipy
```

The VLM steps need the Codex CLI, logged in to an account with GPT-6 Astra access. Download the data from the official [TrackID3x3](https://github.com/open-starlab/TrackID3x3) page.

## Reproducing the submission (per video `V`: IMG_0114, IMG_0115)

```bash
D="path/to/TrackID3x3/Final challenge"; REG="$D/court_registration.csv"; mkdir -p work/$V
# 1. detection + tracking (GPU)
python scripts/ft_detect.py --video "$D/$V.MP4" --checkpoint trackid3x3_rfdetr_medium768_all8.pth \
  --resolution 768 --model medium --no-tta --out work/$V/det.npz
python scripts/track_from_npz.py --video "$D/$V.MP4" --detections work/$V/det.npz --out work/$V/tracks.json
# 2. VLM identity samples: 16 frames per call, one frame shared between calls
python scripts/extract_astra_frames.py --videos "$D/$V.MP4" --out-root work/vlm_$V \
  --every-n-frames 15 --batch-size 16 --batch-overlap 1
TEAM_A_NAME=T1 TEAM_B_NAME=T2 bash scripts/resume_astra.sh work/vlm_$V prompts/rendered_nocolor/$V.txt unused unused
# 3. identities -> tracks
ARGS=(--video "$D/$V.MP4" --tracks work/$V/tracks.json --astra-results work/vlm_$V/results.jsonl
      --clip $V --identity viterbi --court-registration "$REG" --cut-metric jump --ambig-margin 0.3 --ambig-mode foot --no-video)
python scripts/sync_and_render.py "${ARGS[@]}" --out-dir work/$V/base
# 4. occlusion box re-drawing
python scripts/occlusion_fix.py segments --trigger bottom --ident-boxes work/$V/base/ident_boxes.json \
  --video "$D/$V.MP4" --out work/occl_$V
python scripts/occlusion_fix.py run --root work/occl_$V --prompt-file prompts/astra_occlusion_fix.txt
python scripts/occlusion_fix.py boxes --root work/occl_$V --ident-boxes work/$V/base/ident_boxes.json \
  --court-registration "$REG" --out work/$V/fix.json --min-move-cm 30
# 5. final output
python scripts/sync_and_render.py "${ARGS[@]}" --override-boxes work/$V/fix.json --out-dir work/$V/final
python scripts/make_submission.py --pred work/$V/final/predictions.csv --video $V --out-dir submission
```

Keep `work/` outside any directory that has an `AGENTS.md`, because Codex also reads that file. VLM answers can differ between runs. With our recorded answers, steps 3–5 reproduce the submitted files byte for byte.

Training: `python train/resize_pretrain.py --src rf-detr-medium.pth --resolution 768 --out rf-detr-medium-768.pth`, then `train/prepare_dataset.py --data-root <TrackID3x3> --out <ds> --every-n 2 --train-all` and `train/train_rfdetr.py --dataset-dir <ds> --output-dir <run> --resolution 768 --epochs 30 --lr-scheduler cosine --early-stopping-patience 100 --pretrain rf-detr-medium-768.pth`. On one RTX 3090 this takes about 22 h.

## Citation

If you use the data, cite the TrackID3x3 dataset (Yamada et al., ACM MMSports 2025, [doi:10.1145/3728423.3759400](https://doi.org/10.1145/3728423.3759400)).
