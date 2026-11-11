# AngioScope

Fine-grained cerebrovascular parsing in Digital Subtraction Angiography (DSA) with
**Structurally-Grounded Semantic Disentanglement (SD2)**: **7-class vessel segmentation** and
**lesion detection** from one frozen **SAM 3.1** vision encoder.
The same codebase supports two tasks:

| task | needs masks? | what it runs |
|---|---|---|
| `joint` (default) | yes | segmentation (7 classes) + detection |
| `det` | no | detection only — segmentation branch is skipped in `forward()` |

The task is resolved automatically from whether `{split}/masks` exists; force it with
`SD2_TASK=joint|det`.

---

## Contents

- [Quick start](#quick-start)
- [Model at a glance](#model-at-a-glance)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Dataset format](#dataset-format)
- [Training](#training)
- [Inference and evaluation](#inference-and-evaluation)
- [Detection-only variant](#detection-only-variant)
- [Troubleshooting](#troubleshooting)
- [Full configuration reference](#full-configuration-reference)

---

## Quick start

```bash
git clone https://github.com/kylechuuuuu/AngioScope.git
cd AngioScope
conda create -n angioscope python=3.11 -y && conda activate angioscope
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA wheel
pip install -r requirements.txt

# 1. SAM 3.1 weights (gated: accept the licence on Hugging Face first)
huggingface-cli download facebook/sam3.1 --local-dir sam3.1

# 2. train on your dataset (expects <DATA_ROOT>/train/{images,masks})
SD2_DATA_ROOT=/data/DSCA_new python train.py

# 3. evaluate the best segmentation snapshot
SD2_DATA_ROOT=/data/DSCA_new SD2_TEST_MODEL=best_fusion_model_seg.pth python test.py
```

Everything is configured through environment variables — there is no `argparse` in
`train.py`. All examples below are bash; in PowerShell write
`$env:SD2_DATA_ROOT="D:\data\DSCA_new"`.

---

## Model at a glance

1. **Encoder** — SAM 3.1 ViT (`facebook/sam3.1`), input 1008×1008, **frozen**.
2. **AdapterBank on every one of the 32 encoder layers** — a router picks among 6
   structure adapters (one per vessel class) per image, supervised by class presence, so
   thin/rare classes keep a slot instead of collapsing into one expert.
3. **Decoder** — FPN neck (3 levels) → channel reduction → learnable 2× PixelShuffle
   upsampling → segmentation logits for the 7 classes.
4. **Iterative residual refiner** — 2 residual steps on the full-resolution logits
   (`SD2_ITERATIONS`), driven by class entropy.
5. **Detection head** — CenterNet-style heatmap + sub-pixel offset + box-size regression on
   a dilated shared trunk, optionally consuming the class-1 segmentation probability as a
   spatial lesion prior.

Both branches are always constructed, so `joint` and `det` checkpoints stay key-compatible;
`task='det'` only skips the segmentation work in the forward pass. Training saves **two
task-optimal snapshots** (`_seg` / `_det`) rather than one joint checkpoint.

---

## Repository layout

| path | role |
|---|---|
| `train.py` | training entry point; env-var configured; writes `*_seg.pth` / `*_det.pth` |
| `test.py` | inference, overlays and metrics for a checkpoint |
| `fusionet2.py` | model definition: SAM 3.1 encoder, AdapterBank, decoder, refiner, detection head |
| `dataset.py` | dataset, letterbox to 1024, augmentation, COCO box loading |
| `calculate_metrics.py` | per-class Dice / IoU / precision / recall (importable **and** a CLI) |
| `detection_metrics.py` | AP50, precision / recall, matched-box IoU |
| `detonly/` | maskless detection variant (v2) — overlay files, see [below](#detection-only-variant) |
| `docs/ENV_VARS.md` | every `SD2_*` knob with its default |

---

## Installation

**Python 3.11** and a CUDA-capable GPU (24 GB was enough for every run in the paper; the
batch size is fixed at 1). The model needs `transformers` **≥ 5.0**, which is the first
release series shipping the `transformers.models.sam3` module this code imports.

```bash
conda create -n angioscope python=3.11 -y
conda activate angioscope

# PyTorch first, so pip does not pull a CPU-only wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**SAM 3.1 weights.** The encoder is loaded from a local Hugging Face checkpoint directory,
hard-coded as `sam3.1/sam3.1_multiplex.pt` + config `sam3.1/` at the repository root
(`SAM3_CKPT` / `SAM3_CONFIG` in `fusionet2.py`):

```bash
huggingface-cli login          # the facebook/sam3.1 repo is gated
huggingface-cli download facebook/sam3.1 --local-dir sam3.1
ls sam3.1/sam3.1_multiplex.pt  # must exist
```

The loader prints `Loaded N/M SAM3 vision encoder weights` at start-up — if `N` is close to
`M` you are fine. Only the vision backbone and neck are built; the multi-GB video tracker is
never instantiated.

---

## Dataset format

There is no downloader: prepare your own data in this layout.

```
<DATA_ROOT>/
├── label.json                       # colour table (class id / name / hex) — used by the metric scripts
├── train/
│   ├── images/*.png                 # any resolution
│   └── masks/*.png                  # optional; omit the whole directory for a detection-only dataset
├── val/
│   ├── images/*.png
│   └── masks/*.png
├── train_detect/
│   └── annotations/<stem>.json      # per-image COCO detection annotations (pixels, xywh)
└── val_detect/
    └── annotations/<stem>.json
```

`<DATA_ROOT>` defaults to `DSCA_new` and is set with `SD2_DATA_ROOT`.

### Classes and colours

Masks are **RGB PNGs** whose exact colours encode the class; both the dataset and
`calculate_metrics.py` match colours, not palette indices, so keep the files lossless (no
JPEG, no palette quantisation).

| id | name | hex |
|---|---|---|
| 0 | background | `#000000` |
| 1 | noise (ICA bulb) | `#00a2e8` |
| 2 | carotid artery | `#860015` |
| 3 | vertebral artery | `#b97a57` |
| 4 | anterior cerebral artery | `#fff200` |
| 5 | middle cerebral artery | `#c8bfe7` |
| 6 | posterior cerebral artery | `#efe4b0` |

`label.json` maps a key to `{id, color, name}` — the keys are free-form, `id` and `color`
are what matter:

```json
{
  "black":  {"id": 0, "color": "#000000", "name": "background"},
  "#blue":  {"id": 1, "color": "#00a2e8", "name": "noise"},
  "#red":   {"id": 2, "color": "#860015", "name": "carotid_artery"},
  "...":    {"id": 3, "color": "#b97a57", "name": "vertebral_artery"}
}
```

Class 1 is not a vessel: it marks the region the detection boxes point at, and — in a joint
run — its probability is fed to the detection head as a spatial prior.

### Detection annotations

One JSON per image, named after the **image stem** (`pro_HGP_BAP_85.png` →
`annotations/pro_HGP_BAP_85.json`), in COCO form:

```json
{
  "images": [{"id": 1, "file_name": "images/pro_HGP_BAP_85.png", "width": 1024, "height": 1024}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": 8123, "iscrowd": 0}
  ],
  "categories": [{"id": 1, "name": "target", "supercategory": "target"}]
}
```

- Only `bbox` is consumed, in **pixels of the original image**; it is converted to
  normalised `cxcywh` internally and mapped into the letterboxed canvas together with the
  image, so no manual resizing is needed.
- An image with **no** annotation file, or an empty `annotations` list, is a legitimate
  negative sample (the detector is trained to predict nothing there).
- At most 10 boxes per image are kept (`max_det_targets`, the largest areas win).
- `{split}_detect/images/` is **not** read — images always come from `{split}/images/`.

### What the dataset does to your data

- Letterbox to 1024×1024, aspect ratio preserved, padded with the ImageNet mean
  `(124, 116, 104)`; masks are pasted with `NEAREST` so classes stay intact.
- Training only: random scale 0.75–1.25 (below 1 letterboxes, above 1 centre-crops),
  colour jitter, and horizontal / vertical flips with boxes flipped in sync.
- `SD2_DATA_ROOT` must contain **both** `train/` and `val/` — validation runs every
  `SD2_VAL_EVERY` epochs on `val/`.

---

## Training

```bash
cd AngioScope
SD2_DATA_ROOT=/data/DSCA_new \
SD2_EPOCHS=220 \
SD2_WORKERS=8 \
python train.py
```

Defaults: 220 epochs, AdamW (`lr=1e-4`, adapter/router parameters ×5), cosine schedule,
weight decay 5e-3, batch size 1, EMA 0.999, fp16 autocast, validation + checkpointing every
5 epochs. A run writes:

| file | content |
|---|---|
| `<prefix>_seg.pth` | best **Mean Dice (foreground)** snapshot, EMA weights |
| `<prefix>_det.pth` | best **AP50** (rolling median) snapshot, EMA weights |
| `<prefix>_seg.pth.arch.json` | architecture sidecar — **keep it next to the checkpoint** |
| `<prefix>_result.json` | epoch history, per-class Dice, final detection metrics |
| `train.log` | full log (`SD2_LOG`) |
| `results/` | validation overlays, written only with `SD2_SAVE_VIZ=1` (`overall/`, `main/`, `cerebral/`, `det/`) |

`<prefix>` is `SD2_OUT_PREFIX` (default `best_fusion_model`).

**Snapshots are slim by default** (`SD2_SAVE_FULL=0`): the frozen SAM 3.1 encoder weights are
not stored, because they are re-read from `sam3.1/`. This keeps a checkpoint around a few
hundred MB; set `SD2_SAVE_FULL=1` if you need a self-contained file.

The sidecar is not optional bookkeeping: it records the architecture switches
(`extra_upsample`, `bank_deep_layers`, `refiner`, `dilations`, `task`, `seg_prior`, …) that
changed parameter shapes. Without it a checkpoint can load "successfully" into the wrong
architecture and silently produce near-random masks. Copy both files when you move a model.

Useful knobs (full list in [`docs/ENV_VARS.md`](docs/ENV_VARS.md)):

| variable | default | why you would change it |
|---|---|---|
| `SD2_DATA_ROOT` | `DSCA_new` | dataset location |
| `SD2_TASK` | `auto` | `joint` / `det` override |
| `SD2_EPOCHS` | `220` | run length |
| `SD2_LR` | `1e-4` | base learning rate |
| `SD2_WORKERS` | `8` | dataloader workers (`0` when `/dev/shm` is small) |
| `SD2_AMP_DTYPE` | `fp16` | `bf16` if fp16 NaNs out, `fp32` to disable autocast |
| `SD2_EARLY_STOP` / `SD2_STOP_METRIC` | `0` / `dice` | stop on `dice`, `ap50` or `joint` |
| `SD2_SEED` | `42` | RNG seed |
| `SD2_SAVE_VIZ` | `0` | dump validation PNGs during training |
| `SD2_OUT_PREFIX` | `best_fusion_model` | checkpoint prefix |
| `SD2_TAG` | — | label written into `*_result.json` |

---

## Inference and evaluation

```bash
SD2_DATA_ROOT=/data/DSCA_new \
SD2_TEST_MODEL=best_fusion_model_seg.pth \
SD2_TEST_OUT=results \
python test.py
```

- `SD2_TEST_MODEL` defaults to `best_fusion_model_seg.pth`.
- The **architecture is restored from the checkpoint sidecar**, so `SD2_EXTRA_UP`,
  `SD2_BANK_LAYERS`, `SD2_REFINER`, `SD2_TASK`, … do not need to be repeated.
- `SD2_TEST_METRICS=0` skips scoring and only writes overlays.
- `SD2_TTA=1` averages horizontal / vertical / both flips (4 segmentation forwards): about
  +0.5–1 Dice on thin vessels, 4× the inference cost. Detection always uses the original
  pass only.
- `SD2_EVAL_SCORE` (default `0.5`) is the box confidence threshold for the overlays and for
  the evaluation pass.

Output, under `SD2_TEST_OUT`:

| directory | content |
|---|---|
| `overall/` | full 7-class colour prediction |
| `main/` | only carotid + vertebral (classes 2, 3) |
| `cerebral/` | only ACA + MCA + PCA (classes 4, 5, 6) |
| `det/` | original image with detection boxes drawn |

Console output: per-class Dice / IoU / precision / recall plus a foreground mean
(background excluded), then AP50, precision, recall, matched IoU and the TP/FP/FN counts.
Segmentation is skipped automatically — writing `det/` only — when the dataset has no masks
or the checkpoint is detection-only.

`calculate_metrics.py` also runs standalone on any prediction folder:

```bash
python calculate_metrics.py \
  --pred_dir results/overall \
  --gt_dir /data/DSCA_new/val/masks \
  --label_path /data/DSCA_new/label.json
```

---

## Detection-only variant

`detonly/` holds **det-only v2** — box-supervised dense lesion prior + copy-paste
augmentation — for datasets with no masks at all. It is an overlay: copy the files over the
main tree, then train normally.

```bash
cp detonly/{det_only.py,dataset.py,fusionet2.py,train.py,test.py} ./

SD2_DATA_ROOT=/path/to/maskless_split \
SD2_TASK=det SD2_DET_ONLY_V2=1 SD2_COPYPASTE=0.5 \
SD2_EPOCHS=220 python train.py
```

Read [`detonly/README.md`](detonly/README.md) first, and
[`detonly/DET_ONLY_REDESIGN.md`](detonly/DET_ONLY_REDESIGN.md) for the design, ablations and
rollback notes. The main tree is unaffected as long as the overlay is not copied over it.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| Dice ≈ 0.04, masks look random, no error | checkpoint and model were built with different architecture switches. Keep the `.arch.json` sidecar next to the checkpoint, or set `SD2_EXTRA_UP` / `SD2_BANK_LAYERS` / `SD2_REFINER` to the trained values. |
| `Warning: N keys missing (architecture changed?)` | same as above; the warning only appears for non-encoder keys. |
| `Segmentation skipped (...) writing det/ only` | the dataset has no `{split}/masks`, or the checkpoint is a `_det` one. Expected for detection-only runs. |
| loss → NaN around epoch 20 on a large detection subset | fp16 overflow. Use `SD2_AMP_DTYPE=bf16` (or `fp32`). |
| CUDA OOM | keep batch size 1, lower `SD2_ITERATIONS`, and set `SD2_WORKERS=0` if the host has little shared memory. |
| dataloader hangs at epoch start | too many workers for the host: `SD2_WORKERS=0`. |
| AP50 jumps between runs | expected with ~70 box-annotated images; seed is pinned (`SD2_SEED=42`) and the checkpoint is chosen on a *rolling median* AP50. |
| `PRECONDITION`/import error on `transformers.models.sam3` | `transformers` is older than 5.0. |

---

## Full configuration reference

Every knob, its default and what it changes lives in
[`docs/ENV_VARS.md`](docs/ENV_VARS.md) — data, schedule, architecture, loss weights,
detection and inference.

---

## Contact

Questions and bug reports: please open an
[issue](https://github.com/kylechuuuuu/AngioScope/issues).
