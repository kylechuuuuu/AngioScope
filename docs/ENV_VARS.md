# Configuration reference

All options are environment variables — `train.py` and `test.py` have no `argparse`. Values
shown are the built-in defaults, so an unset variable reproduces the paper setting.

> **Shape-changing variables.** `SD2_EXTRA_UP`, `SD2_BANK_LAYERS`, `SD2_REFINER`,
> `SD2_DILATIONS`, `SD2_SEG_PRIOR`, `SD2_TASK` and `SD2_OFFSET_SCALE` / `SD2_SIZE_PRIOR`
> alter parameter shapes. They are written to `<checkpoint>.arch.json` when training ends and
> **restored from that sidecar at load time**, overriding whatever you pass. Only set them by
> hand when the sidecar is missing.

---

## Data and run

| variable | default | meaning |
|---|---|---|
| `SD2_DATA_ROOT` | `DSCA_new` | dataset root containing `train/`, `val/`, `label.json`, `*_detect/annotations/` |
| `SD2_TASK` | `auto` | `joint` (seg + det), `det` (detection only), `auto` = decided by the presence of `{split}/masks` |
| `SD2_SEED` | `42` | seed for `random`, NumPy and Torch; also enables cuDNN benchmark |
| `SD2_WORKERS` | `8` | dataloader workers; `0` fixes hangs/segfaults on hosts with small `/dev/shm` |
| `SD2_OUT_PREFIX` | `best_fusion_model` | prefix of the saved checkpoints |
| `SD2_LOG` | `train.log` | log file (appended) |
| `SD2_TAG` | — | free-form label copied into `<prefix>_result.json` |
| `SD2_SAVE_FULL` | `0` | `0` = slim checkpoint (drop the frozen SAM 3.1 encoder keys), `1` = self-contained |
| `SD2_SAVE_VIZ` | `0` | `1` writes validation PNGs into `results/` during training |

## Schedule and optimisation

| variable | default | meaning |
|---|---|---|
| `SD2_EPOCHS` | `220` | training epochs |
| `SD2_LR` | `1e-4` | base AdamW learning rate |
| `SD2_ADAPTER_LR_MULT` | `5.0` | LR multiplier for adapter / router parameters |
| `SD2_AMP_DTYPE` | `fp16` | autocast dtype: `fp16`, `bf16` (use when fp16 overflows) or `fp32` (off) |
| `SD2_EMA` | `0.999` | EMA decay used for the saved snapshots |
| `SD2_VAL_EVERY` | `5` | validate (and possibly checkpoint) every N epochs |
| `SD2_ITERATIONS` | `2` | residual refinement iterations; `1` = no refinement, faster |
| `SD2_EARLY_STOP` | `0` | patience in epochs on the monitored metric; `0` = run all epochs |
| `SD2_STOP_METRIC` | `dice` | monitored metric: `dice` (Mean Dice FG), `ap50` (rolling median) or `joint` (`0.5·dice + 0.5·ap50`) |

## Architecture

| variable | default | meaning |
|---|---|---|
| `SD2_BANK_LAYERS` | `all` | AdapterBank on every encoder layer (`all` / `-1`), on the N deepest layers (`N`), or a shared adapter only (`0`) |
| `SD2_EXTRA_UP` | `1` | number of extra learnable 2× PixelShuffle stages in the decoder |
| `SD2_REFINER` | `legacy` | residual refiner variant; only `legacy` ships (`off` disables it) |
| `SD2_DILATIONS` | `1,2,4,8` | dilation rates of the detection trunk; empty string drops the trunk |
| `SD2_SEG_PRIOR` | `auto` | whether the detection head consumes the class-1 seg probability; `auto` = on iff masks exist |
| `SD2_ADAPTER_CKPT` | `1` | gradient checkpointing inside the AdapterBank (saves VRAM, slower) |

## Losses and routing

| variable | default | meaning |
|---|---|---|
| `SD2_FG_WEIGHT` | `0.3` | foreground-BCE weight |
| `SD2_CONSIST_W0` / `SD2_CONSIST_W1` | `0.5` / `0.15` | structure-consistency weight at epoch 0 / at the end, linearly annealed |
| `SD2_ROUTER_WEIGHT` | `0.1` | router presence-supervision weight |
| `SD2_ROUTER_BAL_W` | `0.01` | Switch-style router load-balance weight |
| `SD2_ROUTER_TARGET` | `legacy` | presence target: `legacy` (raw binary), `norm` (normalised per image), `none` |
| `SD2_ROUTER_DROP_ALWAYS` | `0.0` | drop classes present in at least this fraction of training images from the routing target |

## Detection

| variable | default | meaning |
|---|---|---|
| `SD2_DET_WEIGHT` | `1.0` | detection loss weight |
| `SD2_DET_GRAD_SCALE` | `1.0` | gradient coupling between the det loss and shared features; `0` detaches (detector floods FPs) |
| `SD2_DET_NEG_K` / `SD2_DET_NEG_W` | `25` / `0.0` | hard-negative top-K term; **off by default** — it collapses the heatmap when enabled |
| `SD2_NEG_FLOOR` | `0.0` | denominator floor of the heatmap focal loss; `0` = old inert normalisation |
| `SD2_SIZE_L1_W` | `5.0` | L1 weight of the box-size channels |
| `SD2_SIZE_PRIOR` | `0.10` | mean normalised box size used to bias-init the size channels |
| `SD2_OFFSET_SCALE` | `4.0` | half-range of the tanh centre-offset regression, in heatmap pixels |
| `SD2_DET_SAMPLE_W` | `0.5` | extra sampler weight for images that carry a box (joint runs) |
| `SD2_DET_ONLY_SAMPLE_W` | `3.0` | same, for detection-only runs |
| `SD2_EVAL_SCORE` | `0.5` | box confidence threshold when heatmaps are turned into boxes for evaluation |

## Inference

| variable | default | meaning |
|---|---|---|
| `SD2_TEST_MODEL` | `best_fusion_model_seg.pth` | checkpoint to evaluate |
| `SD2_TEST_OUT` | `results` | output root (`overall/`, `main/`, `cerebral/`, `det/`) |
| `SD2_TEST_SPLIT` | `val` | split evaluated by `test.py` |
| `SD2_TEST_METRICS` | `1` | `0` = write overlays only |
| `SD2_TTA` | `0` | `1` averages h/v/hv flip variants (4× segmentation cost) |

## Detection-only (v2) overlay

Only meaningful after copying `detonly/*.py` over the main tree; see
[`../detonly/README.md`](../detonly/README.md).

| variable | default | meaning |
|---|---|---|
| `SD2_DET_ONLY_V2` | `0` | enable the v2 path (requires `SD2_TASK=det`) |
| `SD2_DET_PRIOR_W` | `1.0` | weight of the box-supervised dense prior loss |
| `SD2_DET_PRIOR_MIL_W` | `0.5` | weight of the MIL pointing loss |
| `SD2_DET_PRIOR_CORE` | `0.35` | box-core shrink ratio for the dense target |
| `SD2_DET_PRIOR_IGNORE` | `0.25` | ignore-ring expand ratio |
| `SD2_DET_PRIOR_GAUSS` | `1` | centre-Gaussian core target instead of a flat core |
| `SD2_COPYPASTE` | `0.0` | copy-paste augmentation probability (`0.5` recommended) |
| `SD2_COPYPASTE_MARGIN` | `0.2` | lesion crop context margin for copy-paste |
