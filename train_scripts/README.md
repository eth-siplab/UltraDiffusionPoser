# UDP training scripts

One script per model reported in the paper. Each activates the `UIP` conda env,
sets up paths, and launches `Train_model.py` with the verified config. The two
shared backbones (clean / noisy AMASS) are **pretrained once** and reused by the
finetunes, so nothing is trained twice.

## Usage

```bash
./train_scripts/train_amass.sh         # clean-AMASS backbone (DanceDB + TC model), GPU 0
./train_scripts/train_amass.sh 1       # ... on GPU 1
./train_scripts/train_dip.sh           # DIP model (reuses/creates the clean backbone)
./train_scripts/train_all.sh           # all models in sequence
```

Training writes to `output/trainUDP/<timestamp>/` (TensorBoard + wandb logs,
`config.ini`, `model_args.json`, and `ckpt/`). The two pretrains use **fixed**
output dirs so they can be located later:

- clean backbone → `output/trainUDP/pretrain_amass/`
- noisy backbone → `output/trainUDP/pretrain_amass_noisy/`

Finetune runs get a fresh timestamp dir (the trainer forces this).

## Model map

| Script | Backbone (pretrain) | Finetune data | Key run args |
|--------|--------------------|---------------|--------------|
| `train_amass.sh` | clean AMASS | — | epochs 150, no noise, `diffusion_steps 50` → **DanceDB + TC** |
| `train_amass_noisy.sh` | noisy AMASS | — | epochs 50, IMU+UWB noise, `diffusion_steps 200` → **UIP-DB + GIP-DB** |
| `train_dip.sh` | clean AMASS | DIP-IMU | epochs 30, pose-only losses |
| `train_uip_ft.sh` | noisy AMASS | SIGGRAPH UWB-IMU | epochs 5, `UWBIMU_DIR=…/SIGGRAPH_dataset` |
| `train_gip_ft.sh` | noisy AMASS | Multi-UWB-Merged | epochs 5, `UWBIMU_DIR=…/Multi-UWB-Merged` |

`train_uip_ft` and `train_gip_ft` share the **same** noisy backbone and an
otherwise identical config — they differ only in which UWB dataset they finetune
on (`UWBIMU_DIR`).

## Pretrain reuse

The finetune scripts call `ensure_pretrain`, which resolves the backbone in this
order:

1. `CLEAN_PRETRAIN_CKPT` / `NOISY_PRETRAIN_CKPT` if you export a checkpoint path
   (e.g. point at the released checkpoints to skip pretraining entirely);
2. an existing `pretrain_amass[_noisy]/ckpt/baseline_diffusion_all_best_model_*.pt`;
3. otherwise it runs the matching pretrain script once, then finetunes.

Example — finetune DIP straight from the released clean backbone:

```bash
CLEAN_PRETRAIN_CKPT=output/trainUDP/2025_03_03_22_00_38/ckpt/baseline_diffusion_all_best_model_140.pt \
  ./train_scripts/train_dip.sh
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONDA_SH` | `~/miniconda3/etc/profile.d/conda.sh` | conda init script |
| `UDP_ENV` | `UDP` | conda env name |
| `SMPL_MODEL_PATH` | `data/smpl_m_lbs_10_207_0_v1.0.0.pkl` | SMPL body model |
| `WANDB_MODE` | `offline` | `online` to upload, `disabled` to turn off |
| `XVFB` | `0` | set `1` to wrap training in `xvfb-run` (headless preview rendering) |
| `LOG_ROOT` | `output/trainUDP` | output root |
| `AMASS_DIR` | `data/processed_data/AMASS_syn` | synthetic AMASS pretrain data |
| `DIPIMU_DIR` | `data/processed_data/DIP` | DIP finetune data |
| `UWBIMU_DIR` | per-script (SIGGRAPH / Multi-UWB-Merged) | UWB finetune data |
| `UWB_GUIDANCE_LAMBDA` | `50` | inference-time UWB guidance (dataset-dependent) |
| `CLEAN_PRETRAIN_CKPT` / `NOISY_PRETRAIN_CKPT` | _(unset)_ | reuse a specific backbone checkpoint |

## Notes

- The two base configs (`config/train_config_udp.ini` clean,
  `config/train_config_udp_uipgip.ini` noisy) were verified field-by-field
  against the logged wandb run configs.
- `uwb_loss` is **not** a training loss in any of these runs (it was disabled via
  a never-reached start epoch in the originals); it is omitted from the configs.
  UWB is still used at inference through `uwb_guidance_lambda`, which is recorded
  in each run's `model_args.json` for the evaluator to pick up.
- The dataset directories are read from `config.paths`, which now honor the env
  vars above (defaults preserve the previous hardcoded values).
