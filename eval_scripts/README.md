# UDP evaluation scripts

One script per dataset/configuration reported in the paper. Each loads the
correct checkpoint, points it at the matching preprocessed test set, and writes
results to `output/evaluation_res_UDP/<exp_name>/` (a `[Eval_tab]UDP.csv`
table and, where translation is evaluated, a translation-error plot).

## Usage

```bash
./eval_scripts/eval_tc.sh        # evaluate on GPU 0
./eval_scripts/eval_tc.sh 1      # evaluate on GPU 1
./eval_scripts/eval_all.sh       # run all of them in sequence
```

Each script activates the `UIP` conda env and adds the repo to `PYTHONPATH`.
Override the defaults via environment variables if your setup differs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONDA_SH` | `~/miniconda3/etc/profile.d/conda.sh` | conda init script |
| `UDP_ENV` | `UDP` | conda env name |
| `SMPL_MODEL_PATH` | `data/smpl_m_lbs_10_207_0_v1.0.0.pkl` | SMPL body model |
| `DEVICE` | `cuda` | eval device (`cuda` / `cpu`) |
| `EVAL_SAVE_DIR` | `output/evaluation_res_UDP` | output root |
| `RBDL_PYTHON_PATH` | _(unset)_ | rbdl python build, only if using physics |

## Configuration map

Checkpoints live under `data/checkpoints/<name>/ckpt/`; test data under
`data/processed_data/`.

| Script | Checkpoint (`data/checkpoints/…`) | Test data (`data/processed_data/…`) | Extra flags |
|--------|-----------|-----------|-------------|
| `eval_dancedb.sh` | `dancedb_tc` | `AMASS_syn_orig_server/test_split` | `--eval_trans` |
| `eval_tc.sh` | `dancedb_tc` | `TotalCapture` | `--eval_trans` |
| `eval_dip.sh` | `dip` | `DIP_IMU` | — (pose only) |
| `eval_uip.sh` | `uip_gip` | `UWB_IMU/SIGGRAPH_dataset` | `--eval_trans` |
| `eval_uip_ft.sh` | `uip_ft` | `UWB_IMU/SIGGRAPH_dataset` | `--eval_trans` |
| `eval_gip.sh` | `uip_gip` | `Multi-UWB-Merged` | `--eval_trans` |
| `eval_gip_ft.sh` | `gip_ft` | `Multi-UWB-Merged` | `--eval_trans` |

The `dancedb_tc` checkpoint (clean-AMASS model) is shared by `eval_dancedb` and
`eval_tc`; the `uip_gip` checkpoint (noisy-AMASS model) is shared by `eval_uip`
and `eval_gip`; the `_ft` scripts use the fine-tuned checkpoints. The model
architecture is reconstructed automatically from each checkpoint's sibling
`model_args.json`.

Obtain the released checkpoints (the five folders above) and place them under
`data/checkpoints/` before running.
