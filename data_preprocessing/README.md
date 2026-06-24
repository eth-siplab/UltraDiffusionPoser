# UDP data preparation

Helper scripts to fetch the body model and datasets into `data/`.

Use **`download_all.sh`** for the hands-off path: it asks what you want and which
logins are needed **all up front**, then downloads everything unattended so you
can step away (see [Usage](#usage)). Each `download_*.sh` can also be run on its
own, in which case it asks before downloading. If you decline a download, the
script prints which `config/config.py` path to point at your existing copy
instead.

## Scripts

| Script | Gets | Lands at | Account |
|--------|------|----------|---------|
| `download_smpl.sh` | SMPL v1.0.0 male body model | `data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` (+ `smpl_m_lbs_…` alias) | SMPL |
| `download_uip.sh` | UIP-DB processed data (`train.pt`, `test.pt`) | `data/processed_data/UWB_IMU/SIGGRAPH_dataset/` | Google Drive (gdown) |
| `download_dip.sh` | DIP-IMU dataset (~2.6 GB, nested zip) | `data/DIP_IMU/s_01..s_10/*.pkl` | DIP |
| `download_tc.sh` | TotalCapture Vicon GT skeleton `S1..S5` (`sX_vicon_pos_ori`) **and** DIP GT pkls | `data/TotalCapture/S{1..5}/<motion>/gt_skel_gbl_*.txt` and `data/TotalCapture_Real_60FPS/*.pkl` | cvssp **+** DIP |
| `download_amass.sh` | the SMPL-H AMASS datasets used by the project | `config.paths.raw_amass_dir/<dataset>/…` | AMASS |
| `download_checkpoints.sh` | released UDP model weights (5 evaluated checkpoints) | `data/checkpoints/<name>/ckpt/…` | Google Drive (gdown) |
| `download_all.sh` | any/all of the above; all prompts + logins up front, then unattended | — | — |

GIP data is not handled here yet.

### AMASS

`download_amass.sh` handles **only** the datasets in `config.amass_data` +
`config.amass_test_data` (not all of AMASS). Several were renamed on the AMASS
server, e.g. `MPI_HDM05←HDM05`, `MPI_mosh←MoSh`, `Transitions_mocap←Transitions`,
`SSM_synced←SSM`, `Eyes_Japan_Dataset←EyesJapanDataset`, `TCD_handMocap←TCDHands`,
`BioMotionLab_NTroje←BMLrub`, `MPI_Limits←PosePrior`, `DFaust_67←DFaust`.

It builds a **view directory** (`config.paths.raw_amass_dir`, override with
`AMASS_RAW_DIR`) with one entry per dataset, named with the **config** name. For
each dataset:

1. already in the view → left as-is;
2. else, if `AMASS_SRC_DIR` (an existing AMASS install) contains it under the
   config name *or* the current server name → a **symlink** into it. Your
   existing AMASS is **never renamed or modified** — handy if other code relies
   on the original folder names;
3. else → downloaded from `amass.is.tue.mpg.de` into the view under the config name.

`preprocess.py` reads the view dir and follows the symlinks transparently.

```bash
# reuse an existing AMASS install (any naming) without touching it:
AMASS_RAW_DIR=data/AMASS_SMPLH AMASS_SRC_DIR=/path/to/your/AMASS \
  ./data_preprocessing/download_amass.sh
# then point the code at the view:
#   export AMASS_RAW_DIR=data/AMASS_SMPLH   (or set paths.raw_amass_dir in config.py)
```

## Accounts (three separate registrations)

| Account | Used by | Site |
|---------|---------|------|
| **SMPL** | `download_smpl.sh` | smpl.is.tue.mpg.de |
| **DIP** | `download_dip.sh` (DIP data) **and** `download_tc.sh` (TotalCapture GT) | dip.is.tue.mpg.de |
| **AMASS** | `download_amass.sh` | amass.is.tue.mpg.de |
| **cvssp** | `download_tc.sh` (TotalCapture Vicon GT skeleton S1..S5) | cvssp.org (non-MPG) |

The DIP account covers both the DIP dataset and the TotalCapture ground truth.
`download_all.sh` asks for it once and reuses it for both.

**Credentials are kept only in non-exported shell variables** — never `export`ed
into the environment (so other programs in your shell can't read them) and never
written to disk. You can pre-seed them non-interactively via the
`*_USERNAME/_PASSWORD` environment variables below.

## Usage

```bash
./data_preprocessing/download_all.sh    # recommended: choose all-or-some, enter
                                        # the needed logins once, then walk away
```

`download_all.sh` first asks whether to fetch **everything** or **choose
individually**, then checks what's already on disk and which logins are still
needed, collects exactly those credentials once, and finally downloads with no
further prompts.

You can also run a single step directly (it checks whether the target already
exists and prompts before downloading):

```bash
./data_preprocessing/download_smpl.sh
./data_preprocessing/download_uip.sh
./data_preprocessing/download_dip.sh
./data_preprocessing/download_tc.sh
```

Each script activates the `UIP` conda env (for `python`/`gdown`).

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTO_YES` | `0` | set `1` to answer "yes" to every prompt — in `download_all.sh` this selects *everything* (non-interactive; pre-seed logins via the `*_USERNAME/_PASSWORD` vars) |
| `CONDA_SH` / `UDP_ENV` | `~/miniconda3/.../conda.sh` / `UDP` | conda env to activate |
| `DATA_DIR` | `data` | root for downloads/extraction |
| `DOWNLOAD_CACHE` | `data/downloads` | where archives are cached |
| `UWBIMU_DIR` | `data/processed_data/UWB_IMU/SIGGRAPH_dataset` | UIP data target |
| `DIP_ZIP` | _(unset)_ | reuse an already-downloaded `DIPIMUandOthers.zip` (skip the download) |
| `TMPDIR` | `/tmp` | scratch space for the ~2 GB inner DIP zip |
| `TUE_SMPL_USERNAME` / `TUE_SMPL_PASSWORD` | _(unset)_ | pre-seed the SMPL login |
| `TUE_DIP_USERNAME` / `TUE_DIP_PASSWORD` | _(unset)_ | pre-seed the DIP login (covers DIP + TotalCapture GT) |
| `CVSSP_USERNAME` / `CVSSP_PASSWORD` | _(unset)_ | pre-seed the cvssp TotalCapture login |
| `TUE_AMASS_USERNAME` / `TUE_AMASS_PASSWORD` | _(unset)_ | pre-seed the AMASS login |
| `TC_OFFICIAL_DIR` / `TC_GT_DIR` | `data/TotalCapture` / `data/TotalCapture_Real_60FPS` | TotalCapture targets |
| `AMASS_RAW_DIR` | `config.paths.raw_amass_dir` | AMASS target dir |

Example — extract DIP from a local copy without re-downloading the 2.6 GB archive:

```bash
DIP_ZIP=/path/to/DIPIMUandOthers.zip ./data_preprocessing/download_dip.sh
```

## Disk / nesting notes

- `DIPIMUandOthers.zip` nests `DIP_IMU_and_Others/DIP_IMU.zip`, which in turn
  contains `DIP_IMU/s_*/*.pkl`. `download_dip.sh` pulls the inner zip into
  `$TMPDIR` (~2 GB) and unpacks it into `data/`, then deletes the temp copy.
  Peak transient disk usage is roughly outer (2.6 GB) + inner (2 GB) + extracted
  (2 GB); delete the cached archive afterwards to reclaim space.
- `download_uip.sh` stages the Google Drive folder in a temp dir and moves only
  `train.pt`/`test.pt` into place.

## After downloading

These produce the **raw**/processed inputs. To (re)generate the synthetic AMASS
and TotalCapture caches the models train/eval on, run the preprocessing entry
point as described in the top-level docs:

```bash
python modules/dataset/preprocess.py
```
