# CLIP-Based-Person-ReID-with-FL

Reimplementation of **CO-EVO** (arXiv 2604.26363) from the paper; no official code is public.

- `SPEC.md` what the paper states, the protocols and target numbers, and every gap.
- `ASSUMPTIONS.md` each unstated choice as implemented, with the config key that changes it.
- `RESULTS.md` paper vs ours for every reported setting, from real run logs only.
- `REMOTE_SETUP.md` GPU machine: AnyDesk, CUDA 12.8 PyTorch, data, launching, resuming.

## Status

| Verified | Not verified |
|---|---|
| Unit tests pass (laptop CPU and GPU PC) | CUHK02 loader on real data: the dataset is not available yet (tested on a synthetic copy of the layout only) |
| Smoke run end to end on synthetic data: Phase I, style bank, federated rounds, evaluation, checkpoint, resume; RN50 and ViT backbones; FedPav ablation (CPU and GPU) | Every Protocol I and Protocol III setting and the C2 rows of Protocol II: all need CUHK02 |
| GPU PC (RTX 5080, torch 2.7.1+cu128): AMP path, speed, VRAM; full 60-round runs | ViT rows: the separate backbone learning rate (below) has not been run on the GPU yet |
| Market-1501, MSMT17 V2 and CUHK03 loaders on the full real data (counts match, no missing files); `tools/prepare_cuhk03.py` on the real `cuhk-03.mat` | |
| Real CLIP ViT-B/16 and ImageNet RN50 weights load at 256×128 | |
| Two Protocol II settings (RN50) run end to end: see `RESULTS.md` | |

**ViT backbone fix (2026-11-11):** until this fix the ViT tower was frozen during federated training
(it was deep-copied from the CLIP model after that model was frozen for Phase I), so only the BN
neck, the projection and the classifiers trained. All ViT runs made before the fix are invalid.
RN50 runs were not affected.

**ViT learning rate (2026-11-17):** the paper's single SGD learning rate (1e-3) destroys the CLIP ViT,
so `configs/backbone_vit.yaml` trains the pretrained tower at `lr × fed.optimizer.backbone_lr_scale`
(1e-3, i.e. 1e-6) and the new layers at the paper's 1e-3. RN50 keeps one learning rate and its results
are unchanged. Details: `ASSUMPTIONS.md` A15, `SPEC.md` G25.

## Setup (laptop, CPU)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On the GPU machine install the CUDA 12.8 build of PyTorch **first**; see `REMOTE_SETUP.md` §3.

## Tests and smoke test

```powershell
python -m pytest -q                                           # ~1.5 min on CPU
python tools\make_synthetic.py --out data_synthetic            # tiny random datasets in the real layouts
python train.py --config configs\base.yaml configs\smoke.yaml  # ~1 min on CPU
```

The smoke run uses a small random CLIP and an un-pretrained ResNet-50: it checks that the pipeline
runs, not that it learns.

## Data layout

`data.root` defaults to `data/` (git-ignored). Override with `--set data.root=D:/reid`.

```
data/
  market1501/Market-1501-v15.09.15/{bounding_box_train,bounding_box_test,query}/*.jpg
  msmt17/MSMT17_V2/{mask_train_v2,mask_test_v2}/<pid>/*.jpg  + list_{train,val,query,gallery}.txt
  cuhk02/Dataset/{P1..P5}/{cam1,cam2}/*.png
  cuhk03/cuhk03_release/{images/,meta.json,splits.json}      # made by tools\prepare_cuhk03.py from cuhk-03.mat
```

`train.py` checks every file before starting and stops with a count of what is missing.
Sources and download notes: `SPEC.md` §5. Verification command and expected counts: `REMOTE_SETUP.md` §4.

## Running

```
python train.py --config <base> <overlay> [<overlay> ...] [--set key=value ...]
```

The first file is the full config, later files and `--set` items override keys in order. Unknown
keys are an error. Re-running the same command resumes from `<output_dir>\ckpt\last.pt`.

### Protocol I (leave-one-domain-out), RN50

```powershell
python train.py --config configs\base.yaml configs\p1_market.yaml     # MS+C2+C3 -> M    paper 42.4 / 71.2
python train.py --config configs\base.yaml configs\p3_sources.yaml    # M+C2+C3  -> MS   paper 12.9 / 33.7  (also Protocol III)
python train.py --config configs\base.yaml configs\p1_cuhk03.yaml     # MS+C2+M  -> C3   paper 34.9 / 37.1
```

`p3_sources.yaml` trains on M+C2+C3 and evaluates on MSMT17 and on the three source test splits,
so one run covers both Protocol I →MS and Protocol III. `p1_msmt17.yaml` is the same training with
MSMT17 as the only target.

ViT rows of Table 1: add the backbone overlay and a separate output folder. The overlay also sets
the backbone learning-rate scale (`ASSUMPTIONS.md` A15); sweep it with
`--set fed.optimizer.backbone_lr_scale=...`.

```powershell
python train.py --config configs\base.yaml configs\p1_market.yaml configs\backbone_vit.yaml --set output_dir=runs/p1_market_vit
python train.py --config configs\base.yaml configs\p3_sources.yaml configs\backbone_vit.yaml --set output_dir=runs/p3_sources_vit
python train.py --config configs\base.yaml configs\p1_cuhk03.yaml configs\backbone_vit.yaml --set output_dir=runs/p1_cuhk03_vit
```

### Protocol II (fewer sources), RN50

```powershell
python train.py --config configs\base.yaml configs\p2_ms_c3_market.yaml   # MS+C3 -> M    39.3 / 68.4
python train.py --config configs\base.yaml configs\p2_ms_c2_market.yaml   # MS+C2 -> M    36.9 / 67.1
python train.py --config configs\base.yaml configs\p2_ms_m_cuhk03.yaml    # MS+M  -> C3   24.3 / 25.1
python train.py --config configs\base.yaml configs\p2_ms_c2_cuhk03.yaml   # MS+C2 -> C3   29.4 / 34.2
```

The three-source columns of Table 2 are the Protocol I runs.

### Protocol III (source-domain test splits)

```powershell
python train.py --config configs\base.yaml configs\p3_sources.yaml    # M 81.4 / 93.1, C2 89.8 / 91.2, C3 59.7 / 63.1
```

### Ablation rows (Table 7), used to localise a mismatch

```powershell
python train.py --config configs\base.yaml configs\p1_market.yaml configs\ablation_fedpav.yaml   --set output_dir=runs/p1_market_fedpav    # 25.4 / 49.4
python train.py --config configs\base.yaml configs\p1_market.yaml configs\ablation_csa_only.yaml --set output_dir=runs/p1_market_csa_only  # 38.7 / 66.7
python train.py --config configs\base.yaml configs\p1_market.yaml configs\ablation_gsd_only.yaml --set output_dir=runs/p1_market_gsd_only  # 39.8 / 67.6
```

### Evaluate a checkpoint

```powershell
python evaluate.py --config runs\p1_market\config.yaml --checkpoint runs\p1_market\ckpt\last.pt
```

## Outputs (`<output_dir>/`)

| File | Content |
|---|---|
| `train.log` | full config, git commit, per-round losses, evaluations |
| `config.yaml`, `git.json` | resolved config and commit of the run |
| `metrics.csv`, `tb/` | scalars: CSV and TensorBoard |
| `results.json` | per target: final-round metrics, best evaluated round, all evaluations |
| `ckpt/last.pt` | rewritten every round (global model, client classifiers, style bank, RNG state) |
| `ckpt/round_XXX.pt` | kept every `ckpt.keep_every` rounds (default 10) |
| `style_bank.pt` | the GCSB |

Phase I anchors are cached in `cache/anchors/` and shared by all runs with the same Phase I
settings (the MSMT17 anchors are learned once and reused by every protocol).

**Report the `final` numbers.** `best` is selected on the target test set and is logged only to
compare with DACS-style reporting (`ASSUMPTIONS.md` G10).

## Code layout

```
train.py, evaluate.py        entry points
configs/                     base.yaml + overlays (protocols, backbone, ablations, smoke)
coevo/
  config.py                  YAML overlays and --set overrides
  runner.py                  Phase I -> style bank -> federated loop -> evaluation, checkpoint/resume
  phase1.py                  CSA prompt learning, anchor cache
  losses.py                  L_i2t, L_t2i, L_c3, L_id, L_tri, L_align
  style.py                   camera templates, StyleBank (GCSB), re-normalisation (GSD)
  evaluation.py              feature extraction, mAP / CMC
  models/backbones.py        ResNet50Encoder, ClipViTEncoder
  models/clip_text.py        CLIP construction, PromptLearner, TextEncoder
  fed/client.py              Client: local data, classifier, anchors, local update; build_optimizer
  fed/server.py              FedAvgAggregator
  data/datasets.py           the four dataset index builders
  data/loader.py             transforms, identity sampler, loaders
  utils.py                   seeding, logging, checkpoints, git info
tools/                       make_synthetic.py, prepare_cuhk03.py
tests/                       losses, style re-normalisation, aggregation, sampler, metrics, loaders, models, optimizer groups
```

Extension points for later work (nothing below is implemented):

| To add | Where |
|---|---|
| another backbone | a module returning `{"feat", "feat_bn", "proj"}` with a `backbone_parameters()` method (its pretrained weights) + one branch in `models.build_image_encoder` |
| another aggregation rule or a privacy step on uploads | a class with `aggregate(states, num_samples)` + `fed.build_aggregator`; client uploads pass through `runner.run` in one place |
| noise on shared style statistics | `StyleBank.from_client_templates` receives every client's templates before the bank is built |
| a text-query branch | anchors and the CLIP text tower are built in `phase1.py`; `Client.anchors` is the only place they enter Phase II |
| extra loss terms | `losses.py` + the view loop in `Client.train_round` |
