# Results: paper vs ours

Paper: arXiv 2604.26363v1, Tables 1-3. Ours: the **final round (60)** of each run, copied from that
run's `results.json` / `train.log` on the GPU PC (RTX 5080). The run folders are not in the repo.
All numbers are mAP / Rank-1 in %. Success criterion: within 1-2 mAP of the paper on every reported
setting. Every unstated choice behind these numbers is in `ASSUMPTIONS.md`.

Last updated 2026-11-17. Settings run: 2 of 13.

## Protocol II (Table 2), RN50

| Setting | Paper mAP / R1 | Ours mAP / R1 | Diff mAP / R1 | Run dir | Commit |
|---|---|---|---|---|---|
| MS+C3→M | 39.3 / 68.4 | 39.26 / 63.27 | −0.04 / −5.13 | `runs/p2_ms_c3_market` | 59245ee |
| MS+M→C3 | 24.3 / 25.1 | 31.56 / 34.00 | **+7.26** / +8.90 | `runs/p2_ms_m_cuhk03` | 59245ee |
| MS+C2→M | 36.9 / 67.1 | not run: CUHK02 missing | | | |
| MS+C2→C3 | 29.4 / 34.2 | not run: CUHK02 missing | | | |

The three-source columns of Table 2 are the Protocol I settings below.

Notes on the two runs:

- Configs: `configs/base.yaml` plus the protocol overlay, no overrides.
- Best evaluated round = round 60 on both, so best = final.
- `git.json` says `dirty: true` on both. The only cause is an untracked `temp.txt` on the PC (the
  dirty check counts untracked files); no tracked file differed from commit 59245ee.
- **MS+C3→M:** mAP is within tolerance. R1 is 5.1 below the paper: open, not yet investigated.
- **MS+M→C3:** +7.26 mAP, outside tolerance, and not yet a like-for-like comparison. The paper does
  not say which CUHK03 test protocol it uses, and the same checkpoint scores 31.56 under the protocol
  used here (old split 0) and 16.35-17.37 under CUHK03-NP. See `SPEC.md` §3.1 (G4).

## Protocol I (Table 1), RN50

| Setting | Paper mAP / R1 | Ours mAP / R1 | Diff mAP / R1 | Run dir | Commit |
|---|---|---|---|---|---|
| MS+C2+C3→M | 42.4 / 71.2 | not run: CUHK02 missing | | | |
| M+C2+C3→MS | 12.9 / 33.7 | not run: CUHK02 missing | | | |
| MS+C2+M→C3 | 34.9 / 37.1 (37.1 / 38.9 in Tables 4-9) | not run: CUHK02 missing | | | |

## Protocol I (Table 1), ViT

| Setting | Paper mAP / R1 | Ours mAP / R1 | Diff mAP / R1 | Run dir | Commit |
|---|---|---|---|---|---|
| MS+C2+C3→M | 60.7 / 80.2 | not run: CUHK02 missing; ViT LR pending (A15) | | | |
| M+C2+C3→MS | 32.2 / 60.3 | not run: CUHK02 missing; ViT LR pending (A15) | | | |
| MS+C2+M→C3 | 51.3 / 52.7 | not run: CUHK02 missing; ViT LR pending (A15) | | | |

"ViT LR pending": the paper's single SGD learning rate cannot train the CLIP ViT, so the ViT rows use a
separate backbone learning rate (`ASSUMPTIONS.md` A15, `SPEC.md` G25). It has not yet been validated
on the GPU.

## Protocol III (Table 3), RN50, clients {M, C2, C3}

| Test split | Paper mAP / R1 | Ours mAP / R1 | Diff mAP / R1 | Run dir | Commit |
|---|---|---|---|---|---|
| M | 81.4 / 93.1 | not run: CUHK02 missing | | | |
| C2 | 89.8 / 91.2 | not run: CUHK02 missing | | | |
| C3 | 59.7 / 63.1 | not run: CUHK02 missing | | | |
