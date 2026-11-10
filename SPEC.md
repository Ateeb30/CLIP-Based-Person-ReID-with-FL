# CO-EVO reimplementation spec

Source: `CO-EVO paper.pdf` (arXiv 2604.26363v1, 12 pages incl. Appendix A.1), read in full.
Reference code actually read for defaults:

- **CLIP-ReID** (`Syliz517/CLIP-ReID`): prompt learner, stage-1/stage-2 processors, configs.
- **DACS** (`FlyingRoastDuck/DACS_official`): the FedDG-ReID codebase whose numbers CO-EVO's baseline rows copy. CO-EVO's stated optimizer settings (SGD, lr 1e-3, wd 5e-4, momentum 0.9, batch 64, MultiStepLR) are identical to DACS's argparse defaults, so DACS is treated as the most likely base code.
- CO-EVO official repo: `git ls-remote` fails (confirmed not public). SSCU code was not consulted.

Tags used below: **[P]** stated in the paper, **[D]** DACS code default, **[C]** CLIP-ReID code default, **[A]** my assumption.

---

## 1. What the paper states

### 1.1 Overall pipeline (Alg. 1, Fig. 2)

1. **Phase I (CSA)**, per client, in parallel, no aggregation: learn per-identity prompt tokens with frozen CLIP encoders; cache text prototypes `T_k` locally; compute per-camera style templates and upload them.
2. Server unions the templates into the **GCSB** `B` and broadcasts `B` and `θ0`.
3. **Phase II/III (coupled loop)**, `R` rounds: each client sets `θ_k ← θ_{r-1}`, trains `E` local epochs on original + stylized views against its fixed anchors, uploads `θ_k`; server does weighted FedAvg.
4. Output: global image encoder `θ_R`.

"Co-evolution" is explicitly *not* symmetric updating: anchors are frozen, only the image encoder trains in the loop [P, Sec. 1 and 3.1].

### 1.2 Backbone / CLIP

- Vision-language component: **CLIP ViT-B/16** "as the base for semantic anchoring" [P, App.].
- Image encoder in the loop: "compatible with ResNet-50 (RN50) and ViT" [P]. Table 1 reports both. Which RN50 (ImageNet vs CLIP) and how its features meet the text space is **not stated** (Gap G1).
- Text prototype dimension `D` [P, Eq. 4]; 512 for CLIP ViT-B/16.

### 1.3 Phase I: CSA

- Template: `"a photo of a [X_1]...[X_L] person"`, **L = 4** learnable tokens per identity [P]. Same as CLIP-ReID (`"A photo of a X X X X person."`, `n_cls_ctx=4`, init N(0, 0.02)) [C].
- Frozen: both CLIP encoders. Trained: tokens only [P].
- Batch `{(x_i, y_i, c_i)}`, `s(v,t)` = cosine similarity, temperature `τ` [P].

```
L_i2t(i) = -log  exp(s(v_i, t_{y_i})/τ) / Σ_{a=1..B} exp(s(v_i, t_a)/τ)                    (1)
L_t2i(y) = -1/|P(y)| Σ_{p∈P(y)} log  exp(s(v_p, t_y)/τ) / Σ_{a=1..B} exp(s(v_a, t_y)/τ)   (2)
L_c3     = Σ_{y∈Y_batch} Σ_{i,j∈P(y), c_i≠c_j} ( s(v_i,t_y) - s(v_j,t_y) )²               (3)
L_CSA    = Σ L_i2t + Σ L_t2i + λ_c3 · L_c3
T_k      = {t_y}_{y∈Y_k} ∈ R^{|Y_k|×D}   (cached, frozen)                                   (4)
```

- **λ_c3 = 0.1**, **τ = 0.07** (stated for "the semantic alignment loss") [P, App.].
- Duration: "semantic learning phase for **120 rounds**" [P]. (CLIP-ReID ViT stage 1 = 120 epochs.)
- Prototypes never leave the client; the text encoder is never transmitted [P, Sec. 3.1].

### 1.4 Phase II: GSD and the GCSB

- Template per client `k`, camera `c`: channel-wise mean and variance over that camera's images [P]:
  `(μ_{k,c}, σ²_{k,c}) = Stat({x_i^k | c_i^k = c})` (5)
- Bank: `B = ∪_k ∪_{c∈C_k} {(μ_{k,c}, σ²_{k,c})}`. One entry per (client, camera). Global, i.e. includes other clients' cameras [P, Table 9].
- Re-normalization [P]:
  `x̂ = (x − μ(x)) / sqrt(σ²(x) + ε)`, `x' = x̂ ⊙ sqrt(σ²_s) + μ_s`, `(μ_s, σ²_s) ~ B` (6)
- Operates on the **image** (`x`, "stylized views", Fig. 3c shows stylized images; "photometric variation"). No trainable parameters.
- Built **once**, "at the beginning of Phase II" (App.) / uploaded at the end of Phase I (Alg. 1 line 5); ~4 s per client; no further communication in later rounds [P].
- If camera IDs are missing: replace `c` with K-means pseudo-groups (Table 4 only) [P].

### 1.5 Coupled local objective

For each view `x̃ ∈ {x, x'}` [P]:

```
L_id(x̃)    = -1/B Σ_i log p_{i,y_i}(x̃)                                         (7)
L_tri(x̃)   = 1/B Σ_i max( d_p^(i)(x̃) − d_n^(i)(x̃) + α, 0 )                    (8)
L_align(i;x̃)= -log exp(s(v_i(x̃), t_{y_i})/τ) / Σ_{y∈Y_k} exp(s(v_i(x̃), t_y)/τ) (9)
L_loc      = Σ_{x̃∈{x,x'}} ( L_id + L_tri + λ·L_align )                           (10)
```

- Align negatives: **all** of the client's identities `Y_k`, not just the batch.
- Weights: id = 1, tri = 1, align = `λ` (**value not given**, G3). Margin `α` not given (G8).

### 1.6 Federated setup

- **Domain-as-client**: each source dataset is one client; `K` = 2 or 3 [P].
- **R = 60** rounds, **E = 1** local epoch, all clients every round [P].
- Aggregation: `θ^r = Σ_k (n_k/N) θ_k^r` [P].
- Communicated: style templates up once, bank + `θ` down, `θ_k` up each round. Prompts, prototypes, text encoder stay local [P].

### 1.7 Optimisation (federated phase)

- SGD, lr **1e-3**, weight decay **5e-4**, momentum **0.9**, batch **64** [P].
- MultiStepLR "with decay factors at specific communication rounds"; milestones and gamma **not given** (G5).
- Phase I optimizer, LR, batch, sampler: **not given** (G6).
- Sampler, augmentations, input size: **not given** (G7, G9).

### 1.8 Evaluation

- mAP and CMC Rank-1 [P]. Nothing else stated (distance, feature, CUHK03 split, model selection: G10, G11, G12).

---

## 2. Protocols and target numbers

Datasets: Market-1501 (M), CUHK02 (C2), CUHK03 (C3), MSMT17 (MS).

### Protocol I: leave-one-domain-out (Table 1). C2 is never a target.

| Backbone | MS+C2+C3→M | M+C2+C3→MS | MS+C2+M→C3 | Avg |
|---|---|---|---|---|
| RN50 | 42.4 / 71.2 | 12.9 / 33.7 | 34.9 / 37.1 | 30.1 / 47.3 |
| ViT  | 60.7 / 80.2 | 32.2 / 60.3 | 51.3 / 52.7 | 48.1 / 64.4 |

(mAP / R1.) Reference rows, RN50: SSCU 39.5/66.4, 11.9/32.3, 32.8/34.1; DACS 36.3/61.2, 10.4/27.5, 30.7/34.1; FedPav 25.4/49.4, 5.2/15.5, 22.5/24.3. ViT: DACS 45.4/70.7, 20.3/44.2, 36.6/42.1.

### Protocol II: fewer sources, MS always a source (Table 2, RN50 numbers)

| Sources→Target | mAP / R1 |
|---|---|
| MS+C3→M | 39.3 / 68.4 |
| MS+C2→M | 36.9 / 67.1 |
| MS+C2+C3→M | 42.4 / 71.2 |
| MS+M→C3 | 24.3 / 25.1 |
| MS+C2→C3 | 29.4 / 34.2 |
| MS+C2+M→C3 | 34.9 / 37.1 |

### Protocol III: source-domain test splits, clients {M, C2, C3} (Table 3)

| Test on | mAP / R1 |
|---|---|
| M  | 81.4 / 93.1 |
| C2 | 89.8 / 91.2 |
| C3 | 59.7 / 63.1 |

One training run (M+C2+C3), which is also the Protocol I →MS run.

### Ablation rows usable as intermediate checkpoints (RN50, →M / →MS / →C3)

| Config | →M | →MS | →C3 |
|---|---|---|---|
| no CSA, no GSD (= FedPav row) | 25.4/49.4 | 5.2/15.5 | 22.5/24.3 |
| CSA only | 38.7/66.7 | 9.1/30.7 | 33.8/35.2 |
| GSD only | 39.8/67.6 | 10.8/30.3 | 34.1/35.9 |
| both | 42.4/71.2 | 12.9/33.7 | 37.1/38.9 |
| GSD local bank | 40.2/68.5 | 10.5/31.4 | 35.1/36.3 |
| λ_c3 = 0 | 41.2/69.8 | 11.5/31.8 | 35.8/37.5 |
| L = 1 / 8 / 16 | 40.8/69.4, 41.9/70.7, 41.5/70.2 | 10.3/29.5, 12.1/32.8, 11.5/31.6 | 35.5/36.8, 36.6/38.2, 36.2/37.7 |

### Inconsistencies in the paper (not resolvable from the text)

1. **→C3, RN50**: Tables 1 and 2 give **34.9 / 37.1**; Tables 4-9 give **37.1 / 38.9** for the same full model. Table 1's average (30.1/47.3) is computed from 34.9/37.1. Target: Table 1; treat 34.9-37.1 mAP as the acceptable band for this setting.
2. **GCSB update**: Sec. 3.3, Alg. 1 and the Appendix say the bank is built once; the last paragraph of Sec. 3.4 says the server "synchronizes the GCSB to incorporate evolving camera statistics". Following the three-against-one majority: built once.
3. The "no CSA, no GSD" ablation row is digit-for-digit the FedPav row, which itself matches DACS's published FedPav numbers. So the ablation baseline was probably not re-run, and it implies the RN50 pipeline is the DACS one (ImageNet RN50).
4. Phase I is described in "rounds" although nothing is aggregated in Phase I.
5. Table 2 FedPav MS+C2+C3→M R1 is 49.1 vs 49.4 in Table 1 (cosmetic).

---

## 3. GAPS

Sensitivity: **H** = plausibly > 2 mAP, **M** = 0.5-2, **L** = < 0.5. These are judgement calls, not measurements.

| # | Unspecified detail | Proposed default | Reason | Sens. |
|---|---|---|---|---|
| G1 | RN50 variant, and how a 2048-d CNN feature is aligned with 512-d CLIP ViT-B/16 text anchors | **RN50 rows**: ImageNet torchvision RN50 exactly as DACS (last stride 1, GAP, BN w/o affine, 2048-d) + one bias-free linear 2048→512 used only for `L_align`, aggregated with `θ`. Phase I always uses frozen CLIP ViT-B/16. **ViT rows**: CLIP ViT-B/16 image encoder as the trainable backbone (stride 16), its native 512-d projection used for `L_align`. | Paper says ViT-B/16 is the anchoring base and RN50/ViT are swappable backbones; ablation baseline = DACS FedPav (ImageNet RN50). ViT row's +15 mAP over DACS(ViT) is only plausible with CLIP init. Alternative: CLIP RN50 end to end (CLIP-ReID CNN config, 1024-d). | **H** |
| G2 | Meaning of "E = 1 epoch" | One pass of the PK sampler = `num_ids × 4 / 64` iterations (≈65 MS, 46 M, 98 C2, 85 C3). | Literal reading. DACS instead runs a fixed **200 iterations** per round per client (≈2-4× more steps); if CO-EVO kept that and called it an epoch, we under-train. | **H** |
| G3 | `λ` (weight of `L_align`) | 1.0 | CLIP-ReID `I2T_LOSS_WEIGHT = 1.0`. | **M-H** |
| G4 | CUHK03 version and split | DACS loader: open-reid format (`meta.json`, `splits.json`), split 0, 1,367 train IDs, 100 test IDs, query = last image per camera per ID, gallery = the rest. | Baseline numbers come from this loader. Alternative: CUHK03-NP 767/700. Labeled vs detected boxes also unknown (default: whatever the open-reid conversion yields; must be confirmed when the data arrives). | **H** for every →C3 number and for C3-as-source |
| G5 | MultiStepLR milestones / gamma | milestones **[20, 30]**, gamma **0.5**, stepped per round | DACS defaults, which stay in force if DACS is run with `--epochs 60`. Alternative: [30, 50], 0.1 (CLIP-ReID stage 2, also 60 epochs). | **M** |
| G6 | Phase I optimisation | Adam, lr 3.5e-4, wd 1e-4, cosine schedule, 5 warm-up epochs from 1e-5, **120 local epochs**, batch 64, image features pre-extracted once from un-augmented images, AMP | CLIP-ReID ViT stage 1; "120 rounds" read as 120 epochs because Phase I has no aggregation. | **M** |
| G7 | Phase I batch sampler | PK sampler (16 IDs × 4, other-camera positives preferred) | `L_c3` needs same-ID / different-camera pairs in the batch; CLIP-ReID's random-permutation batches almost never contain them. Deviation from CLIP-ReID forced by Eq. 3. | **M** |
| G8 | Triplet margin `α`, mining, distance | Batch-hard, Euclidean, margin **0.5**, on the BN feature | DACS `TripletLoss(margin=0.5)`. Alternative 0.3 (CLIP-ReID). | **L-M** |
| G9 | Input size, augmentation, normalisation | 256×128; resize (bicubic), h-flip 0.5, pad 10 + random crop, normalise, random erasing 0.5 (mean fill). ImageNet mean/std for RN50; CLIP-ReID's 0.5/0.5 for CLIP ViT and for Phase I feature extraction. | DACS / CLIP-ReID, which agree except on normalisation. | **L-M** |
| G10 | Model selection | Report **final round (60)**. Also evaluate every 10 rounds and log the best round separately. | DACS keeps the best-on-target checkpoint (evaluated every 10 rounds), which selects on the test set; the paper is silent. Both numbers go in RESULTS.md, labelled. | **M** |
| G11 | Test feature and distance | RN50: 2048-d BN feature, Euclidean distance, no L2-norm, no re-ranking. ViT: 768-d BN-neck feature, same. Same-ID-same-camera gallery entries removed; Market junk (pid −1) dropped. | DACS evaluator. | **L-M** |
| G12 | MSMT17 version and train split | Paper/DACS: **MSMT17_V1**, train+val merged (1,041 IDs, 32,621 images), standard query/gallery lists. We have V2. | DACS loader. V2 has the same identities and split lists but blurred faces and a different folder/filename layout; the size of the shift is unknown, not expected to be large. | **L-M** |
| G13 | Template statistics (Eq. 5) | Computed on the normalised, un-augmented 256×128 tensor; `μ_{k,c}` = mean over the camera's images of per-image channel means, `σ²_{k,c}` = mean of per-image channel variances. 3+3 numbers per camera. | Eq. 6 uses per-image `μ(x), σ²(x)` (same as DACS's `norm_image`), so the template should live in the same per-image space. Alternative: pooled-pixel variance (larger σ²). | **M** |
| G14 | Template sampling | One template drawn uniformly from the whole bank **per image**, every step, own cameras included; `x'` made from the augmented `x`; stylisation probability 1; `ε = 1e-8` | "(μ_s, σ²_s) ~ B"; Alg. 1 line 12. Alternatives: one per batch; exclude own client. | **M** |
| G15 | Classifier for `L_id` | One bias-free linear per client on the BN feature, kept local, never aggregated; plain CE, no label smoothing | DACS; identity sets are disjoint; Eq. 7 has no smoothing. | **L-M** |
| G16 | Normalisation of the CSA sums | `L_i2t` mean over batch, `L_t2i` mean over batch IDs, `L_c3` literal sum over ordered pairs | CLIP-ReID averages; Eq. 3 is written as a plain sum. Only the ratio to `λ_c3` matters. | **L-M** |
| G17 | Duplicate IDs in the Eq. 1 denominator | Implement literally (denominator runs over batch samples, so each ID's prototype appears 4× under PK) | Paper equation. CLIP-ReID's SupCon form is the alternative. | **L** |
| G18 | Two views in one forward pass or two | Two separate forward passes, losses summed, one backward | DACS forwards original and augmented images separately; matters only through BatchNorm statistics. | **L-M** |
| G19 | Local optimizer state, BN buffers | Optimizer (momentum) re-created every round; all float buffers incl. BN running stats averaged; `n_k` = number of training images | DACS. | **L** |
| G20 | ViT head details | Single BN-neck on the 768-d CLS feature; `L_id` and `L_tri` on it as in Eqs. 7-8 (not CLIP-ReID's summed multi-feature losses, ID weight 0.25, SIE, or stride 12) | Paper's equations have one term each. | **M** (ViT rows only) |
| G21 | `τ` in Phase I | 0.07, same as `L_align` | Only one τ is defined. CLIP-ReID stage 1 effectively uses un-normalised features with temperature 1. | **M** |
| G22 | CUHK02 split | P1-P4 train (1,577 IDs), P5 test (cam1 = query, cam2 = gallery) | DACS loader. | **L** |
| G23 | AMP, seeds | AMP on in Phase II (your 16 GB requirement); seed 1 | Infra choice, not in paper. DACS seed = 1. | **L** |
| G24 | Table 4 settings (noise model, K for K-means) | Not implemented in the first pass | Robustness study, not needed for Protocols I-III. | n/a |

---

## 4. Gaps ranked by likely mAP impact, with the cheapest test

All tests: Protocol I, **MS+C2+C3→M**, RN50, one seed, change one thing against the default. Market is the cheapest target to evaluate and has the widest ablation coverage in the paper. CUHK03 tests use MS+C2+M→C3.

| Rank | Gap | Cheapest experiment | What decides it |
|---|---|---|---|
| 1 | G1 backbone / projection | (a) default vs (b) CLIP RN50 end to end. Run the "no CSA, no GSD" config first: it must land near 25.4 / 49.4. | If our FedPav baseline is already far above 25.4, the backbone is not ImageNet RN50 as DACS uses it. |
| 2 | G2 local epoch length | sampler-epoch vs fixed 200 iterations, 20 rounds each | Compare the trajectory to the baseline row and to 42.4; under-training shows as a curve still rising at round 60. |
| 3 | G4 CUHK03 split | Evaluate the same →C3 checkpoint under the DACS split and under CUHK03-NP | No retraining. Whichever protocol puts FedPav near 22.5 / 24.3 is the one in use. |
| 4 | G3 `λ` | λ ∈ {0.5, 1, 2} on the CSA-only config | Target is the CSA-only row, 38.7 / 66.7. |
| 5 | G10 model selection | None needed: log final and best-of-every-10 | If only "best" matches, the paper used DACS-style selection. |
| 6 | G5 LR milestones | [20,30]/0.5 vs [30,50]/0.1 | Full 60-round run each; do after ranks 1-4 are settled. |
| 7 | G13/G14 bank statistics and sampling | GSD-only config: per-image-mean variance vs pooled variance; per-image vs per-batch sampling | Target is the GSD-only row, 39.8 / 67.6. Visual check of stylised images against Fig. 3(c) costs nothing. |
| 8 | G6/G7/G21 Phase I recipe | Re-run Phase I only (frozen encoders, minutes on GPU) with τ ∈ {0.07, 1.0} and PK vs random batches, then a 20-round Phase II | Phase I is cheap; Phase II cost dominates. |
| 9 | G8 triplet margin | 0.5 vs 0.3 | Low priority. |
| 10 | G12 MSMT17 V1 vs V2 | Cannot be tested without V1 | Affects every setting (MS is a source or the target in all of them). Ask the authors which version they used. |

Order of work this implies: FedPav baseline row → CSA-only row → GSD-only row → full model. Each row has a paper number, so a mismatch is localised before the next component is added.

---

## 5. Data needed

| Dataset | Used as | Where to get it | Notes |
|---|---|---|---|
| MSMT17 | source in all but one run, target in one | Request from the PKU VMC lab (dataset page of Wei et al., agreement form by e-mail). Only V2 is distributed now. | You have V2, **but the copy in this folder is incomplete**, see below. |
| Market-1501 | source / target | Liang Zheng's project page (`Market-1501-v15.09.15.zip`); no application needed | Standard 751 / 750 split. |
| CUHK03 | source / target | CUHK person re-ID datasets page (Xiaogang Wang's group), request form; file `cuhk03_release.zip` containing `cuhk-03.mat` | Needs conversion to the open-reid `meta.json` / `splits.json` layout (G4). CUHK03-NP split files come from Zhong Zhun's `person-re-ranking` repo if we need the alternative. |
| CUHK02 | source only (target only in Protocol III) | Same CUHK page and form | Folder `Dataset/P1..P5/cam1,cam2/*.png`. |

Links are from memory and were not checked today; the CUHK page has been intermittently unavailable in the past, so request early.

## 6. Problems found in this folder

1. **`MSMT17_V2` is incomplete.** `mask_train_v2/` is missing entirely. `mask_test_v2/` has 2,017 identity folders (0000-2016) and 63,090 images; the lists reference 3,060 identities and 93,820 test images. `list_train.txt` (30,248) + `list_val.txt` (2,373) are present but point at files that are not here. Looks like an interrupted extraction. The partial test set is still enough for CPU checks of the loader and evaluator.
2. Not a git repository yet. `__MACOSX/` and `.DS_Store` are archive junk and will be ignored along with the dataset.
3. Laptop has Python 3.13.7 and no PyTorch. A CPU build of a current PyTorch is needed for the smoke tests.
