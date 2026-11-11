# Assumptions

Everything the paper does not state, as implemented. `G#` refers to the GAPS table in `SPEC.md`.
Each row names the config key (in `configs/base.yaml`) that changes it, so every assumption can be
tested without touching code. Rows marked **code** are structural and need a code change.

## Carried over from SPEC.md

| G# | Assumption as implemented | Config key | Source |
|---|---|---|---|
| G1 | RN50 rows: ImageNet torchvision ResNet-50 (`IMAGENET1K_V1`), last stride 1, GAP, BatchNorm without affine, plus one bias-free linear 2048→512 on the BN feature used only by `L_align`; it is part of `θ` and is aggregated. ViT rows: CLIP ViT-B/16 image tower, stride 16, CLIP's own 768→512 projection for `L_align`. Phase I always uses frozen CLIP ViT-B/16. | `model.backbone`, `model.resnet50.*` | DACS / inference |
| G2 | One local epoch = one pass of the identity sampler = `floor(ids × 4 / 64)` iterations. | `fed.local_epochs`, `fed.local_iters` (set to 200 for DACS behaviour) | literal reading |
| G3 | `λ = 1.0` | `loss.align_weight` | CLIP-ReID |
| G4 | CUHK03 in open-reid format, split 0, labeled + detected boxes merged, query = last image per (id, camera), gallery = the rest. | `data.datasets.cuhk03.split_id`; other protocols: **code** | DACS |
| G5 | LR × 0.5 at rounds 20 and 30 (0-based round index, as DACS's `scheduler.step(epoch)`). | `fed.lr_milestones`, `fed.lr_gamma` | DACS |
| G6 | Phase I: Adam, lr 3.5e-4, wd 1e-4, cosine to 1e-6, 5 warm-up epochs from 1e-5, 120 epochs, batch 64, image features extracted once from un-augmented images. | `phase1.*` | CLIP-ReID stage 1 |
| G7 | Phase I batches come from the identity sampler (16 ids × 4, other cameras preferred) so that `L_c3` has pairs to act on. | **code** | forced by Eq. 3 |
| G8 | Batch-hard triplet, Euclidean, margin 0.5, on the BN feature. | `loss.triplet_margin`, `loss.triplet_feature` | DACS |
| G9 | 256×128, bicubic resize, flip 0.5, pad 10 + crop, random erasing 0.5. ImageNet mean/std for RN50; 0.5/0.5 for CLIP ViT and for Phase I. | `data.*`, `model.resnet50.pixel_*`, `clip.pixel_*` | DACS / CLIP-ReID |
| G10 | Evaluation every 10 rounds and at round 60. `results.json` holds both the final-round numbers and the best evaluated round; the final round is the one to report. | `eval.every` | — |
| G11 | Test feature = BN feature (2048-d RN50, 768-d ViT), squared Euclidean ranking, no L2-norm, no re-ranking, no flip. Same-id-same-camera gallery entries excluded. | `eval.metric` | DACS |
| G12 | MSMT17 V2 in the official layout, train + val merged (32,621 images, 1,041 ids). | `data.datasets.msmt17.*` | DACS (which used V1) |
| G13 | Camera template = mean over the camera's images of per-image channel mean and per-image channel variance, computed on resized, normalised, un-augmented images. 3 + 3 numbers per camera. | **code** (`style.extract_camera_templates`) | consistency with Eq. 6 |
| G14 | One template per image, drawn uniformly with replacement from the whole bank (own cameras included), every step; `x'` is made from the augmented `x`; `ε = 1e-8`. | `style.per_image`, `style.eps` | Eq. 6, Alg. 1 |
| G15 | One bias-free linear classifier per client on the BN feature, kept on the client, never aggregated, kept across rounds; plain cross-entropy. | **code** | DACS |
| G16 | `L_i2t` mean over the batch, `L_t2i` mean over batch identities, `L_c3` plain sum over ordered pairs. | `phase1.c3_reduction` | CLIP-ReID / Eq. 3 |
| G17 | Eq. 1 implemented literally: denominator over batch samples. With 4 images per identity the own prototype appears 4 times in the denominator, so `L_i2t` (and likewise `L_t2i`) cannot go below `ln 4 ≈ 1.39`; a Phase I loss that plateaus near 2.8 is converged, not stuck. | **code** | paper |
| G18 | `x` and `x'` go through the network in two separate forward passes; one backward on the summed loss. | **code** | DACS |
| G19 | Fresh SGD optimizer (zero momentum) for every client every round. FedAvg over all floating-point entries including BN running statistics, weights `n_k / N` with `n_k` = training images; `num_batches_tracked` copied from the first client. | `fed.aggregator` | DACS |
| G20 | ViT head: one BatchNorm neck (bias frozen) on the class token; `L_id` on it, `L_tri` on it. No SIE, no overlapping patches, no multi-feature loss sums. | `loss.triplet_feature` | Eqs. 7-8 |
| G21 | `τ = 0.07` in Phase I as well. | `phase1.tau` | single τ in paper |
| G22 | CUHK02: P1-P4 train, P5 test. | `data.datasets.cuhk02.test_pair` | DACS |
| G23 | AMP on (CUDA only), seed 1. | `amp`, `seed` | infra |
| G24 | Table 4 (noisy / missing camera IDs) not implemented. | — | out of scope |

## Added during implementation

| # | Assumption | Config key | Why |
|---|---|---|---|
| A1 | CLIP comes from `open_clip` (`ViT-B-16`, `openai` weights, QuickGELU), built at 256×128. open_clip resizes the 14×14 position embedding to 16×8 with **bicubic** interpolation; CLIP-ReID uses bilinear. | `clip.*` | pip-installable, pinned; OpenAI's own package is not on PyPI |
| A2 | Phase I epoch length = `N_images / 64` iterations of identity-sampler batches. This keeps CLIP-ReID's number of updates and LR schedule; only the batch composition differs (G7). A pure sampler epoch would be 5-8× fewer updates. | `phase1.iters_per_epoch` | keep the CLIP-ReID optimisation budget |
| A3 | Phase I image feature = CLIP's projected class token (`encode_image`), same as CLIP-ReID stage 1. Text feature = EOT token through `text_projection`. Only the unique identities of a batch are passed through the text encoder. | **code** | CLIP-ReID |
| A4 | ViT `feat` is the class token **after** CLIP's final LayerNorm. CLIP-ReID applies its BN neck to the token before that LayerNorm. | **code** | open_clip exposes the post-LN token |
| A5 | Random erasing fills with 0 in normalised space (= the dataset mean colour). DACS's own implementation writes the un-normalised mean values (0.485, 0.456, 0.406) into the normalised tensor. | `data.augment.random_erasing_value` | torchvision implementation |
| A6 | Identity sampler is a port of DACS's `RandomMultipleGallerySampler`. An identity with a single image contributes one sample instead of four (as in DACS). | **code** | DACS |
| A7 | New layers use PyTorch default initialisation: per-client classifiers (as DACS's federated trainer) and the RN50 text projection. | **code** | DACS |
| A8 | Anchors are stored un-normalised; all similarities are cosine. Anchors are cached per dataset under `cache/anchors/`, keyed by a hash of the Phase I settings, CLIP settings, seed and the training list, and reused across protocols. | `phase1.cache_dir` | Phase I depends only on the client |
| A9 | Loss terms are computed in float32 outside autocast. | **code** | numerical safety under AMP |
| A10 | The style bank uses the Phase II normalisation (so RN50 and ViT runs have different banks), and stays fixed for the whole run. | **code** | Sec. 3.3, Alg. 1, Appendix |
| A11 | Round evaluation uses the aggregated global model in eval mode (aggregated BN running statistics). | **code** | DACS |
| A12 | MSMT17 V1 is supported only in its official layout (`train/`, `test/`, same file names as V2) by changing two config keys. The Market-style V1 re-packaging that DACS reads is not supported. | `data.datasets.msmt17.train_dir/test_dir` | no V1 copy available |
| A13 | `tools/prepare_cuhk03.py` re-implements open-reid's conversion from memory. It has not been run on the real `cuhk-03.mat`. | — | no CUHK03 copy available |
| A14 | The MSMT17_V2 zip used has three truncated JPEGs: `mask_train_v2/1029/1029_018_13_0114afternoon_0972_0.jpg` (train), `mask_test_v2/2590/2590_006_07_0114noon_0223_2.jpg` (query), `mask_test_v2/2592/2592_003_07_0114noon_0229_0_ex.jpg` (gallery). They were re-saved with the missing tail padded (PIL `LOAD_TRUNCATED_IMAGES`), outside the code. Effect expected negligible (3 of 126,441 images). | — | damaged archive |

## Known departures from bit-exact reproducibility

- Resuming restores the Python / NumPy / PyTorch / CUDA and GSD generators and all model state, but DataLoader worker processes are re-created, so a resumed run is statistically, not bit-wise, identical to an uninterrupted one when `num_workers > 0`.
- `torch.use_deterministic_algorithms(True, warn_only=True)`: operations without a deterministic CUDA implementation warn instead of failing.
