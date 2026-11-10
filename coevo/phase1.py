"""Phase I: Camera-Invariant Semantic Anchoring (Sec. 3.2, Alg. 1 lines 1-5).

Runs locally on one client: frozen CLIP image and text encoders, only the per-identity
prompt tokens are optimised with L_CSA. The result is the cached prototype matrix T_k.
Training recipe follows CLIP-ReID stage 1 (image features extracted once from
un-augmented images, Adam, cosine schedule with linear warm-up)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .data import IdentitySampler, ReIDData, build_eval_loader, build_test_transform
from .losses import csa_loss
from .models import PromptLearner, TextEncoder
from .utils import atomic_save


def phase1_lr(cfg, epoch: int) -> float:
    """CLIP-ReID stage-1 schedule; `epoch` is 1-based."""
    if epoch < cfg.warmup_epochs:
        return cfg.warmup_lr_init + epoch * (cfg.lr - cfg.warmup_lr_init) / cfg.warmup_epochs
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * epoch / cfg.epochs))


def anchor_cache_path(cfg, data: ReIDData) -> Path:
    """Anchors depend only on the client's data, the CLIP model and the Phase I settings,
    so they are shared between runs (e.g. the MSMT17 client of every protocol)."""
    key = json.dumps({
        "phase1": cfg.phase1.to_dict(), "clip": cfg.clip.to_dict(), "seed": cfg.seed,
        "size": [cfg.data.height, cfg.data.width], "sampler": cfg.data.sampler.to_dict(),
        "interpolation": cfg.data.augment.interpolation,
        "train": [(Path(p).name, pid, cam) for p, pid, cam in data.train],
    }, sort_keys=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return Path(cfg.phase1.cache_dir) / f"{data.name}_{digest}.pt"


def _index_batches(sampler: IdentitySampler, batch_size: int):
    """Endless stream of index batches from the identity sampler."""
    while True:
        batch = []
        for index in sampler:
            batch.append(index)
            if len(batch) == batch_size:
                yield batch
                batch = []


@torch.no_grad()
def _extract_clip_image_features(clip_model, data: ReIDData, cfg, device, amp: bool) -> torch.Tensor:
    transform = build_test_transform(cfg.data, cfg.clip.pixel_mean, cfg.clip.pixel_std)
    loader = build_eval_loader(data.train, transform, cfg.eval.batch_size, cfg.data.num_workers,
                               pin_memory=device.type == "cuda")
    feats = []
    for images, _, _, _ in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            feats.append(clip_model.encode_image(images.to(device, non_blocking=True)).float())
    return torch.cat(feats)


def learn_anchors(cfg, data: ReIDData, clip_model, tokenizer, device, logger, writer) -> torch.Tensor:
    """Return T_k as a [num_train_pids, D] float tensor on the CPU, training it if no
    cached copy exists."""
    p1 = cfg.phase1
    cache = anchor_cache_path(cfg, data)
    if cache.is_file():
        logger.info(f"[phase1] {data.name}: loading cached anchors {cache}")
        return torch.load(cache, map_location="cpu")["anchors"]

    amp = cfg.amp and device.type == "cuda"
    clip_model.eval()
    logger.info(f"[phase1] {data.name}: extracting CLIP image features for {len(data.train)} images")
    image_feats = _extract_clip_image_features(clip_model, data, cfg, device, amp)
    pids = torch.tensor([pid for _, pid, _ in data.train], device=device)
    cams = torch.tensor([cam for _, _, cam in data.train], device=device)

    prompt_learner = PromptLearner(data.num_train_pids, p1.num_tokens, clip_model, tokenizer).to(device)
    text_encoder = TextEncoder(clip_model)
    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=p1.lr, weight_decay=p1.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    sampler = IdentitySampler(data.train, cfg.data.sampler.num_instances)
    batches = _index_batches(sampler, p1.batch_size)
    if p1.iters_per_epoch == "images":
        iters = max(1, len(data.train) // p1.batch_size)
    elif p1.iters_per_epoch == "sampler":
        iters = max(1, len(sampler) // p1.batch_size)
    else:
        iters = int(p1.iters_per_epoch)

    for epoch in range(1, p1.epochs + 1):
        lr = phase1_lr(p1, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        sums = {"loss": 0.0, "i2t": 0.0, "t2i": 0.0, "c3": 0.0}
        for _ in range(iters):
            idx = torch.tensor(next(batches), device=device)
            b_pids, b_cams, b_img = pids[idx], cams[idx], image_feats[idx]
            uniq, inverse = torch.unique(b_pids, return_inverse=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                text = text_encoder(prompt_learner(uniq), prompt_learner.eot_index)
            text = text.float()[inverse]  # prototype of each sample's identity
            loss, parts = csa_loss(b_img, text, b_pids, b_cams, p1.tau, p1.lambda_c3, p1.c3_reduction)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += loss.item()
            for k, v in parts.items():
                sums[k] += v.item()
        scalars = {f"{data.name}/{k}": v / iters for k, v in sums.items()}
        scalars[f"{data.name}/lr"] = lr
        writer.write("phase1", epoch, scalars)
        if epoch % p1.log_every == 0 or epoch == p1.epochs:
            logger.info(f"[phase1] {data.name} epoch {epoch}/{p1.epochs} lr {lr:.2e} "
                        + " ".join(f"{k} {v / iters:.4f}" for k, v in sums.items()))

    anchors = []
    with torch.no_grad():
        all_ids = torch.arange(data.num_train_pids, device=device)
        for chunk in all_ids.split(p1.batch_size):
            with torch.autocast(device_type=device.type, enabled=amp):
                anchors.append(text_encoder(prompt_learner(chunk), prompt_learner.eot_index).float().cpu())
    anchors = torch.cat(anchors)
    atomic_save({"anchors": anchors, "tokens": prompt_learner.cls_ctx.detach().float().cpu(),
                 "dataset": data.name}, cache)
    logger.info(f"[phase1] {data.name}: saved {tuple(anchors.shape)} anchors to {cache}")
    return anchors
