"""Single-query ReID evaluation: mAP and CMC, Market-1501 protocol (gallery images with
the query's identity AND camera are excluded). Matches the DACS evaluator."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(model, loader, device, amp: bool) -> torch.Tensor:
    model.eval()
    feats = []
    for images, _, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(images)
        feats.append(out["feat_bn"].float().cpu())
    return torch.cat(feats)


def rank_metrics(sorted_gallery_pids, sorted_gallery_cams, query_pid, query_cam):
    """AP and first-hit rank for one query given its gallery sorted by distance.
    Returns None when the query has no valid match."""
    valid = (sorted_gallery_pids != query_pid) | (sorted_gallery_cams != query_cam)
    hits = (sorted_gallery_pids[valid] == query_pid)
    if not hits.any():
        return None
    positions = np.flatnonzero(hits)
    precision = np.arange(1, len(positions) + 1) / (positions + 1)
    return precision.mean(), positions[0]


@torch.no_grad()
def evaluate_ranking(query_feats, gallery_feats, query_pids, gallery_pids, query_cams, gallery_cams,
                     metric: str, device, topk=(1, 5, 10), chunk: int = 256) -> dict:
    query_pids, gallery_pids = np.asarray(query_pids), np.asarray(gallery_pids)
    query_cams, gallery_cams = np.asarray(query_cams), np.asarray(gallery_cams)
    if metric == "cosine":
        query_feats, gallery_feats = F.normalize(query_feats, dim=1), F.normalize(gallery_feats, dim=1)
    elif metric != "euclidean":
        raise ValueError(f"unknown metric '{metric}'")
    gallery = gallery_feats.to(device)
    g_sq = gallery.pow(2).sum(dim=1)

    aps, cmc, valid_queries = [], np.zeros(max(topk)), 0
    for start in range(0, len(query_feats), chunk):
        q = query_feats[start:start + chunk].to(device)
        # squared Euclidean distance; for unit vectors its ordering equals cosine ordering
        dist = q.pow(2).sum(dim=1, keepdim=True) + g_sq[None, :] - 2 * q @ gallery.t()
        order = torch.argsort(dist, dim=1, stable=True).cpu().numpy()
        for row, idx in enumerate(order):
            qi = start + row
            res = rank_metrics(gallery_pids[idx], gallery_cams[idx], query_pids[qi], query_cams[qi])
            if res is None:
                continue
            ap, first = res
            aps.append(ap)
            if first < len(cmc):
                cmc[first:] += 1
            valid_queries += 1
    if valid_queries == 0:
        raise RuntimeError("no valid query")
    cmc = cmc / valid_queries
    out = {"mAP": float(np.mean(aps)), "valid_queries": valid_queries}
    out.update({f"R{k}": float(cmc[k - 1]) for k in topk})
    return out


def evaluate_dataset(model, data, transform, eval_cfg, data_cfg, device, amp: bool) -> dict:
    """Extract features for `data.query` / `data.gallery` and score them."""
    from .data import build_eval_loader

    def feats(samples):
        loader = build_eval_loader(samples, transform, eval_cfg.batch_size, data_cfg.num_workers,
                                   pin_memory=device.type == "cuda")
        return extract_features(model, loader, device, amp)

    return evaluate_ranking(
        feats(data.query), feats(data.gallery),
        [p for _, p, _ in data.query], [p for _, p, _ in data.gallery],
        [c for _, _, c in data.query], [c for _, _, c in data.gallery],
        metric=eval_cfg.metric, device=device,
    )
