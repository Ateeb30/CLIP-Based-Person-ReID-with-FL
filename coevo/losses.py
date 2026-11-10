"""Loss terms of CO-EVO. Equation numbers refer to the paper.

All functions take float tensors and compute in float32 (callers running under AMP
pass `.float()` features)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(a, dim=1) @ F.normalize(b, dim=1).t()


def csa_i2t_loss(image_feats, text_feats, tau: float) -> torch.Tensor:
    """Eq. 1. `text_feats[a]` is the prototype of sample a's identity, so the denominator
    runs over the batch samples exactly as written. Mean over the batch."""
    logits = _cosine(image_feats, text_feats) / tau  # [i, a] = s(v_i, t_{y_a}) / tau
    return (torch.logsumexp(logits, dim=1) - logits.diagonal()).mean()


def csa_t2i_loss(image_feats, text_feats, pids, tau: float) -> torch.Tensor:
    """Eq. 2. For each identity y in the batch: average over its positives of
    -log softmax over all batch images. Mean over identities."""
    uniq, inverse = torch.unique(pids, return_inverse=True)
    first = torch.stack([(inverse == k).nonzero()[0, 0] for k in range(len(uniq))])
    logits = _cosine(image_feats, text_feats[first]) / tau  # [a, y] = s(v_a, t_y) / tau
    log_prob = logits - torch.logsumexp(logits, dim=0, keepdim=True)
    pos = F.one_hot(inverse, len(uniq)).to(log_prob.dtype)  # [a, y]
    per_id = -(pos * log_prob).sum(dim=0) / pos.sum(dim=0)
    return per_id.mean()


def cross_camera_consistency_loss(image_feats, text_feats, pids, cams, reduction: str = "sum") -> torch.Tensor:
    """Eq. 3. Squared difference of image-to-own-prototype similarity over ordered pairs
    (i, j) with the same identity and different cameras."""
    sim = F.cosine_similarity(image_feats, text_feats, dim=1)  # s(v_i, t_{y_i})
    pair = (pids[:, None] == pids[None, :]) & (cams[:, None] != cams[None, :])
    sq = (sim[:, None] - sim[None, :]).pow(2)
    total = (sq * pair).sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        return total / pair.sum().clamp(min=1)
    raise ValueError(f"unknown reduction '{reduction}'")


def csa_loss(image_feats, text_feats, pids, cams, tau: float, lambda_c3: float, c3_reduction: str = "sum"):
    """L_CSA = L_i2t + L_t2i + lambda_c3 * L_c3. Returns (total, parts)."""
    i2t = csa_i2t_loss(image_feats, text_feats, tau)
    t2i = csa_t2i_loss(image_feats, text_feats, pids, tau)
    c3 = cross_camera_consistency_loss(image_feats, text_feats, pids, cams, c3_reduction)
    return i2t + t2i + lambda_c3 * c3, {"i2t": i2t.detach(), "t2i": t2i.detach(), "c3": c3.detach()}


def triplet_loss(feats, pids, margin: float) -> torch.Tensor:
    """Eq. 8. Batch-hard triplet loss on Euclidean distance."""
    sq = feats.pow(2).sum(dim=1, keepdim=True)
    dist = (sq + sq.t() - 2 * feats @ feats.t()).clamp(min=1e-12).sqrt()
    same = pids[:, None] == pids[None, :]
    d_pos = dist.masked_fill(~same, float("-inf")).max(dim=1).values
    d_neg = dist.masked_fill(same, float("inf")).min(dim=1).values
    return F.relu(d_pos - d_neg + margin).mean()


def id_loss(logits, pids) -> torch.Tensor:
    """Eq. 7."""
    return F.cross_entropy(logits, pids)


def align_loss(image_feats, anchors, pids, tau: float) -> torch.Tensor:
    """Eq. 9. Cross-entropy of each image against all of the client's frozen anchors."""
    return F.cross_entropy(_cosine(image_feats, anchors) / tau, pids)
