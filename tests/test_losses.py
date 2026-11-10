import math

import torch
import torch.nn.functional as F

from coevo import losses


def _batch(seed=0, ids=4, per_id=3, dim=16):
    g = torch.Generator().manual_seed(seed)
    pids = torch.arange(ids).repeat_interleave(per_id)
    cams = torch.arange(per_id).repeat(ids)
    img = torch.randn(ids * per_id, dim, generator=g)
    protos = torch.randn(ids, dim, generator=g)
    return img, protos[pids], pids, cams, protos


def test_i2t_matches_equation_1():
    img, text, pids, _, _ = _batch()
    tau = 0.07
    sim = F.cosine_similarity(img[:, None], text[None], dim=2)
    expected = torch.stack([
        -torch.log(torch.exp(sim[i, i] / tau) / torch.exp(sim[i] / tau).sum()) for i in range(len(img))
    ]).mean()
    assert torch.allclose(losses.csa_i2t_loss(img, text, tau), expected, atol=1e-5)


def test_t2i_matches_equation_2():
    img, text, pids, _, protos = _batch()
    tau = 0.07
    per_id = []
    for y in pids.unique():
        sim = F.cosine_similarity(img, protos[y][None], dim=1) / tau
        log_prob = sim - torch.logsumexp(sim, dim=0)
        per_id.append(-log_prob[pids == y].mean())
    expected = torch.stack(per_id).mean()
    assert torch.allclose(losses.csa_t2i_loss(img, text, pids, tau), expected, atol=1e-5)


def test_c3_counts_only_same_id_cross_camera_pairs():
    # similarities to the own prototype: 1, 0, -1, 1 (prototype = e0)
    text = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
    img = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])
    pids = torch.tensor([0, 0, 0, 1])
    cams = torch.tensor([0, 1, 0, 1])
    # cross-camera same-id unordered pairs: (0,1) diff 1, (1,2) diff 1; (0,2) shares a camera
    # ordered pairs double the sum: 2 * (1 + 1) = 4
    assert losses.cross_camera_consistency_loss(img, text, pids, cams, "sum").item() == 4.0
    assert losses.cross_camera_consistency_loss(img, text, pids, cams, "mean").item() == 1.0


def test_c3_zero_without_cross_camera_pairs_and_csa_total():
    img, text, pids, cams, _ = _batch()
    same_cam = torch.zeros_like(cams)
    assert losses.cross_camera_consistency_loss(img, text, pids, same_cam).item() == 0.0
    total, parts = losses.csa_loss(img, text, pids, cams, 0.07, 0.1)
    assert torch.allclose(total, parts["i2t"] + parts["t2i"] + 0.1 * parts["c3"])


def test_triplet_hand_example():
    feats = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0], [5.0, 0.0]])
    pids = torch.tensor([0, 0, 1, 1])
    # hardest positive / negative per anchor:
    #   0: pos 3,          neg 1            -> 3 - 1 + m
    #   1: pos 3,          neg 2            -> 3 - 2 + m
    #   2: pos sqrt(26),   neg 1            -> sqrt(26) - 1 + m
    #   3: pos sqrt(26),   neg 2            -> sqrt(26) - 2 + m
    m = 0.5
    expected = (2 + m + 1 + m + math.sqrt(26) - 1 + m + math.sqrt(26) - 2 + m) / 4
    assert math.isclose(losses.triplet_loss(feats, pids, m).item(), expected, rel_tol=1e-5)


def test_triplet_zero_when_separated_and_has_finite_grad():
    feats = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]], requires_grad=True)
    pids = torch.tensor([0, 0, 1, 1])
    assert losses.triplet_loss(feats, pids, 0.5).item() == 0.0
    close = torch.randn(8, 4, requires_grad=True)
    losses.triplet_loss(close, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), 0.5).backward()
    assert torch.isfinite(close.grad).all()


def test_align_is_cross_entropy_over_all_anchors():
    g = torch.Generator().manual_seed(1)
    img, anchors = torch.randn(6, 8, generator=g), torch.randn(10, 8, generator=g)
    pids = torch.tensor([0, 3, 3, 9, 5, 1])
    tau = 0.07
    logits = F.normalize(img, dim=1) @ F.normalize(anchors, dim=1).t() / tau
    expected = -torch.log_softmax(logits, dim=1)[torch.arange(6), pids].mean()
    assert torch.allclose(losses.align_loss(img, anchors, pids, tau), expected, atol=1e-5)
    # scale invariance: cosine similarity ignores feature norms
    assert torch.allclose(losses.align_loss(img * 7, anchors * 0.1, pids, tau), expected, atol=1e-4)


def test_id_loss_is_plain_cross_entropy():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    expected = math.log(1 + math.exp(-2.0))
    assert math.isclose(losses.id_loss(logits, torch.tensor([0, 1])).item(), expected, rel_tol=1e-6)
