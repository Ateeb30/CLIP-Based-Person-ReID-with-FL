import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from coevo.config import load_config
from coevo.data import IdentitySampler, load_dataset, missing_files
from coevo.evaluation import evaluate_ranking
from coevo.models import PromptLearner, TextEncoder, build_clip, build_image_encoder
from coevo.phase1 import phase1_lr
from coevo.runner import round_lr

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic.py"), "--out", str(out)], check=True)
    return out


@pytest.fixture(scope="session")
def smoke_cfg(synthetic_root):
    return load_config([CONFIGS / "base.yaml", CONFIGS / "smoke.yaml"], [f"data.root={synthetic_root.as_posix()}"])


def test_config_overlays_and_overrides():
    cfg = load_config([CONFIGS / "base.yaml", CONFIGS / "p1_cuhk03.yaml", CONFIGS / "backbone_vit.yaml"],
                      ["fed.optimizer.lr=1e-4", "fed.lr_milestones=[30,50]"])
    assert cfg.data.targets == ["cuhk03"] and cfg.model.backbone == "clip_vit"
    assert cfg.fed.optimizer.lr == 1e-4 and cfg.fed.lr_milestones == [30, 50]
    assert isinstance(cfg.phase1.lr, float) and isinstance(cfg.style.eps, float)
    with pytest.raises(KeyError):
        load_config([CONFIGS / "base.yaml"], ["fed.round=3"])


def test_every_overlay_loads():
    for path in CONFIGS.glob("*.yaml"):
        if path.name != "base.yaml":
            load_config([CONFIGS / "base.yaml", path])


def test_schedules():
    cfg = load_config([CONFIGS / "base.yaml"])
    assert round_lr(cfg.fed, 0) == 1e-3 and round_lr(cfg.fed, 19) == 1e-3
    assert round_lr(cfg.fed, 20) == 5e-4 and round_lr(cfg.fed, 59) == 2.5e-4
    assert phase1_lr(cfg.phase1, 0) == pytest.approx(1e-5)
    assert phase1_lr(cfg.phase1, 5) == pytest.approx(3.5e-4, rel=1e-2)
    assert phase1_lr(cfg.phase1, 120) == pytest.approx(1e-6)


def test_dataset_loaders_on_synthetic_layouts(smoke_cfg):
    expected = {  # (train ids, train imgs, train cams, query imgs, gallery imgs)
        "market1501": (8, 48, 3, 24, 49),
        "msmt17": (8, 64, 4, 32, 32),
        "cuhk02": (12, 48, 8, 6, 6),
        "cuhk03": (8, 48, 2, 8, 16),
    }
    for name, (ids, imgs, cams, nq, ng) in expected.items():
        data = load_dataset(name, smoke_cfg.data)
        assert (data.num_train_pids, len(data.train), len(data.train_cams)) == (ids, imgs, cams), name
        assert (len(data.query), len(data.gallery)) == (nq, ng), name
        assert sorted({p for _, p, _ in data.train}) == list(range(ids)), name
        assert not missing_files(data.train + data.query + data.gallery), name
    market = load_dataset("market1501", smoke_cfg.data)
    assert all(pid != -1 for _, pid, _ in market.gallery) and any(pid == 0 for _, pid, _ in market.gallery)


def test_msmt17_parses_real_v2_names(tmp_path):
    base = tmp_path / "msmt17" / "MSMT17_V2"
    base.mkdir(parents=True)
    (base / "list_train.txt").write_text("0000/0000_045_12_0303morning_0006_2.jpg 0\n", encoding="utf-8")
    (base / "list_val.txt").write_text("1040/1040_168_07_0302morning_0340_3_ex.jpg 1040\n", encoding="utf-8")
    (base / "list_query.txt").write_text("0000/0000_000_01_0303morning_0015_0.jpg 0\n", encoding="utf-8")
    (base / "list_gallery.txt").write_text("3059/3059_042_04_0113noon_1651_0.jpg 3059\n", encoding="utf-8")
    cfg = load_config([CONFIGS / "base.yaml"], [f"data.root={tmp_path.as_posix()}"])
    data = load_dataset("msmt17", cfg.data)
    assert [(pid, cam) for _, pid, cam in data.train] == [(0, 11), (1, 6)]  # relabelled, 0-based cameras
    assert data.query[0][1:] == (0, 0) and data.gallery[0][1:] == (3059, 3)
    assert Path(data.train[0][0]).parent.parent.name == "mask_train_v2"
    assert Path(data.gallery[0][0]).parent.parent.name == "mask_test_v2"


def test_identity_sampler_prefers_other_cameras():
    samples = [("", pid, cam) for pid in range(5) for cam in range(3) for _ in range(2)]
    samples.append(("", 5, 0))  # identity with a single image
    sampler = IdentitySampler(samples, 4)
    order = list(sampler)
    assert len(order) == 5 * 4 + 1
    pos = 0
    while pos < len(order):
        pid, cam = samples[order[pos]][1:]
        if pid == 5:
            pos += 1
            continue
        group = order[pos:pos + 4]
        assert all(samples[i][1] == pid for i in group)
        assert all(samples[i][2] != cam for i in group[1:])
        pos += 4


def test_ranking_metrics_hand_example():
    # query pid 1 cam 0. Gallery sorted by distance:
    #   g0 pid 1 cam 0 -> same id and camera: ignored
    #   g1 pid 2       -> miss
    #   g2 pid 1 cam 1 -> hit at rank 2 (precision 1/2)
    #   g3 pid 3       -> miss
    #   g4 pid 1 cam 2 -> hit at rank 4 (precision 2/4)
    query = torch.tensor([[0.0]])
    gallery = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
    res = evaluate_ranking(query, gallery, [1], [1, 2, 1, 3, 1], [0], [0, 5, 1, 5, 2],
                           metric="euclidean", device=torch.device("cpu"))
    assert res["mAP"] == pytest.approx(0.5)
    assert (res["R1"], res["R5"]) == (0.0, 1.0)


def test_ranking_perfect_and_skips_queries_without_match():
    feats = torch.eye(3)
    res = evaluate_ranking(feats, feats, [0, 1, 9], [0, 1, 2], [0, 0, 0], [1, 1, 1],
                           metric="cosine", device=torch.device("cpu"))
    assert res["valid_queries"] == 2 and res["mAP"] == 1.0 and res["R1"] == 1.0


def test_ranking_matches_sklearn_free_reference():
    rng = np.random.default_rng(0)
    q, g = torch.tensor(rng.normal(size=(20, 8))).float(), torch.tensor(rng.normal(size=(60, 8))).float()
    qp, gp = rng.integers(0, 5, 20), rng.integers(0, 5, 60)
    qc, gc = rng.integers(0, 3, 20), rng.integers(0, 3, 60)
    res = evaluate_ranking(q, g, qp, gp, qc, gc, metric="euclidean", device=torch.device("cpu"), chunk=7)
    aps, r1 = [], []
    dist = torch.cdist(q, g).numpy()
    for i in range(20):
        order = np.argsort(dist[i], kind="stable")
        keep = [j for j in order if not (gp[j] == qp[i] and gc[j] == qc[i])]
        hits = [gp[j] == qp[i] for j in keep]
        if not any(hits):
            continue
        precisions = [sum(hits[:k + 1]) / (k + 1) for k in range(len(hits)) if hits[k]]
        aps.append(np.mean(precisions))
        r1.append(float(hits[0]))
    assert res["mAP"] == pytest.approx(np.mean(aps), abs=1e-6)
    assert res["R1"] == pytest.approx(np.mean(r1), abs=1e-6)


def test_prompt_text_encoder_matches_clip_encode_text(smoke_cfg):
    torch.manual_seed(0)
    clip, tokenizer = build_clip(smoke_cfg.clip, (smoke_cfg.data.height, smoke_cfg.data.width))
    clip.eval()
    learner = PromptLearner(num_ids=3, num_tokens=4, clip_model=clip, tokenizer=tokenizer)
    assert learner.cls_ctx.shape == (3, 4, 64)
    prompts = learner(torch.tensor([0, 2]))
    assert prompts.shape == (2, 77, 64)
    # With the learnable slots filled by the embedding of the placeholder word, the prompt
    # path must reproduce CLIP's own text encoding of the template.
    tokens = tokenizer(["A photo of a X X X X person."])
    with torch.no_grad():
        learner.cls_ctx.copy_(clip.token_embedding(tokens)[:, 5:9].expand(3, -1, -1))
        mine = TextEncoder(clip)(learner(torch.tensor([1])), learner.eot_index)
        reference = clip.encode_text(tokens)
    assert torch.allclose(mine, reference, atol=1e-5)
    # only the tokens of the requested identities receive gradient
    out = TextEncoder(clip)(learner(torch.tensor([0])), learner.eot_index).sum()
    out.backward()
    assert learner.cls_ctx.grad[0].abs().sum() > 0 and learner.cls_ctx.grad[1:].abs().sum() == 0


def test_vit_backbone_is_trainable_when_clip_is_frozen(smoke_cfg):
    """Regression: runner freezes CLIP for Phase I before building the encoder; the copied
    ViT tower must still be trained by the client optimizer."""
    from coevo.data import build_train_transform, load_dataset
    from coevo.fed import Client

    cfg = load_config([CONFIGS / "base.yaml", CONFIGS / "smoke.yaml"],
                      [f"data.root={smoke_cfg.data.root}", "model.backbone=clip_vit", "fed.local_iters=1"])
    clip, _ = build_clip(cfg.clip, (cfg.data.height, cfg.data.width))
    clip.requires_grad_(False)  # as in runner.run
    model, mean, std = build_image_encoder(cfg, clip)

    assert all(p.requires_grad for p in model.visual.parameters())
    assert model.proj.requires_grad and model.feat_bn.weight.requires_grad
    assert not model.feat_bn.bias.requires_grad  # BNNeck: bias frozen on purpose
    assert not any(p.requires_grad for p in clip.parameters())  # the original stays frozen

    data = load_dataset("msmt17", cfg.data)
    client = Client("msmt17", data, build_train_transform(cfg.data, mean, std), model.feat_dim, cfg,
                    torch.device("cpu"))
    before = model.visual.conv1.weight.detach().clone()
    state, _ = client.train_round(model, model.state_dict(), lr=0.1, gsd=None)
    assert not torch.equal(state["visual.conv1.weight"], before)


@pytest.mark.parametrize("backbone", ["resnet50", "clip_vit"])
def test_image_encoder_interface(smoke_cfg, backbone):
    cfg = load_config([CONFIGS / "base.yaml", CONFIGS / "smoke.yaml"], [f"model.backbone={backbone}"])
    clip, _ = build_clip(cfg.clip, (cfg.data.height, cfg.data.width))
    reference = clip.encode_image(torch.zeros(2, 3, 64, 32)).detach()
    model, mean, std = build_image_encoder(cfg, clip)
    out = model.train()(torch.randn(4, 3, 64, 32))
    assert out["feat"].shape == out["feat_bn"].shape == (4, model.feat_dim)
    assert out["proj"].shape == (4, 32) and model.proj_dim == 32
    assert len(mean) == len(std) == 3
    if backbone == "clip_vit":
        # the wrapped tower must still produce CLIP's image embedding, and must be a copy
        assert torch.allclose(model.eval()(torch.zeros(2, 3, 64, 32))["proj"], reference, atol=1e-5)
        assert clip.visual.proj is not None
