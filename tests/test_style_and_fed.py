from collections import OrderedDict

import torch

from coevo.fed import FedAvgAggregator
from coevo.style import (GlobalStyleDiversification, StyleBank, extract_camera_templates, image_stats,
                         renormalize)


def test_renormalize_gives_each_image_the_template_statistics():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 3, 16, 8, generator=g) * 2 + 1
    s_mean = torch.randn(5, 3, generator=g)
    s_var = torch.rand(5, 3, generator=g) + 0.1
    out = renormalize(x, s_mean, s_var, eps=1e-8)
    mean, var = image_stats(out)
    assert torch.allclose(mean, s_mean, atol=1e-5)
    assert torch.allclose(var, s_var, atol=1e-4)


def test_renormalize_with_own_statistics_is_identity_and_keeps_structure():
    x = torch.randn(4, 3, 16, 8, generator=torch.Generator().manual_seed(1))
    mean, var = image_stats(x)
    assert torch.allclose(renormalize(x, mean, var, eps=1e-8), x, atol=1e-5)
    # a different template is a per-channel affine map: correlation with the input stays 1
    out = renormalize(x, mean + 1, var * 4, eps=1e-8)
    a, b = x[0, 0].flatten(), out[0, 0].flatten()
    assert torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.9999


def test_renormalize_constant_image_is_finite():
    x = torch.full((1, 3, 4, 4), 0.3)
    out = renormalize(x, torch.zeros(1, 3), torch.ones(1, 3), eps=1e-8)
    assert torch.isfinite(out).all()


def test_camera_templates_average_per_image_statistics():
    g = torch.Generator().manual_seed(2)
    images = torch.randn(6, 3, 8, 4, generator=g)
    cams = torch.tensor([0, 1, 0, 1, 1, 2])
    loader = [(images[:4], None, cams[:4], None), (images[4:], None, cams[4:], None)]
    templates = extract_camera_templates(loader, torch.device("cpu"))
    assert sorted(templates) == [0, 1, 2]
    mean, var = image_stats(images)
    for cam in (0, 1, 2):
        assert torch.allclose(templates[cam][0], mean[cams == cam].mean(dim=0), atol=1e-6)
        assert torch.allclose(templates[cam][1], var[cams == cam].mean(dim=0), atol=1e-6)


def _bank():
    templates = {
        "a": {0: (torch.zeros(3), torch.ones(3)), 1: (torch.ones(3), torch.ones(3) * 2)},
        "b": {0: (torch.ones(3) * 2, torch.ones(3) * 3)},
    }
    return StyleBank.from_client_templates(templates)


def test_bank_is_the_union_over_clients_and_cameras():
    bank = _bank()
    assert bank.keys == [("a", 0), ("a", 1), ("b", 0)]
    assert bank.mean.shape == (3, 3)
    restored = StyleBank.from_state_dict(bank.state_dict())
    assert restored.keys == bank.keys and torch.equal(restored.var, bank.var)
    # uniform sampling reaches every template
    mean, _ = bank.sample(300, torch.Generator().manual_seed(0))
    assert set(mean[:, 0].tolist()) == {0.0, 1.0, 2.0}


def test_gsd_is_deterministic_and_per_image():
    x = torch.randn(64, 3, 8, 4, generator=torch.Generator().manual_seed(3))
    a = GlobalStyleDiversification(_bank(), 1e-8, per_image=True, seed=7)(x)
    b = GlobalStyleDiversification(_bank(), 1e-8, per_image=True, seed=7)(x)
    assert torch.equal(a, b)
    assert len(set(image_stats(a)[0][:, 0].round(decimals=3).tolist())) == 3  # all templates used
    c = GlobalStyleDiversification(_bank(), 1e-8, per_image=False, seed=7)(x)
    assert len(set(image_stats(c)[0][:, 0].round(decimals=3).tolist())) == 1  # one template per batch


def test_fedavg_weights_by_sample_count():
    s1 = OrderedDict(w=torch.tensor([1.0, 2.0]), n=torch.tensor(5))
    s2 = OrderedDict(w=torch.tensor([3.0, 6.0]), n=torch.tensor(9))
    out = FedAvgAggregator().aggregate([s1, s2], [1, 3])
    assert torch.allclose(out["w"], torch.tensor([2.5, 5.0]))
    assert out["n"].item() == 5 and out["n"].dtype == torch.int64  # integer buffers are not averaged
    assert out["w"].dtype == torch.float32


def test_fedavg_of_identical_models_is_the_model():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    state = model.state_dict()
    out = FedAvgAggregator().aggregate([state, state, state], [10, 20, 30])
    for key, value in state.items():
        assert torch.allclose(out[key].float(), value.float(), atol=1e-6)
    model.load_state_dict(out)
