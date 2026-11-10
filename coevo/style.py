"""Global Style Diversification: camera-style templates (Eq. 5), the Global
Camera-Style Bank and template-based re-normalisation (Eq. 6)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def image_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image channel-wise mean and variance of a [B, C, H, W] batch -> two [B, C]."""
    return x.mean(dim=(2, 3)), x.var(dim=(2, 3))


@torch.no_grad()
def extract_camera_templates(loader, device) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Eq. 5 for one client. `loader` yields (image, pid, group, index) with un-augmented,
    normalised images; `group` is the camera id (or any pseudo-group id). The template of
    a group is the average of its images' per-image channel means and variances."""
    sums: dict[int, list] = {}
    for images, _, groups, _ in loader:
        mean, var = image_stats(images.to(device, non_blocking=True).float())
        mean, var = mean.double().cpu(), var.double().cpu()
        for g in groups.unique().tolist():
            sel = groups == g
            acc = sums.setdefault(g, [torch.zeros(mean.shape[1], dtype=torch.double),
                                      torch.zeros(mean.shape[1], dtype=torch.double), 0])
            acc[0] += mean[sel].sum(dim=0)
            acc[1] += var[sel].sum(dim=0)
            acc[2] += int(sel.sum())
    return {g: ((m / n).float(), (v / n).float()) for g, (m, v, n) in sorted(sums.items())}


@dataclass
class StyleBank:
    """B = union over clients and cameras of (mu, sigma^2). One row per (client, camera)."""

    mean: torch.Tensor  # [N, C]
    var: torch.Tensor  # [N, C]
    keys: list[tuple[str, int]]  # (client name, camera id) of each row

    @classmethod
    def from_client_templates(cls, templates: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]):
        keys, means, variances = [], [], []
        for client, per_cam in templates.items():
            for cam, (mean, var) in per_cam.items():
                keys.append((client, cam))
                means.append(mean)
                variances.append(var)
        return cls(torch.stack(means), torch.stack(variances), keys)

    def __len__(self) -> int:
        return len(self.keys)

    def sample(self, n: int, generator: torch.Generator | None = None):
        """Draw n templates uniformly with replacement."""
        idx = torch.randint(len(self), (n,), generator=generator)
        return self.mean[idx], self.var[idx]

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "keys": self.keys}

    @classmethod
    def from_state_dict(cls, state: dict):
        return cls(state["mean"], state["var"], [tuple(k) for k in state["keys"]])


def renormalize(x: torch.Tensor, style_mean: torch.Tensor, style_var: torch.Tensor, eps: float) -> torch.Tensor:
    """Eq. 6: x_hat = (x - mu(x)) / sqrt(var(x) + eps);  x' = x_hat * sqrt(var_s) + mu_s."""
    mean, var = image_stats(x)
    x_hat = (x - mean[:, :, None, None]) / torch.sqrt(var[:, :, None, None] + eps)
    return x_hat * torch.sqrt(style_var)[:, :, None, None] + style_mean[:, :, None, None]


class GlobalStyleDiversification:
    """Produces the stylised view x' of a batch from templates sampled out of the bank."""

    def __init__(self, bank: StyleBank, eps: float, per_image: bool, seed: int):
        self.bank = bank
        self.eps = eps
        self.per_image = per_image
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0] if self.per_image else 1
        mean, var = self.bank.sample(n, self.generator)
        mean, var = mean.to(x.device, x.dtype), var.to(x.device, x.dtype)
        if not self.per_image:
            mean, var = mean.expand(x.shape[0], -1), var.expand(x.shape[0], -1)
        return renormalize(x, mean, var, self.eps)
