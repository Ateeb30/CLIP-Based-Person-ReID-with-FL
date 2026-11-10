"""Client side: local data, local classifier, frozen anchors and the local update
(Alg. 1 lines 10-15, Eq. 10)."""
from __future__ import annotations

import torch
from torch import nn

from ..data import ReIDData, build_train_loader
from ..losses import align_loss, id_loss, triplet_loss


def cycle(loader, num_iters: int):
    """Yield exactly `num_iters` batches, restarting the loader when it runs out."""
    done = 0
    while done < num_iters:
        for batch in loader:
            yield batch
            done += 1
            if done == num_iters:
                return


class Client:
    def __init__(self, name: str, data: ReIDData, transform, feat_dim: int, cfg, device):
        self.name = name
        self.cfg = cfg
        self.device = device
        self.num_samples = len(data.train)
        self.num_pids = data.num_train_pids
        self.loader = build_train_loader(
            data.train, transform, cfg.fed.batch_size, cfg.data.sampler.num_instances,
            cfg.data.num_workers, pin_memory=device.type == "cuda",
        )
        if len(self.loader) == 0:
            raise ValueError(f"{name}: fewer than one batch of training data")
        # The identity classifier never leaves the client (identity sets are disjoint).
        self.classifier = nn.Linear(feat_dim, self.num_pids, bias=False).to(device)
        self.anchors: torch.Tensor | None = None  # T_k, [num_pids, D], frozen

    def set_anchors(self, anchors: torch.Tensor) -> None:
        if anchors.shape[0] != self.num_pids:
            raise ValueError(f"{self.name}: {anchors.shape[0]} anchors for {self.num_pids} identities")
        self.anchors = anchors.detach().float().to(self.device)

    def iters_per_round(self) -> int:
        fixed = self.cfg.fed.local_iters
        return fixed if fixed is not None else self.cfg.fed.local_epochs * len(self.loader)

    def train_round(self, model: nn.Module, global_state: dict, lr: float, gsd) -> tuple[dict, dict]:
        """Start from the global model, run the local objective, return the new weights."""
        cfg, loss_cfg = self.cfg, self.cfg.loss
        amp = cfg.amp and self.device.type == "cuda"
        model.load_state_dict(global_state)
        model.train()
        self.classifier.train()
        params = [p for p in model.parameters() if p.requires_grad] + list(self.classifier.parameters())
        optimizer = torch.optim.SGD(params, lr=lr, momentum=cfg.fed.optimizer.momentum,
                                    weight_decay=cfg.fed.optimizer.weight_decay)
        scaler = torch.amp.GradScaler(self.device.type, enabled=amp)

        sums = {"loss": 0.0, "id": 0.0, "tri": 0.0, "align": 0.0}
        num_iters = self.iters_per_round()
        for images, pids, _, _ in cycle(self.loader, num_iters):
            images = images.to(self.device, non_blocking=True)
            pids = pids.to(self.device, non_blocking=True)
            views = [images] if gsd is None else [images, gsd(images)]

            loss = 0.0
            for view in views:
                with torch.autocast(device_type=self.device.type, enabled=amp):
                    out = model(view)
                    logits = self.classifier(out["feat_bn"])
                l_id = id_loss(logits.float(), pids)
                l_tri = triplet_loss(out[loss_cfg.triplet_feature].float(), pids, loss_cfg.triplet_margin)
                loss = loss + loss_cfg.id_weight * l_id + loss_cfg.triplet_weight * l_tri
                sums["id"] += l_id.item()
                sums["tri"] += l_tri.item()
                if self.anchors is not None:
                    l_align = align_loss(out["proj"].float(), self.anchors, pids, loss_cfg.tau)
                    loss = loss + loss_cfg.align_weight * l_align
                    sums["align"] += l_align.item()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += loss.item()

        stats = {k: v / num_iters / (1 if k == "loss" else len(views)) for k, v in sums.items()}
        stats["iters"] = num_iters
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return state, stats
