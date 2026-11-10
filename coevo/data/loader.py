"""Transforms, the identity sampler and DataLoader construction."""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from ..utils import seed_worker
from .datasets import Sample

_INTERPOLATION = {"bicubic": InterpolationMode.BICUBIC, "bilinear": InterpolationMode.BILINEAR}


def build_train_transform(data_cfg, mean, std) -> T.Compose:
    aug = data_cfg.augment
    size = (data_cfg.height, data_cfg.width)
    return T.Compose([
        T.Resize(size, interpolation=_INTERPOLATION[aug.interpolation]),
        T.RandomHorizontalFlip(p=aug.hflip_prob),
        T.Pad(aug.pad),
        T.RandomCrop(size),
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
        T.RandomErasing(p=aug.random_erasing_prob, scale=(0.02, 0.4), ratio=(0.3, 3.3),
                        value=aug.random_erasing_value),
    ])


def build_test_transform(data_cfg, mean, std) -> T.Compose:
    size = (data_cfg.height, data_cfg.width)
    return T.Compose([
        T.Resize(size, interpolation=_INTERPOLATION[data_cfg.augment.interpolation]),
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ])


class ImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, pid, cam = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), pid, cam, index


class IdentitySampler(Sampler):
    """Port of DACS `RandomMultipleGallerySampler`: for every identity, one random anchor
    image plus `num_instances - 1` images of the same identity, taken from other cameras
    when the identity has any, otherwise from the same camera. An identity with a single
    image contributes one sample only (as in DACS)."""

    def __init__(self, samples: list[Sample], num_instances: int):
        self.samples = samples
        self.num_instances = num_instances
        self.pid_index = defaultdict(list)
        for index, (_, pid, _) in enumerate(samples):
            self.pid_index[pid].append(index)
        self.pids = list(self.pid_index)

    def __len__(self) -> int:
        return len(self.pids) * self.num_instances

    def __iter__(self):
        order = torch.randperm(len(self.pids)).tolist()
        out = []
        extra = self.num_instances - 1
        for k in order:
            index = self.pid_index[self.pids[k]]
            anchor = random.choice(index)
            out.append(anchor)
            anchor_cam = self.samples[anchor][2]
            pool = [i for i in index if self.samples[i][2] != anchor_cam]
            if not pool:
                pool = [i for i in index if i != anchor]
            if not pool:
                continue
            picks = np.random.choice(len(pool), size=extra, replace=len(pool) < extra)
            out.extend(pool[j] for j in picks)
        return iter(out)


def build_train_loader(samples: list[Sample], transform, batch_size: int, num_instances: int,
                       num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        ImageDataset(samples, transform),
        batch_size=batch_size,
        sampler=IdentitySampler(samples, num_instances),
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )


def build_eval_loader(samples: list[Sample], transform, batch_size: int, num_workers: int,
                      pin_memory: bool) -> DataLoader:
    return DataLoader(
        ImageDataset(samples, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
