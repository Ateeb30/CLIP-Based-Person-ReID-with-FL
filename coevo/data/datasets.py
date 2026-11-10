"""Dataset index builders. Each returns lists of (image_path, pid, camid).

Splits follow the DACS loaders (the code base the paper's baseline numbers come from):
  market1501  standard 751/750 split, junk (pid -1) dropped
  msmt17      train = list_train + list_val, test = list_query / list_gallery
  cuhk02      P1-P4 train, P5 test (cam1 = query, cam2 = gallery)
  cuhk03      open-reid format (meta.json + splits.json), one of the 20 official splits
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

Sample = tuple[str, int, int]


@dataclass
class ReIDData:
    name: str
    train: list[Sample] = field(default_factory=list)
    query: list[Sample] = field(default_factory=list)
    gallery: list[Sample] = field(default_factory=list)

    @property
    def num_train_pids(self) -> int:
        return len({pid for _, pid, _ in self.train})

    @property
    def train_cams(self) -> list[int]:
        return sorted({cam for _, _, cam in self.train})

    def summary(self) -> str:
        def stat(samples):
            return f"{len({p for _, p, _ in samples}):5d} ids {len(samples):7d} imgs {len({c for _, _, c in samples}):3d} cams"

        return f"{self.name}: train {stat(self.train)} | query {stat(self.query)} | gallery {stat(self.gallery)}"


def _relabel(samples: list[Sample]) -> list[Sample]:
    mapping = {pid: i for i, pid in enumerate(sorted({pid for _, pid, _ in samples}))}
    return [(path, mapping[pid], cam) for path, pid, cam in samples]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"dataset path not found: {path}")
    return path


def load_market1501(root: Path, cfg) -> ReIDData:
    base = _require(root / cfg.dir)
    pattern = re.compile(r"([-\d]+)_c(\d)")

    def scan(sub: str) -> list[Sample]:
        out = []
        for path in sorted(_require(base / sub).glob("*.jpg")):
            pid, cam = map(int, pattern.search(path.name).groups())
            if pid == -1:  # junk detections
                continue
            out.append((str(path), pid, cam - 1))
        return out

    return ReIDData("market1501", _relabel(scan("bounding_box_train")), scan("query"), scan("bounding_box_test"))


def load_msmt17(root: Path, cfg) -> ReIDData:
    """MSMT17 V1/V2 in the official layout: `<dir>/<train_dir>/<pid>/<pid>_<idx>_<cam>_....jpg`
    with list files of `relative/path label` lines. V1 and V2 differ only in directory
    names (train/test vs mask_train_v2/mask_test_v2) and face masking."""
    base = _require(root / cfg.dir)

    def read(list_name: str, sub: str) -> list[Sample]:
        out = []
        with _require(base / list_name).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, label = line.rsplit(" ", 1)
                cam = int(Path(rel).name.split("_")[2])
                out.append((str(base / sub / rel), int(label), cam - 1))
        return out

    train = read("list_train.txt", cfg.train_dir)
    if cfg.merge_val:
        train += read("list_val.txt", cfg.train_dir)
    return ReIDData(
        "msmt17", _relabel(train), read("list_query.txt", cfg.test_dir), read("list_gallery.txt", cfg.test_dir)
    )


def load_cuhk02(root: Path, cfg) -> ReIDData:
    base = _require(root / cfg.dir)
    train, query, gallery = [], [], []
    num_train_pids, camid = 0, 0
    for pair in ["P1", "P2", "P3", "P4", "P5"]:
        cams = [sorted(_require(base / pair / cam).glob("*.png")) for cam in ("cam1", "cam2")]
        if pair == cfg.test_pair:
            for paths, dest in zip(cams, (query, gallery)):
                dest.extend((str(p), int(p.name.split("_")[0]), camid) for p in paths)
                camid += 1
            continue
        pids = sorted({p.name.split("_")[0] for paths in cams for p in paths})
        label = {pid: i + num_train_pids for i, pid in enumerate(pids)}
        for paths in cams:
            train.extend((str(p), label[p.name.split("_")[0]], camid) for p in paths)
            camid += 1
        num_train_pids += len(pids)
    return ReIDData("cuhk02", train, query, gallery)


def load_cuhk03(root: Path, cfg) -> ReIDData:
    """open-reid layout produced by tools/prepare_cuhk03.py. Query is the last image of
    each (identity, camera); gallery is the remaining images (DACS convention)."""
    base = _require(root / cfg.dir)
    images = base / "images"
    with _require(base / "meta.json").open("r", encoding="utf-8") as f:
        identities = json.load(f)["identities"]
    with _require(base / "splits.json").open("r", encoding="utf-8") as f:
        split = json.load(f)[cfg.split_id]

    train, query, gallery = [], [], []
    for label, pid in enumerate(sorted(split["trainval"])):
        for cam, names in enumerate(identities[pid]):
            train.extend((str(images / n), label, cam) for n in names)
    for pid in split["query"]:
        for cam, names in enumerate(identities[pid]):
            query.extend((str(images / n), pid, cam) for n in names[-1:])
    for pid in split["gallery"]:
        for cam, names in enumerate(identities[pid]):
            keep = names[:-1] if len(names) > 1 else names
            gallery.extend((str(images / n), pid, cam) for n in keep)
    return ReIDData("cuhk03", train, query, gallery)


_LOADERS = {
    "market1501": load_market1501,
    "msmt17": load_msmt17,
    "cuhk02": load_cuhk02,
    "cuhk03": load_cuhk03,
}


def dataset_names() -> list[str]:
    return sorted(_LOADERS)


def load_dataset(name: str, data_cfg) -> ReIDData:
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset '{name}', expected one of {dataset_names()}")
    return _LOADERS[name](Path(data_cfg.root), data_cfg.datasets[name])


def missing_files(samples: list[Sample]) -> list[str]:
    return [path for path, _, _ in samples if not Path(path).is_file()]
