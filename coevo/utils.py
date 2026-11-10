"""Seeding, run logging (file + CSV + TensorBoard), checkpoints, git info."""
from __future__ import annotations

import csv
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmul; must be set before the first CUDA call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def git_info(repo_dir: Path) -> dict:
    def run(*args):
        try:
            out = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit or "unknown", "dirty": bool(status) if status is not None else None}


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("coevo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class MetricWriter:
    """Appends scalar rows to a CSV file and mirrors them to TensorBoard."""

    def __init__(self, out_dir: Path, tensorboard: bool):
        self.csv_path = Path(out_dir) / "metrics.csv"
        self.new_file = not self.csv_path.exists()
        self.tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(log_dir=str(Path(out_dir) / "tb"))

    def write(self, phase: str, step: int, scalars: dict) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.new_file:
                writer.writerow(["phase", "step", "name", "value"])
                self.new_file = False
            for name, value in scalars.items():
                writer.writerow([phase, step, name, f"{float(value):.6f}"])
        if self.tb is not None:
            for name, value in scalars.items():
                self.tb.add_scalar(f"{phase}/{name}", float(value), step)
            self.tb.flush()

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()


def atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename, so a crash never leaves a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
