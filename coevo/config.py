"""YAML config loading: ordered overlays plus `a.b=value` command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """dict with attribute access; nested dicts are converted recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = __setitem__

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}


def _merge(base: dict, overlay: dict, path: str = "") -> None:
    """Merge `overlay` into `base` in place. Keys must already exist in `base`,
    so a typo in an overlay or override fails instead of being silently ignored."""
    for key, value in overlay.items():
        where = f"{path}.{key}" if path else key
        if key not in base:
            raise KeyError(f"unknown config key '{where}'")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value, where)
        else:
            base[key] = value


def _parse_override(item: str) -> dict:
    if "=" not in item:
        raise ValueError(f"override must look like key.sub=value, got '{item}'")
    dotted, raw = item.split("=", 1)
    nested: dict[str, Any] = {}
    cursor = nested
    keys = dotted.strip().split(".")
    for key in keys[:-1]:
        cursor[key] = {}
        cursor = cursor[key]
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 reads "1e-3" as a string
        try:
            value = float(value)
        except ValueError:
            pass
    cursor[keys[-1]] = value
    return nested


def load_config(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> Config:
    """First file is the full base config; later files and overrides are overlays."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("at least one config file is required")
    with paths[0].open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for path in paths[1:]:
        with path.open("r", encoding="utf-8") as f:
            _merge(data, yaml.safe_load(f) or {})
    for item in overrides:
        _merge(data, _parse_override(item))
    return Config(data)


def dump_config(cfg: Config, path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
