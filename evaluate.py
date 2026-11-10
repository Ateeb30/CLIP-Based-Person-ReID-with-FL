"""Evaluate a saved global model on the configured targets.

    python evaluate.py --config runs/p1_market_rn50/config.yaml --checkpoint runs/p1_market_rn50/ckpt/last.pt
    python evaluate.py --config runs/p3_rn50/config.yaml --checkpoint runs/p3_rn50/ckpt/last.pt --set data.targets=[msmt17]
"""
import argparse
import json

import torch

from coevo.config import load_config
from coevo.data import build_test_transform, load_dataset
from coevo.evaluation import evaluate_dataset
from coevo.models import build_clip, build_image_encoder
from coevo.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    device = resolve_device(cfg.device)

    clip_model = None
    if cfg.model.backbone == "clip_vit":
        clip_model, _ = build_clip(cfg.clip, (cfg.data.height, cfg.data.width))
    model, mean, std = build_image_encoder(cfg, clip_model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["global_state"])
    model.to(device)

    transform = build_test_transform(cfg.data, mean, std)
    amp = cfg.amp and device.type == "cuda"
    for name in cfg.data.targets:
        data = load_dataset(name, cfg.data)
        res = evaluate_dataset(model, data, transform, cfg.eval, cfg.data, device, amp)
        print(json.dumps({"round": checkpoint["round"], "target": name, **res}))


if __name__ == "__main__":  # required for DataLoader workers on Windows
    main()
