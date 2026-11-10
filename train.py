"""Train CO-EVO.

    python train.py --config configs/base.yaml configs/p1_market.yaml
    python train.py --config configs/base.yaml configs/p1_market.yaml --set fed.rounds=20 output_dir=runs/tmp

The first config file is the base; later files and --set items are overlays.
Re-running the same command resumes from <output_dir>/ckpt/last.pt.
"""
import argparse

from coevo.config import load_config
from coevo.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":  # required for DataLoader workers on Windows
    main()
