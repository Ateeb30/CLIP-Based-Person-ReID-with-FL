"""Write tiny random-image copies of all four datasets in their real folder layouts,
for CPU smoke tests. Images of one identity share a base colour and every camera adds
its own tint, so there is an identity signal and a camera-style signal to learn.

    python tools/make_synthetic.py --out data_synthetic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _image(rng, pid_colour, cam_tint, size) -> Image.Image:
    h, w = size
    base = np.clip(pid_colour + cam_tint, 0, 255)
    noise = rng.normal(0, 20, (h, w, 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))


def make_market(root: Path, rng, size, ids=8, cams=3, per_cam=2) -> None:
    base = root / "market1501" / "Market-1501-v15.09.15"
    tints = rng.uniform(-40, 40, (cams, 3))
    for sub, first_pid in (("bounding_box_train", 1), ("query", 101), ("bounding_box_test", 101)):
        (base / sub).mkdir(parents=True, exist_ok=True)
        for pid in range(first_pid, first_pid + ids):
            colour = np.random.default_rng(pid).uniform(40, 215, 3)
            for cam in range(cams):
                n = 1 if sub == "query" else per_cam
                for k in range(n):
                    frame = k if sub != "bounding_box_test" else k + 10
                    _image(rng, colour, tints[cam], size).save(
                        base / sub / f"{pid:04d}_c{cam + 1}s1_{frame:06d}_00.jpg")
    # one junk and one distractor image, as in the real gallery
    _image(rng, np.full(3, 128.0), tints[0], size).save(base / "bounding_box_test" / "-1_c1s1_000000_00.jpg")
    _image(rng, np.full(3, 128.0), tints[0], size).save(base / "bounding_box_test" / "0000_c1s1_000000_00.jpg")


def make_msmt17(root: Path, rng, size, ids=8, cams=4, per_cam=2) -> None:
    base = root / "msmt17" / "MSMT17_V2"
    tints = rng.uniform(-40, 40, (cams, 3))

    def write(sub, pids, colour_offset):
        lines = []
        for pid in pids:
            colour = np.random.default_rng(pid + colour_offset).uniform(40, 215, 3)
            folder = base / sub / f"{pid:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            idx = 0
            for cam in range(cams):
                for _ in range(per_cam):
                    name = f"{pid:04d}_{idx:03d}_{cam + 1:02d}_0303morning_{idx:04d}_0.jpg"
                    _image(rng, colour, tints[cam], size).save(folder / name)
                    lines.append((f"{pid:04d}/{name}", pid, idx))
                    idx += 1
        return lines

    def dump(name, lines):
        (base / name).write_text("".join(f"{rel} {pid}\n" for rel, pid, _ in lines), encoding="utf-8")

    train = write("mask_train_v2", range(ids), 1000)
    dump("list_train.txt", [x for x in train if x[2] % 4 != 3])
    dump("list_val.txt", [x for x in train if x[2] % 4 == 3])
    test = write("mask_test_v2", range(ids), 2000)
    dump("list_query.txt", [x for x in test if x[2] % per_cam == 0])
    dump("list_gallery.txt", [x for x in test if x[2] % per_cam != 0])


def make_cuhk02(root: Path, rng, size, ids_per_pair=3, per_cam=2) -> None:
    base = root / "cuhk02" / "Dataset"
    for p, pair in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        tints = rng.uniform(-40, 40, (2, 3))
        for c, cam in enumerate(["cam1", "cam2"]):
            folder = base / pair / cam
            folder.mkdir(parents=True, exist_ok=True)
            for pid in range(1, ids_per_pair + 1):
                colour = np.random.default_rng(3000 + 100 * p + pid).uniform(40, 215, 3)
                for k in range(per_cam):
                    _image(rng, colour, tints[c], size).save(folder / f"{pid:03d}_{c * per_cam + k + 1}.png")


def make_cuhk03(root: Path, rng, size, ids=12, test_ids=4, per_cam=3) -> None:
    base = root / "cuhk03" / "cuhk03_release"
    (base / "images").mkdir(parents=True, exist_ok=True)
    tints = rng.uniform(-40, 40, (2, 3))
    identities = []
    for pid in range(ids):
        colour = np.random.default_rng(5000 + pid).uniform(40, 215, 3)
        cams = []
        for cam in range(2):
            names = []
            for k in range(per_cam):
                name = f"{pid:08d}_{cam:02d}_{k:04d}.jpg"
                _image(rng, colour, tints[cam], size).save(base / "images" / name)
                names.append(name)
            cams.append(names)
        identities.append(cams)
    meta = {"name": "cuhk03", "shot": "multiple", "num_cameras": 2, "identities": identities}
    (base / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    test = list(range(ids - test_ids, ids))
    split = {"trainval": list(range(ids - test_ids)), "query": test, "gallery": test}
    (base / "splits.json").write_text(json.dumps([split]), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    size = (args.height, args.width)
    make_market(args.out, rng, size)
    make_msmt17(args.out, rng, size)
    make_cuhk02(args.out, rng, size)
    make_cuhk03(args.out, rng, size)
    print(f"synthetic datasets written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
