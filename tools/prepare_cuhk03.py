"""Convert the official `cuhk-03.mat` into the open-reid layout that the DACS loader
(and `coevo.data.load_cuhk03`) reads:

    <out>/images/{pid:08d}_{cam:02d}_{idx:04d}.jpg
    <out>/meta.json      identities[pid][cam] -> list of file names
    <out>/splits.json    the 20 official test sets (100 test identities each)

Re-implementation of the conversion in open-reid's `reid/datasets/cuhk03.py`
(labeled and detected boxes are both kept, cameras 1-5 of each pair collapse to
camera 0 / 1). NOT yet run against the real file: check the printed counts
(1,467 identities, 20 splits, 100 test identities per split) before training.

    python tools/prepare_cuhk03.py --mat data/cuhk03/cuhk03_release/cuhk-03.mat --out data/cuhk03/cuhk03_release
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.mat, "r") as mat:
        def deref(ref):
            return mat[ref][:].T

        def dump(refs, pid, cam, names):
            for ref in refs:
                img = deref(ref)
                if img.size == 0 or img.ndim < 2:
                    break
                name = f"{pid:08d}_{cam:02d}_{len(names):04d}.jpg"
                Image.fromarray(img).save(images_dir / name)
                names.append(name)

        identities = []
        for labeled, detected in zip(mat["labeled"][0], mat["detected"][0]):
            labeled, detected = deref(labeled), deref(detected)
            assert labeled.shape == detected.shape
            for i in range(labeled.shape[0]):
                pid = len(identities)
                names = [[], []]
                dump(labeled[i, :5], pid, 0, names[0])
                dump(detected[i, :5], pid, 0, names[0])
                dump(labeled[i, 5:], pid, 1, names[1])
                dump(detected[i, 5:], pid, 1, names[1])
                identities.append(names)

        view_counts = [deref(ref).shape[0] for ref in mat["labeled"][0]]
        offsets = np.r_[0, np.cumsum(view_counts)]
        splits = []
        for ref in mat["testsets"][0]:
            test_info = deref(ref).astype(np.int32)
            test_pids = sorted(int(offsets[i - 1] + j - 1) for i, j in test_info)
            trainval = sorted(set(range(int(offsets[-1]))) - set(test_pids))
            splits.append({"trainval": trainval, "query": test_pids, "gallery": test_pids})

    meta = {"name": "cuhk03", "shot": "multiple", "num_cameras": 2, "identities": identities}
    (args.out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (args.out / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    num_images = sum(len(cam) for person in identities for cam in person)
    print(f"identities: {len(identities)}  images: {num_images}  splits: {len(splits)}  "
          f"test ids in split 0: {len(splits[0]['query'])}")


if __name__ == "__main__":
    main()
