"""Read the original MNIST IDX files without depending on torchvision."""

from __future__ import annotations

import gzip
import hashlib
import ssl
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
import certifi

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def download_mnist(root: str | Path) -> Path:
    raw = Path(root) / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in FILES.items():
        destination = raw / filename
        if destination.exists() and hashlib.md5(destination.read_bytes()).hexdigest() == expected_md5:
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {filename}", flush=True)
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(BASE_URL + filename, timeout=60, context=context) as response:
            temporary.write_bytes(response.read())
        if hashlib.md5(temporary.read_bytes()).hexdigest() != expected_md5:
            raise RuntimeError(f"Checksum mismatch for {filename}")
        temporary.replace(destination)
    return raw


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        zero, datatype, dimensions = struct.unpack(">HBB", handle.read(4))
        if zero != 0 or datatype != 8 or dimensions not in (1, 3):
            raise ValueError(f"Unexpected MNIST IDX header: {path}")
        shape = struct.unpack(">" + "I" * dimensions, handle.read(4 * dimensions))
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(shape).copy()


def load_subset(root: str | Path, classes: list[int], per_class: int,
                seed: int, test: bool = False) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    raw = Path(root) / "MNIST" / "raw"
    prefix = "t10k" if test else "train"
    images = read_idx(raw / f"{prefix}-images-idx3-ubyte.gz")
    labels = read_idx(raw / f"{prefix}-labels-idx1-ubyte.gz")
    generator = np.random.default_rng(seed)
    indices = []
    for digit in classes:
        candidates = np.flatnonzero(labels == digit)
        if len(candidates) < per_class:
            raise ValueError(f"Only {len(candidates)} examples of digit {digit}")
        indices.extend(generator.choice(candidates, per_class, replace=False).tolist())
    indices = np.asarray(indices, dtype=np.int64)
    generator.shuffle(indices)
    mapping = {digit: i for i, digit in enumerate(classes)}
    targets = np.asarray([mapping[int(label)] for label in labels[indices]])
    x = torch.from_numpy(images[indices].reshape(len(indices), -1).astype(np.float32) / 255.0)
    y = torch.from_numpy(targets).long()
    return x, y, indices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    print(download_mnist(args.root))
