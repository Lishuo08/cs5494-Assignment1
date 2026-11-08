"""Download original MNIST and prepare the data portion of the GAN assignment."""

import argparse
import csv
import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

from mnist_data import DEFAULT_DATA_DIR, PROJECT_DIR, make_train_loader


def check_loader(data_dir: Path, *, debug: bool) -> dict:
    """One data-only pass to confirm shapes and that no samples are dropped."""
    loader = make_train_loader(data_dir, debug=debug)
    sample_count = 0
    for images in loader:
        assert images.dtype == torch.float32
        assert images.shape[1:] == (1, 28, 28)
        sample_count += len(images)
        last_batch_size = len(images)
    expected_count = 6_000 if debug else 60_000
    assert sample_count == expected_count
    assert last_batch_size == expected_count % 128
    return {
        "samples_seen": sample_count,
        "batches": len(loader),
        "last_batch_size": last_batch_size,
    }


def save_plots(images: torch.Tensor, full_counts: list, debug_counts: list, output_dir: Path):
    # Display the first 64 samples in original order, without selecting examples.
    grid = make_grid(images[:64].add(1).div(2), nrow=8, padding=2)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).numpy(), vmin=0, vmax=1)
    ax.set_title("MNIST training samples: first 64 (displayed in [0, 1])")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "training_samples.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, counts, title in zip(
        axes,
        (full_counts, debug_counts),
        ("Full training set (60,000)", "Fixed debug subset (6,000; seed 42)"),
    ):
        ax.bar(range(10), counts)
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit label (statistics only)")
        ax.set_ylabel("Number of images")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "label_distribution.png", dpi=160)
    plt.close(fig)


def prepare(data_dir: Path, output_dir: Path, raw_cache: Path | None = None) -> dict:
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if raw_cache is not None:
        raw_dir = data_dir / "MNIST" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for filename, _ in MNIST.resources:
            shutil.copy2(raw_cache / filename, raw_dir / filename)

    print("Loading original MNIST training split...", flush=True)
    raw = MNIST(root=str(data_dir), train=True, download=True)
    assert raw.data.shape == (60_000, 28, 28)

    # Preserve all 256 grayscale levels. No thresholding or augmentation.
    images = raw.data.unsqueeze(1).to(torch.float32).div_(255).mul_(2).sub_(1)
    debug_indices = torch.randperm(
        len(images), generator=torch.Generator().manual_seed(42)
    )[:6_000]
    torch.save(images, processed_dir / "train_images.pt")
    torch.save(debug_indices, processed_dir / "debug_indices.pt")

    full_counts = torch.bincount(raw.targets, minlength=10).tolist()
    debug_counts = torch.bincount(raw.targets[debug_indices], minlength=10).tolist()
    with (output_dir / "label_counts.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["digit", "train_count", "debug_count"])
        writer.writerows((digit, full_counts[digit], debug_counts[digit]) for digit in range(10))

    # Check data requirements only; no GAN model or training is run here.
    assert images.min().item() == -1.0 and images.max().item() == 1.0
    assert raw.data[0].unique().numel() > 2
    assert torch.allclose((images[:64] + 1) / 2, raw.data[:64, None].float() / 255)
    loader_checks = {
        "formal": check_loader(data_dir, debug=False),
        "debug": check_loader(data_dir, debug=True),
    }
    save_plots(images, full_counts, debug_counts, output_dir)

    summary = {
        "dataset": "MNIST",
        "split_used": "train",
        "download_method": "torchvision.datasets.MNIST(train=True, download=True)",
        "torchvision_primary_mirror": MNIST.mirrors[0],
        "raw_archive_cache": str(raw_cache.resolve()) if raw_cache is not None else None,
        "download_note": "torchvision also caches test files; they are not loaded or used here.",
        "train_count": len(images),
        "image_shape": list(images.shape[1:]),
        "dtype": str(images.dtype),
        "raw_pixel_range": [int(raw.data.min()), int(raw.data.max())],
        "processed_pixel_range": [float(images.min()), float(images.max())],
        "preprocessing": "uint8 -> float32 / 255 -> 2*x - 1",
        "binarization": False,
        "augmentation": False,
        "labels_used_for": "distribution statistics only; DataLoader yields images only",
        "train_label_counts": full_counts,
        "debug_count": len(debug_indices),
        "debug_subset_seed": 42,
        "debug_sampling": "torch.randperm(60000, dedicated generator with seed 42)[:6000]",
        "debug_label_counts": debug_counts,
        "loader": {
            "batch_size": 128,
            "shuffle_each_epoch": True,
            "drop_last": False,
            "shuffle_seed": 42,
            "num_workers": 0,
            "rng": "dedicated torch.Generator; does not consume the GAN global RNG",
        },
        "loader_checks": loader_checks,
        "environment": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "data")
    parser.add_argument("--raw-cache", type=Path, help="Reuse original MNIST .gz archives from this directory")
    args = parser.parse_args()
    prepare(args.data_dir, args.output_dir, args.raw_cache)
