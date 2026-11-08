"""MNIST image-only batches for the unconditional GAN in 实验方案.md."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"


def load_train_images(data_dir: str | Path = DEFAULT_DATA_DIR) -> torch.Tensor:
    """Load float32 training images with shape [60000, 1, 28, 28], in [-1, 1]."""
    return torch.load(
        Path(data_dir) / "processed" / "train_images.pt",
        map_location="cpu",
        weights_only=True,
    )


def make_train_loader(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    debug: bool = False,
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Return shuffled image tensors, retaining the last incomplete batch.

    debug=True selects the saved, fixed 6,000-image subset. Its membership is
    independent of `seed`, which controls batch order with a dedicated RNG.
    Labels are deliberately absent from the batches for unconditional training.
    Reuse the loader across epochs so its shuffle RNG advances each epoch.
    """
    images = load_train_images(data_dir)
    dataset = images
    if debug:
        indices = torch.load(
            Path(data_dir) / "processed" / "debug_indices.pt",
            map_location="cpu",
            weights_only=True,
        )
        dataset = Subset(images, indices.tolist())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
