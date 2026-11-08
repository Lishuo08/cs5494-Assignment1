"""Train the MNIST GAN with the E0--E3 settings in 实验方案.md."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter


# cuDNN also searches PATH when Python is launched directly by a Windows kernel.
if sys.platform == "win32" and (Path(sys.prefix) / "Library" / "bin").is_dir():
    _dll_dir = str(Path(sys.prefix) / "Library" / "bin")
    _path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry != _dll_dir]
    os.environ["PATH"] = os.pathsep.join([_dll_dir, *_path_entries])

import torch
from torch import nn
from matplotlib.figure import Figure
from torchvision.utils import save_image

from mnist_data import DEFAULT_DATA_DIR, PROJECT_DIR, make_train_loader
from models import Discriminator, Generator


@dataclass(frozen=True)
class TrainConfig:
    experiment: str = "E0"
    seed: int = 42
    debug: bool = True
    epochs: int = 2
    batch_size: int = 128
    latent_dim: int = 100
    learning_rate: float = 0.0002
    betas: tuple[float, float] = (0.5, 0.999)
    weight_decay: float = 0.0
    use_batchnorm: bool = True
    sample_seed: int = 2026
    sample_count: int = 64


def get_experiment_config(experiment: str = "E0", seed: int = 42) -> TrainConfig:
    settings = {
        "E0": dict(debug=True, epochs=2),
        "E1": dict(debug=False, epochs=20),
        "E2": dict(debug=False, epochs=20, learning_rate=0.0001),
        "E3": dict(debug=False, epochs=20, use_batchnorm=False),
    }
    return TrainConfig(experiment=experiment, seed=seed, **settings[experiment])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_fixed_noise(config: TrainConfig) -> torch.Tensor:
    """Create CPU display noise without consuming the training RNG."""
    return torch.randn(
        config.sample_count,
        config.latent_dim,
        generator=torch.Generator().manual_seed(config.sample_seed),
    )


def save_samples(generator: Generator, noise: torch.Tensor, path: Path) -> None:
    """Sample in eval mode, save [0, 1] pixels, then restore the previous mode."""
    was_training = generator.training
    generator.eval()
    try:
        with torch.no_grad():
            images = generator(noise).cpu().add(1).div(2)
        save_image(images, path, nrow=8)
    finally:
        generator.train(was_training)


def train_one_epoch(
    generator: Generator,
    discriminator: Discriminator,
    loader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """One D step and one G step per batch; report sample-weighted means."""
    generator.train()
    discriminator.train()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    samples = batches = 0

    for real_images in loader:
        real_images = real_images.to(device, non_blocking=device.type == "cuda")
        batch_size = len(real_images)

        # D: real -> 1, detached fake -> 0. G receives no new gradients here.
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        fake_images = generator(z)
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images.detach())
        d_loss = criterion(real_logits, torch.ones_like(real_logits)) + criterion(
            fake_logits, torch.zeros_like(fake_logits)
        )
        d_loss.backward()
        optimizer_d.step()

        # G: fresh noise, fake -> 1. Freeze D weights but keep its input gradient.
        optimizer_d.zero_grad(set_to_none=True)
        discriminator.requires_grad_(False)
        try:
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            generator_logits = discriminator(generator(z))
            g_loss = criterion(generator_logits, torch.ones_like(generator_logits))
            g_loss.backward()
            optimizer_g.step()
        finally:
            discriminator.requires_grad_(True)

        # D probabilities are from the D-stage forward passes, before its step.
        values = torch.stack(
            [
                g_loss.detach(),
                d_loss.detach(),
                real_logits.detach().sigmoid().mean(),
                fake_logits.detach().sigmoid().mean(),
            ]
        )
        totals += values.to(torch.float64) * batch_size
        samples += batch_size
        batches += 1
        last_batch_size = batch_size

    means = (totals / samples).cpu().tolist()
    return dict(
        zip(("g_loss", "d_loss", "d_real_prob", "d_fake_prob"), means),
        samples=samples,
        batches=batches,
        last_batch_size=last_batch_size,
        g_updates=batches,
        d_updates=batches,
    )


def save_training_curves(history: list[dict], path: Path) -> None:
    """Save raw epoch means; use Figure directly to preserve notebook backend."""
    figure = Figure(figsize=(10, 4), layout="constrained")
    loss_ax, probability_ax = figure.subplots(1, 2)
    epochs = [row["epoch"] for row in history]
    for key, label in (("g_loss", "G loss"), ("d_loss", "D loss")):
        loss_ax.plot(epochs, [row[key] for row in history], marker="o", label=label)
    loss_ax.set(xlabel="Epoch", ylabel="Sample-weighted mean loss", xticks=epochs)
    loss_ax.legend()
    for key, label in (("d_real_prob", "D(real)"), ("d_fake_prob", "D(fake)")):
        probability_ax.plot(epochs, [row[key] for row in history], marker="o", label=label)
    probability_ax.set(xlabel="Epoch", ylabel="D-stage mean probability", ylim=(0, 1), xticks=epochs)
    probability_ax.legend()
    figure.savefig(path, dpi=160)


def run_experiment(
    experiment: str = "E0",
    *,
    seed: int = 42,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = None,
    device: str = "auto",
) -> dict:
    """Start a fresh run; existing result directories are kept intact."""
    config = get_experiment_config(experiment, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    data_dir = Path(data_dir)

    torch.manual_seed(config.seed)
    generator = Generator(config.latent_dim, config.use_batchnorm).to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = make_train_loader(
        data_dir,
        debug=config.debug,
        batch_size=config.batch_size,
        seed=config.seed,
        pin_memory=device.type == "cuda",
    )

    output_dir = Path(output_dir) if output_dir is not None else PROJECT_DIR / "outputs" / experiment / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    sample_dir = output_dir / "samples"
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir.mkdir()
    checkpoint_dir.mkdir()
    environment = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    record = {
        **asdict(config),
        "data_dir": str(data_dir.resolve()),
        "sample_count_per_epoch": len(loader.dataset),
        "batches_per_epoch": len(loader),
        "optimizer": "Adam",
        "d_to_g_update_ratio": "1:1",
        "loss": "BCEWithLogitsLoss",
        "environment": environment,
    }
    (output_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    fixed_noise_cpu = make_fixed_noise(config)
    torch.save(fixed_noise_cpu, output_dir / "fixed_noise.pt")
    fixed_noise = fixed_noise_cpu.to(device)
    save_samples(generator, fixed_noise, sample_dir / "epoch_000.png")

    history = []
    print(f"{experiment}: {len(loader.dataset)} samples, {config.epochs} epochs, device={device}", flush=True)
    synchronize(device)
    training_start = perf_counter()
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file:
        writer = None
        for epoch in range(1, config.epochs + 1):
            synchronize(device)
            epoch_start = perf_counter()
            metrics = train_one_epoch(
                generator, discriminator, loader, optimizer_g, optimizer_d, criterion, device
            )
            save_samples(generator, fixed_noise, sample_dir / f"epoch_{epoch:03d}.png")
            checkpoint = {
                "epoch": epoch,
                "config": record,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "loader_rng_state": loader.generator.get_state(),
                "metrics": metrics,
            }
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
            synchronize(device)
            epoch_end = perf_counter()
            row = {
                "epoch": epoch,
                **metrics,
                "epoch_seconds": epoch_end - epoch_start,
                "cumulative_seconds": epoch_end - training_start,
            }
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            metrics_file.flush()
            print(
                f"Epoch {epoch}/{config.epochs}: G={row['g_loss']:.4f}, D={row['d_loss']:.4f}, "
                f"D(real)={row['d_real_prob']:.4f}, D(fake)={row['d_fake_prob']:.4f}, "
                f"samples={row['samples']}, batches={row['batches']}, {row['epoch_seconds']:.2f}s",
                flush=True,
            )
    synchronize(device)
    training_seconds = perf_counter() - training_start

    # Plotting aggregate curves and writing the final summary are postprocessing.
    save_training_curves(history, output_dir / "training_curves.png")
    summary = {
        "status": "completed",
        "experiment": experiment,
        "seed": seed,
        "completed_epochs": len(history),
        "samples_per_epoch": len(loader.dataset),
        "total_samples_seen": sum(row["samples"] for row in history),
        "total_g_updates": sum(row["g_updates"] for row in history),
        "total_d_updates": sum(row["d_updates"] for row in history),
        "training_seconds": training_seconds,
        "timing_scope": "Epoch loop including batch updates, epoch sampling, checkpoint and metric writes; excludes data/model setup, epoch-0 sampling, final curve plotting and summary writing.",
        "epoch_timing_scope": "Batch updates, epoch sampling and checkpoint writes; each timing row is written just after its measurement.",
        "d_probability_scope": "D-stage forward outputs before the D optimizer step",
        "environment": environment,
        "final_metrics": history[-1],
        "final_checkpoint": f"checkpoints/epoch_{config.epochs:03d}.pt",
        "final_samples": f"samples/epoch_{config.epochs:03d}.png",
        "is_formal_comparison": experiment != "E0",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {experiment}: {training_seconds:.2f}s; results: {output_dir.resolve()}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E0", "E1", "E2", "E3"), default="E0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run_experiment(
        args.experiment, seed=args.seed, data_dir=args.data_dir, output_dir=args.output_dir, device=args.device
    )
