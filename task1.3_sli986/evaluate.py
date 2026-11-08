"""Common-noise sampling, inference timing, and comparison for E1--E3."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
from statistics import mean, stdev
from time import perf_counter

# The training module configures the current Conda DLL path before torch import.
from train import PROJECT_DIR, synchronize

import torch
from matplotlib.figure import Figure
from torchvision.utils import make_grid, save_image

from models import Generator


EXPERIMENTS = ("E1", "E2", "E3")


def load_generator(run_dir: Path, device: torch.device, epoch: int = 20):
    checkpoint = torch.load(
        run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    generator = Generator(config["latent_dim"], config["use_batchnorm"]).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, checkpoint


def benchmark_inference(generator: Generator, device: torch.device) -> dict:
    """Time 1,000-image forward passes: 10 warmups, 30 synchronized repeats."""
    noise = torch.randn(
        1000, generator.latent_dim, generator=torch.Generator().manual_seed(2028)
    ).to(device)
    seconds = []
    with torch.no_grad():
        for _ in range(10):
            generator(noise)
        synchronize(device)
        for _ in range(30):
            synchronize(device)
            start = perf_counter()
            generator(noise)
            synchronize(device)
            seconds.append(perf_counter() - start)
    batch_mean_ms = mean(seconds) * 1000
    return {
        "batch_size": 1000,
        "warmups": 10,
        "repeats": 30,
        "noise_seed": 2028,
        "batch_mean_ms": batch_mean_ms,
        "batch_std_ms": stdev(seconds) * 1000,
        "amortized_image_mean_ms": batch_mean_ms / 1000,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "repeat_seconds": seconds,
        "scope": "Generator eval/no_grad forward only; input allocated on device before timing; GPU synchronized at both boundaries.",
    }


def evaluate_run(run_dir: Path, device: torch.device) -> dict:
    """Reload epoch 20, check completed training, sample 200, and benchmark."""
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as file:
        history = list(csv.DictReader(file))
    assert training["status"] == "completed" and len(history) == 20
    assert all(
        (int(row["samples"]), int(row["batches"]), int(row["last_batch_size"])) == (60000, 469, 96)
        for row in history
    )
    assert training["total_g_updates"] == training["total_d_updates"] == 9380
    generator, checkpoint = load_generator(run_dir, device)
    assert checkpoint["epoch"] == 20
    for key in ("optimizer_g_state_dict", "optimizer_d_state_dict"):
        assert all(int(state["step"].item()) == 9380 for state in checkpoint[key]["state"].values())

    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    noise = torch.randn(200, generator.latent_dim, generator=torch.Generator().manual_seed(2027))
    with torch.no_grad():
        images = generator(noise.to(device)).cpu()
    assert images.shape == (200, 1, 28, 28) and torch.isfinite(images).all()
    torch.save(noise, evaluation_dir / "noise.pt")
    torch.save(images, evaluation_dir / "images.pt")
    save_image(images[:64].add(1).div(2), evaluation_dir / "first_64.png", nrow=8)
    timing = benchmark_inference(generator, device)
    with (evaluation_dir / "inference_timings.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["repeat", "seconds"])
        writer.writerows(enumerate(timing["repeat_seconds"], start=1))
    config = checkpoint["config"]
    result = {
        "experiment": config["experiment"],
        "epoch": 20,
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "use_batchnorm": config["use_batchnorm"],
        "evaluation_noise_seed": 2027,
        "evaluation_samples": 200,
        "training_seconds": training["training_seconds"],
        "epoch_seconds_min": min(float(row["epoch_seconds"]) for row in history),
        "epoch_seconds_max": max(float(row["epoch_seconds"]) for row in history),
        "final_g_loss": training["final_metrics"]["g_loss"],
        "final_d_loss": training["final_metrics"]["d_loss"],
        "final_d_real_prob": training["final_metrics"]["d_real_prob"],
        "final_d_fake_prob": training["final_metrics"]["d_fake_prob"],
        "training_verified": {"epochs": 20, "samples_per_epoch": 60000, "g_updates": 9380, "d_updates": 9380},
        "inference": timing,
        "environment": training["environment"],
    }
    (evaluation_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Evaluated {config['experiment']}: 200 samples; inference 1000 images = "
        f"{timing['batch_mean_ms']:.3f} +/- {timing['batch_std_ms']:.3f} ms",
        flush=True,
    )
    return result


def save_comparison_figures(outputs_dir: Path, comparison_dir: Path, seed: int, device: torch.device):
    figure = Figure(figsize=(12, 4.3), layout="constrained")
    for experiment, axis in zip(EXPERIMENTS, figure.subplots(1, 3)):
        images = torch.load(outputs_dir / experiment / f"seed_{seed}" / "evaluation" / "images.pt", weights_only=True)
        # The displayed 64 are exactly the first 64 of the common 200, without selection.
        grid = make_grid(images[:64].add(1).div(2), nrow=8)
        axis.imshow(grid.permute(1, 2, 0).numpy(), vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"{experiment}: epoch 20, evaluation seed 2027")
        axis.axis("off")
    figure.savefig(comparison_dir / "final_samples.png", dpi=160)

    figure = Figure(figsize=(12, 8), layout="constrained")
    axes = figure.subplots(2, 2).flatten()
    keys = ("g_loss", "d_loss", "d_real_prob", "d_fake_prob")
    for experiment in EXPERIMENTS:
        with (outputs_dir / experiment / f"seed_{seed}" / "metrics.csv").open(encoding="utf-8", newline="") as file:
            history = list(csv.DictReader(file))
        for axis, key in zip(axes, keys):
            axis.plot([int(row["epoch"]) for row in history], [float(row[key]) for row in history], label=experiment)
            axis.set(xlabel="Epoch", ylabel=key)
            axis.legend()
            if key.endswith("prob"):
                axis.set_ylim(0, 1)
    figure.savefig(comparison_dir / "training_comparison.png", dpi=160)

    run_dir = outputs_dir / "E1" / f"seed_{seed}"
    fixed_noise = torch.load(run_dir / "fixed_noise.pt", weights_only=True)[:4].to(device)
    figure = Figure(figsize=(5, 7), layout="constrained")
    axes = figure.subplots(5, 4)
    for row, epoch in enumerate((1, 5, 10, 15, 20)):
        generator, _ = load_generator(run_dir, device, epoch)
        with torch.no_grad():
            images = generator(fixed_noise).cpu().add(1).div(2)
        for column, axis in enumerate(axes[row]):
            axis.imshow(images[column, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(f"Epoch {epoch}")
    figure.savefig(comparison_dir / "baseline_progress.png", dpi=160)


def prepare_annotations(outputs_dir: Path, comparison_dir: Path, seed: int) -> dict:
    """Prepare blind sheets and preserve any labels already entered by the user."""
    key_path = comparison_dir / "annotation_key.json"
    if key_path.exists():
        aliases = json.loads(key_path.read_text(encoding="utf-8"))
    else:
        order = random.Random(2029).sample(list(EXPERIMENTS), len(EXPERIMENTS))
        aliases = dict(zip(("A", "B", "C"), order))
        key_path.write_text(json.dumps(aliases, indent=2) + "\n", encoding="utf-8")
    annotation_path = comparison_dir / "annotations.csv"
    if not annotation_path.exists():
        with annotation_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["set_id", "sample_id", "label", "notes"])
            writer.writerows((alias, index, "", "") for alias in aliases for index in range(200))

    for alias, experiment in aliases.items():
        images = torch.load(outputs_dir / experiment / f"seed_{seed}" / "evaluation" / "images.pt", weights_only=True)
        for page in range(2):
            figure = Figure(figsize=(10, 11), layout="constrained")
            figure.suptitle(f"Set {alias}, samples {page * 100:03d}--{page * 100 + 99:03d}")
            for offset, axis in enumerate(figure.subplots(10, 10).flatten()):
                index = page * 100 + offset
                axis.imshow(images[index, 0].add(1).div(2).numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                axis.set_title(f"{index:03d}", fontsize=8)
                axis.axis("off")
            figure.savefig(comparison_dir / f"annotation_{alias}_{page + 1}.png", dpi=150)
    return aliases


def summarize_annotations(comparison_dir: Path, aliases: dict) -> dict:
    with (comparison_dir / "annotations.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    results = {}
    valid_labels = [str(digit) for digit in range(10)] + ["unreadable"]
    for alias, experiment in aliases.items():
        selected = [row for row in rows if row["set_id"] == alias]
        identifiers = [int(row["sample_id"]) for row in selected]
        missing_ids = sorted(set(range(200)) - set(identifiers))
        duplicate_ids = sorted(index for index, count in Counter(identifiers).items() if count > 1)
        unexpected_ids = sorted(set(identifiers) - set(range(200)))
        invalid_labels = []
        blank_ids = []
        labels = []
        for row in selected:
            label = row["label"].strip().replace("无法辨认", "unreadable")
            if not label:
                blank_ids.append(int(row["sample_id"]))
            elif label not in valid_labels:
                invalid_labels.append({"sample_id": int(row["sample_id"]), "label": row["label"]})
            else:
                labels.append(label)
        invalid = bool(duplicate_ids or unexpected_ids or invalid_labels)
        complete = not invalid and not missing_ids and not blank_ids and len(selected) == len(labels) == 200
        counts = {label: labels.count(label) for label in valid_labels} if complete else None
        results[experiment] = {
            "status": "completed" if complete else "invalid_annotation" if invalid else "pending_annotation",
            "labeled_count": len(labels),
            "recognizable_rate": (200 - counts["unreadable"]) / 200 if complete else None,
            "class_coverage": sum(counts[str(digit)] > 0 for digit in range(10)) if complete else None,
            "unreadable_count": counts["unreadable"] if complete else None,
            "counts": counts,
            "issues": {
                "missing_ids": missing_ids,
                "blank_ids": blank_ids,
                "duplicate_ids": duplicate_ids,
                "unexpected_ids": unexpected_ids,
                "invalid_labels": invalid_labels,
            },
        }
    return results


def save_class_distribution(annotations: dict, path: Path) -> None:
    """Show all 11 classes; incomplete groups are explicitly marked."""
    figure = Figure(figsize=(13, 4), layout="constrained")
    labels = [str(digit) for digit in range(10)] + ["unreadable"]
    maximum = max((max(item["counts"].values()) for item in annotations.values() if item["counts"] is not None), default=1)
    for experiment, axis in zip(EXPERIMENTS, figure.subplots(1, 3)):
        quality = annotations[experiment]
        if quality["counts"] is None:
            axis.text(0.5, 0.5, f"{experiment}: {quality['labeled_count']}/200 valid\n{quality['status']}", ha="center", va="center")
            axis.axis("off")
            continue
        counts = [quality["counts"][label] for label in labels]
        axis.bar(range(11), counts, color=["tab:blue"] * 10 + ["tab:orange"])
        axis.set(xticks=range(11), xticklabels=[str(digit) for digit in range(10)] + ["U"], xlabel="Human label (U = unreadable)", ylabel="Count", ylim=(0, maximum + 5), title=f"{experiment}: n=200, readable={quality['recognizable_rate']:.1%}")
    figure.savefig(path, dpi=160)


def save_comparison_summary(comparison_dir: Path, summary: dict) -> None:
    """Update tables and the label plot while retaining existing timing values."""
    table = []
    for experiment, result in summary["results"].items():
        quality = summary["annotations"][experiment]
        table.append({
            "experiment": experiment,
            "training_seconds": result["training_seconds"],
            "epoch_seconds_min": result["epoch_seconds_min"],
            "epoch_seconds_max": result["epoch_seconds_max"],
            "final_g_loss": result["final_g_loss"],
            "final_d_loss": result["final_d_loss"],
            "inference_1000_mean_ms": result["inference"]["batch_mean_ms"],
            "inference_1000_std_ms": result["inference"]["batch_std_ms"],
            "amortized_image_mean_ms": result["inference"]["amortized_image_mean_ms"],
            **{key: quality[key] for key in ("status", "labeled_count", "recognizable_rate", "class_coverage", "unreadable_count")},
        })
    with (comparison_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (comparison_dir / "class_counts.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment", "label", "count"])
        for experiment in EXPERIMENTS:
            counts = summary["annotations"][experiment]["counts"]
            writer.writerows((experiment, label, counts[label] if counts is not None else "") for label in [str(digit) for digit in range(10)] + ["unreadable"])
    save_class_distribution(summary["annotations"], comparison_dir / "class_distribution.png")
    (comparison_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def refresh_annotations(*, outputs_dir: str | Path = PROJECT_DIR / "outputs", seed: int = 42) -> dict:
    """Refresh human metrics only: no training, image generation, or benchmarking."""
    comparison_dir = Path(outputs_dir) / "comparison" / f"seed_{seed}"
    summary = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    aliases = json.loads((comparison_dir / "annotation_key.json").read_text(encoding="utf-8"))
    summary["annotations"] = summarize_annotations(comparison_dir, aliases)
    save_comparison_summary(comparison_dir, summary)
    return summary


def evaluate_experiments(
    *, outputs_dir: str | Path = PROJECT_DIR / "outputs", seed: int = 42, device: str = "auto"
) -> dict:
    outputs_dir = Path(outputs_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    comparison_dir = outputs_dir / "comparison" / f"seed_{seed}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    results = {experiment: evaluate_run(outputs_dir / experiment / f"seed_{seed}", device) for experiment in EXPERIMENTS}
    noises = [torch.load(outputs_dir / experiment / f"seed_{seed}" / "evaluation" / "noise.pt", weights_only=True) for experiment in EXPERIMENTS]
    assert all(torch.equal(noises[0], noise) for noise in noises[1:])
    save_comparison_figures(outputs_dir, comparison_dir, seed, device)
    aliases = prepare_annotations(outputs_dir, comparison_dir, seed)
    annotations = summarize_annotations(comparison_dir, aliases)
    summary = {"seed": seed, "device": str(device), "evaluation_noise_seed": 2027, "results": results, "annotations": annotations}
    save_comparison_summary(comparison_dir, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--annotations-only", action="store_true", help="Refresh human metrics and class plot without re-running inference timing")
    args = parser.parse_args()
    if args.annotations_only:
        result = refresh_annotations(outputs_dir=args.outputs_dir, seed=args.seed)
        print(json.dumps(result["annotations"], ensure_ascii=False, indent=2))
    else:
        evaluate_experiments(outputs_dir=args.outputs_dir, seed=args.seed, device=args.device)
