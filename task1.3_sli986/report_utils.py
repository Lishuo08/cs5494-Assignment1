"""Presentation helpers for the English GAN report; no training or benchmarking."""

import csv
import json

from train import PROJECT_DIR
from evaluate import EXPERIMENTS, load_generator

import torch
from IPython.display import Image, Markdown, display
from matplotlib.figure import Figure
from torchvision.utils import make_grid


def load_report_data() -> dict:
    """Read the completed seed-42 artifacts without changing their contents."""
    comparison_dir = PROJECT_DIR / "outputs" / "comparison" / "seed_42"
    summary = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    if any(summary["annotations"][name]["status"] != "completed" for name in EXPERIMENTS):
        raise ValueError("Complete and refresh the human annotations before rendering the final report.")
    configs = {
        name: json.loads((PROJECT_DIR / "outputs" / name / "seed_42" / "config.json").read_text(encoding="utf-8"))
        for name in ("E0", *EXPERIMENTS)
    }
    data = json.loads((PROJECT_DIR / "outputs" / "data" / "data_summary.json").read_text(encoding="utf-8"))
    return {"root": PROJECT_DIR, "comparison_dir": comparison_dir, "summary": summary, "configs": configs, "data": data}


def _display_table(headers: list[str], rows: list[list]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    display(Markdown("\n".join(lines)))


def show_table(report: dict, table: str) -> None:
    """Render small report tables directly from the recorded results."""
    summary = report["summary"]
    if table == "experiments":
        roles = {"E0": "Debug only", "E1": "Baseline", "E2": "Lower learning rate", "E3": "Generator BN ablation"}
        headers = ["Run", "Role", "Training images", "Epochs", "G / D learning rate", "Generator BatchNorm"]
        rows = [[name, roles[name], f"{config['sample_count_per_epoch']:,}", config["epochs"], f"{config['learning_rate']:.4f} / {config['learning_rate']:.4f}", "Enabled" if config["use_batchnorm"] else "Removed"] for name, config in report["configs"].items()]
    elif table == "quality":
        headers = ["Run", "Recognizable / 200", "Recognizability", "Class coverage", "Unreadable / 200"]
        rows = []
        for name in EXPERIMENTS:
            quality = summary["annotations"][name]
            rows.append([name, 200 - quality["unreadable_count"], f"{quality['recognizable_rate']:.1%}", f"{quality['class_coverage']}/10", quality["unreadable_count"]])
    elif table == "runtime":
        headers = ["Run", "Training (s)", "Epoch range (s)", "1,000-image inference (ms, mean +/- SD)", "Amortized ms/image"]
        rows = []
        for name in EXPERIMENTS:
            result = summary["results"][name]
            timing = result["inference"]
            rows.append([name, f"{result['training_seconds']:.2f}", f"{result['epoch_seconds_min']:.2f}-{result['epoch_seconds_max']:.2f}", f"{timing['batch_mean_ms']:.3f} +/- {timing['batch_std_ms']:.3f}", f"{timing['amortized_image_mean_ms']:.6f}"])
    elif table == "losses":
        headers = ["Run", "Final G loss", "Final D loss", "D(real), D stage", "D(fake), D stage"]
        rows = [[name, *[f"{summary['results'][name][key]:.4f}" for key in ("final_g_loss", "final_d_loss", "final_d_real_prob", "final_d_fake_prob")]] for name in EXPERIMENTS]
    elif table == "environment":
        environment = summary["results"]["E1"]["environment"]
        headers = ["Item", "Recorded value"]
        rows = [["Scope", "Original recorded training and benchmark runs"], ["Python", environment["python"]], ["PyTorch", environment["torch"]], ["torchvision", report["data"]["environment"]["torchvision"]], ["GPU", environment["device_name"]], ["CUDA", environment["cuda"]], ["cuDNN version identifier", environment["cudnn"]], ["Data type", "float32"]]
    else:
        raise ValueError(f"Unknown report table: {table}")
    _display_table(headers, rows)


def show_figure(report: dict, relative_path: str, width: int = 1000) -> None:
    """Embed an existing PNG in the notebook's saved outputs."""
    display(Image(filename=str(report["root"] / relative_path), width=width))


def _save_report_figure(report: dict, figure: Figure, filename: str, width: int) -> None:
    output_dir = report["root"] / "outputs" / "report" / "seed_42"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=160)
    figure.clear()
    display(Image(filename=str(path), width=width))


def show_checkpoint_demo(report: dict, experiment: str = "E1", count: int = 8) -> None:
    """Regenerate a few fixed examples from saved weights; do not time them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = report["root"] / "outputs" / experiment / "seed_42"
    generator, _ = load_generator(run_dir, device, epoch=20)
    noise = torch.load(run_dir / "evaluation" / "noise.pt", map_location="cpu", weights_only=True)[:count].to(device)
    with torch.no_grad():
        images = generator(noise).cpu()
    figure = Figure(figsize=(10, 1.7), layout="constrained")
    axis = figure.subplots()
    grid = make_grid(images.add(1).div(2), nrow=count)
    axis.imshow(grid.permute(1, 2, 0).numpy(), vmin=0, vmax=1, interpolation="nearest")
    axis.axis("off")
    _save_report_figure(report, figure, "checkpoint_demo.png", width=900)
    print(f"Reloaded {experiment}, epoch 20, on {device}; generated tensor shape: {tuple(images.shape)}.")


def show_failure_examples(report: dict) -> None:
    """Display the first four human-unreadable examples per run, in ID order."""
    aliases = json.loads((report["comparison_dir"] / "annotation_key.json").read_text(encoding="utf-8"))
    reverse_aliases = {experiment: alias for alias, experiment in aliases.items()}
    with (report["comparison_dir"] / "annotations.csv").open(encoding="utf-8-sig", newline="") as file:
        annotations = list(csv.DictReader(file))
    figure = Figure(figsize=(8, 6), layout="constrained")
    axes = figure.subplots(3, 4)
    for row, experiment in enumerate(EXPERIMENTS):
        indices = sorted(int(item["sample_id"]) for item in annotations if item["set_id"] == reverse_aliases[experiment] and item["label"].strip() in ("unreadable", "\u65e0\u6cd5\u8fa8\u8ba4"))[:4]
        images = torch.load(report["root"] / "outputs" / experiment / "seed_42" / "evaluation" / "images.pt", map_location="cpu", weights_only=True)
        for column, index in enumerate(indices):
            axis = axes[row, column]
            axis.imshow(images[index, 0].add(1).div(2).numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(f"{experiment}, sample {index:03d}")
        for axis in axes[row]:
            axis.axis("off")
    _save_report_figure(report, figure, "failure_examples.png", width=850)
