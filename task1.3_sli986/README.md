# MNIST GAN: Data, Training, Evaluation, and Report

This project implements an unconditional convolutional GAN for MNIST handwritten digit generation (CS5494 Assignment 1, Task 1.3). It includes data preparation, generator and discriminator models, four experiment configurations (E0–E3), checkpoint-based generation, human evaluation, runtime measurements, and an English notebook report.

All four experiments are complete. E0 is a short debugging run; E1–E3 are the formal comparisons. Each formal run trained on 60,000 images for 20 epochs, with 9,380 updates to each network. The final checkpoints were reloaded to generate 200 evaluation images per model, and all 600 human annotations have been summarized.

## Main entry points

| File | Purpose |
|---|---|
| [GAN_report.ipynb](GAN_report.ipynb) | English report with module explanations, generated examples, training curves, generation progress, human evaluation, and runtime analysis. Start here to review the completed work. |
| [debug.ipynb](debug.ipynb) | The single notebook maintained for debugging data, models, gradients, training, and evaluation. |
| [Code structure guide](代码结构说明.md) | It helps me record and comprehense functions, code sections, their relationships, and the corresponding GAN principles. |
| [Experiment plan](实验方案.md) | Help me plan experiment specification, controlled comparisons, and evaluation definitions. |

## Repository contents

This GitHub repository includes final checkpoints only: E0 epoch 2 and E1-E3 epoch 20. All saved sample grids and experiment logs are included. Training from scratch saves checkpoints for every epoch as described below. Raw and processed MNIST files are excluded; run `prepare_data.py` before training or data-dependent debugging.

## Environment setup

Use **Python 3.10 or newer** in a virtual environment or Conda environment of your choice. No environment name or local interpreter path is required. Run the following command from the project root, the folder containing this README:

```shell
python -m pip install -r requirements.txt
```

The dependencies cover PyTorch, torchvision, plotting, and notebook execution. Use compatible versions of PyTorch and torchvision. Open the notebooks in your notebook editor and select the Python kernel in which these dependencies are installed. Both notebooks use generic `python3` kernel metadata.

Training, evaluation, and the report's generation demonstration automatically use CUDA when available and otherwise use the CPU. The training and evaluation commands also accept `--device cpu` or `--device cuda`.

The original results were recorded with Python 3.11.16, PyTorch 2.10.0, torchvision 0.25.0, and an NVIDIA GeForce RTX 4060 Laptop GPU. These describe the recorded runs; the project does not require that exact environment. It has also been checked in another Python environment with CUDA disabled.

## View or rerun the report

Open [GAN_report.ipynb](GAN_report.ipynb) to read the saved English report. Its tables and figures are embedded in the notebook and can be viewed without rerunning the code.

To rerun its demonstrations:

1. Keep the project Python modules and the supplied `outputs/` directory together.
2. Open the notebook with the project root as the working directory, select a kernel with the dependencies installed, and restart the kernel.
3. Run all cells from top to bottom.

The report reads the existing seed-42 results and reloads the E1 epoch-20 checkpoint to generate eight examples. It does not train models or repeat the inference benchmark. CPU execution is supported. The small display calls are implemented in `report_utils.py`; newly rendered demonstration and failure figures are saved under `outputs/report/seed_42/`.

The report currently reads the original `outputs/` tree and seed 42. Results produced under a different output root are separate experiments and are not automatically substituted into the report or its written analysis.

## Prepare the data

Skip this step if the supplied `data/processed/` files are already present and you only need to train. The debugging notebook also reads the raw MNIST files for its data checks.

```shell
python -u prepare_data.py
```

On the first run, torchvision downloads the original MNIST files. Later runs reuse the local raw files and regenerate the processed tensors, statistics, and preview figures. Only the 60,000-image training split is used; torchvision also caches test files, but this project does not load the test split for training or evaluation.

To reuse an offline cache, pass the directory containing the four original MNIST `.gz` archives:

```shell
python -u prepare_data.py --raw-cache path/to/MNIST/raw
```

The original run reused such archives from another assignment directory and let torchvision check and extract them. It did not reuse that assignment's binarized data. No external assignment folder is required to run this project.

Data preparation produces:

- `data/processed/train_images.pt`: all 60,000 images as `float32`, shape `[60000, 1, 28, 28]`, normalized by `uint8 / 255 * 2 - 1` to `[-1, 1]`. There is no binarization or augmentation.
- `data/processed/debug_indices.pt`: a fixed, randomly selected subset of 6,000 training indices, sampled without replacement using a dedicated seed-42 generator. These images remain part of the full training set.
- `outputs/data/`: preprocessing metadata, class counts, a class-distribution plot, and a preview of the first 64 real images in their original order.

Digit labels are used only for data-distribution statistics. GAN training batches contain images only.

## Run E0–E3 and evaluate

### Experiment configurations

| Run | Purpose | Images per epoch | Epochs | G and D learning rate | Generator BatchNorm | Updates per network |
|---|---|---:|---:|---:|---|---:|
| E0 | Debug the complete training pipeline | 6,000 | 2 | 0.0002 | Enabled | 94 |
| E1 | Baseline | 60,000 | 20 | 0.0002 | Enabled | 9,380 |
| E2 | Lower-learning-rate comparison | 60,000 | 20 | 0.0001 | Enabled | 9,380 |
| E3 | Generator BatchNorm ablation | 60,000 | 20 | 0.0002 | Removed | 9,380 |

All runs use seed 42, batch size 128, 100-dimensional Gaussian noise, separate Adam optimizers with betas `(0.5, 0.999)`, and one discriminator update followed by one generator update per batch. Each experiment initializes its own models, optimizers, and data loader. E0 is excluded from the formal quality comparison.

### Train from scratch

**The supplied default output directories already contain completed runs.** The commands below are for a fresh project copy without those run directories. Training requires a new run directory and does not overwrite an existing one. To keep the supplied results and train again, use the separate-output example in the next section.

After preparing the data, run:

```shell
python -u train.py --experiment E0 --device auto
python -u train.py --experiment E1 --device auto
python -u train.py --experiment E2 --device auto
python -u train.py --experiment E3 --device auto
```

The default run directories are `outputs/E0/seed_42/` through `outputs/E3/seed_42/`. After E1–E3 have all completed, run the common evaluation:

```shell
python -u evaluate.py --device auto
```

Evaluation reloads each epoch-20 generator, generates 200 images using the same noise inputs (seed 2027), measures inference time, saves comparison figures, and prepares anonymous annotation sheets. The inference benchmark uses a batch of 1,000 images, 10 warm-up passes, and 30 measured passes.

Running this evaluation command again regenerates evaluation artifacts and repeats the inference benchmark. To retain the recorded timings and update human statistics only, use `--annotations-only` as described below.

### Train again while keeping the supplied results

Use a new output root consistently for training and evaluation. For example, if `outputs_repeat/` has not already been used:

```shell
python -u train.py --experiment E0 --output-dir outputs_repeat/E0/seed_42 --device auto
python -u train.py --experiment E1 --output-dir outputs_repeat/E1/seed_42 --device auto
python -u train.py --experiment E2 --output-dir outputs_repeat/E2/seed_42 --device auto
python -u train.py --experiment E3 --output-dir outputs_repeat/E3/seed_42 --device auto
python -u evaluate.py --outputs-dir outputs_repeat --device auto
```

The training option `--output-dir` specifies one experiment's run directory. The evaluation option `--outputs-dir` specifies the common parent containing `E1/`, `E2/`, and `E3/`. Both commands accept `--seed`; keep the seed and directory names consistent across the three formal runs.

### Human annotation and summary refresh

For the default output root, annotation files are in `outputs/comparison/seed_42/`:

1. View `annotation_A_1.png` through `annotation_C_2.png`, with two pages per anonymous set.
2. Fill the `label` column of `annotations.csv` with `0`–`9` or `unreadable`, matching each `set_id` and `sample_id`. Avoid opening `annotation_key.json` before labeling.
3. Refresh the statistics:

```shell
python -u evaluate.py --annotations-only
```

For a separate output root, use `python -u evaluate.py --outputs-dir outputs_repeat --annotations-only`.

This refresh updates the human-evaluation summaries and class-count plot while retaining the recorded training and inference times. Missing, duplicate, or invalid entries are reported with their IDs; final quality rates are calculated only for complete, valid groups. All 600 supplied annotations are already complete. New training runs require their own annotations.

## Use the debugging notebook

Open [debug.ipynb](debug.ipynb) from the project root, select a kernel with the dependencies installed, restart the kernel, and run cells in order.

- Sections 1–10 cover data, preprocessing, model dimensions, BatchNorm ablation, losses, and gradient flow.
- Sections 11–13 run E0 or load its completed results, then inspect its checkpoints and samples.
- Sections 14–17 run or load E1–E3, perform or read the common evaluation, refresh human statistics, and display comparisons.

Completed training runs are loaded instead of trained again. When a comparison summary exists, the notebook refreshes annotations and retains the previous inference measurements. For new runs, use the command-line workflow above or update the notebook's output-directory variables consistently.

## Code structure and module relationships

Implementation lives in Python modules; notebooks call those modules and display results.

| File | Main responsibilities and connections |
|---|---|
| `prepare_data.py` | Downloads or reuses raw MNIST, normalizes images, selects the fixed debugging subset, and saves tensors and data figures. Uses the shared paths and loader from `mnist_data.py`. |
| `mnist_data.py` | Defines project-relative paths, loads processed tensors with `load_train_images()`, and creates shuffled image-only batches with `make_train_loader()`. Supplies data to training and debugging. |
| `models.py` | Defines `Generator` and `Discriminator`. Training uses both; evaluation reconstructs the generator from checkpoint settings. |
| `train.py` | Defines `TrainConfig` and `get_experiment_config()`. `train_one_epoch()` performs alternating updates; `run_experiment()` manages initialization, logging, fixed-noise samples, checkpoints, and training time. Shared by the command line and debugging notebook. |
| `evaluate.py` | Reloads checkpoints with `load_generator()`, samples and benchmarks models, draws comparisons, creates annotation sheets, and summarizes human labels. `refresh_annotations()` updates human statistics without generation or benchmarking. |
| `report_utils.py` | Loads recorded results and renders report tables, figures, checkpoint demonstrations, and human-labeled failure examples. Reuses `evaluate.load_generator()` and existing artifacts. |
| `debug.ipynb` | The single debugging notebook connecting data, models, training, and evaluation. |
| `GAN_report.ipynb` | The English presentation notebook. Short calls to `report_utils.py` demonstrate existing results. |
| `requirements.txt` | Lists dependencies for the modules and notebooks. |

The main data and result flow is:

```text
Raw MNIST -> prepare_data.py -> data/processed/ -> mnist_data.py
                                                        |
models.py (G and D) --------------------------------> train.py
                                                        |
                                             checkpoints and logs
                                                        |
                                                   evaluate.py
                                                        |
                                  generated samples, timings, annotations
                                                        |
                                                 report_utils.py
                                                        |
                                                GAN_report.ipynb
```

### Model and training behavior

- **Generator:** `[B,100]` noise → `Linear(6272)` → ReLU → `[B,128,7,7]` → `ConvTranspose2d(128,64)` → BatchNorm → ReLU → `ConvTranspose2d(64,1)` → Tanh. Output: `[B,1,28,28]` in `[-1,1]`. Both transposed convolutions use kernel size 4, stride 2, and padding 1. E3 replaces only the generator BatchNorm layer with an identity operation.
- **Discriminator:** `[B,1,28,28]` → `Conv2d(1,64)` → LeakyReLU(0.2) → `Conv2d(64,128)` → LeakyReLU(0.2) → Flatten → `Linear(6272,1)`. Both convolutions use kernel size 4, stride 2, and padding 1. Output: `[B,1]` logits, with no BatchNorm, Dropout, or final Sigmoid.
- **Loss and updates:** `BCEWithLogitsLoss` handles the logits directly. The discriminator learns real → 1 and detached fake → 0; the generator learns fake → 1 using fresh noise. Discriminator parameters are frozen during the generator update while gradients still flow through its input. Learnable layers retain bias and use PyTorch's default initialization.
- **Data loader:** batches are shuffled each epoch with a dedicated random generator; the last incomplete batch is retained. The full dataset gives 469 batches per epoch, ending with 96 images; E0 gives 47 batches, ending with 112 images. The default is `num_workers=0`.
- **Logging and checkpoints:** epoch metrics are weighted by sample count. Each epoch saves the same 64 display-noise inputs as an image grid, G/D weights, optimizer states, and random-generator states. An additional grid is saved before training. Display-noise generation uses seed 2026 independently of the training random stream.

For custom code, import `make_train_loader` from `mnist_data`, `Generator` and `Discriminator` from `models`, and `run_experiment` from `train`. Use `make_train_loader(debug=True)` for E0's fixed subset. Detailed function-by-function explanations are in the [code structure guide](代码结构说明.md).

## Output files

| Location | Contents |
|---|---|
| `data/MNIST/raw/` | Original MNIST archives and extracted files. |
| `data/processed/` | `train_images.pt` and `debug_indices.pt`. |
| `outputs/data/` | `data_summary.json`, `label_counts.csv`, `label_distribution.png`, and `training_samples.png`. |
| `outputs/E*/seed_42/` | `config.json`, `metrics.csv`, `summary.json`, `fixed_noise.pt`, and `training_curves.png`. |
| `outputs/E*/seed_42/samples/` | Fixed-noise image grids: epoch 0 through 2 for E0, or epoch 0 through 20 for E1–E3. |
| `outputs/E*/seed_42/checkpoints/` | One checkpoint per completed epoch, including G/D and optimizer states. |
| `outputs/E1/seed_42/evaluation/` (also E2 and E3) | The 200-image noise and image tensors, `first_64.png`, 30 inference timing records, and the evaluation summary. |
| `outputs/comparison/seed_42/` | Combined CSV/JSON summaries, final samples, training curves, the 5×4 baseline progress grid, annotation sheets, labels, and class counts. |
| `outputs/report/seed_42/` | Figures rendered by the report's checkpoint and failure-example demonstrations. |

To run the project on another machine, copy the Python modules, notebooks, `requirements.txt`, and the required data/results folders together. Reading the saved report needs only the notebook; rerunning it needs the modules and supplied `outputs/` artifacts. Training needs processed data, and the debugging notebook also needs raw MNIST. Missing data can be recreated with `prepare_data.py`.

Default paths are resolved from the module directory. Historical absolute paths in archived JSON files record the original run and do not select the interpreter or data location on a new machine. Windows-specific DLL setup uses the current environment's `Library/bin` only when that directory exists.

## Recorded results and interpretation

See the [combined metrics](outputs/comparison/seed_42/comparison.csv), [final sample comparison](outputs/comparison/seed_42/final_samples.png), [training curves](outputs/comparison/seed_42/training_comparison.png), [baseline progress](outputs/comparison/seed_42/baseline_progress.png), and [class-count plot](outputs/comparison/seed_42/class_distribution.png).

| Run | Human recognizability | Class coverage | Unreadable / 200 | Training loop (s) | Inference per 1,000 images (ms, mean ± SD) |
|---|---:|---:|---:|---:|---:|
| E1 | 89.5% | 10/10 | 21 | 335.65 | 6.222 ± 0.321 |
| E2 | 84.0% | 10/10 | 32 | 102.36 | 5.319 ± 0.145 |
| E3 | 82.0% | 10/10 | 36 | 48.32 | 4.583 ± 0.239 |

E1 has the highest recognizability in this run; E2 and E3 are lower by 5.5 and 7.5 percentage points. These are descriptive results from one training seed, one annotator, and 200 images per model. All ten classes appear, but the generated class distributions are uneven. Class coverage alone does not establish balanced generation or rule out partial mode collapse.

Training time includes the epoch loop, sampling, checkpoint saving, and metric writes. It excludes data/model setup, the epoch-0 preview, and final plotting/summary writing. Epoch times varied substantially, including 2.45–18.24 seconds for E2. The recorded total-time differences cannot be attributed directly to learning rate or BatchNorm; quality comparisons use the same epoch budget.

Inference measurements cover only generator forward passes in evaluation mode with gradients disabled. Inputs are allocated on the device before timing, and CUDA is synchronized at timing boundaries. Loading, noise generation, transfers, plotting, and disk writes are excluded. The standard deviation describes 30 timing repetitions. Batch time divided by 1,000 is an amortized per-image time, not single-request latency.

## Maintenance

Maintain only `debug.ipynb` for debugging. Use `GAN_report.ipynb` for the English report and keep its display logic in `report_utils.py`. Put reusable implementation in Python modules and update the code structure guide when the implementation changes. Keep this README focused on runnable entry points, module relationships, and result locations.
