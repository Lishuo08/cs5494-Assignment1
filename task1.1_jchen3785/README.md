# CS5494 Assignment 1 - Task 1.1: FVSBN

Code, notebooks and figures for task 1.1 (fully visible sigmoid belief network) plus
the shared data module used by all three tasks.  Owner: CHEN Jiaqi (jchen3785).

**New to this folder?** Open `index.html` first (one page with every file and its
purpose), then `START_HERE.md` (reading order and the 3-command quickstart), then
`report/section_1_1_fvsbn.pdf`.

## Contents

1. [Files](#files)
2. [Environment](#environment)
3. [How to run](#how-to-run)
4. [Figures](#figures)
5. [Data contract](#data-contract)
6. [Numbers](#numbers)
7. [Notes](#notes)

## Files

| file | what it is |
| --- | --- |
| `START_HERE.md` | reading order, 3-command quickstart, 中文摘要 |
| `index.html` | one-page map of every file, open it in a browser |
| `HANDOFF.md` | what is done, the numbers, and what I still need from tasks 1.2 / 1.3 |
| `message_to_teammates.txt` | ready-to-paste message for the other two members |
| `check_package.py` | verifies that the folder / zip is complete |
| `make_index.py` | regenerates `index.html` |
| `data.py` | shared MNIST loader and the binarisation rule (`load_gray`, `load_binarized`, `binarize`, `pixel_order`, `take_subset`, `show_grid`) |
| `fvsbn.py` | the model: `logits(x)`, `nll(x)`, `sample(n)` |
| `train_fvsbn.py` | one training run (Adam, staged samples, timing, JSON/npz/pt output) |
| `run_experiments.py` | main run + the three ablations (training size, threshold, pixel order) |
| `make_figures.py` | loss curve, stage grid, sample grid, ablation bars |
| `analyze_fvsbn.py` | unconditional baseline, pixel accuracy, predictive maps |
| `summarize.py` | collects every run into `build/results_summary.csv` / `.md` |
| `build_demo_nb.py` | creates and executes `demo_data.ipynb` |
| `demo_data.ipynb` | short tour of the shared data module (already executed) |
| `report/section_1_1_fvsbn.md` | my draft of the report section for task 1.1 |
| `report/section_1_1_fvsbn.pdf` | the same section rendered with the figures |
| `report/report_skeleton.md` | outline of the team report and where this section goes |
| `readme.txt` | the same run instructions in plain text |
| `requirements.txt` | versions of the packages I used |
| `build/` | data, run outputs, figures - regenerated, not part of the source |

## Environment

Python 3.9, PyTorch 2.5.1 (CUDA 12.1), torchvision 0.20.1, matplotlib 3.7.2,
numpy 1.26.4.  The interpreter used here is
`C:\Users\Lenovo\.conda\envs\pytorch\python.exe`; a plain `python3` kernel may not
have torch installed.  A RTX 3070 Laptop GPU was used, but everything also runs
on CPU (slower, the model has 615,440 parameters).

## How to run

```bash
python run_experiments.py     # all runs, about 3 minutes on the GPU
python make_figures.py        # build/figs/*.png
python analyze_fvsbn.py       # baseline + predictive maps
python summarize.py           # build/results_summary.csv
```

MNIST is downloaded automatically into `build/data` the first time
(torchvision tries `yann.lecun.com` first; if that returns 404 it falls back to
the S3 mirror and it still works).

## Figures

| file | content |
| --- | --- |
| `build/figs/fig_loss.png` | train / test NLL per image and test bits per dimension |
| `build/figs/fig_stages.png` | 5 rows (epoch 1, 5, 10, 25, 50) x 4 samples from the same model |
| `build/figs/fig_samples.png` | 40 samples of the final model |
| `build/figs/fig_ablation.png` | bits/dim for the three ablation axes |
| `build/figs/fig_ablation_samples.png` | six samples for every ablation setting |
| `build/figs/fig_predmap.png` | test image, teacher-forced p(x_i = 1 given x_<i) and its argmax |
| `build/figs/fig_pixel_acc.png` | per-pixel teacher-forcing accuracy map |

## Data contract (the same for tasks 1.1, 1.2 and 1.3)

- Grayscale images are `float32` in [0, 1], flattened row-major to 784 values
  (`p = row * 28 + col`).
- Tasks 1.1 and 1.2 use `load_binarized(split, threshold=0.5)`: deterministic
  threshold, pixels become exactly 0.0 / 1.0, mean ink is 0.132 (train) and
  0.134 (test).
- Task 1.3 (GAN) uses `load_gray` - the original grayscale images.
- Training uses all 60,000 images, the test split is only read for the numbers
  in the report, seed is fixed at 0 (`take_subset` uses the same seed).
- 100 generated images are timed after one warm-up call; log the median of 5
  repetitions.

## Numbers (test split, binarised at 0.5, 10k images)

| model | NLL / image | bits / dim | pixel accuracy | train time | 100 samples |
| --- | --- | --- | --- | --- | --- |
| unconditional Bernoulli | 205.85 | 0.3788 | - | - | - |
| FVSBN, 60k / 50 epochs | 75.75 | 0.1394 | 0.9603 | 30-50 s (GPU) | 0.06 s (median of 5) |

Ablations (30 epochs unless stated):

| change | bits / dim |
| --- | --- |
| 1k training images | 0.4537 |
| 5k | 0.2888 |
| 20k | 0.1656 |
| 60k | 0.1402 |
| threshold 0.3 (20k) | 0.1604 |
| threshold 0.7 (20k) | 0.1717 |
| column-major order (20k) | 0.1677 |

The threshold rows change the data as well, so those numbers are only
comparable with each other, not with the threshold-0.5 rows.

## Notes

- Training is plain supervised learning on the binarised pixels (teacher
  forcing): the NLL is a sum of 784 binary cross entropies.
- Sampling is ancestral, one pixel at a time, 784 sequential steps; it is the
  slow part of the model and the reason the timing is reported separately.
- The full report is assembled by the team from `report/section_1_1_fvsbn.md`
  plus the equivalent sections of tasks 1.2 and 1.3.
