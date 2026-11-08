# START HERE - CS5494 Assignment 1, task 1.1 (FVSBN)

Owner: **CHEN Jiaqi (jchen3785)**. Deadline for the merged report: **Sunday 8 November 2026**.
This folder is the complete handoff for task 1.1 plus the shared data loader.

## Read in this order (10 minutes)

| # | file | why |
| --- | --- | --- |
| 1 | `index.html` | one page listing every file, what it is, and how big it is - open it in a browser |
| 2 | `report/section_1_1_fvsbn.pdf` | my report section, 5 pages, ready to paste into the team report |
| 3 | `report/report_skeleton.md` | where the section goes, and the two `<TODO>` slots for tasks 1.2 / 1.3 |
| 4 | `HANDOFF.md` | what is done, the numbers, and the four things I still need from B and C |
| 5 | `README.md` | environment, run order, data contract, all ablation numbers |
| 6 | `message_to_teammates.txt` | a ready-to-paste message to send to B and C with the link and the numbering |

## Reproduce in 3 commands

```bash
pip install -r requirements.txt          # torch, torchvision, numpy, matplotlib, nbformat, nbclient
python run_experiments.py                # ~2-3 min on the GPU, ~10 min on CPU
python make_figures.py && python analyze_fvsbn.py && python summarize.py
```

`python check_package.py` prints a status table (files present, encodings, PDF page count,
zip integrity) if you want to confirm the folder arrived complete.

## What is in the box

| folder / file | content |
| --- | --- |
| `data.py` | shared MNIST loader + binarisation + pixel order (tasks 1.1, 1.2 and 1.3 must all use this) |
| `fvsbn.py`, `train_fvsbn.py` | the model and one training run |
| `run_experiments.py` | main run + 3 ablations (train size, threshold, pixel order), 13 runs |
| `make_figures.py`, `analyze_fvsbn.py`, `summarize.py` | the 7 figures, the baseline / pixel-accuracy numbers, `results_summary.csv` |
| `demo_data.ipynb` | executed tour of the data module, open it first |
| `report/` | section source (md), rendered PDF, team report skeleton |
| `build/figs/` | every figure used in the report, PNG, 300 dpi class |
| `build/results_summary.csv` | all 13 runs in one table (config + NLL + bpd + timings) |
| `HANDOFF.md`, `README.md`, `readme.txt`, `requirements.txt` | handoff notes and run instructions |
| `CS5494_task1.1_jchen3785.zip` | the same folder as a zip, for sending to the team |

## Key numbers (test split, binarised at 0.5)

| model | NLL / image | bits / dim | params | train time | 100 samples |
| --- | --- | --- | --- | --- | --- |
| unconditional Bernoulli | 205.85 | 0.3788 | - | - | - |
| FVSBN, 60k images, 50 epochs | 75.75 | **0.1394** | 615,440 | 30-50 s (GPU) | 0.06 s |

Ablations: 1k 0.4537 / 5k 0.2888 / 20k 0.1656 / 60k 0.1402 bits/dim;
threshold 0.3 (0.1604) vs 0.7 (0.1717) needs care because the data changes;
column-major order 0.1677 vs row-major 0.1656.

## What I still need from the other two members

1. Their sections as Markdown, same structure as 3.1 (model, setup, results, discussion, limitations).
2. One row each for the comparison table in `report/report_skeleton.md`.
3. Their entry-script names for section 6 of the merged report.
4. Confirmation that they used `data.load_binarized` (1.2) and `data.load_gray` (1.3) - if the
   preprocessing differs, the bits/dim values cannot be compared in one table.

Numbering is already fixed so the three sections merge without re-indexing: task 1 is
section 3 (3.1-3.5), task 2 becomes section 4 (4.1-4.5), task 3 becomes section 5
(5.1-5.5), then section 6 comparison, 7 discussion, 8 code. Details are in
`report/report_skeleton.md`; `message_to_teammates.txt` is the message to send them.

## 中文摘要

这份文件夹是我负责的 CS5494 作业 1.1（FVSBN）全部交付物，外加三个人共用的数据读取模块
（`data.py`）。阅读顺序：先开 `index.html` 看文件清单，再看 `report/section_1_1_fvsbn.pdf`
（我的 5 页报告节），然后 `report/report_skeleton.md`（合并报告框架，1.2/1.3 的位置留了 TODO），
`HANDOFF.md` 写了我还缺什么。复现只要三条命令（见上）。所有数字都可以用
`python run_experiments.py` 在 2-3 分钟内重新跑出来，结果逐位一致（0.1394 bpd）。
