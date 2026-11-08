# Handoff - task 1.1 (FVSBN) and the shared data module

Read `index.html` (file map) or `START_HERE.md` (reading order + quickstart) first;
this page is the summary of what is done and what is still needed.

Everything for my part is in this folder. What is done, what B and C need from
me, and what I still need from them.

## Done

- `data.py` - the shared MNIST loader and binarisation rule (threshold 0.5 by
  default, `pixel_order` helpers, `show_grid`). Tasks 1.2 and 1.3 should import
  this file instead of writing their own loader, otherwise the samples are not
  comparable.
- `fvsbn.py`, `train_fvsbn.py`, `run_experiments.py` - the model, one training
  run, and the main run plus three ablations. Re-running the zip reproduces all
  numbers in about 2 minutes.
- `make_figures.py`, `analyze_fvsbn.py`, `summarize.py` - the five figures, the
  baseline/predictive-map numbers, and `build/results_summary.csv`.
- `demo_data.ipynb` - executed tour of the data module, run this first.
- `README.md` / `readme.txt` - environment, run order, data contract.
- `report/section_1_1_fvsbn.md` + `.pdf` - my report section, 5 pages with
  figures, ready to be pasted into the team report.
- `report/report_skeleton.md` - the frame for the merged report; the `<TODO>`
  markers show exactly where the other two sections and the comparison table go.

## Numbers to reuse in the shared table

| item | value |
| --- | --- |
| test bpd (main run) | 0.1394 |
| test NLL / image | 75.75 nats |
| unconditional baseline | 0.3788 bpd |
| pixel accuracy | 96.03% |
| parameters | 615,440 |
| training time | 30-50 s (50 epochs) |
| 100 samples | 0.06 s (median of 5) |
| hyperparameters | Adam, lr 1e-3, batch 128, seed 0 |

## What I need from B (task 1.2) and C (task 1.3)

1. Their report sections as Markdown, in the same order as section 3 (model, setup,
   results, discussion, limitations). Numbering is already fixed in
   `report/report_skeleton.md`: task 2 is section 4 (4.1-4.5) and task 3 is section
   5 (5.1-5.5), so the three sections read the same way. Figures go to
   `build/figs/` with `fig_<task>_<what>.png`.
2. One row each for the comparison table: model, test bpd (or n/a), params,
   training time, time for 100 samples, one-line note. If the GAN needs a
   different quality metric, say which one and define it.
3. Their entry script names so I can update section 6 of the report.
4. The exact preprocessing they used - it must be `data.load_binarized` for 1.2
   and `data.load_gray` for 1.3. If either of them used a different threshold or
   image size, I have to know before I merge, because the bpd values then are not
   comparable.

## Open question for the whole team

The report is due Sunday 8 November and the file must be named
`<id>_<id>_<id>.pdf`. We also need one shared-folder link that the marker can
open, and the run-time numbers for all three tasks; I can assemble it once the
two sections above land.
