# CS5494 Assignment 1 - Task 1.2 (NADE)

## Start here

Open `CS5494_Assignment1_Task2_NADE.ipynb` with the `CUHK` kernel.

1. Run the notebook first with `QUICK_RUN=True`.
2. For the final experiment, set `QUICK_RUN=False`, restart the kernel and run all cells.
3. Set `RUN_ABLATION=True` only after the main run succeeds.

## Files copied from Task 1.1

- `data.py`: required shared MNIST loader and threshold-0.5 binarization.
- `requirements.txt`: Task 1.1 reproducibility environment; the existing CUHK environment already contains the needed packages.
- `reference/task1_HANDOFF.md`: integration requirements from Task 1.1.
- `reference/task1_README.md`: Task 1.1 data contract and run notes.
- `reference/report_skeleton.md`: section numbering and shared comparison table.
- `reference/section_1_1_fvsbn.md`: reference style for the Task 1.2 report section.
- `reference/task1_results_summary.csv`: FVSBN numbers for comparison.
- `reference/task1_analysis.json`: FVSBN baseline and analysis values.

## Generated output

The notebook writes model checkpoints and metrics to `build/runs/`, figures to
`build/figs/`, and the final comparison-table values to
`build/task2_handoff_values.md`.

Do not report values from the quick run.
