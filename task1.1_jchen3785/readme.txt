CS5494 Assignment 1 - Task 1.1 (FVSBN) - run instructions
=========================================================

Author: CHEN Jiaqi (jchen3785)

1. Environment
   Python 3.9 with the packages in requirements.txt:
       pip install -r requirements.txt
   Tested with torch 2.5.1 (CUDA 12.1), torchvision 0.20.1, numpy 1.26.4,
   matplotlib 3.7.2 on a laptop RTX 3070 (CUDA) and on CPU.
   A GPU is not required; on CPU the main run takes roughly 10 minutes
   instead of about 30 seconds.

2. Run everything (from this folder)
       python run_experiments.py     # main run + 3 ablations, ~3 min on the GPU
       python make_figures.py        # writes build/figs/*.png
       python analyze_fvsbn.py       # baseline / pixel accuracy / predictive maps
       python summarize.py           # writes build/results_summary.csv and .md

   MNIST is downloaded into build/data automatically the first time.
   All numbers in report/section_1_1_fvsbn.md come from these four scripts,
   with seed 0 and the default arguments; no manual step is needed.

3. Notebook
       demo_data.ipynb is already executed; open it to see the shared data
       module in 8 short cells (shapes, binarisation, threshold effect,
       one batch, pixel orders). It can also be rebuilt with
       python build_demo_nb.py.

4. Outputs
       build/runs/<name>.json   per-run config, history, NLL, bits/dim, timings
       build/runs/<name>.npz    sampled images used in the figures
       build/runs/<name>.pt     model weights
       build/results_summary.csv  every run in one table
       build/figs/*.png         figures used in the report section

5. Report section
       report/section_1_1_fvsbn.md   source text
       report/section_1_1_fvsbn.pdf  rendered version with the figures
       report/report_skeleton.md     where this section goes in the team report
