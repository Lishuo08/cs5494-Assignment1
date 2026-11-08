# CS5494 Assignment 1 - report skeleton (integration draft)

Student IDs in the file name: `<id>_<id>_<id>.pdf`. Members, IDs and the code link
go on the first page.

Numbering convention: each task is one top-level section with the same five
subsections. Task 1 (FVSBN) is section 3 and its text is already written in
`section_1_1_fvsbn.md` (3.1 model, 3.2 setup, 3.3 results, 3.4 discussion,
3.5 limitations). Task 2 goes to section 4, task 3 to section 5, with the same
5-subsection pattern, so the merged report reads consistently.

## 1  Introduction

Generative models for images can be grouped by what they model directly. This
report compares three of them on MNIST: a fully visible sigmoid belief network
(FVSBN), a neural autoregressive model, and a generative adversarial network. The
first two describe the joint distribution of the pixels as a product of
conditionals and differ in how expressive each conditional is; the GAN drops the
likelihood completely and learns a generator through an adversarial game. We use
one binarisation rule and one training/test split for the likelihood models so
that the numbers can be compared, and report training time, sampling time and
sample quality for all three.

## 2  Shared experimental setup

- Data: MNIST, 60,000 training and 10,000 test images. Tasks 1 and 2 use the
  binarised version (`data.py`, threshold 0.5, 13.2% ink), task 3 uses the
  original grayscale images.
- Hardware: RTX 3070 Laptop GPU, PyTorch 2.5.1. Seeds are fixed; timings are
  wall-clock on the same machine and therefore comparable between the three
  sections.
- Metrics: bits per dimension (bpd = NLL / (784 ln 2)) on the test split for the
  likelihood models; for the GAN, bpd is not defined, so we report FID/inception
  score or a nearest-neighbour check instead (task 3 decides).
- Timing protocol: training time and the time to generate 100 images, measured
  with CUDA synchronised, one warm-up call, median of 5.

## 3  Task 1: fully visible sigmoid belief network (FVSBN)

Draft ready, complete text and five figures in `section_1_1_fvsbn.md`
(rendered: `section_1_1_fvsbn.pdf`). Subsections: 3.1 model, 3.2 experimental
setup, 3.3 results, 3.4 discussion, 3.5 limitations.

Key row for the comparison table: test bpd 0.1394, 615,440 parameters, 30-50 s
training, 0.06 s for 100 samples (median of 5).

## 4  Task 2: neural autoregressive model

<!-- TODO(task 1.2): same subsection pattern as section 3.
     4.1 model (hidden layer, masking / directed graph, objective)
     4.2 experimental setup (must be data.load_binarized, threshold 0.5, seed 0)
     4.3 results (test bpd, samples, stage grid, timings) with its own figures
     4.4 discussion against section 3.3
     4.5 limitations -->

## 5  Task 3: generative adversarial network

<!-- TODO(task 1.3): same subsection pattern.
     5.1 architecture and loss
     5.2 experimental setup (grayscale images via data.load_gray)
     5.3 results (samples, stage grids, training curves, timings)
     5.4 discussion: training stability, mode collapse, what bpd cannot say
     5.5 limitations -->

## 6  Comparison

<!-- TODO: fill the rows for 1.2 and 1.3; the FVSBN row is final. -->

| model | test bpd | params | train time | 100 samples | notes |
| --- | --- | --- | --- | --- | --- |
| FVSBN (section 3) | 0.1394 | 615,440 | 30-50 s | 0.06 s (median of 5) | linear conditionals, exposure bias |
| neural autoregressive (section 4) | TODO | TODO | TODO | TODO | |
| GAN (section 5) | n/a | TODO | TODO | TODO | likelihood not available; stability |

Points that belong to this section (keep them short, one paragraph each):

- Accuracy: FVSBN against the stronger autoregressive model, and what the gap
  says about the linear conditional (see the ablation table in 3.3).
- Cost: parameters, training time and sampling time. The autoregressive models
  pay for sampling with 784 sequential steps; the GAN pays in training
  instability instead.
- Sample quality: the FVSBN samples are recognisable but thick; the GAN should be
  sharper, so it is worth one sentence on why likelihood and perceptual quality
  do not have to agree.

## 7  Discussion

<!-- 3-5 sentences, not a summary of the sections: which model we would use for
which purpose, and what the experiment does not show. -->

## 8  Code and how to run it

All code is in the shared folder (link in the cover page). Reproduction order:

1. `pip install -r requirements.txt`, then `python data.py` - downloads MNIST
   into `build/data` and prints the shapes.
2. `python run_experiments.py` - all runs of task 1.1, about 3 minutes on the GPU.
3. `python make_figures.py`, `python analyze_fvsbn.py`, `python summarize.py` -
   figures and the numbers table.
4. Task 1.2 and 1.3 have their own entry scripts (see sections 4 and 5).

`demo_data.ipynb` walks through the shared preprocessing and is the first file to
open. `README.md` has the file list and the full data contract; `index.html` is
the one-page map of the folder.
