"""Create and execute demo_data.ipynb, the notebook the teammates run first."""

import json
import os

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'build')
KERNEL_DIR = os.path.join(BUILD, 'jupyter', 'kernels', 'pytorch')
NB_PATH = os.path.join(HERE, 'demo_data.ipynb')

PY = r'C:\Users\Lenovo\.conda\envs\pytorch\python.exe'


def make_kernelspec():
    os.makedirs(KERNEL_DIR, exist_ok=True)
    spec = {'argv': [PY, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'pytorch (cs5494)', 'language': 'python'}
    with open(os.path.join(KERNEL_DIR, 'kernel.json'), 'w') as f:
        json.dump(spec, f)


def nb(cells):
    out = nbformat.v4.new_notebook()
    out.cells = cells
    out.metadata['kernelspec'] = {'name': 'pytorch', 'display_name': 'pytorch (cs5494)',
                                  'language': 'python'}
    return out


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(src):
    return nbformat.v4.new_code_cell(src)


CELLS = [
    md('# Shared data module\n'
       'This notebook is the short version of `data.py`, so we all preprocess MNIST the '
       'same way. Run it from the assignment folder (`python -m jupyter nbconvert --execute` '
       'is not needed, just open it). It needs `torch`, `torchvision` and `matplotlib`; '
       'the MNIST files are downloaded once into `build/data`.'),
    code('import data as D\n'
         'import torch\n'
         '\n'
         'D.summary()'),
    md('The arrays are float32 of shape (N, 784) and the pixel index is row-major '
       '(`p = row * 28 + col`). Task 1 and task 2 use `load_binarized`, task 3 uses '
       '`load_gray` because the GAN should see the original grayscale images.'),
    code('xb, yb = D.load_binarized("train")\n'
         'xg, _ = D.load_gray("train")\n'
         'xt, yt = D.load_binarized("test")\n'
         'print(xb.shape, xt.shape, xg.shape)\n'
         'print("unique values in xb:", torch.unique(torch.from_numpy(xb)).tolist())'),
    md('A quick look at twelve binarised training images. Binarisation is a fixed '
       'threshold at 0.5, so the same digit always gives the same 0/1 image.'),
    code('D.show_grid(xb[:12], 2, 6, path="build/demo_grid.png", title="binarised training images")'),
    md('The threshold is a design choice, not a law. Raising it removes ink (and part of '
       'the stroke), lowering it keeps shadows and background noise. The numbers below are '
       'the fraction of pixels that are 1:'),
    code('for thr in (0.3, 0.5, 0.7):\n'
         '    x, _ = D.load_binarized("train", threshold=thr)\n'
         '    print("threshold", thr, "mean pixel", round(float(x.mean()), 4))'),
    md('One training batch, the way the autoregressive models consume it:'),
    code('batch = torch.from_numpy(xb[:128])\n'
         'print(batch.shape, batch.dtype)\n'
         'print("pixels set in this batch:", int(batch.sum().item()))'),
    md('The order of the pixels matters for the autoregressive models, so `data.py` also '
       'has `pixel_order("row")`, `pixel_order("col")`, `apply_order` and `undo_order`. '
       'For the shared comparison table each task should log: preprocessing, learning rate, '
       'batch size, epochs, optimiser, seed, train time, sampling time for 100 images, and '
       'test bits/dim. The FVSBN numbers are in `build/results_summary.csv`.'),
]


def main():
    make_kernelspec()
    runtime = os.path.join(BUILD, 'jupyter', 'runtime')
    os.makedirs(runtime, exist_ok=True)
    os.environ['JUPYTER_PATH'] = os.path.join(BUILD, 'jupyter')
    os.environ['JUPYTER_RUNTIME_DIR'] = runtime
    os.environ['IPYTHONDIR'] = os.path.join(BUILD, 'ipython')

    notebook = nb(CELLS)
    client = NotebookClient(notebook, timeout=600, kernel_name='pytorch',
                            resources={'metadata': {'path': HERE}})
    client.execute()
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    errs = [c for c in notebook.cells if c.cell_type == 'code'
            and any(o.output_type == 'error' for o in c.get('outputs', []))]
    print('executed cells', sum(1 for c in notebook.cells if c.cell_type == 'code'),
          'errors', len(errs))


if __name__ == '__main__':
    main()
