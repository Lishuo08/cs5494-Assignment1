"""Shared data utilities for the CS5494 assignment.

Task 1 (FVSBN) and task 2 (neural autoregressive) are trained on the same
binarised MNIST, so the preprocessing is defined once, here. Task 3 (GAN) uses
the grayscale version through load_gray().

Arrays are float32 with shape (N, 784). Pixel index p = row * 28 + col, i.e. the
images are flattened in row-major order. Binarised pixels are 0.0 / 1.0.
"""

import os

import numpy as np
from torchvision.datasets import MNIST

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, 'build', 'data')

TRAIN_FLAG = {'train': True, 'test': False}


def load_gray(split='train'):
    """Load one split as (x, y). x is float32 in [0, 1] with shape (N, 784)."""
    ds = MNIST(root=DATA_ROOT, train=TRAIN_FLAG[split], download=True)
    x = ds.data.numpy().astype(np.float32).reshape(len(ds), -1) / 255.0
    y = ds.targets.numpy()
    return x, y


def load_raw(split='train'):
    """Same as load_gray but keeps the 0..255 integers."""
    x, y = load_gray(split)
    return np.rint(x * 255.0).astype(np.uint8), y


def binarize(x, threshold=0.5, mode='threshold', seed=0):
    """Map pixels in [0, 1] to {0, 1}.

    mode='threshold' is the default: a pixel becomes 1 when it is brighter than
    `threshold`. The result is deterministic, so training samples stay fixed.
    mode='bernoulli' samples each pixel with probability equal to its intensity;
    it keeps more of the grayscale information but the samples change between
    runs. I keep it only for the comparison in the report.
    """
    if mode == 'threshold':
        return (x > threshold).astype(np.float32)
    if mode == 'bernoulli':
        rng = np.random.RandomState(seed)
        return (rng.rand(*x.shape) < x).astype(np.float32)
    raise ValueError('unknown mode: ' + str(mode))


def load_binarized(split='train', threshold=0.5, mode='threshold', seed=0):
    """Load one split and binarise it with the shared rule."""
    x, y = load_gray(split)
    return binarize(x, threshold, mode, seed), y


def take_subset(x, y, n, seed=0):
    """Random subset with a fixed seed. n=None keeps everything."""
    if n is None or n >= len(x):
        return x, y
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))[:n]
    return x[idx], y[idx]


def pixel_order(order='row'):
    """Permutation of the 784 pixel indices.

    FVSBN is not invariant to the order in which the pixels are generated, so
    the ablations swap between row-major and column-major visiting order.
    """
    if order == 'row':
        return np.arange(784)
    if order == 'col':
        return np.arange(784).reshape(28, 28).T.reshape(-1)
    raise ValueError('unknown order: ' + str(order))


def apply_order(x, perm):
    """Reorder the pixels of every image. perm[i] is the pixel shown at step i."""
    return x[:, perm]


def undo_order(x, perm):
    """Inverse of apply_order, used to plot the generated images."""
    out = np.empty_like(x)
    out[:, perm] = x
    return out


def to_tensor(x):
    import torch
    return torch.from_numpy(np.ascontiguousarray(x))


def show_grid(imgs, rows, cols, path=None, titles=None, figsize=None, title=None):
    """Save a grid of images as one figure.

    imgs: array of shape (N, 784) or (N, 28, 28), values in {0, 1} or [0, 1].
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    imgs = np.asarray(imgs)
    if imgs.ndim == 2:
        imgs = imgs.reshape(-1, 28, 28)

    if figsize is None:
        figsize = (1.25 * cols, 1.4 * rows + (0.4 if titles else 0))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis('off')
            idx = r * cols + c
            if idx < len(imgs):
                ax.imshow(imgs[idx], cmap='gray', vmin=0, vmax=1)
            if titles is not None and c == 0:
                ax.set_ylabel(titles[r], fontsize=9)
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        return path
    return fig


def summary():
    """Print the shape and value range of both versions, for a quick check."""
    for split in ('train', 'test'):
        xg, y = load_gray(split)
        xb, _ = load_binarized(split)
        print(split, 'gray', xg.shape, round(float(xg.min()), 3), round(float(xg.max()), 3))
        print(split, 'bin ', xb.shape, int(xb.min()), int(xb.max()),
              'mean pixel', round(float(xb.mean()), 4), 'labels', y.shape)


if __name__ == '__main__':
    summary()
