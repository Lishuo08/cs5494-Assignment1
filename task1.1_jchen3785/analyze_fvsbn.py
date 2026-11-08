"""Extra numbers for the FVSBN section: baseline, pixel accuracy, predictive maps.

The trained model of the main run is reloaded from build/runs/main_60k_ep50.pt,
so nothing is retrained here.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import data as D
from fvsbn import FVSBN

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'build', 'runs')
FIGS = os.path.join(HERE, 'build', 'figs')
OUT = os.path.join(HERE, 'build', 'analysis.json')


def load_model(name='main_60k_ep50'):
    model = FVSBN()
    model.load_state_dict(torch.load(os.path.join(RUNS, name + '.pt')))
    model.eval()
    return model


def unconditional_bpd(x_train, x_test):
    """Baseline: one Bernoulli per pixel, fitted on the training marginals."""
    p = np.clip(x_train.mean(0), 1e-6, 1 - 1e-6)
    nll = -(x_test * np.log(p) + (1 - x_test) * np.log(1 - p)).sum(1).mean()
    return float(nll), float(nll / 784 / np.log(2))


def ink_stats():
    out = {}
    for thr in (0.3, 0.5, 0.7):
        x, _ = D.load_binarized('test', threshold=thr)
        out[str(thr)] = round(float(x.mean()), 4)
    return out


def pixel_accuracy(model, x_test, batch=2000):
    acc = []
    with torch.no_grad():
        for i in range(0, len(x_test), batch):
            xb = torch.from_numpy(x_test[i:i+batch]).float()
            pred = (model.logits(xb) > 0).float()
            acc.append((pred == xb).float().mean(0).numpy())
    return np.mean(np.stack(acc), 0)


def predictive_maps(model, x_test, idx, path):
    with torch.no_grad():
        xb = torch.from_numpy(x_test[idx]).float()
        p = torch.sigmoid(model.logits(xb)).numpy()
    n = len(idx)
    fig, axes = plt.subplots(n, 3, figsize=(4.6, 1.55 * n))
    for r in range(n):
        for c, (img, title) in enumerate(zip(
                [x_test[idx[r]], p[r], (p[r] > 0.5).astype(np.float32)],
                ['test image', 'p(x_i = 1 | x_<i)', 'argmax of p'])):
            ax = axes[r, c]
            ax.imshow(img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def accuracy_map(acc, path):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    im = ax.imshow(acc.reshape(28, 28), cmap='viridis', vmin=0.8, vmax=1.0)
    ax.set_title('per-pixel accuracy (teacher forcing)', fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    x_train, _ = D.load_binarized('train', threshold=0.5)
    x_test, _ = D.load_binarized('test', threshold=0.5)
    model = load_model()

    base_nll, base_bpd = unconditional_bpd(x_train, x_test)
    acc = pixel_accuracy(model, x_test)
    with torch.no_grad():
        xb = torch.from_numpy(x_test).float()
        model_nll = model.nll(xb).mean().item()

    idx = [0, 3, 7, 13]
    predictive_maps(model, x_test, idx, os.path.join(FIGS, 'fig_predmap.png'))
    accuracy_map(acc, os.path.join(FIGS, 'fig_pixel_acc.png'))

    rows = acc.reshape(28, 28).mean(1)
    res = {
        'test_nll_main': round(model_nll, 2),
        'test_bpd_main': round(model_nll / 784 / np.log(2), 4),
        'unconditional_nll': round(base_nll, 2),
        'unconditional_bpd': round(base_bpd, 4),
        'pixel_accuracy_mean': round(float(acc.mean()), 4),
        'pixel_accuracy_first_row': round(float(rows[0]), 4),
        'pixel_accuracy_last_row': round(float(rows[-1]), 4),
        'ink_fraction_by_threshold': ink_stats(),
    }
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
