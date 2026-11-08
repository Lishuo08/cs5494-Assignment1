"""Turn the saved FVSBN runs into the figures used in the report."""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'build', 'runs')
FIGS = os.path.join(HERE, 'build', 'figs')
os.makedirs(FIGS, exist_ok=True)

STAGE_EPOCHS = [1, 5, 10, 25, 50]


def load(name):
    with open(os.path.join(RUNS, name + '.json')) as f:
        meta = json.load(f)
    npz = np.load(os.path.join(RUNS, name + '.npz'))
    return meta, npz


def loss_curve():
    meta, _ = load('main_60k_ep50')
    h = meta['history']
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax1.plot(h['epoch'], h['train_nll'], label='train (per image)', color='#2c7fb8')
    ax1.plot(h['epoch'], h['val_nll'], label='test (per image)', color='#d95f0e')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('negative log likelihood (nats / image)')
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(h['epoch'], h['val_bpd'], color='#d95f0e', ls=':', lw=1.2, label='test bits/dim')
    ax2.set_ylabel('bits / pixel (test)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.set_title('FVSBN training on 60k binarised MNIST images', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_loss.png'), dpi=180)
    plt.close(fig)


def stage_grid():
    _, npz = load('main_60k_ep50')
    rows = [npz['stage_%d' % ep] for ep in STAGE_EPOCHS]
    fig, axes = plt.subplots(len(rows), 4, figsize=(4.2, 5.4))
    for r, (ep, imgs) in enumerate(zip(STAGE_EPOCHS, rows)):
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(imgs[c].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel('epoch %d' % ep, fontsize=9)
    fig.suptitle('Samples drawn by the same model during training', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_stages.png'), dpi=180)
    plt.close(fig)


def final_samples():
    _, npz = load('main_60k_ep50')
    imgs = npz['final']
    fig, axes = plt.subplots(4, 10, figsize=(10, 4.3))
    for i in range(4):
        for j in range(10):
            ax = axes[i, j]
            ax.imshow(imgs[i*10 + j].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('40 images sampled from the trained FVSBN', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_samples.png'), dpi=180)
    plt.close(fig)


def ablation_bars():
    groups = [
        ('training set size', ['size_1000', 'size_5000', 'size_20000', 'size_60000'],
         ['1k', '5k', '20k', '60k']),
        ('binarisation threshold', ['thr_03', 'thr_05', 'thr_07'], ['0.3', '0.5', '0.7']),
        ('pixel order', ['order_row', 'order_col'], ['row-major', 'column-major']),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    all_vals = [load(n)[0]['final_bpd'] for _, names, _ in groups for n in names]
    top = max(all_vals) * 1.2
    for ax, (title, names, labels) in zip(axes, groups):
        vals = [load(n)[0]['final_bpd'] for n in names]
        bars = ax.bar(range(len(vals)), vals, color='#4c72b0')
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, top)
        ax.grid(axis='y', alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, '%.3f' % v, ha='center', va='bottom',
                    fontsize=8)
    axes[0].set_ylabel('test bits / pixel')
    fig.suptitle('Ablations of the FVSBN (30 epochs unless stated)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_ablation.png'), dpi=180)
    plt.close(fig)


def ablation_samples():
    groups = [
        ('training size', [('size_1000', '1k'), ('size_5000', '5k'),
                           ('size_20000', '20k'), ('size_60000', '60k')]),
        ('threshold', [('thr_03', '0.3'), ('thr_05', '0.5'), ('thr_07', '0.7')]),
        ('order', [('order_row', 'row'), ('order_col', 'col')]),
    ]
    rows = sum(len(g[1]) for g in groups)
    fig, axes = plt.subplots(rows, 6, figsize=(6.4, 1.05 * rows))
    r = 0
    for title, items in groups:
        for name, label in items:
            _, npz = load(name)
            imgs = npz['final'][:6]
            if 'perm' in npz.files:
                imgs = D.undo_order(imgs, npz['perm'])
            for c in range(6):
                ax = axes[r, c]
                ax.imshow(imgs[c].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[r, 0].set_ylabel('%s %s' % (title.split()[0], label), fontsize=8)
            r += 1
    fig.suptitle('Generated samples for each ablation setting', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_ablation_samples.png'), dpi=170)
    plt.close(fig)


def main():
    loss_curve()
    stage_grid()
    final_samples()
    ablation_bars()
    ablation_samples()
    print('figures written to', FIGS)
    for f in sorted(glob.glob(os.path.join(FIGS, '*.png'))):
        print(' ', os.path.basename(f), os.path.getsize(f))


if __name__ == '__main__':
    main()
