"""All FVSBN runs used in the report (task 1.1).

Each run is saved to build/runs/<name>.json (numbers) and .npz (samples), so
make_figures.py can be re-run without training again.

Runs:
  main_60k_ep50      the model I report as the final FVSBN
  size_*             training set size ablation, 30 epochs, same everything else
  thr_*              binarisation threshold ablation, 20k images, 30 epochs
  order_*            pixel visiting order ablation, 20k images, 30 epochs
"""

import os
import time

import numpy as np

import data as D
from train_fvsbn import get_device, save_run, train_model

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS_MAIN = 50
EPOCHS_ABL = 30
BATCH = 128
LR = 1e-3
SEED = 0


def run(name, n_train, epochs, threshold=0.5, order='row', seed=SEED, verbose=True):
    t0 = time.time()
    perm = D.pixel_order(order)
    x_val, _ = D.load_binarized('test', threshold=threshold)
    x_tr, y_tr = D.load_binarized('train', threshold=threshold)
    x_tr, y_tr = D.take_subset(x_tr, y_tr, n_train, seed=seed)
    x_tr = D.apply_order(x_tr, perm)
    x_val = D.apply_order(x_val, perm)
    res = train_model(x_tr, x_val, epochs=epochs, batch=BATCH, lr=LR, seed=seed,
                      device=get_device(), verbose=verbose)
    cfg = {'n_train': int(len(x_tr)), 'epochs': epochs, 'batch': BATCH, 'lr': LR,
           'threshold': threshold, 'order': order, 'seed': seed}
    payload = save_run(name, res, cfg, perm=perm)
    if verbose:
        print('%-18s n=%5d ep=%2d thr=%.1f %s  bpd %.3f  train %.1fs  sample100 %.3fs'
              % (name, len(x_tr), epochs, threshold, order, payload['final_bpd'],
                 payload['train_time'], payload['sample_time_100']))
    return payload


def main():
    print('device', get_device())
    t0 = time.time()

    print('--- main run')
    run('main_60k_ep50', 60000, EPOCHS_MAIN)

    print('--- training size ablation (30 epochs)')
    for n in (1000, 5000, 20000, 60000):
        run('size_%d' % n, n, EPOCHS_ABL, verbose=False)
        print('done size', n)

    print('--- threshold ablation (20k, 30 epochs)')
    for thr in (0.3, 0.5, 0.7):
        run('thr_%s' % str(thr).replace('.', ''), 20000, EPOCHS_ABL, threshold=thr,
            verbose=False)
        print('done thr', thr)

    print('--- pixel order ablation (20k, 30 epochs)')
    for order in ('row', 'col'):
        run('order_%s' % order, 20000, EPOCHS_ABL, order=order, verbose=False)
        print('done order', order)

    print('all runs finished in %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
