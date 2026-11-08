"""Collect every run into one CSV and one Markdown table.

The team report has a common comparison table; the CSV here is the FVSBN part
of it, with the same column names the other tasks should use.
"""

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'build', 'runs')
OUT_CSV = os.path.join(HERE, 'build', 'results_summary.csv')
OUT_MD = os.path.join(HERE, 'build', 'results_summary.md')

COLS = ['run', 'model', 'n_train', 'epochs', 'batch', 'lr', 'binarisation',
        'pixel_order', 'seed', 'params', 'train_time_s', 'sample_100_ms',
        'test_nll', 'test_bpd']


def rows():
    for path in sorted(glob.glob(os.path.join(RUNS, '*.json'))):
        with open(path) as f:
            d = json.load(f)
        c = d['config']
        yield {
            'run': d['name'],
            'model': 'FVSBN',
            'n_train': c['n_train'],
            'epochs': c['epochs'],
            'batch': c['batch'],
            'lr': c['lr'],
            'binarisation': 'threshold %.1f' % c['threshold'],
            'pixel_order': c['order'],
            'seed': c['seed'],
            'params': d['params'],
            'train_time_s': round(d['train_time'], 1),
            'sample_100_ms': round(1000 * d['sample_time_100'], 1),
            'test_nll': round(d['history']['val_nll'][-1], 2),
            'test_bpd': round(d['final_bpd'], 4),
        }


def main():
    data = list(rows())
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(data)

    head = '| ' + ' | '.join(COLS) + ' |'
    sep = '|' + '---|' * len(COLS)
    lines = [head, sep]
    for r in data:
        lines.append('| ' + ' | '.join(str(r[c]) for c in COLS) + ' |')
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
