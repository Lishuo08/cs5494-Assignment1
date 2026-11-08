"""Training and sampling for the FVSBN (task 1.1).

The same function is used by run_experiments.py, so the numbers in the report
all come from one code path. Results are written as JSON plus a small npz with
the intermediate samples, which make_figures.py then turns into figures.
"""

import json
import os
import time

import numpy as np
import torch

import data as D
from fvsbn import FVSBN

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'build')
RUNS = os.path.join(BUILD, 'runs')
FIGS = os.path.join(BUILD, 'figs')


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def val_nll(model, x, batch=2000, device='cpu'):
    """Mean negative log likelihood per image (summed over the 784 pixels)."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i+batch]).to(device)
            total += model.nll(xb).sum().item()
            count += len(xb)
    return total / count


def time_sampling(model, n=100, device='cpu', repeats=5):
    """Median seconds needed to draw n images, with CUDA synchronised."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    model.sample(n, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        model.sample(n, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    return float(np.median(times))


def train_model(x_train, x_val, epochs=50, batch=128, lr=1e-3, seed=0,
                stage_epochs=(1, 5, 10, 25, 50), stage_samples=4, device=None,
                verbose=True):
    """Train one FVSBN and return the history, stage samples and timings."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or get_device()
    model = FVSBN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_all = torch.from_numpy(np.ascontiguousarray(x_train)).float()
    n = len(x_all)

    history = {'epoch': [], 'train_nll': [], 'val_nll': [], 'val_bpd': []}
    stages = {}
    t_epochs = []

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n)
        total, count = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb = x_all[idx].to(device)
            loss = model.nll(xb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        train_nll = total / count
        v_nll = val_nll(model, x_val, device=device)
        history['epoch'].append(ep)
        history['train_nll'].append(train_nll)
        history['val_nll'].append(v_nll)
        history['val_bpd'].append(v_nll / 784.0 / np.log(2))
        if ep in stage_epochs:
            stages[ep] = model.sample(stage_samples, device=device).cpu().numpy()
        t_epochs.append(time.time() - t_ep)
        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print('ep %d  train nll %.1f  val nll %.1f  val bpd %.3f  %.1fs'
                  % (ep, train_nll, v_nll, history['val_bpd'][-1], t_epochs[-1]))

    train_time = float(sum(t_epochs))
    sample_time = time_sampling(model, 100, device=device)
    final = model.sample(40, device=device).cpu().numpy()

    return {
        'model': model,
        'history': history,
        'stages': stages,
        'final': final,
        'train_time': train_time,
        'sample_time_100': sample_time,
        'device': str(device),
        'params': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def save_run(name, result, config, perm=None):
    """Write one run to build/runs/<name>.json and .npz."""
    os.makedirs(RUNS, exist_ok=True)
    payload = {
        'name': name,
        'config': config,
        'optimizer': 'Adam',
        'torch': torch.__version__,
        'numpy': np.__version__,
        'gpu': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
        'history': result['history'],
        'train_time': result['train_time'],
        'sample_time_100': result['sample_time_100'],
        'device': result['device'],
        'params': result['params'],
        'final_bpd': result['history']['val_bpd'][-1],
    }
    with open(os.path.join(RUNS, name + '.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    np.savez_compressed(os.path.join(RUNS, name + '.npz'),
                        final=result['final'],
                        **{'stage_%d' % k: v for k, v in result['stages'].items()},
                        perm=np.arange(784) if perm is None else perm)
    torch.save(result['model'].state_dict(), os.path.join(RUNS, name + '.pt'))
    return payload


def main():
    device = get_device()
    print('device', device)
    x_val, _ = D.load_binarized('test', threshold=0.5)
    x_train, _ = D.load_binarized('train', threshold=0.5)
    res = train_model(x_train, x_val, epochs=50, device=device)
    payload = save_run('main_60k_ep50', res,
                       {'n_train': 60000, 'epochs': 50, 'batch': 128, 'lr': 1e-3,
                        'threshold': 0.5, 'order': 'row', 'seed': 0})
    print('final val nll %.1f  bpd %.3f  train %.1fs  sample100 %.3fs'
          % (payload['history']['val_nll'][-1], payload['final_bpd'],
             payload['train_time'], payload['sample_time_100']))


if __name__ == '__main__':
    main()
