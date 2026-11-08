"""Fully visible sigmoid belief network (task 1.1).

For a fixed pixel order the model factorises

    p(x) = prod_i p(x_i | x_<i)

and every factor is its own logistic regression on the pixels that come before
i:

    p(x_i = 1 | x_<i) = sigmoid( sum_{j<i} W[i, j] * x_j + b[i] )

So W is 784x784 and only its strictly lower triangle is used (pixel 0 has no
inputs and is just b[0]). Training minimises the sum of the 784 binary cross
entropies, sampling goes pixel by pixel and only uses the already generated
prefix.

I also keep a masked linear layer version in `logits()`, which lets one forward
pass compute all 784 logits at once during training. That is what makes the
training loop fast; sampling is still sequential (784 steps) because each pixel
needs the one before it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FVSBN(nn.Module):
    def __init__(self, dim=784):
        super().__init__()
        self.dim = dim
        # one weight row per pixel; only the lower triangle survives
        self.W = nn.Parameter(torch.zeros(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.register_buffer('mask', torch.tril(torch.ones(dim, dim), diagonal=-1))

    def logits(self, x):
        """All 784 conditional logits for a batch, shape (N, 784)."""
        return x @ (self.W * self.mask).t() + self.b

    def nll(self, x):
        """Summed negative log likelihood per image, shape (N,)."""
        return F.binary_cross_entropy_with_logits(self.logits(x), x, reduction='none').sum(1)

    @torch.no_grad()
    def sample(self, n, device=None):
        """Ancestral sampling: fill pixel 0, then 1, ... and use the prefix."""
        device = device or self.b.device
        out = torch.zeros(n, self.dim, device=device)
        for i in range(self.dim):
            # only row i of the weight matrix matters for pixel i
            w = self.W[i] * self.mask[i]
            p = torch.sigmoid(out @ w + self.b[i])
            out[:, i] = (torch.rand(n, device=device) < p).float()
        return out
