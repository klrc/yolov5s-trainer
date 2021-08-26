import torch.nn as nn
import torch
import numpy as np


def prune(model, compress_rate=0.1, device='cuda', test_mode=False):
    dist = nn.PairwiseDistance(p=2)
    masks = {}
    print('pruning model...')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if hasattr(m, 'no_fpgm'):
                print('skip layer', name, type(m))
                continue
            N = m.weight.shape[0]
            filters = m.weight.data.view(N, -1)
            distances = torch.zeros(N, N, device=device, requires_grad=False)
            __cache = {}
            for i, fi in enumerate(filters):
                print(f'\rpruning layer {name} [{i}/{N}]             ', end='')
                for j, fj in enumerate(filters):
                    # check if filter is already pruned
                    # set (i, i) to inf, set (i, j)(j!=i) to 0
                    if fi.abs().sum() == 0:
                        distances[i][i] += float('inf')
                        continue
                    if fj.abs().sum() == 0:
                        distances[j][j] += float('inf')
                        continue
                    # calculate distance
                    table_index = (i, j) if i < j else (j, i)
                    if table_index in __cache:
                        distances[i][j] += __cache[table_index]
                    else:
                        # write into cache
                        distance = dist(fi.view(1, -1), fj.view(1, -1))[0]
                        __cache[table_index] = distance
                        distances[i][j] += distance
            # get filter(s) with minimum sum-distance
            distances = distances.sum(dim=1)
            to_prune = distances.argsort(descending=True) < N*compress_rate
            # add previous mask
            to_prune = torch.logical_or(to_prune, filters.abs().sum(dim=1) == 0)
            mask = torch.masked_fill(torch.ones(N, device=device, requires_grad=False), to_prune, 0)
            masks[name] = mask.unsqueeze(1)
            if test_mode:
                break
    print('FPGM finished')
    return masks


def cut_model(model, mask):
    for name, m in model.named_modules():
        if name in mask:
            N = m.weight.shape[0]
            a = m.weight.data.view(N, -1)
            b = a * mask[name]
            m.weight.data = b.view(m.weight.data.shape)


def cut_grad(model, mask):
    for name, m in model.named_modules():
        if name in mask:
            N = m.weight.shape[0]
            a = m.weight.grad.data.view(N, -1)
            b = a * mask[name]
            m.weight.grad.data = b.view(m.weight.grad.data.shape)


def if_zero(model, mask):
    nz = 0
    total = 0
    print('check overall compress rate...', end='')
    for name, m in model.named_modules():
        if name in mask:
            a = m.weight.data.view(-1)
            b = a.cpu().numpy()
            nz += np.count_nonzero(b)
            total += len(b)
            # print("layer: %d, number of nonzero weight is %d, zero is %d" % (
            #     index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))
    print(f'{nz/total:.2%}')
    return nz/total


class Counter():
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.max = 0
    
    def step(self, x):
        if x >= self.max:
            self.max = x
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                return True

