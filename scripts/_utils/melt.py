# 指定一个稀疏（部分nn.Conv2d的filter置0）的神经网络，
# 输入：model，
# 输出：model

import torch
from torch._C import device
from torch.autograd.grad_mode import F
import torch.nn as nn
from torch.nn import modules
from torch.nn.modules import activation, module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Identity
from torchvision import models
import numpy as np
import random
import torchvision
import types

from torchvision.models.resnet import resnet18


def _locate_zeroized(conv, zero_inputs):
    num_filters = conv.weight.size(0)
    zero_mask = conv.weight.data[:, ~zero_inputs, ...]
    zero_mask = zero_mask.view(num_filters, -1).abs().sum(dim=1) == 0
    return zero_mask


def _fix_all_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs)
    # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def melt(model: nn.Module, verbose=True):

    # _fix_all_seed(0)

    modules = {}
    target_layers = []

    def print_if_verbose(msg):
        if verbose:
            print(msg)

    def replace_layer(module_name, layer_name, layer):
        print_if_verbose(f'replace layer {module_name}/{layer_name} -> ({type(layer)})')
        modules[module_name].add_module(layer_name, layer)

    def get_layer_name(layer):
        module_name = layer._meta['module_name']
        layer_name = layer._meta['layer_name']
        return f'{module_name}/{layer_name} ({type(layer)})'

    # scan model and register necessary information
    for module_name, m in model.named_modules():
        for layer_name, layer in m.named_children():
            # pre-compile dict for melting
            modules[module_name] = m
            layer._meta = {
                'module_name': module_name,
                'layer_name': layer_name,
                'is_relu': type(layer) == nn.ReLU,
                'is_linear': type(layer) == nn.Linear,
                'is_conv': type(layer) == nn.Conv2d,
                'is_bn': type(layer) == nn.BatchNorm2d,
                'is_sensitive': hasattr(layer, 'in_channels') or hasattr(layer, 'num_features') or hasattr(layer, 'in_features'),
                'has_weight': hasattr(layer, 'weight') and layer.weight is not None,
                'has_bias': hasattr(layer, 'bias') and layer.bias is not None,
                'zero_filters': None,
                'zero_inputs': None,
            }
            if type(layer) in (nn.Conv2d, nn.ReLU, nn.BatchNorm2d, nn.Linear):
                target_layers.append(layer)

    # disable relu
    relu_backup = []
    for layer in target_layers:
        if layer._meta['is_relu']:
            relu_backup.append(layer)
            module_name = layer._meta['module_name']
            layer_name = layer._meta['layer_name']
            replace_layer(module_name, layer_name, nn.Identity())
            print_if_verbose(f'disable relu {get_layer_name(layer)}')

    # disable bias
    for layer in target_layers:
        if layer._meta['has_bias']:
            layer._meta['bias_backup'] = layer.bias.data.clone()
            layer.bias.data *= 0
            print_if_verbose(f'disable bias for {get_layer_name(layer)}')

    # register pre-forward hook
    def pre_hook(layer, input):
        batch_size = input[0].size(0)
        in_channels = input[0].size(1)
        zero_inputs = (input[0].view(batch_size, in_channels, -1).abs().sum(2).sum(0) == 0)
        layer._meta['zero_inputs'] = zero_inputs
        if type(layer) == nn.BatchNorm2d:
            layer._meta['zero_filters'] = zero_inputs
        elif type(layer) == nn.Conv2d:
            layer._meta['zero_filters'] = _locate_zeroized(layer, zero_inputs)
        print_if_verbose(f'{torch.count_nonzero(zero_inputs)} dead inputs detected for layer {get_layer_name(layer)}')
        return input

    _hooks = []
    for layer in target_layers:
        if layer._meta['is_sensitive']:
            handle = layer.register_forward_pre_hook(pre_hook)
            _hooks.append(handle)

    # forward with random input(s) (to trace zero inputs & update zero filters for conv)
    x = torch.rand(4, 3, 224, 224)
    print_if_verbose(f'tracing with random input {x.size()}')
    _ = model(x)

    # clean hooks
    for handle in _hooks:
        handle.remove()

    # recover relu
    for layer in relu_backup:
        module_name = layer._meta['module_name']
        layer_name = layer._meta['layer_name']
        replace_layer(module_name, layer_name, layer)
        print_if_verbose(f'recover relu {module_name}/{layer_name} -> {type(layer)}')
    del relu_backup

    # register after-forward hook
    def after_hook(layer, input, result):
        zero_filters = layer._meta['zero_filters']
        zero_inputs = layer._meta['zero_inputs']
        if type(layer) is nn.BatchNorm2d:  # clean batchnorm non-zero outputs
            result *= zero_inputs.view(1, -1, 1, 1)

        # collect upstream data
        batch_size = result.size(0)
        out_channels = result.size(1)
        upstream = result.data.view(batch_size, out_channels, -1)[0, :, 0]
        layer._meta['bias_backup'] += upstream
        _data = upstream.abs().sum().item()
        if _data > 0:
            print_if_verbose(f'-> {_data:.4f} dying biases flowing into {get_layer_name(layer)}')
        result *= 0

        # add additional bias to next layer input
        if zero_filters is not None:
            additional_stream = (layer._meta['bias_backup'] * zero_filters).view(1, -1)
            while len(additional_stream.size()) < len(result.size()):
                additional_stream = additional_stream.unsqueeze(-1)
            result += additional_stream
            _data = additional_stream.abs().sum().item()
            if _data > 0:
                print_if_verbose(f'{_data:.4f} data leaking from {get_layer_name(layer)}')

        return result

    # collect dead biases (into next layer with bias)
    _hooks = []
    for layer in target_layers:
        if layer._meta['has_bias']:
            handle = layer.register_forward_hook(after_hook)
            _hooks.append(handle)

    # forward with zero input(s)
    x = torch.zeros(4, 3, 224, 224)
    print_if_verbose(f'tracing with zero input {x.size()}')

    # make sure no homeless bias remains
    y = model(x)
    if isinstance(y, list):
        for _y in y:
            assert _y.abs().sum() == 0
    else:
        assert y.abs().sum() == 0, print(y)

    # clean hooks
    for handle in _hooks:
        handle.remove()
    del _hooks

    # slaughter time!
    for layer in target_layers:
        if layer._meta['has_weight']:
            zero_filters = layer._meta['zero_filters']
            zero_inputs = layer._meta['zero_inputs']
            module_name = layer._meta['module_name']
            layer_name = layer._meta['layer_name']
            weight = layer.weight.clone()
            if type(layer) == nn.Conv2d:
                if zero_filters is None:
                    zero_filters = torch.tensor([False for _ in range(weight.size(1))], device=weight.device)
                if zero_inputs is None:
                    zero_inputs = torch.tensor([False for _ in range(weight.size(0))], device=weight.device)
                non_zero_filters = ~zero_filters
                non_zero_inputs = ~zero_inputs
                weight = weight[non_zero_filters, ...]
                weight = weight[:, non_zero_inputs, ...]
                new_conv = nn.Conv2d(
                    in_channels=weight.size(1),
                    out_channels=weight.size(0),
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=layer.bias is not None,
                    device=layer.weight.device,
                )
                new_conv.weight.data = weight
                if layer._meta['has_bias']:
                    bias = layer._meta['bias_backup']
                    bias = bias[non_zero_filters]
                    new_conv.bias.data = bias
                replace_layer(module_name, layer_name, new_conv)
            elif type(layer) == nn.BatchNorm2d:
                if zero_filters is None:
                    zero_filters = torch.tensor([False for _ in range(weight.size(1))], device=weight.device)
                if zero_inputs is None:
                    zero_inputs = torch.tensor([False for _ in range(weight.size(0))], device=weight.device)
                non_zero_filters = ~zero_filters
                non_zero_inputs = ~zero_inputs
                weight = weight[non_zero_filters]
                new_bn = nn.BatchNorm2d(
                    num_features=weight.size(0),
                    eps=layer.eps,
                    momentum=layer.momentum,
                    device=layer.weight.device,
                )
                new_bn.train(layer.training)
                new_bn.weight.data = weight
                new_bn.running_mean = layer.running_mean[non_zero_filters]
                new_bn.running_var = layer.running_var[non_zero_filters]
                if layer._meta['has_bias']:
                    bias = layer._meta['bias_backup']
                    bias = bias[non_zero_filters]
                    new_bn.bias.data = bias
                replace_layer(module_name, layer_name, new_bn)
            elif type(layer) == nn.Linear:
                if zero_filters is None:
                    zero_filters = torch.tensor([False for _ in range(weight.size(0))], device=weight.device)
                if zero_inputs is None:
                    zero_inputs = torch.tensor([False for _ in range(weight.size(1))], device=weight.device)
                non_zero_filters = ~zero_filters
                non_zero_inputs = ~zero_inputs
                weight = weight[non_zero_filters, :]
                weight = weight[:, non_zero_inputs]
                new_linear = nn.Linear(
                    weight.size(1),
                    weight.size(0),
                    bias=layer.bias is not None,
                    device=weight.device,
                )
                new_linear.weight.data = weight
                if layer._meta['has_bias']:
                    bias = layer._meta['bias_backup']
                    bias = bias[non_zero_filters]
                    new_linear.bias.data = bias
                replace_layer(module_name, layer_name, new_linear)

        if hasattr(layer, '_meta'):
            del layer._meta
    del modules, target_layers

    return model


if __name__ == "__main__":
    # debug sample
    # model = resnet18()
    # model.conv1.weight.data[[0,2,4],...] = 0
    # model.fc = nn.Linear(model.fc.in_features, 8)
    # model.eval()
    # x = torch.rand(4, 3, 224, 224)
    # y = model(x).abs().sum().item()
    # _y = melt(model, False)(x).abs().sum().item()
    # print(y, _y)
    # print()

    # test sample 1
    print('=============================')
    print('test sample 1')
    model = nn.Sequential(nn.Conv2d(3, 12, 1, bias=False),)
    model[0].weight.data[[0, 2, 4], ...] *= 0
    model.eval()
    x = torch.rand(2, 3, 4, 4)
    y = model(x).abs().sum().item()
    _y = melt(model)(x).abs().sum().item()
    print(y, _y)
    print()

    # test sample 2
    print('=============================')
    print('test sample 2')
    model = nn.Sequential(
        nn.Conv2d(3, 12, 1, bias=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(12, 5, bias=True),
    )
    model.eval()
    model[0].weight.data[[0, 2, 4], ...] *= 0
    x = torch.rand(2, 3, 4, 4)
    # y = model(x).abs().sum().item()
    y = nn.Sequential(*[m for m in model])(x).abs().sum().item()
    model = melt(model)
    # _y = model(x).abs().sum().item()
    _y = nn.Sequential(*[m for m in model])(x).abs().sum().item()
    print(y, _y)
    print()

    # test sample 3
    print('=============================')
    print('test sample 3')
    model = nn.Sequential(
        nn.Conv2d(3, 12, 1, bias=True),
        nn.BatchNorm2d(12),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(12, 5, bias=True),
    )
    model.eval()
    model[0].weight.data[[0, 2, 4], ...] *= 0
    x = torch.rand(2, 3, 4, 4)
    y = model(x).abs().sum().item()
    model = melt(model)
    _y = model(x).abs().sum().item()
    print(y, _y)
    print()

    # test sample 4
    print('=============================')
    print('test sample 4')
    model = nn.Sequential(
        nn.Conv2d(3, 120, 1, bias=False),
        nn.ReLU(),
        nn.Conv2d(120, 60, 1, bias=True),
        nn.BatchNorm2d(60),
        nn.ReLU(),
        nn.Conv2d(60, 60, 1, bias=False),
        nn.BatchNorm2d(60),
        nn.ReLU(),
        nn.Conv2d(60, 60, 1, bias=True),
        nn.BatchNorm2d(60),
        nn.ReLU(),
        nn.Conv2d(60, 120, 1, bias=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(120, 5),
    )
    model[0].weight.data[[0, 2, 4], ...] *= 0
    model[2].weight.data[[1, 3, 5], ...] *= 0
    model[5].weight.data[[0, 2, 4], ...] *= 0
    model[8].weight.data[[0, 3, 5], ...] *= 0
    model[11].weight.data[[0, 2, 4], ...] *= 0
    model.eval()
    x = torch.rand(64, 3, 4, 4)
    y = model(x).abs().sum().item()
    model = melt(model)
    _y = model(x).abs().sum().item()
    print(y, _y)
