

import torch
import torch.nn as nn
import networkx as nx
import pylab
import random
import matplotlib.pyplot as plt
from functools import reduce

# Utils ----------------------------------------
# for plt visualization issue
pylab.ion()
# visualizations


def render(G: nx.Graph, pause=5, pos=None):
    # render function using networkx
    if pos is None:
        pos = nx.spring_layout(G)
        pos = nx.kamada_kawai_layout(G, pos=pos)
    cmap = plt.cm.plasma
    nx.draw_networkx_nodes(
        G, pos,
        node_color=range(G.number_of_nodes()),
        cmap=cmap,
        node_size=8,
    )
    nx.draw_networkx_edges(
        G, pos,
        node_size=12,
        edge_color=range(G.number_of_edges()),
        edge_cmap=cmap,
        arrowstyle="->",
        arrowsize=6,
        width=0.5,
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=6,
        alpha=0.4,
        horizontalalignment='right',
    )
    plt.gca().set_axis_off()
    plt.show()
    plt.pause(pause)
    plt.close()
    return pos


class Handle():
    def __init__(self, cls, name, raw_func):
        self.cls = cls
        self.name = name
        self.raw_func = raw_func

    def remove(self):
        setattr(self.cls, self.name, self.raw_func)


class Hooks():
    def __init__(self) -> None:
        self.handles = []

    def append(self, handle):
        self.handles.append(handle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        del self


def removable_hooks():
    return Hooks()


class TracerUtils():
    supported_ops = dict(
        cat_ops=['cat'],
        pass_ops=['flatten'],
        join_ops=['add', 'sub', 'mul', 'div'],
        join_ops_tensor=[
            '__add__', '__iadd__', '__radd__',
            '__sub__', '__isub__', '__rsub__',
            '__mul__', '__imul__', '__rmul__',
            '__div__', '__idiv__', '__rdiv__',
        ],
    )

    def __init__(self) -> None:
        pass

    @staticmethod
    def self_increase_id(id, graph):
        i = 0
        while f'{id}#{i}' in graph:
            i += 1
        return f'{id}#{i}'

    @staticmethod
    def get_layer_op_id(layer):
        op_order = len(layer.trace_stack)
        return layer.lid + ('' if op_order == 1 else f'#{op_order}')

    def register_op_hook(self, cls, name, hook):
        raw_func = getattr(cls, name)

        def hooked_func(*args, **kwargs):
            result = raw_func(*args, **kwargs)
            result = hook(name, result, *args, **kwargs)
            return result
        setattr(cls, name, hooked_func)
        self.hooks.append(Handle(cls, name, raw_func))

    def bind_hooks(self, hooks, graph: nx.DiGraph):

        def single_source_op_hook(layer, result, input, *args, **kwargs):
            op = self.self_increase_id(layer, graph)
            graph.add_edge(input.trace_op, op, args=args, **kwargs)
            result.trace_op = op
            return result

        def binary_source_op_hook(layer, result, tensor, other, *args, **kwargs):
            op = self.self_increase_id(layer, graph)
            meta = [(input.trace_op, i) for i, input in enumerate((tensor, other))]
            for input in (tensor, other):
                graph.add_edge(input.trace_op, op, meta=meta, args=args, **kwargs)
            result.trace_op = op
            return result

        def multi_source_op_hook(layer, result, inputs, *args, **kwargs):
            op = self.self_increase_id(layer, graph)
            meta = [(input.trace_op, i) for i, input in enumerate(inputs)]
            for input in inputs:
                graph.add_edge(input.trace_op, op, meta=meta, args=args, **kwargs)
            result.trace_op = op
            return result

        def forward_pre_hook(layer, input):
            if not hasattr(layer, 'trace_stack'):
                layer.trace_stack = []
            layer.trace_stack.append(input[0].trace_op)
            graph.add_edge(input[0].trace_op, self.get_layer_op_id(layer))
            del input[0].trace_op  # block tracer during forward

        def forward_hook(layer, input, result):
            input[0].trace_op = layer.trace_stack[-1]  # recover blocked tracer
            result.trace_op = self.get_layer_op_id(layer)
            return result

        self.hooks = hooks
        self.single_source_op_hook = single_source_op_hook
        self.binary_source_op_hook = binary_source_op_hook
        self.multi_source_op_hook = multi_source_op_hook
        self.forward_pre_hook = forward_pre_hook
        self.forward_hook = forward_hook

        return graph


def isolate(graph: nx.DiGraph):
    clusters = []
    for nodes in graph.edges():
        next_gen = []
        home = [x for x in nodes]
        while len(clusters) > 0:
            cluster = clusters.pop()
            if any([(node in cluster) for node in nodes]):
                home.extend(cluster)
            else:
                next_gen.append(cluster)
        next_gen.append(home)
        clusters = next_gen
    return clusters


class ClusterUtils():
    def __init__(self) -> None:
        pass

    @staticmethod
    def leaf_layers(model, graph):
        # create {lid: layer} dict
        ll = {}
        for lid, layer in model.named_modules():
            if lid in graph:
                ll[lid] = layer
        return ll

    @staticmethod
    def layer_types(ll, graph):
        # register layer type
        lt = {}
        T = TracerUtils.supported_ops
        ops = reduce(lambda x, y: x+y, T.values())
        for lid in graph.nodes():
            true_lid = lid.split('#')[0]
            if true_lid in ll:
                lt[lid] = ll[true_lid]._get_name()
            else:
                for op in ops:
                    if lid.startswith(op):
                        lt[lid] = op
                        break
                else:
                    lt[lid] = lid
        return lt

    @staticmethod
    def dependency_graph(lt, graph):
        dep_graph = nx.DiGraph()
        cat_graph = nx.DiGraph()
        T = TracerUtils.supported_ops
        ops = reduce(lambda x, y: x+y, T.values())
        cat_ops = ['cat']
        split_ops = ['Conv2d', 'Linear', 'input.']
        fuse_ops = [x for x in ops if x not in split_ops]
        fuse_ops += ['BatchNorm2d', 'ReLU', 'MaxPool2d', 'Upsample', 'AdaptiveAvgPool2d']
        for nodes in graph.edges():
            data = graph.get_edge_data(*nodes)
            ntypes = [lt[node] for node in nodes]
            ns = []
            for node, ntype, iotype in zip(nodes, ntypes, ('out', 'in')):
                if ntype in fuse_ops + cat_ops:
                    ns.append((node, 'pass'))
                elif ntype in split_ops:
                    ns.append((node, iotype))
                else:
                    print(f'unsupported operation {ntype}')
                    raise NotImplementedError
            if ntypes[1] == 'cat':
                # record cat ops specially
                for i, (v, order) in enumerate(data['meta']):
                    if v == nodes[0]:
                        data['meta'][i] = (ns[0], order)
                cat_graph.add_edge(*ns, **data)
            else:
                # record normal ops
                dep_graph.add_edge(*ns, **data)
        return dep_graph, cat_graph


def broadcast_infection(clusters, infected):
    unresolved = [True for _ in clusters]
    while any(unresolved):
        for i, (source, infection, cluster_type) in enumerate(clusters):
            if any([(x not in infected) for x in source]):
                # unresolved source exists, skip
                continue
            if len(source) > 0:
                # start infection
                deads = [infected[x] for x in source]
                if cluster_type == 'join':
                    dead_channels = reduce(lambda x, y: x*y, deads)
                    for x in infection + source:  # infect both source and target
                        infected[x] = dead_channels
                elif cluster_type == 'cat':   # infect(cat) target only
                    dead_channels = torch.cat(deads)
                    for x in infection:
                        infected[x] = dead_channels
                else:
                    raise NotImplementedError
            unresolved[i] = False
        print('unresolved:', unresolved.count(True))
    return infected


def replace_infected_layers(infected, ll, fm):
    for lid, layer in ll.items():
        if type(layer) in (nn.Conv2d, nn.Linear):
            zero_in = torch.zeros(layer.weight.size(1))
            zero_out = torch.zeros(layer.weight.size(0))
            in_mask = ~infected.setdefault((lid, 'in'), zero_in).bool()
            out_mask = ~infected.setdefault((lid, 'out'), zero_out).bool()
            if type(layer) == nn.Conv2d:
                new_layer = nn.Conv2d(
                    in_channels=torch.count_nonzero(in_mask),
                    out_channels=torch.count_nonzero(out_mask),
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=layer.bias is not None,
                    padding_mode=layer.padding_mode,
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
            elif type(layer) == nn.Linear:
                new_layer = nn.Linear(
                    in_features=torch.count_nonzero(in_mask),
                    out_features=torch.count_nonzero(out_mask),
                    bias=layer.bias is not None,
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
            new_layer.weight.data.copy_(layer.weight.data[out_mask, ...][:, in_mask, ...])
            if hasattr(new_layer, 'bias') and new_layer.bias is not None:
                new_layer.bias.data.copy_(layer.bias.data[out_mask])
            cid, m = fm[lid]
            setattr(m, cid, new_layer)
        elif type(layer) == nn.BatchNorm2d:
            mask = ~infected[(lid, 'pass')].bool()
            new_layer = nn.BatchNorm2d(
                num_features=torch.count_nonzero(mask),
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            new_layer.weight.data.copy_(layer.weight.data[mask])
            new_layer.bias.data.copy_(layer.bias.data[mask])
            new_layer.running_mean.data.copy_(layer.running_mean.data[mask])
            new_layer.running_var.data.copy_(layer.running_var.data[mask])
            cid, m = fm[lid]
            setattr(m, cid, new_layer)
    return

# ----------------------------------------
    # register common ops in:
    #   torch, torch.tensor, torch.nn.functional
    # in order to trace forward compute graph


def trace_compute_graph(model: nn.Module, tracer_shape=(4, 3, 224, 224)):
    with removable_hooks() as hooks:
        tracer = TracerUtils()
        graph = tracer.bind_hooks(hooks, nx.DiGraph())
        for op in tracer.supported_ops['pass_ops']:
            tracer.register_op_hook(torch, op, tracer.single_source_op_hook)
        for op in tracer.supported_ops['join_ops']:
            tracer.register_op_hook(torch, op, tracer.binary_source_op_hook)
        for op in tracer.supported_ops['cat_ops']:
            tracer.register_op_hook(torch, op, tracer.multi_source_op_hook)
        # for op in tracer.supported_ops['pass_o ps_functional']:
        #     tracer.register_op_hook(torch.nn.functional, op, tracer.single_source_op_hook)
        for op in tracer.supported_ops['join_ops_tensor']:
            tracer.register_op_hook(torch.Tensor, op, tracer.binary_source_op_hook)
        for lid, layer in model.named_modules():
            if not any(layer.children()):  # for all leaf layers
                # register layer id
                layer: nn.Module
                layer.lid = lid
                # register tracer hook for layer ops
                hooks.append(layer.register_forward_hook(tracer.forward_hook))
                hooks.append(layer.register_forward_pre_hook(tracer.forward_pre_hook))
        # trace
        x = torch.rand(*tracer_shape)
        x.trace_op = 'input.'
        model.eval().forward(x)
    # render(graph, 100)
    return graph


def find_clusters(model, graph: nx.DiGraph):
    T = ClusterUtils
    ll = T.leaf_layers(model, graph)
    lt = T.layer_types(ll, graph)
    dep_graph, cat_graph = T.dependency_graph(lt, graph)
    clusters = []
    for c in isolate(dep_graph):
        source, infection = [], []
        for node in c:
            lid, iotype = node
            ntype = lt[lid]
            if ntype == 'cat' or (ntype == 'Conv2d' and iotype == 'out'):
                source.append(node)
            else:
                infection.append(node)
        clusters.append([source, infection, 'join'])
    for _, v, data in cat_graph.edges(data=True):
        source, infection = sorted(data['meta'], key=lambda x: x[1]), [v]
        source = [x[0] for x in source]
        clusters.append([source, infection, 'cat'])
    # display --------------------------------------------
    print('clustering dependencies..')
    # sizel = max([len(str(lid)) for lid in dep_graph.nodes()])
    # for src, inf, _ in clusters:
    #     maxlen = max(len(src), len(inf))
    #     for i in range(maxlen):
    #         left = '' if i > len(src)-1 else src[i]
    #         right = '' if i > len(inf)-1 else f'-> {inf[i]}'
    #         line = f'{str(left):{sizel}s} {right}'
    #         print(line)
    #     print()
    return clusters


def find_dead_channels(model, clusters, mode, compress_rate=0.1, blacklist=[]):
    ll = {}
    infected = {}
    pairwise_d = nn.PairwiseDistance(p=2)
    for lid, layer in model.named_modules():
        if any(layer.children()):
            continue
        ll[lid] = layer
        if type(layer) == nn.Conv2d:
            num_filters = layer.out_channels
            zeros = torch.zeros(num_filters)
            if lid in blacklist:
                dead_channels = zeros.bool()
            elif mode == 'fpgm':
                device = layer.weight.device
                N = layer.weight.data.size(0)
                FS = layer.weight.data.view(N, -1)
                D = torch.zeros(N, N, device=device, requires_grad=False)
                __cache = {}
                for i, Fi in enumerate(FS):
                    print(f'\rpruning layer {lid} [{i}/{N}]             ', end='')
                    for j, Fj in enumerate(FS):
                        # calculate distance
                        __cache_index = (i, j) if i < j else (j, i)
                        if __cache_index in __cache:
                            # read from cache
                            D[i][j] += __cache[__cache_index]
                        else:
                            __cache[__cache_index] = d = pairwise_d(Fi.view(1, -1), Fj.view(1, -1))[0]
                            D[i][j] += d
                # get filter(s) with minimum sum-distance
                D = D.sum(dim=1)
                dead_channels = D.argsort() < N*compress_rate
            elif mode == 'zeroized':
                dead_channels = (layer.weight.data.view(num_filters, -1).abs().sum(dim=1) == 0).bool()
            elif mode == 'random':
                dead_channels = random.choices([i for i in range(num_filters)], k=num_filters//2)
                zeros = torch.zeros(num_filters)
                zeros[dead_channels] = 1
                dead_channels = zeros.bool()
            else:
                raise NotImplementedError
            # make sure no branch is completely removed
            if torch.count_nonzero(dead_channels) == dead_channels.size(0):
                dead_channels = zeros.bool()
            infected[(lid, 'out')] = dead_channels
    # strict searching of dead channels
    infected = broadcast_infection(clusters, infected)
    # for x, d in infected.items():
    #     print(x, torch.count_nonzero(d))
    return infected


def prune_dead_channels(model, infected: dict):
    # register layer info
    ll = {}
    for lid, layer in model.named_modules():
        if any(layer.children()):
            continue
        ll[lid] = layer
        layer.lid = lid
    # register father module
    fm = {}
    for m in model.modules():
        for cid, layer in m.named_children():
            if hasattr(layer, 'lid'):
                fm[layer.lid] = (cid, m)
                del layer.lid
    # technically,
    # dead biases are NOT allowed to pass nonlinear layers unless the channel is fixed,
    # e.g. nn.ReLU, nn.Sigmoid...
    # also, some operations like mul/div can cause critical issue.
    # as a conclusion, lossless compression is NOT a realistic target,
    # we just prune here and leave it to fine-tune.
    replace_infected_layers(infected, ll, fm)


def shrink(model, mode, blacklist=[]):
    # -----------------------------------------------------
    # Transform a sparse model into a compact model
    # to reduce size and computational cost.
    # it works in following steps:
    #   1. trace compute graph for leaf layers (e.g. nn.Conv2d, nn.Maxpool2d, nn.Linear...)
    #   2. find clusters with same dependency
    #   2. find dead channels (or target channels for FPGM)
    #   3. prune dead channels
    #
    graph = trace_compute_graph(model)
    clusters = find_clusters(model, graph)
    infected = find_dead_channels(model, clusters, mode=mode, blacklist=blacklist)
    prune_dead_channels(model, infected)
    return model


if __name__ == '__main__':
    # from torchvision.models import resnet18
    # model = resnet18(pretrained=True)
    from models import Xyolov5s
    model = Xyolov5s(nc=6).dsp()
    # for lid, layer in model.named_modules():
    #     if type(layer) == nn.Conv2d and 'detect.m.' not in lid:
    #         rand_p_index = random.choices([x for x in range(layer.out_channels)], k=layer.out_channels//2)
    #         layer.weight.data[rand_p_index,...] *= 0
    model.eval()
    model = shrink(
        model,
        mode='fpgm',
        blacklist=['backbone_stage_1.0.focus', 'detect.m.0', 'detect.m.1', 'detect.m.2'],
    )
    model.forward(torch.rand(4, 3, 224, 224))
