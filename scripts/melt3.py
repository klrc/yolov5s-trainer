from functools import reduce
from inspect import stack
import random
import torch.nn as nn
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pylab
from torch.nn import parameter
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet50

# visualizations ------------------------------------
# for plt visualization issue
pylab.ion()

# render function using networkx
# pretty slow, need to be replaced
def render(G:nx.Graph, pause=5):
    pos = nx.kamada_kawai_layout(G)
    cmap = plt.cm.plasma
    nx.draw_networkx_nodes(
        G, pos,
        node_color='indigo',
        node_size=12,    
        # node_size=[12 if 'parameters' not in G.nodes[n] else (6 + 4 * len(str(G.nodes[n]['parameters']))) for n in G.nodes()],    
    )
    nx.draw_networkx_edges(
        G, pos,
        node_size=12,
        edge_color=range(2, G.number_of_edges() + 2),
        edge_cmap=cmap,
        arrowstyle="->",
        arrowsize=6,
        width=0.5,
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=6,
        horizontalalignment='right',
    )
    plt.gca().set_axis_off()
    plt.show()
    plt.pause(pause)
    plt.close()
    return pos

# removable tensor op tracers -----------------------------------
# ops
__cat_ops = ['cat']
__pass_ops = ['flatten']
__join_ops = ['add', 'sub', 'mul', 'div']
__chunk_ops = ['chunk']
__functional_pass_ops = ['adaptive_avg_pool2d']
__tensor_join_ops = [
    '__add__', '__iadd__', '__radd__', 
    '__sub__', '__isub__', '__rsub__',
    '__mul__', '__imul__', '__rmul__',
    '__div__', '__idiv__', '__rdiv__',
]
__tensor_chunk_ops = ['chunk']

# removable handle for hook functions
class Handle():
    def __init__(self, cls, name, raw_func):
        self.cls = cls
        self.name = name
        self.raw_func = raw_func
    
    def remove(self):
        setattr(self.cls, self.name, self.raw_func)

# for operations as equal/flatten/sigmoid/..., just pass through tracer
def register_op_pass(cls, name):
    def func_wrapper(func):
        def new_func(tensor, *args, **kwargs):
            output = func(tensor, *args, **kwargs)
            if hasattr(tensor, 'from_id'):
                output.from_id = tensor.from_id
            return output
        return new_func
    raw_func = getattr(cls, name)
    setattr(cls, name, func_wrapper(raw_func))
    return Handle(cls, name, raw_func)

# for operations as add/sub/mul/div, register a new node as op_join_i
__op_join_iid = -1
def register_op_join(cls, name, nx_graph):
    def func_wrapper(func):
        def new_func(self, other, *args, **kwargs):
            output = func(self, other, *args, **kwargs)
            if not any([hasattr(x, 'from_id') for x in (self, other)]):
                return output
            else:
                global __op_join_iid
                __op_join_iid += 1
                self_id = f'op_join_{__op_join_iid}'
                nx_graph.add_node(self_id)
                for x in (self, other):
                    if hasattr(x, 'from_id'):
                        nx_graph.add_edge(x.from_id, self_id)
                output.from_id = self_id
                return output
        return new_func
    raw_func = getattr(cls, name)
    setattr(cls, name, func_wrapper(raw_func))
    return Handle(cls, name, raw_func)

# for operations as concat, register a new node as op_cat_i
__op_cat_iid = -1
def register_op_cat(cls, name, nx_graph:nx.DiGraph):
    def func_wrapper(func):
        def new_func(tensors, *args, **kwargs):
            output = func(tensors, *args, **kwargs)
            if any([not hasattr(x, 'from_id') for x in tensors]):
                return output
            else:
                global __op_cat_iid
                __op_cat_iid += 1
                self_id = f'op_cat_{__op_cat_iid}'
                nx_graph.add_node(self_id)
                for i, x in enumerate(tensors):
                    nx_graph.add_edge(x.from_id, self_id, order=i)
                output.from_id = self_id
                return output
        return new_func
    raw_func = getattr(cls, name)
    setattr(cls, name, func_wrapper(raw_func))
    return Handle(cls, name, raw_func)

# for chunk() operation, register a new node as op_chunk_i if dim=1
__op_chunk_iid = -1
def register_op_chunk(cls, name, nx_graph:nx.DiGraph):
    def func_wrapper(func):
        def new_func(input, chunks, dim=0, *args, **kwargs):
            outputs = func(input, chunks, dim, *args, **kwargs)
            if not hasattr(input, 'from_id'):
                return outputs
            if dim == 1:  # warning: unstable check
                print('warning: unstable check for op_chunk')
                global __op_chunk_iid
                __op_chunk_iid += 1
                self_id = f'op_chunk_{__op_chunk_iid}'
                nx_graph.add_node(self_id)
                nx_graph.add_edge(input.from_id, self_id, chunks=chunks)
                for i, output in enumerate(outputs):
                    nx_graph.add_node(f'{self_id}_{i}')
                    nx_graph.add_edge(self_id, f'{self_id}_{i}', order=i)
                    output.from_id = f'{self_id}_{i}'
                return outputs
                
            else:
                for output in outputs:
                    output.from_id = input.from_id
                return outputs
        return new_func
    raw_func = getattr(cls, name)
    setattr(cls, name, func_wrapper(raw_func))
    return Handle(cls, name, raw_func)

# register nature ops to supplement operations in forward()
def register_nature_ops(nx_graph):
    # environments
    hooks = []
    for op in __pass_ops:
        hooks.append(register_op_pass(torch, op))
    for op in __functional_pass_ops:
        hooks.append(register_op_pass(torch.nn.functional, op))
    for op in __cat_ops:
        hooks.append(register_op_cat(torch, op, nx_graph))
    for op in __join_ops:
        hooks.append(register_op_join(torch, op, nx_graph))
    for op in __tensor_join_ops:
        hooks.append(register_op_join(torch.Tensor, op, nx_graph))
    for op in __chunk_ops:
        hooks.append(register_op_chunk(torch, op, nx_graph))        
    for op in __tensor_chunk_ops:
        hooks.append(register_op_chunk(torch.Tensor, op, nx_graph))        
    return hooks

# trace tensor-ops graph ------------------------------------------
__nn_ignore_ops = [ 
    nn.ReLU, nn.SiLU, nn.Sigmoid, 
    nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    nn.Identity, nn.Flatten, nn.Upsample,
]
def _trace(model: nn.Module, x: torch.Tensor, nx_graph:nx.DiGraph):
    # receive and disable tracer during forward
    def forward_pre_hook(layer, input):
        assert len(input) == 1, 'only support tracing with single input.'
        from_id = input[0].from_id
        if type(layer) not in __nn_ignore_ops:
            self_id = layer.meta_id
            nx_graph.add_edge(from_id, self_id)
        layer._trace_from_id = from_id
        del input[0].from_id
    # attach tracer to the output
    def forward_hook(layer, input, result):
        input[0].from_id = layer._trace_from_id
        if type(layer) in __nn_ignore_ops:
            result.from_id = layer._trace_from_id
        else:
            self_id = layer.meta_id
            result.from_id = self_id
        del layer._trace_from_id
    # register hooks
    hooks = []
    for module in model.modules():
        if any(module.children()):
            continue
        # nx graph visualize
        if type(module) not in __nn_ignore_ops:
            parameters = 0
            if hasattr(module, 'weight'):
                parameters += torch.ones_like(module.weight.data).sum()
            if hasattr(module, 'bias') and module.bias is not None:
                parameters += torch.ones_like(module.bias.data).sum()
            nx_graph.add_node(module.meta_id, parameters=parameters)
            # print('->', module.meta_id)
        hooks.append(module.register_forward_pre_hook(forward_pre_hook))
        hooks.append(module.register_forward_hook(forward_hook))
    # trace
    x.from_id = 'input.'
    with torch.no_grad():
        model.forward(x)
    # remove hooks
    for handle in hooks:
        handle.remove()
    del hooks
    return nx_graph

def register_layer_meta(model: nn.Module):
    for layer_name, layer in model.named_modules():
        setattr(layer, 'meta_id', layer_name)

def trace(model: nn.Module, tracer_shape=(4, 3, 224, 224)):
    # register full layer name for all layers
    register_layer_meta(model)
    # trace recursively
    graph = nx.DiGraph()
    hooks = register_nature_ops(graph)
    graph = _trace(model, torch.rand(*tracer_shape), graph)
    # remove hooks
    for handle in hooks:
        handle.remove()
    del hooks
    return graph

# binding all dependencies ---------------------------------------------------------
__nn_zeroize_ops = [nn.Conv2d, nn.Linear]
class Meta():
    def __init__(self, lid):
        self.lid = lid
        self.pre = []
        self.next = []
        self.zeroized_in = None
        self.zeroized_out = None
        self.type = None
        if 'op_cat' in lid:
            self.type = 'op_cat'
        if 'op_join' in lid:
            self.type = 'op_join'

    def receive(self, metas):
        # print('-> receive:', self.lid, self.pre)
        # with self.pre & self.type
        inputs = [metas[x].receive(metas) for x in self.pre]
        if self.lid == 'input.':
            pass
        elif self.type == 'op_cat':
            zeroized = torch.cat(inputs)
            self._trace_back_splits = [x.size(0) for x in inputs]
            self.zeroized_in = zeroized
            self.zeroized_out = zeroized
        elif self.type == 'op_join':
            zeroized = reduce(lambda x, y: x*y, inputs)
            self.zeroized_in = zeroized
            self.zeroized_out = zeroized
            self.backward(zeroized, metas)
        elif self.type == 'BatchNorm2d':
            zeroized = inputs[0]
            self.zeroized_in = zeroized
            self.zeroized_out = zeroized
        else:
            self.zeroized_in = inputs[0]
        # # output nodes
        if len(self.next) > 0:
            # print(self.next, self.type, self.lid)
            assert self.zeroized_out is not None
        return self.zeroized_out

    def backward(self, value, metas):
        if self.type == 'op_cat':
            c1 = 0
            for x, c2 in zip(self.pre, self._trace_back_splits):
                x = metas[x]
                assert x.zeroized_out is not None
                x.zeroized_out *= value[c1:c2]
                c1 += c2
        else:
            for x in self.pre:
                x = metas[x]
                assert x.zeroized_out is not None
                x.zeroized_out *= value
        for x in self.pre:
            x = metas[x]
            if x.type in ('op_cat', 'op_join', 'BatchNorm2d'):
                x.zeroized_in *= value
                x.backward(x.zeroized_in, metas)

    def __str__(self):
        ret = f'lid: {self.lid}\n'
        ret += f'type: {self.type}\n'
        ret += f'pre: {self.pre}\n'
        ret += f'next: {self.next}\n'
        ret += f'zeroized_in: {None if self.zeroized_in is None else self.zeroized_in.shape}\n'
        ret += f'zeroized_out: {None if self.zeroized_out is None else self.zeroized_out.shape}\n'
        if hasattr(self, '_trace_back_splits'):
            ret += f'_trace_back_splits: {self._trace_back_splits}\n'
        return ret

def leaf_layers(model: nn.Module):
    ll = {}
    for lid, layer in model.named_modules():
        if any(layer.children()):
            continue
        ll[lid] = layer
    return ll

def bind(model, nx_graph:nx.DiGraph, tracer_shape=(4, 3, 224, 224)):
    # init binding meta
    metas = {}
    for node in nx_graph.nodes():
        metas[node] = Meta(node)
    for from_node, to_node in nx_graph.edges():
        edge_data = nx_graph.get_edge_data(from_node, to_node)
        if 'order' in edge_data:  # special settings for concat layer
            metas[from_node].next.append(to_node)
            if type(metas[to_node].pre) is list:   # init state
                metas[to_node].pre = {}
            metas[to_node].pre[edge_data['order']] = from_node
        else:
            metas[from_node].next.append(to_node)
            metas[to_node].pre.append(from_node)
    for m in metas.values():
        if type(m.pre) is dict:
            stack = []
            for i in range(max(m.pre.keys()) + 1):
                if i in m.pre:
                    stack.append(m.pre[i])
                else:
                    stack.append(None)
            pre = []
            while stack:
                item = stack.pop()
                if item is None:
                    pre.append(pre[-1])
                else:
                    pre.append(item)
            m.pre = pre
    # searching possible zeroized filters
    metas['input.'].zeroized_out = torch.zeros(tracer_shape[1]).bool()
    ll = leaf_layers(model)
    for lid, layer in ll.items():
        if lid in metas:
            metas[lid].type = layer._get_name()
        if type(layer) in __nn_zeroize_ops:
            num_filters = layer.weight.size(0)
            if len(metas[lid].next) > 0:
                zeroized = (layer.weight.data.view(num_filters, -1).abs().sum(dim=1) == 0)
                # print('--->', sum(zeroized).item(), 'zeroizable filter found in', lid)
            else:
                zeroized = torch.zeros(num_filters)
            metas[lid].zeroized_out = zeroized
    # forward & backward
    for m in metas.values():
        print('tracing:', m.lid)
        m.receive(metas)
    return metas

# melt biases ---------------------------------------------------------------
def _melt(model, metas, tracer_shape=(4, 3, 224, 224)):
    
    def forward_hook(layer, input, result):
        shape = [1, -1]
        while len(shape) < len(result.shape):
            shape.append(1)
        # freeze
        if hasattr(layer, 'bias') and layer.bias is not None:
            if len(result.shape) >= 4:
                h, w = result.size(2), result.size(3)
                collected_biases = result[0,:, h//2, w//2]
            else:
                collected_biases = result[0]
            if type(layer) in (nn.BatchNorm2d,):
                collected_biases *= metas[layer._lid].zeroized_out
            layer.bias.data.copy_(collected_biases)
        # melt
        if layer._lid in metas:
            zeroized_out = metas[layer._lid].zeroized_out
            if zeroized_out is not None:
                # cut virtually for testing ------------------------------------
                if type(layer) in (nn.BatchNorm2d,):
                    layer.weight.data *= ~zeroized_out.bool()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data *= ~zeroized_out.bool()
                # --------------------------------------------------------------
                result *= zeroized_out.view(shape)
                # print(layer._lid, zeroized_out.sum().item(), input[0].abs().sum(), result.abs().sum())
        return result
    # test: output sample A ---------------
    test_x = torch.rand(*tracer_shape)
    with torch.no_grad():
        model.eval()
        y = model(test_x)
        if type(y) == list:  # for yolov5s outputs as list
            checksum_a = sum([_y.abs().sum() for _y in y])
        else:
            checksum_a = y.abs().sum()    
    # register forward hooks
    ll = leaf_layers(model)
    hooks = []
    for lid, layer in ll.items():
        layer._lid = lid
        hooks.append(layer.register_forward_hook(forward_hook))
    # trace
    with torch.no_grad():
        x = torch.zeros(*tracer_shape)
        model.eval()
        y = model(x)
        if type(y) == list:  # for yolov5s outputs as list
            checksum_leaked = sum([_y.abs().sum() for _y in y])
        else:
            checksum_leaked = y.abs().sum()    

        print('leaked:', checksum_leaked)
    # clean model tracer
    for handle in hooks:
        handle.remove()
    del hooks
    for layer in ll.values():
        del layer._lid
    # test: output sample B ---------------
    with torch.no_grad():
        model.eval()
        y = model(test_x)
        if type(y) == list:  # for yolov5s outputs as list
            checksum_b = sum([_y.abs().sum() for _y in y])
        else:
            checksum_b = y.abs().sum()    
    print('checksum_loss:', abs((checksum_a - checksum_b) / checksum_a).item())
    

# lets kill all dead weights! ------------------------------------
def freeze_states(model, metas:list[Meta]):
    # masked-select weight & biases
    for lid, layer in leaf_layers(model).items():
        if lid in metas:
            m = metas[lid]
            _in = ~m.zeroized_in.bool()
            _out = ~m.zeroized_out.bool()
            reshape_if_available_mean_var(layer, _out)
            reshape_if_available_weight(layer, _in, _out)
            reshape_if_available_bias(layer, _out)
            reset_if_available_io_channels(layer, _in, _out)
    return model

def reshape_if_available_mean_var(layer, outputs_mask=None):
    if outputs_mask is not None:
        if hasattr(layer, 'running_mean'):
            layer.running_mean = layer.running_mean[outputs_mask]
        if hasattr(layer, 'running_var'):
            layer.running_var = layer.running_var[outputs_mask]

def reshape_if_available_weight(layer, inputs_mask=None, outputs_mask=None):
    if hasattr(layer, 'weight') and layer.weight is not None:
        if len(layer.weight.data.size()) == 1:
            if inputs_mask is not None:
                layer.weight.data = layer.weight.data[inputs_mask]
                layer.weight.grad = None  # clean gradients
        else:
            if inputs_mask is not None:
                layer.weight.data = layer.weight.data[:,inputs_mask,...]
                layer.weight.grad = None
            if outputs_mask is not None:
                layer.weight.data = layer.weight.data[outputs_mask,:,...]
                layer.weight.grad = None

def reshape_if_available_bias(layer, outputs_mask=None):
    if hasattr(layer, 'bias') and layer.bias is not None:
        if outputs_mask is not None:
            layer.bias.data = layer.bias.data[outputs_mask]
            layer.bias.grad = None

def set_layer_attr(layer, attr, value):
    if hasattr(layer, attr):
        setattr(layer, attr, value)

def reset_if_available_io_channels(layer, inputs_mask=None, outputs_mask=None):
    __in_channel_attrs = ['in_channels', 'num_features', 'in_features',]
    __out_channel_attrs = ['out_channels', 'out_features',]
    if inputs_mask is not None:
        _in = torch.count_nonzero(inputs_mask)
        for attr in __in_channel_attrs:
            set_layer_attr(layer, attr, _in)
    if outputs_mask is not None:
        _out = torch.count_nonzero(outputs_mask)
        for attr in __out_channel_attrs:
            set_layer_attr(layer, attr, _out)  

def melt(model, test_shape=(4,3,224,416)):
    x = torch.rand(*test_shape)
    with torch.no_grad():
        model.eval()
        y = model(x)
        checksum_a = sum([_y.abs().sum() for _y in y])


    graph = trace(model, tracer_shape=test_shape)
    metas = bind(model, graph, tracer_shape=test_shape)
    _melt(model, metas, tracer_shape=test_shape)
    freeze_states(model, metas)

    with torch.no_grad():
        model.eval()
        y = model(x)
        checksum_b = sum([_y.abs().sum() for _y in y])
    print('checksum_loss:', abs((checksum_a - checksum_b) / checksum_a).item())

    render(graph, 100)
    return model

if __name__ == '__main__':
    from _utils.models import Xyolov5s
    model = Xyolov5s().dsp()

    for layer in model.modules():
        if type(layer) == nn.Conv2d:
            ind = [x for x in range(layer.out_channels)]
            ind = random.choices(ind, k=len(ind)//2)
            layer.weight.data[ind,...] *= 0

    x = torch.rand(4,3,224,224)
    with torch.no_grad():
        model.eval()
        y = model(x)
        checksum_a = sum([_y.abs().sum() for _y in y])


    graph = trace(model)
    metas = bind(model, graph)
    _melt(model, metas)
    freeze_states(model, metas)

    with torch.no_grad():
        model.eval()
        y = model(x)
        checksum_b = sum([_y.abs().sum() for _y in y])
    print('checksum_loss:', abs((checksum_a - checksum_b) / checksum_a).item())

    render(graph, 100)
