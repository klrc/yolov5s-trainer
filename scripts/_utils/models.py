import torch
import torch.nn as nn
from common import Focus, Conv, C3, SPP, Detect


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class Xyolov5s(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.backbone_stage_1 = nn.Sequential(
            Focus(3, 32),
            Conv(32, 64, 3, 2),
            C3(64, 64, 1),
            Conv(64, 128, 3, 2),
            C3(128, 128, 3),
        )
        self.backbone_stage_2 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C3(256, 256, 3),
        )
        self.backbone_stage_3 = nn.Sequential(
            Conv(256, 512, 3, 2),
            SPP(512),
            C3(512, 512, 1, False),
            Conv(512, 256, 1, 1),
        )
        self.up_p1 = nn.Upsample(None, 2, 'nearest')
        self.up_p2 = nn.Upsample(None, 2, 'nearest')
        self.conv_p3 = Conv(128, 128, 3, 2)
        self.conv_p4 = Conv(256, 256, 3, 2)

        self.head_p1 = nn.Sequential(
            C3(512, 256, 1, False),
            Conv(256, 128, 1, 1),
        )
        self.c3_p2 = C3(256, 128, 1, False)
        self.c3_p3 = C3(256, 256, 1, False)
        self.c3_p4 = C3(512, 512, 1, False)

        self.detect = Detect(nc)

    def forward(self, x):
        x1 = self.backbone_stage_1(x)  # (1, 128, .., ..)
        x2 = self.backbone_stage_2(x1)  # (1, 256, .., ..)
        x3 = self.backbone_stage_3(x2)  # (1, 256, .., ..)

        f1 = self.head_p1(torch.cat((self.up_p1(x3), x2), dim=1))  # (1, 128, .., ..)
        f2 = self.c3_p2(torch.cat((self.up_p2(f1), x1), dim=1))  # (1, 128, .., ..)
        f3 = self.c3_p3(torch.cat((self.conv_p3(f2), f1), dim=1))  # (1, 256, .., ..)
        f4 = self.c3_p4(torch.cat((self.conv_p4(f3), x3), dim=1))  # (1, 512, .., ..)

        return self.detect([f2, f3, f4])

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        return self

    def dsp(self):
        self.detect.forward = self.detect._dsp_forward
        return self

    def torch(self):
        self.detect.forward = self.detect._torch_forward
        return self

    def squeeze_loader(self, state_dict):
        # register layer info
        ll = {}
        for lid, layer in self.named_modules():
            if any(layer.children()):
                continue
            ll[lid] = layer
            layer.lid = lid
        # register father module
        fm = {}
        for m in self.modules():
            for cid, layer in m.named_children():
                if hasattr(layer, 'lid'):
                    fm[layer.lid] = (cid, m)
                    del layer.lid
        # squeeze model layers
        _squeeze_layers(ll, fm, state_dict)
        self.load_state_dict(state_dict)
        # check nc
        loaded_no = self.detect.m[0].weight.size(0)//3
        loaded_nc = loaded_no - 5
        # if loaded model does not satisfy dataset
        if self.nc != loaded_nc:
            # reload detect
            self.detect = Detect(self.nc)
        return self


def _squeeze_layers(ll, fm, state_dict):
    for lid, layer in ll.items():
        if type(layer) in (nn.Conv2d, nn.Linear):
            w = state_dict[lid + '.weight']
            ic, oc = w.size(1), w.size(0)
            if type(layer) == nn.Conv2d:
                new_layer = nn.Conv2d(
                    in_channels=ic,
                    out_channels=oc,
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
                    in_features=torch.count_nonzero(ic),
                    out_features=torch.count_nonzero(oc),
                    bias=layer.bias is not None,
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
            cid, m = fm[lid]
            setattr(m, cid, new_layer)
        elif type(layer) == nn.BatchNorm2d:
            w = state_dict[lid + '.weight']
            n = w.size(0)
            new_layer = nn.BatchNorm2d(
                num_features=n,
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            cid, m = fm[lid]
            setattr(m, cid, new_layer)
    return


def load_model(weights, device, nc) -> Xyolov5s:
    # Model
    model = Xyolov5s(nc=nc).to(device)  # create
    if weights.endswith('.pt'):
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        # for different version of checkpoint files
        if 'model' in ckpt:
            # exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        elif 'state_dict' in ckpt:
            csd = ckpt['state_dict']
        else:
            csd = ckpt
        # replace detection head for new nc
        ckpt_nc = 0
        detectors = []
        for k, v in csd.items():
            if 'detect.m' in k:
                detectors.append(k)
                ckpt_nc = v.size(0)//3 - 5
                if ckpt_nc != nc:
                    if '.weight' in k:
                        ic = v.size(1)
                        csd[k] = model.state_dict()[k][:, :ic, ...]
                    else:
                        csd[k] = model.state_dict()[k]
        model.squeeze_loader(csd)
        model.to(device)
    # Freeze
    freeze = [
        'backbone_stage_1.0.focus.weight',
        'detect.anchors',
        'detect.anchor_grids',
    ]
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    return model, ckpt, csd



if __name__ == '__main__':
    model = load_model('/Volumes/ASM236X NVM/yolov5s-e106.pt', 'cpu', 80)[0]
    print(model.detect.anchors)
    x = torch.rand(4,3,224,224)
    model(x)