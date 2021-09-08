import torch
import torch.nn as nn
from .common import Focus, Conv, C3, SPP, Detect


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
        for m in self.modules():
            if type(m) is Detect:
                m.forward = m.dspforward  # update forward
        return self

    def squeeze_loader(self, state_dict):
        pass