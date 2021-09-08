import torch
import torch.nn as nn


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = in_channels
        self.cv1 = Conv(in_channels, hidden, 1, 1)
        self.cv2 = Conv(hidden, in_channels, 3, 1)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class Res(nn.Module):
    # Residual bottleneck
    def __init__(self, in_channels, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = in_channels
        self.cv1 = Conv(in_channels, hidden, 1, 1)
        self.cv2 = Conv(hidden, in_channels, 3, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channels, out_channels, repeats=1, shortcut=True, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = int(0.5*out_channels)
        self.cv1 = Conv(in_channels, hidden, 1, 1)
        self.cv2 = Conv(in_channels, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, out_channels, 1, 1)
        if shortcut:
            self.m = nn.Sequential(*[Res(hidden, hidden) for _ in range(repeats)])
        else:
            self.m = nn.Sequential(*[Bottleneck(hidden, hidden) for _ in range(repeats)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.focus = self._make_layer(in_channels)
        self.conv = Conv(in_channels*4, out_channels, 3, 1)

    def _make_layer(self, c1):
        conv = torch.nn.Conv2d(c1,  c1*4, 2, 2, bias=False)
        # set conv kernel
        weight = torch.zeros(c1*4, c1, 2, 2, dtype=conv.weight.dtype)
        for c in range(c1):
            weight[c+0*c1, c, 0, 0] = 1
            weight[c+1*c1, c, 1, 0] = 1
            weight[c+2*c1, c, 0, 1] = 1
            weight[c+3*c1, c, 1, 1] = 1
        conv.weight = torch.nn.Parameter(weight, requires_grad=False)
        return conv

    # def forward(self, x):
    #     return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

    def forward(self, x):
        return self.conv(self.focus(x))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_channels, k=(5, 9, 13)):
        super().__init__()
        hidden = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, hidden, 1, 1)
        self.cv2 = Conv(hidden * (len(k) + 1), in_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Detect(nn.Module):
    def __init__(self,
                 nc=80,
                 anchors=([10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]),
                 ch=(128, 256, 512)):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.stride = torch.tensor((8, 16, 32))  # strides computed during build
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.forward = self._torch_forward

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _torch_forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _dsp_forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
        return x
