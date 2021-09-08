from _utils.models import Xyolov5s
from melt3 import melt
import torch

if __name__ == '__main__':

    device = 'cpu'
    weights = 'runs/release/last.pt'

    # load model
    nc = 6
    model = Xyolov5s(nc=6).to(device)  # create
    if weights.endswith('.pt'):
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if 'model' in ckpt:
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        else:
            csd = ckpt
        # if nc != 80:
        #     new_csd = model.state_dict()
        #     for k in csd.keys():
        #         if 'detect.m' not in k:
        #             new_csd[k] = csd[k]
        #     csd = new_csd
        model.load_state_dict(csd, strict=False)  # load
    # for layer in model.modules():
    #     if type(layer) == torch.nn.BatchNorm2d:
    #         layer.bias.data *= 0
    model.dsp()

    # forward with random input
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        y = model(x)
        _y = melt(model)(x)
    
    # check results
    for i, (y1, y2) in enumerate(zip(y, _y)):
        float_ratio = ((y1 - y2)/y1).abs().mean().item()
        print(f'output{i} error: {float_ratio:.16f}')
        # assert float_ratio < 1e-4


