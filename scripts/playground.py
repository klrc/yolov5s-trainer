from _utils.models import Xyolov5s
from _utils.melt import melt
import torch

if __name__ == '__main__':

    device = 'cpu'
    weights = 'runs/train/exp58/weights/last.pt'

    # Model
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

    x = torch.rand(64, 3, 224, 224)

    model.eval()
    model.detect.train()
    y = model(x)
    print(model.detect.anchors)

    model = melt(model)
    _y = model(x)

    for yi, _yi in zip(y, _y):
        print(yi.abs().sum(), _yi.abs().sum())
