from _utils.models import Xyolov5s
from _utils.melt import melt
import torch

if __name__ == '__main__':

    device = 'cpu'
    weights = 'runs/train/exp35/weights/last.pt'

    # Model
    model = Xyolov5s().to(device)  # create
    if weights.endswith('.pt'):
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if 'model' in ckpt:
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        else:
            csd = ckpt
        model.load_state_dict(csd, strict=False)  # load

    model.eval()
    model.detect.train()

    x = torch.rand(4, 3, 224, 224)
    y = model(x)

    model = melt(model)
    _y = model(x)

    for yi, _yi in zip(y, _y):
        print(yi.abs().sum(), _yi.abs().sum())