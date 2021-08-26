import torch
from _utils.models import Xyolov5s
from _utils.melt import melt


def export_onnx(model, img, f, opset):
    # ONNX model export
    prefix = 'ONNX:'
    try:
        import onnx
        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['output'])

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        print(f'{prefix} export success, saved as {f}')
        print(f"{prefix} run --dynamic ONNX model inference with detect.py: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def run(model, output_path):
    model = model.to('cpu')
    img = torch.zeros(1, 3, 224, 416).to('cpu')

    for _ in range(2):
        model(img)  # dry runs

    export_onnx(model, img, output_path, opset=9)


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

    model.eval().fuse().dsp()
    # model.detect.train()
    y = model(x)

    model = melt(model)
    _y = model(x)

    for yi, _yi in zip(y, _y):
        print(yi.abs().sum(), _yi.abs().sum())

    run(model, '../yolov5s-builder/build/yolov5s.onnx')
