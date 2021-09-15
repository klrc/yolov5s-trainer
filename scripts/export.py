import torch
from _utils.models import load_model


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
    weights = 'runs/train/exp106/weights/last.pt'

    # Model
    nc = 6
    model, _, _ = load_model(weights, device, nc=6)
    x = torch.rand(4, 3, 224, 224)

    model.eval().fuse().dsp()
    run(model, 'runs/release/yolov5s-e106.onnx')
