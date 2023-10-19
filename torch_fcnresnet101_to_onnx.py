import torch
import torch.onnx
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights


def get_pytorch_onnx_model(pytorch_model):
    # define the name of further converted model
    onnx_model_path = "fcn_resnet101.onnx"

    input_tensor = torch.randn(1, 3, 500, 500)

    # model export into ONNX format
    torch.onnx.export(
        pytorch_model,
        input_tensor,
        onnx_model_path,
        verbose=False,
        input_names=['input'],
        output_names=['out'],
        opset_version=11
    )

    return onnx_model_path

def main():
    # initialize fcn_resnet50 with default weight
    weights = FCN_ResNet101_Weights.DEFAULT
    pytorch_model = fcn_resnet101(weights=weights)
    pytorch_model.eval()
    
    # get the path to the converted into ONNX PyTorch model
    onnx_model = get_pytorch_onnx_model(pytorch_model)

if __name__ == "__main__":
    main()