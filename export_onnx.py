import torch
import yaml
from models import ViTUnified


class ViTUnifiedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        embedding, pad_outputs = self.model(x)
        pad_logits = pad_outputs[5]
        return embedding, pad_logits


def main():
    config_path = "default_joint_config.yaml"
    checkpoint_path = "ckpts/joint.pth"
    output_onnx_path = "outputs/vitunified.onnx"

    device = torch.device("cpu")

    cfg = yaml.safe_load(open(config_path, "r"))
    model_cfg = cfg["model"]

    model = ViTUnified(
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"]
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    model.eval()
    wrapped_model = ViTUnifiedWrapper(model).eval().to(device)

    dummy = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        wrapped_model,
        dummy,
        output_onnx_path,
        input_names=["input"],
        output_names=["embedding", "pad_logits"],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
            "pad_logits": {0: "batch"},
        },
)

    print(f"Saved ONNX to {output_onnx_path}")


if __name__ == "__main__":
    main()