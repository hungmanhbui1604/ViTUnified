import torch
import torch.nn as nn
import yaml
import os
from models import ViTUnified
import argparse


class ViTUnifiedWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        embedding, pad_outputs = self.model(x)
        return embedding, pad_outputs

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, args.filename)

    cfg = yaml.safe_load(open(args.config, "r"))
    model_cfg = cfg["model"]

    model = ViTUnified(
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"],
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    model.eval().cuda()

    wrapped_model = ViTUnifiedWrapper(model).eval().cuda()

    dummy = torch.randn(1, 3, 224, 224).cuda()

    torch.onnx.export(
        wrapped_model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["embedding", "pad_outputs"],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
            "pad_outputs": {0: "batch"},
        },
)

    print(f"Saved ONNX to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ONNX")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/",
        help="Directory for output",
    )
    parser.add_argument(
        "--filename",
        default="vu.onnx",
        help="ONNX filename",
    )
    args = parser.parse_args()

    main(args)