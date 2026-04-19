import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PADDataset
from model import ViTUnified
from transforms import get_transforms


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, model_cfg: dict, device: torch.device) -> ViTUnified:
    model = ViTUnified(
        model_name=model_cfg.get("model_name", "vit_small_patch16_224"),
        pretrained=False,
        num_classes=model_cfg.get("num_classes", 2),
        pad_dropout=model_cfg.get("pad_dropout", 0.0),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[checkpoint] loaded from '{ckpt_path}'")
    return model


@torch.no_grad()
def collect_preds(
    model: ViTUnified,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Inference", unit="batch"):
        images = images.to(device, non_blocking=True)

        _, pad_outputs = model(images)

        # ensemble: average logits across all PAD heads
        ensemble_logits = torch.stack(pad_outputs, dim=0).mean(dim=0)  # (B, C)

        preds = ensemble_logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_pad_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    n_total = len(labels)
    n_live = int((labels == 0).sum())
    n_spoof = int((labels == 1).sum())

    # True Positives / True Negatives / False Positives / False Negatives
    # Positive class = spoof (1)
    tp = int(((preds == 1) & (labels == 1)).sum())  # spoof correctly classified
    tn = int(((preds == 0) & (labels == 0)).sum())  # live  correctly classified
    fp = int(((preds == 1) & (labels == 0)).sum())  # live misclassified as spoof
    fn = int(((preds == 0) & (labels == 1)).sum())  # spoof misclassified as live

    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    apcer = fn / n_spoof if n_spoof > 0 else 0.0  # spoof→live errors
    bpcer = fp / n_live if n_live > 0 else 0.0  # live→spoof errors
    ace = (apcer + bpcer) / 2.0

    return {
        "n_total": n_total,
        "n_live": n_live,
        "n_spoof": n_spoof,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": float(accuracy),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACE": float(ace),
    }


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def plot_confusion_matrix(
    metrics: dict,
    output_dir: str,
    title: str = "Confusion Matrix",
) -> str:
    _style()
    cm = np.array(
        [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Live", "Pred: Spoof"], fontsize=11)
    ax.set_yticklabels(["True: Live", "True: Spoof"], fontsize=11)
    ax.set_title(title, pad=14)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = os.path.join(output_dir, "pad_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] confusion matrix saved → {path}")
    return path


def main(args: argparse.Namespace) -> None:
    # ── setup ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    evaluation_cfg = cfg["evaluation"]
    training_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint_path, model_cfg, device)

    # ── transforms ───────────────────────────────────────────────────────────
    _, eval_transform = get_transforms("all")

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset = PADDataset(
        split_path=args.split_path,
        split="test",
        transform=eval_transform,
    )

    print(f"\n{dataset}")

    loader = DataLoader(
        dataset,
        batch_size=evaluation_cfg["pad_batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    # ── inference ────────────────────────────────────────────────────────────
    preds, labels = collect_preds(model, loader, device)

    # ── metrics ──────────────────────────────────────────────────────────────
    metrics = compute_pad_metrics(preds, labels)

    print("\n" + "=" * 50)
    print(f"Split path: {args.split_path}")
    print("Split: 'test'")
    print(
        f"Samples: {metrics['n_total']:,} (live={metrics['n_live']:,}, spoof={metrics['n_spoof']:,})"
    )
    print("-" * 50)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"APCER    : {metrics['APCER']:.4f}  (argmax threshold)")
    print(f"BPCER    : {metrics['BPCER']:.4f}  (argmax threshold)")
    print(f"ACE      : {metrics['ACE']:.4f}  (argmax threshold)")
    print("-" * 50)
    print("=" * 50)

    # ── save JSON ─────────────────────────────────────────────────────────────
    results = {
        "checkpoint": args.checkpoint_path,
        "split_path": args.split_path,
        "split": "test",
        "metrics": metrics,
    }

    json_path = os.path.join(args.output_dir, "pad_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── plots ────────────────────────────────────────────────────────────────
    _style()
    plot_confusion_matrix(metrics, args.output_dir)

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fingerprint PAD Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="joint_config.yaml",
        help="Path to the PAD YAML config",
    )
    parser.add_argument(
        "--split-path",
        required=True,
        help="Path to the PAD split JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/pad/",
        help="Directory for metrics JSON and plot PNGs",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the checkpoint",
    )

    main(parser.parse_args())
