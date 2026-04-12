import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PADDataset
from metrics import compute_recog_metrics
from model import ViTUnified
from transforms import get_transforms


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, model_cfg: dict, device: torch.device) -> ViTUnified:
    model = ViTUnified(
        model_name=model_cfg["model_name"],
        pretrained=False,
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[checkpoint] loaded from '{ckpt_path}'  (epoch {ckpt.get('epoch', '?')})")
    return model


@torch.no_grad()
def collect_scores(
    model: ViTUnified,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Inference",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    
    all_scores, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc=desc, unit="batch"):
        images = images.to(device, non_blocking=True)

        _, pad_outputs = model(images)

        # ensemble: average logits across all PAD heads
        ensemble_logits = torch.stack(pad_outputs, dim=0).mean(dim=0)   # (B, C)
        
        preds = ensemble_logits.argmax(dim=1).cpu().numpy()
        scores = F.softmax(ensemble_logits, dim=1)[:, 1].cpu().numpy()  # prob of spoof

        all_scores.append(scores)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_scores), np.concatenate(all_preds), np.concatenate(all_labels)


def compute_pad_metrics(preds: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> dict:
    n_total  = len(labels)
    n_live   = int((labels == 0).sum())
    n_spoof  = int((labels == 1).sum())

    # True Positives / True Negatives / False Positives / False Negatives
    # Positive class = spoof (1)
    tp = int(((preds == 1) & (labels == 1)).sum())   # spoof correctly classified
    tn = int(((preds == 0) & (labels == 0)).sum())   # live  correctly classified
    fp = int(((preds == 1) & (labels == 0)).sum())   # live misclassified as spoof
    fn = int(((preds == 0) & (labels == 1)).sum())   # spoof misclassified as live

    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    apcer    = fn / n_spoof if n_spoof > 0 else 0.0   # spoof→live errors
    bpcer    = fp / n_live  if n_live  > 0 else 0.0   # live→spoof errors
    ace      = (apcer + bpcer) / 2.0

    # ROC/DET metrics (from metrics.py, mapping FMR->BPCER, FNMR->APCER since spoof=1)
    curve_metrics = compute_recog_metrics(scores, labels)

    return {
        "n_total"  : n_total,
        "n_live"   : n_live,
        "n_spoof"  : n_spoof,
        "TP"       : tp,
        "TN"       : tn,
        "FP"       : fp,
        "FN"       : fn,
        "accuracy" : float(accuracy),
        "APCER"    : float(apcer),
        "BPCER"    : float(bpcer),
        "ACE"      : float(ace),
        # Curve metrics
        "thresholds" : curve_metrics["thresholds"],
        "BPCER_curve": curve_metrics["FMR"],    # FPR
        "APCER_curve": curve_metrics["FNMR"],   # FNR
        "EER"        : curve_metrics["EER"],
        "EER_threshold": curve_metrics["EER_threshold"],
        "AUC"        : curve_metrics["AUC"],
    }


def _style():
    plt.rcParams.update({
        "figure.dpi"       : 150,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "font.size"        : 11,
    })


def plot_confusion_matrix(
    metrics: dict,
    output_dir: str,
    title: str = "Confusion Matrix",
) -> str:
    _style()
    cm = np.array(
        [[metrics["TN"], metrics["FP"]],
         [metrics["FN"], metrics["TP"]]],
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
    thresh  = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
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


def plot_roc(metrics: dict, output_dir: str) -> str:
    bpcer = np.array(metrics["BPCER_curve"])
    apcer = np.array(metrics["APCER_curve"])
    eer = metrics["EER"]
    auc = metrics["AUC"]

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Normally ROC is TPR vs FPR, so 1 - APCER vs BPCER
    # We plot (1 - APCER) which is True Spoof Rate vs BPCER
    tpr = 1.0 - apcer
    ax.plot(bpcer, tpr, lw=2, color="#2563EB", label=f"ROC  (AUC={auc:.4f})")
    
    # Plot EER point
    ax.scatter(
        [eer],
        [1 - eer],
        zorder=6,
        s=80,
        color="grey",
        marker="x",
        label=f"EER point ({eer:.4f}, {1 - eer:.4f})",
    )

    ax.set_xlabel("BPCER  (False Positive Rate)")
    ax.set_ylabel("True Spoof Rate  (1 - APCER)")
    ax.set_title("ROC Curve — Fingerprint PAD")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.3)

    path = os.path.join(output_dir, "pad_roc_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] ROC curve saved → {path}")
    return path


def plot_det(metrics: dict, output_dir: str) -> str:
    """DET curve: APCER vs BPCER on a log-log scale."""
    bpcer = np.array(metrics["BPCER_curve"])
    apcer = np.array(metrics["APCER_curve"])
    eer = metrics["EER"]

    # clip to avoid log(0)
    bpcer_plot = np.clip(bpcer, 1e-5, 1.0)
    apcer_plot = np.clip(apcer, 1e-5, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(bpcer_plot, apcer_plot, lw=2, color="#7C3AED", label="DET")
    ax.plot(
        [eer],
        [eer],
        "o",
        color="#DC2626",
        zorder=6,
        label=f"EER = {eer:.4f}",
        markersize=8,
    )
    ax.plot([1e-5, 1.0], [1e-5, 1.0], "--", color="grey", lw=1, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("BPCER  (False Positive Rate)")
    ax.set_ylabel("APCER  (False Negative Rate)")
    ax.set_title("DET Curve — Fingerprint PAD")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    path = os.path.join(output_dir, "pad_det_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] DET curve saved → {path}")
    return path


def plot_score_dist(
    scores: np.ndarray,
    labels: np.ndarray,
    eer_thr: float,
    output_dir: str,
) -> str:
    spoof_scores = scores[labels == 1]
    live_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 80)
    ax.hist(
        live_scores,
        bins=bins,
        alpha=0.6,
        color="#16A34A",
        label=f"Live  (n={len(live_scores):,})",
        density=True,
    )
    ax.hist(
        spoof_scores,
        bins=bins,
        alpha=0.6,
        color="#DC2626",
        label=f"Spoof (n={len(spoof_scores):,})",
        density=True,
    )
    ax.axvline(
        eer_thr, color="black", ls="--", lw=1.5, label=f"EER threshold = {eer_thr:.4f}"
    )

    ax.set_xlabel("Predicted Probability of Spoof")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions — Fingerprint PAD")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = os.path.join(output_dir, "pad_score_distributions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Score distributions saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── setup ────────────────────────────────────────────────────────────────
    cfg       = load_config(args.config)
    model_cfg = cfg["model"]

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nDevice: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, model_cfg, device)

    # ── transforms ───────────────────────────────────────────────────────────
    _, eval_transform = get_transforms("all")

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset = PADDataset(
        split_path=args.split_path,
        split=args.split,
        transform=eval_transform,
    )
    
    if len(dataset) == 0:
        print(f"No samples found in split '{args.split}' from '{args.split_path}'.")
        return
        
    print(f"\n{dataset}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── inference ────────────────────────────────────────────────────────────
    scores, preds, labels = collect_scores(model, loader, device, desc=args.split)
    
    # ── metrics ──────────────────────────────────────────────────────────────
    metrics = compute_pad_metrics(preds, scores, labels)

    print("\n" + "=" * 50)
    print(f"Split path: {args.split_path}")
    print(f"Split: '{args.split}'")
    print(f"Samples: {metrics['n_total']:,} (live={metrics['n_live']:,}, spoof={metrics['n_spoof']:,})")
    print("-" * 50)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"APCER    : {metrics['APCER']:.4f}  (argmax threshold)")
    print(f"BPCER    : {metrics['BPCER']:.4f}  (argmax threshold)")
    print(f"ACE      : {metrics['ACE']:.4f}  (argmax threshold)")
    print("-" * 50)
    print(f"EER      : {metrics['EER']:.4f}  (threshold={metrics['EER_threshold']:.4f})")
    print(f"AUC (ROC): {metrics['AUC']:.4f}")
    print("=" * 50)

    # ── save JSON ─────────────────────────────────────────────────────────────
    # Remove large arrays before saving JSON
    summary_metrics = {k: v for k, v in metrics.items() if k not in ("thresholds", "BPCER_curve", "APCER_curve")}

    results = {
        "checkpoint": args.checkpoint,
        "split_path": args.split_path,
        "split": args.split,
        "metrics": summary_metrics,
    }
    
    json_path = os.path.join(args.output_dir, "pad_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── plots ────────────────────────────────────────────────────────────────
    _style()
    plot_confusion_matrix(metrics, args.output_dir)
    plot_roc(metrics, args.output_dir)
    plot_det(metrics, args.output_dir)
    plot_score_dist(scores, labels, metrics["EER_threshold"], args.output_dir)

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fingerprint PAD Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config_pad.yaml",
        help="Path to the PAD YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained PAD model checkpoint (.pth)",
    )
    parser.add_argument(
        "--split-path",
        required=True,
        help="Path to the PAD split JSON file",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to evaluate (e.g. 'test', 'val')",
    )
    parser.add_argument(
        "--output-dir",
        default="results/pad/",
        help="Directory for metrics JSON and plot PNGs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="'cuda', 'cpu', or a specific device string. Auto-detected if omitted.",
    )

    main(parser.parse_args())