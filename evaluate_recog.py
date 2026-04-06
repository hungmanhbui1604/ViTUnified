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

from data import RecogEvaluationDataset
from model import ViTUnified
from transforms import get_transforms
from metrics import compute_roc


# ──────────────────────────────────────────────────────────────────────────────
#  Config / checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────


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
    state = ckpt.get("model", ckpt)  # handle bare state-dicts too
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[checkpoint] loaded from '{ckpt_path}'  (epoch {ckpt.get('epoch', '?')})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Inference — collect cosine-similarity scores and labels
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_scores(
    model: ViTUnified,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_scores, all_labels = [], []

    for img_a, img_b, labels in tqdm(loader, desc="Inference", unit="batch"):
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)

        emb_a, _ = model(img_a)
        emb_b, _ = model(img_b)

        emb_a = F.normalize(emb_a, p=2, dim=1)
        emb_b = F.normalize(emb_b, p=2, dim=1)

        cos_sim = (emb_a * emb_b).sum(dim=1).cpu().numpy()
        all_scores.append(cos_sim)
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def plot_roc(metrics: dict, output_dir: str) -> str:
    _style()
    fmr = np.array(metrics["FMR"])
    tar = np.array(metrics["TAR"])
    eer = metrics["EER"]
    auc = metrics["AUC"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fmr, tar, lw=2, color="#2563EB", label=f"ROC  (AUC={auc:.4f})")
    ax.axline((eer, 0), (eer, 1), color="grey", ls="--", lw=1, label=f"EER={eer:.4f}")

    # TAR@FAR operating points
    colors = ["#DC2626", "#D97706", "#16A34A"]
    for far_t, color in zip([0.01, 0.001, 0.0001], colors):
        key = f"TAR@FAR={far_t}"
        tar_val = metrics[key]
        ax.scatter(
            [far_t],
            [tar_val],
            zorder=5,
            s=60,
            color=color,
            label=f"TAR@FAR={far_t:.4f} = {tar_val:.4f}",
        )

    ax.set_xlabel("FMR  (False Match Rate)")
    ax.set_ylabel("TAR  (True Accept Rate)")
    ax.set_title("ROC Curve — Fingerprint Recognition")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.3)

    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] ROC curve saved → {path}")
    return path


def plot_det(metrics: dict, output_dir: str) -> str:
    """DET curve: FNMR vs FMR on a log-log scale (ISO/IEC 19795-1 style)."""
    _style()
    fmr = np.array(metrics["FMR"])
    fnmr = np.array(metrics["FNMR"])
    eer = metrics["EER"]

    # clip to avoid log(0)
    fmr_plot = np.clip(fmr, 1e-5, 1.0)
    fnmr_plot = np.clip(fnmr, 1e-5, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fmr_plot, fnmr_plot, lw=2, color="#7C3AED", label="DET")
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
    ax.set_xlabel("FMR  (False Match Rate)")
    ax.set_ylabel("FNMR  (False Non-Match Rate)")
    ax.set_title("DET Curve — Fingerprint Recognition")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    path = os.path.join(output_dir, "det_curve.png")
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
    _style()
    genuine_scores = scores[labels == 0]
    impostor_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(scores.min(), scores.max(), 80)
    ax.hist(
        genuine_scores,
        bins=bins,
        alpha=0.6,
        color="#16A34A",
        label=f"Genuine  (n={len(genuine_scores):,})",
        density=True,
    )
    ax.hist(
        impostor_scores,
        bins=bins,
        alpha=0.6,
        color="#DC2626",
        label=f"Impostor (n={len(impostor_scores):,})",
        density=True,
    )
    ax.axvline(
        eer_thr, color="black", ls="--", lw=1.5, label=f"EER threshold = {eer_thr:.4f}"
    )

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions — Fingerprint Recognition")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = os.path.join(output_dir, "score_distributions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Score distributions saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    # ── setup ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    general_cfg = cfg["general"]
    model_cfg = cfg["model"]
    evaluation_cfg = cfg["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    model = load_model(evaluation_cfg["checkpoint_path"], model_cfg, device)

    # ── dataset ──────────────────────────────────────────────────────────────
    _, eval_transform = get_transforms("all")

    dataset = RecogEvaluationDataset(
        splits_path=args.split_file,
        split=args.split,
        n_genuine_impressions=evaluation_cfg["n_genuine_impressions"],
        n_impostor_impressions=evaluation_cfg["n_impostor_impressions"],
        impostor_mode=evaluation_cfg["impostor_mode"],
        n_impostor_subset=evaluation_cfg["n_impostor_subset"],
        transform=eval_transform,
        seed=general_cfg["seed"],
    )
    print(f"\n{dataset}\n")

    loader = DataLoader(
        dataset,
        batch_size=evaluation_cfg["batch_size"],
        shuffle=False,
        num_workers=evaluation_cfg["num_workers"],
        pin_memory=evaluation_cfg["pin_memory"],
    )

    # ── inference ────────────────────────────────────────────────────────────
    scores, labels = collect_scores(model, loader, device)

    # ── metrics ──────────────────────────────────────────────────────────────
    metrics = compute_roc(scores, labels, n_thresholds=2000)

    print("\n" + "=" * 55)
    print(f"  Split           : {args.split}")
    print(
        f"  Total pairs     : {len(scores):,}"
        f"  (genuine={int((labels == 0).sum()):,}  impostor={int((labels == 1).sum()):,})"
    )
    print("-" * 55)
    print(
        f"  EER             : {metrics['EER']:.4f}  (threshold={metrics['EER_threshold']:.4f})"
    )
    print(f"  AUC (ROC)       : {metrics['AUC']:.4f}")
    print(f"  TAR @ FAR=0.01  : {metrics['TAR@FAR=0.01']:.4f}")
    print(f"  TAR @ FAR=0.001 : {metrics['TAR@FAR=0.001']:.4f}")
    print(f"  TAR @ FAR=0.0001: {metrics['TAR@FAR=0.0001']:.4f}")
    print("=" * 55)

    # ── save JSON (exclude the heavy curve arrays unless requested) ───────────
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_pairs": int(len(scores)),
        "n_genuine": int((labels == 0).sum()),
        "n_impostor": int((labels == 1).sum()),
        "EER": metrics["EER"],
        "EER_threshold": metrics["EER_threshold"],
        "AUC": metrics["AUC"],
        "TAR@FAR=0.01": metrics["TAR@FAR=0.01"],
        "TAR@FAR=0.001": metrics["TAR@FAR=0.001"],
        "TAR@FAR=0.0001": metrics["TAR@FAR=0.0001"],
    }
    json_path = os.path.join(args.output_dir, "recog_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[results] metrics saved → {json_path}")

    curves_path = os.path.join(args.output_dir, "recog_curves.json")
    with open(curves_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[results] full curve data saved → {curves_path}")

    # ── plots ────────────────────────────────────────────────────────────────
    plot_roc(metrics, args.output_dir)
    plot_det(metrics, args.output_dir)
    plot_score_dist(scores, labels, metrics["EER_threshold"], args.output_dir)

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fingerprint Recognition Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config_recog.yaml",
        help="Path to the recognition YAML config",
    )
    parser.add_argument(
        "--split-file",
        required=True,
        help="Path to the split file",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split to evaluate on: 'val' or a test key",
    )
    parser.add_argument(
        "--output-dir",
        default="results/recog/",
        help="Directory for metrics JSON and plot PNGs",
    )

    main(parser.parse_args())
