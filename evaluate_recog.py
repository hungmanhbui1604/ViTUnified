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

from data import RecogEvaluationDataset, UniqueImageDataset
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
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"Checkpoint loaded from '{ckpt_path}'")
    return model


@torch.no_grad()
def collect_scores(
    model: ViTUnified,
    loader: DataLoader,
    unique_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    embed_dim = model.embed_dim
    n_unique_images = len(unique_loader.dataset)

    embeddings = torch.zeros((n_unique_images, embed_dim), device=device)
    for idxs, imgs in tqdm(unique_loader, desc="Extracting embeddings", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            emb, _ = model(imgs)
        emb = F.normalize(emb, dim=1).float()
        embeddings[idxs] = emb

    all_scores, all_labels = [], []
    for idx_a, idx_b, labels in tqdm(loader, desc="Inference", unit="batch"):
        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]

        cos_sim = (emb_a * emb_b).sum(dim=1).cpu().numpy()

        all_scores.append(cos_sim)
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


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
    fmr = np.array(metrics["FMR"])
    tar = np.array(metrics["TAR"])
    eer = metrics["EER"]
    auc = metrics["AUC"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fmr, tar, lw=2, color="#2563EB", label=f"ROC  (AUC={auc:.4f})")
    ax.axvline(eer, color="grey", ls="--", lw=1, label=f"EER={eer:.4f}")

    # EER operating point on the curve
    ax.scatter(
        [eer],
        [1 - eer],
        zorder=6,
        s=80,
        color="grey",
        marker="x",
        label=f"EER point ({eer:.4f}, {1 - eer:.4f})",
    )

    # TAR@FAR operating points
    colors = ["#DC2626", "#D97706", "#16A34A"]
    for far_t, color in zip([0.1, 0.01, 0.001], colors):
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
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(1e-5, 1.0)
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
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

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


def main(args: argparse.Namespace) -> None:
    # ── setup ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    general_cfg = cfg["general"]
    model_cfg = cfg["model"]
    evaluation_cfg = cfg["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    model = load_model(evaluation_cfg["checkpoint_path"], model_cfg, device)

    # ── dataset ──────────────────────────────────────────────────────────────
    _, eval_transform = get_transforms("all")

    dataset = RecogEvaluationDataset(
        split_path=args.split_path,
        split="test",
        n_genuine_impressions=evaluation_cfg["n_genuine_impressions"],
        n_impostor_impressions=evaluation_cfg["n_impostor_impressions"],
        impostor_mode=evaluation_cfg["impostor_mode"],
        n_impostor_subset=None
        if evaluation_cfg["n_impostor_subset"] in ("None", None)
        else evaluation_cfg["n_impostor_subset"],
        seed=general_cfg["seed"],
    )
    print(f"\n{dataset}")

    unique_dataset = UniqueImageDataset(
        idx_to_path=dataset.idx_to_path,
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=evaluation_cfg["batch_size"],
        shuffle=False,
        num_workers=evaluation_cfg["num_workers"],
        pin_memory=evaluation_cfg["pin_memory"],
    )

    unique_loader = DataLoader(
        unique_dataset,
        batch_size=evaluation_cfg["unique_batch_size"],
        shuffle=False,
        num_workers=evaluation_cfg["num_workers"],
        pin_memory=evaluation_cfg["pin_memory"],
    )

    # ── inference ────────────────────────────────────────────────────────────
    scores, labels = collect_scores(model, loader, unique_loader, device)

    # ── metrics ──────────────────────────────────────────────────────────────
    metrics = compute_recog_metrics(scores, labels)

    print("\n" + "=" * 50)
    print(f"Split path: {args.split_path}")
    print("Split: 'test'")
    print(
        f"Total pairs: {len(dataset):,} "
        f"(genuine={dataset.n_genuine:,}, impostor={dataset.n_impostor:,})"
    )
    print("-" * 50)
    print(f"EER: {metrics['EER']:.4f} (threshold={metrics['EER_threshold']:.4f})")
    print(f"AUC (ROC): {metrics['AUC']:.4f}")
    print(f"TAR @ FAR=0.1: {metrics['TAR@FAR=0.1']:.4f}")
    print(f"TAR @ FAR=0.01: {metrics['TAR@FAR=0.01']:.4f}")
    print(f"TAR @ FAR=0.001: {metrics['TAR@FAR=0.001']:.4f}")
    print("=" * 50)

    # ── save results to JSON ──────────────────────────────────────────────────
    summary = {
        "split_path": args.split_path,
        "split": "test",
        "n_pairs": len(dataset),
        "n_genuine": dataset.n_genuine,
        "n_impostor": dataset.n_impostor,
        "EER": metrics["EER"],
        "EER_threshold": metrics["EER_threshold"],
        "AUC": metrics["AUC"],
        "TAR@FAR=0.1": metrics["TAR@FAR=0.1"],
        "TAR@FAR=0.01": metrics["TAR@FAR=0.01"],
        "TAR@FAR=0.001": metrics["TAR@FAR=0.001"],
    }
    json_path = os.path.join(args.output_dir, "recog_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── plots ────────────────────────────────────────────────────────────────
    _style()
    plot_roc(metrics, args.output_dir)
    plot_det(metrics, args.output_dir)
    plot_score_dist(scores, labels, metrics["EER_threshold"], args.output_dir)

    print(f"\nAll outputs written to: {args.output_dir}")


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
        "--split-path",
        required=True,
        help="Path to the split file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/recog/",
        help="Directory for result JSON and plot PNGs",
    )

    main(parser.parse_args())
