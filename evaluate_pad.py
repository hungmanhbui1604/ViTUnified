"""
evaluate_pad.py — Fingerprint PAD Evaluation
=============================================
Loads a trained ViTUnified checkpoint and evaluates it on the LivDet PAD splits,
producing the following metrics and artefacts per sensor and overall:

Metrics (printed + saved to JSON)
  • Accuracy
  • APCER  (Attack Presentation Classification Error Rate) — spoof→live errors
  • BPCER  (Bona-fide Presentation Classification Error Rate) — live→spoof errors
  • ACE    (Average Classification Error) = (APCER + BPCER) / 2

Plots (saved as PNG)
  • Grouped bar chart of APCER / BPCER / ACE per sensor
  • Confusion matrix (overall)

Usage
-----
python evaluate_pad.py --config config_pad_default.yaml \
                       --checkpoint ckpts/pad/pad_best_ace.pth

# choose a single sensor test key
python evaluate_pad.py --config config_pad_default.yaml \
                       --checkpoint ckpts/pad/pad_best_ace.pth \
                       --sensor livdet2015_CrossMatch

# evaluate on the shared val split instead of sensor test splits
python evaluate_pad.py --config config_pad_default.yaml \
                       --checkpoint ckpts/pad/pad_best_ace.pth \
                       --split val
"""

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


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
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
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[checkpoint] loaded from '{ckpt_path}'  (epoch {ckpt.get('epoch', '?')})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Inference — collect predictions and labels for one split / sensor key
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_preds(
    model: ViTUnified,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Inference",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    preds  : (N,) predicted class (0=live, 1=spoof)
    labels : (N,) ground-truth class
    """
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc=desc, unit="batch", leave=False):
        images = images.to(device, non_blocking=True)

        _, pad_outputs = model(images)

        # ensemble: average logits across all PAD heads
        ensemble_logits = torch.stack(pad_outputs, dim=0).mean(dim=0)   # (B, C)
        preds = ensemble_logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
#  Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_pad_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    Labels: 0 = live (bona fide), 1 = spoof (attack)

    APCER  = spoof samples classified as live / total spoof samples
    BPCER  = live  samples classified as spoof / total live  samples
    ACE    = (APCER + BPCER) / 2
    """
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
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "figure.dpi"       : 150,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "font.size"        : 10,
    })


def plot_sensor_bars(
    sensor_metrics: dict[str, dict],
    overall_metrics: dict,
    output_dir: str,
) -> str:
    """Grouped bar chart: APCER / BPCER / ACE per sensor + overall."""
    _style()

    sensor_names = list(sensor_metrics.keys()) + ["OVERALL"]
    apcer_vals   = [m["APCER"] for m in sensor_metrics.values()] + [overall_metrics["APCER"]]
    bpcer_vals   = [m["BPCER"] for m in sensor_metrics.values()] + [overall_metrics["BPCER"]]
    ace_vals     = [m["ACE"]   for m in sensor_metrics.values()] + [overall_metrics["ACE"]]

    x      = np.arange(len(sensor_names))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(sensor_names) * 1.4), 5))

    bars_apcer = ax.bar(x - width, apcer_vals, width, label="APCER", color="#DC2626", alpha=0.85)
    bars_bpcer = ax.bar(x,         bpcer_vals, width, label="BPCER", color="#2563EB", alpha=0.85)
    bars_ace   = ax.bar(x + width, ace_vals,   width, label="ACE",   color="#7C3AED", alpha=0.85)

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    _label_bars(bars_apcer)
    _label_bars(bars_bpcer)
    _label_bars(bars_ace)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("livdet", "LD") for s in sensor_names],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Error Rate")
    ax.set_ylim(0, min(1.15, max(apcer_vals + bpcer_vals + ace_vals) * 1.25 + 0.05))
    ax.set_title("PAD Metrics per Sensor")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # highlight the OVERALL group
    ax.axvspan(
        x[-1] - width * 1.7,
        x[-1] + width * 1.7,
        color="gold",
        alpha=0.15,
        zorder=0,
    )

    path = os.path.join(output_dir, "pad_sensor_bars.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] sensor bar chart saved → {path}")
    return path


def plot_confusion_matrix(
    metrics: dict,
    output_dir: str,
    title: str = "Confusion Matrix (Overall)",
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


# ──────────────────────────────────────────────────────────────────────────────
#  Sensor-level evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_split(
    model: ViTUnified,
    json_path: str,
    split_key: str,
    eval_transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict | None:
    """
    Evaluates a single split key from the PAD JSON.
    Returns the metrics dict, or None if the split is empty / missing.
    """
    try:
        dataset = PADDataset(
            json_path=json_path,
            split=split_key,
            transform=eval_transform,
        )
    except AssertionError:
        return None   # split key not in JSON

    if len(dataset) == 0:
        return None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds, labels = collect_preds(model, loader, device, desc=split_key)
    return compute_pad_metrics(preds, labels)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

SENSOR_TEST_KEYS = [
    "test_livdet2011_Biometrika",
    "test_livdet2011_Digital",
    "test_livdet2011_Italdata",
    "test_livdet2011_Sagem",
    "test_livdet2013_Biometrika",
    "test_livdet2013_CrossMatch",
    "test_livdet2013_Italdata",
    "test_livdet2015_CrossMatch",
    "test_livdet2015_DigitalPersona",
    "test_livdet2015_GreenBit",
    "test_livdet2015_HiScan",
]


def main(args: argparse.Namespace) -> None:
    # ── setup ────────────────────────────────────────────────────────────────
    cfg       = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, model_cfg, device)

    # ── transforms ───────────────────────────────────────────────────────────
    _, eval_transform = get_transforms("all")

    pad_json = data_cfg["pad_splits"]

    # ── determine which splits to evaluate ───────────────────────────────────
    if args.split:
        # user specified a single split key directly (e.g. 'val')
        split_keys = [args.split]
        per_sensor = False
    elif args.sensor:
        # user specified a sensor shorthand (e.g. 'livdet2015_CrossMatch')
        split_keys = [f"test_{args.sensor}"]
        per_sensor = False
    else:
        # default: evaluate all sensor test splits
        split_keys = SENSOR_TEST_KEYS
        per_sensor = True

    # ── evaluate ─────────────────────────────────────────────────────────────
    all_preds_list, all_labels_list = [], []
    sensor_metrics: dict[str, dict] = {}

    print()
    for key in split_keys:
        m = evaluate_split(
            model, pad_json, key, eval_transform, device,
            args.batch_size, args.num_workers,
        )
        if m is None:
            print(f"  [skip] '{key}' — not found or empty in {pad_json}")
            continue

        sensor_metrics[key] = m
        print(
            f"  {key:<42s}  "
            f"n={m['n_total']:>5,}  "
            f"ACC={m['accuracy']:.4f}  "
            f"APCER={m['APCER']:.4f}  "
            f"BPCER={m['BPCER']:.4f}  "
            f"ACE={m['ACE']:.4f}"
        )

        # accumulate for overall stats
        # rebuild pred / label arrays from TP/TN/FP/FN counts
        _preds  = np.array(
            [1] * m["TP"] + [0] * m["FN"] +   # true spoof
            [0] * m["TN"] + [1] * m["FP"]      # true live
        )
        _labels = np.array(
            [1] * m["TP"] + [1] * m["FN"] +
            [0] * m["TN"] + [0] * m["FP"]
        )
        all_preds_list.append(_preds)
        all_labels_list.append(_labels)

    if not sensor_metrics:
        print("No valid splits found. Check --config / --split / --sensor.")
        return

    # ── overall metrics (pooled across all evaluated splits) ─────────────────
    all_preds  = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    overall    = compute_pad_metrics(all_preds, all_labels)

    print()
    print("=" * 60)
    print(f"  OVERALL  ({len(sensor_metrics)} sensor(s))")
    print(f"  Samples  : {overall['n_total']:,}  "
          f"(live={overall['n_live']:,}  spoof={overall['n_spoof']:,})")
    print(f"  Accuracy : {overall['accuracy']:.4f}")
    print(f"  APCER    : {overall['APCER']:.4f}")
    print(f"  BPCER    : {overall['BPCER']:.4f}")
    print(f"  ACE      : {overall['ACE']:.4f}")
    print("=" * 60)

    # ── save JSON ─────────────────────────────────────────────────────────────
    results = {
        "checkpoint" : args.checkpoint,
        "overall"    : overall,
        "per_sensor" : sensor_metrics,
    }
    json_path = os.path.join(args.output_dir, "pad_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[results] metrics saved → {json_path}")

    # ── plots ────────────────────────────────────────────────────────────────
    if per_sensor and len(sensor_metrics) > 1:
        plot_sensor_bars(sensor_metrics, overall, args.output_dir)

    plot_confusion_matrix(overall, args.output_dir, title="Confusion Matrix (Overall)")

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")


# ──────────────────────────────────────────────────────────────────────────────

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

    # ── split selection (mutually exclusive) ─────────────────────────────────
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split",
        default=None,
        help="Evaluate a single named split key (e.g. 'val', 'train')",
    )
    split_group.add_argument(
        "--sensor",
        default=None,
        help=(
            "Evaluate a single sensor by shorthand, e.g. 'livdet2015_CrossMatch'. "
            "The key 'test_<sensor>' will be looked up in the JSON."        
        ),
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