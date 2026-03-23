import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from tqdm import tqdm

from data import (
    RecogEvaluationDataset,
    RecogTrainingDataset,
    create_dataloaders,
    unify_splits
)
from loss import ArcFaceLoss
from model import ViTUnified


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_polynomial_scheduler(
    optimizer: AdamW,
    total_steps: int,
    lr_start: float,
    lr_min: float,
    power: float,
) -> LambdaLR:
    """
        lr(t) = (lr_start - lr_min) * (1 - t / total_steps) ** power + lr_min
    """
    def _lr_lambda(current_step: int) -> float:
        if current_step >= total_steps:
            return lr_min / lr_start
        decay = (1 - current_step / total_steps) ** power
        return (lr_start - lr_min) * decay / lr_start + lr_min / lr_start

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ──────────────────────────────────────────────────────────────
#  EER computation
# ──────────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    genuine_scores  = scores[labels == 0]
    impostor_scores = scores[labels == 1]

    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    min_diff   = float("inf")
    eer        = 1.0
    best_thr   = thresholds[0]

    for thr in thresholds:
        # FAR: impostors accepted (score >= thr)
        far = (impostor_scores >= thr).mean()
        # FRR: genuines rejected (score < thr)
        frr = (genuine_scores  <  thr).mean()
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer      = (far + frr) / 2.0
            best_thr = thr

    return float(eer), float(best_thr)


# ──────────────────────────────────────────────────────────────
#  Evaluation loop
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: ViTUnified,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    all_scores, all_labels = [], []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]", leave=False, unit="batch")
    for img_a, img_b, labels in pbar:
        img_a  = img_a.to(device, non_blocking=True)
        img_b  = img_b.to(device, non_blocking=True)

        emb_a, _ = model(img_a)
        emb_b, _ = model(img_b)

        emb_a = F.normalize(emb_a, p=2, dim=1)
        emb_b = F.normalize(emb_b, p=2, dim=1)

        cos_sim = (emb_a * emb_b).sum(dim=1).cpu().numpy()
        all_scores.append(cos_sim)
        all_labels.append(labels.numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    return compute_eer(all_scores, all_labels)


# ──────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    epoch: int,
    model: ViTUnified,
    arcface_loss: ArcFaceLoss,
    optimizer: AdamW,
    scheduler: LambdaLR,
    best_eer: float,
) -> None:
    torch.save(
        {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "arcface":    arcface_loss.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_eer":   best_eer,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")


def save_best(
    ckpt_dir: str,
    best_name: str,
    epoch: int,
    model: ViTUnified,
    arcface_loss: ArcFaceLoss,
    eer: float,
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch":    epoch,
            "model":    model.state_dict(),
            "arcface":  arcface_loss.state_dict(),
            "eer":      eer,
        },
        path,
    )
    tqdm.write(f"  [best model] EER={eer:.4f} → saved → {path}")


# ──────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ViTUnified,
    arcface_loss: ArcFaceLoss,
    train_loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    arcface_loss.train()

    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False, unit="batch")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward – only use the final embedding for recognition
        embeddings, _pad_logits = model(images)

        loss = arcface_loss(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(arcface_loss.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(train_loader)


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── transforms ────────────────────────────────────────────
    img_size = cfg["data"]["image_size"]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── datasets ──────────────────────────────────────────────
    if not os.path.exists(cfg["data"]["splits"]):
        unify_splits(
            data_root=cfg["data"]["data_root"],
            datasets=cfg["data"]["datasets"],
            output_path=cfg["data"]["splits"],
            split_ratio=cfg["data"]["split_ratio"],
            seed=cfg["training"]["seed"],
        )

    train_dataset = RecogTrainingDataset(
        json_path=cfg["data"]["splits"],
        transform=train_transform,
    )
    print(f"\n{train_dataset}")

    val_dataset = RecogEvaluationDataset(
        json_path=cfg["data"]["splits"],
        split="val",
        n_genuine_impressions=cfg["data"]["n_genuine_impressions"],
        n_impostor_impressions=cfg["data"]["n_impostor_impressions"],
        transform=eval_transform,
    )
    print(f"{val_dataset}")

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )

    # ── model ─────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = ViTUnified(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        in_channels=model_cfg["in_channels"],
        num_classes_pad=model_cfg["num_classes_pad"],
    ).to(device)

    num_identities = len(train_dataset.key_to_label)
    embedding_dim  = model.embed_dim
    print(f"\nEmbedding dim : {embedding_dim}")
    print(f"Num identities: {num_identities}")

    # ── loss ──────────────────────────────────────────────────
    loss_cfg = cfg["loss"]
    arcface_loss = ArcFaceLoss(
        embedding_dim=embedding_dim,
        num_identities=num_identities,
        margin=loss_cfg["margin"],
        scale=loss_cfg["scale"],
    ).to(device)

    # ── optimizer ─────────────────────────────────────────────
    opt_cfg   = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    epochs    = cfg["training"]["epochs"]

    optimizer = AdamW(
        list(model.parameters()) + list(arcface_loss.parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )

    total_steps = epochs * len(train_loader)
    scheduler   = make_polynomial_scheduler(
        optimizer,
        total_steps=total_steps,
        lr_start=opt_cfg["lr"],
        lr_min=sched_cfg["min_lr"],
        power=sched_cfg["power"],
    )

    # ── training ──────────────────────────────────────────────
    best_eer        = float("inf")
    ckpt_interval   = cfg["training"]["checkpoint_interval"]
    best_model_name = cfg["output"]["best_model_name"]

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        avg_loss = train_one_epoch(
            model, arcface_loss, train_loader,
            optimizer, scheduler, device, epoch,
        )

        eer, thr = evaluate(model, val_loader, device, epoch)

        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}", eer=f"{eer:.4f}", thr=f"{thr:.4f}")
        tqdm.write(
            f"Epoch {epoch:03d} | avg loss: {avg_loss:.4f} | "
            f"val EER: {eer:.4f}  (threshold={thr:.4f})"
        )

        # Save best model
        if eer < best_eer:
            best_eer = eer
            save_best(ckpt_dir, best_model_name, epoch, model, arcface_loss, eer)

        # Periodic checkpoint
        if epoch % ckpt_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:03d}.pth")
            save_checkpoint(
                ckpt_path, epoch, model, arcface_loss,
                optimizer, scheduler, best_eer,
            )

    print("=" * 60)
    print(f"Training complete. Best val EER: {best_eer:.4f}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint Recognition Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config_recog_default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)