import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import wandb
from data import (
    RecogEvaluationDataset,
    RecogTrainingDataset,
    create_dataloaders,
    unify_recog_splits,
)
from loss import ArcFaceLoss
from model import ViTUnified
from transforms import get_transforms

# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────


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
    genuine_scores = scores[labels == 0]
    impostor_scores = scores[labels == 1]

    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    min_diff = float("inf")
    eer = 1.0
    best_thr = thresholds[0]

    for thr in thresholds:
        # FAR: impostors accepted (score >= thr)
        far = (impostor_scores >= thr).mean()
        # FRR: genuines rejected (score < thr)
        frr = (genuine_scores < thr).mean()
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
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
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)

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
    use_wandb: bool = True,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "arcface": arcface_loss.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_eer": best_eer,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")

    if use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"checkpoint-epoch{epoch:03d}",
            type="checkpoint",
            metadata={"epoch": epoch, "best_eer": best_eer},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        tqdm.write("  [wandb] checkpoint artifact logged")


def save_best(
    ckpt_dir: str,
    best_name: str,
    epoch: int,
    model: ViTUnified,
    arcface_loss: ArcFaceLoss,
    eer: float,
    use_wandb: bool = True,
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "arcface": arcface_loss.state_dict(),
            "eer": eer,
        },
        path,
    )
    tqdm.write(f"  [best model] EER={eer:.4f} → saved → {path}")

    if use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(
            name="best-model",
            type="model",
            metadata={"epoch": epoch, "val_eer": eer},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

        # Keep the run summary always reflecting the best EER seen so far
        wandb.run.summary["best_val_eer"] = eer
        wandb.run.summary["best_val_eer_epoch"] = epoch
        tqdm.write("  [wandb] best-model artifact logged")


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
    global_step: int,
    use_wandb: bool = True,
) -> tuple[float, int]:
    model.train()
    arcface_loss.train()

    total_loss = 0.0
    pbar = tqdm(
        train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False, unit="batch"
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward – only use the final embedding for recognition
        embeddings, _pad_logits = model(images)

        loss, _ = arcface_loss(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(arcface_loss.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        lr_val = scheduler.get_last_lr()[0]
        total_loss += loss_val
        global_step += 1

        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_val:.2e}")

        # ── per-step wandb log ──────────────────────────────────────────
        if use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/loss_step": loss_val,
                    "train/lr": lr_val,
                },
                step=global_step,
            )

    return total_loss / len(train_loader), global_step


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────


def main(cfg: dict, use_wandb: bool = True) -> None:
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    output_cfg = cfg["output"]

    set_seed(general_cfg["seed"])

    # ────────── wandb login & init ──────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    if use_wandb:
        api_key = wandb_cfg.get("api_key")
        if not api_key or api_key == "your_wandb_api_key_here":
            raise ValueError(
                "wandb.api_key is not set in the config. "
                "Add your key to config_recog.yaml or pass --no-wandb to skip."
            )
        wandb.login(key=api_key)

        wandb.init(
            project=wandb_cfg.get("project", "fingerprint-recognition"),
            config=cfg,
            save_code=True,
        )
        tqdm.write(f"[wandb] run: {wandb.run.url}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = output_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ────────── transforms ──────────────────────────────────────────────────
    train_transform, eval_transform = get_transforms("all")

    # ────────── datasets & dataloaders ──────────────────────────────────────
    if not os.path.exists(data_cfg["splits"]):
        unify_recog_splits(
            data_root=data_cfg["data_root"],
            datasets=data_cfg["datasets"],
            output_path=data_cfg["splits"],
        )

    train_dataset = RecogTrainingDataset(
        json_path=data_cfg["splits"],
        transform=train_transform,
    )
    print(f"\n{train_dataset}")

    val_dataset = RecogEvaluationDataset(
        json_path=data_cfg["splits"],
        split="val",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=data_cfg["n_impostor_subset"],
        transform=eval_transform,
        seed=general_cfg["seed"],
    )
    print(f"{val_dataset}")

    # Log dataset sizes as wandb summary stats
    if use_wandb and wandb.run is not None:
        wandb.run.summary.update(
            {
                "dataset/train_samples": len(train_dataset),
                "dataset/train_ids": len(train_dataset.key_to_label),
                "dataset/val_pairs": len(val_dataset),
            }
        )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    # ────────── model ──────────────────────────────────────────────────
    model = ViTUnified(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"],
    ).to(device)

    if use_wandb and wandb.run is not None:
        wandb.watch(model, log="gradients", log_freq=100)

    # ────────── loss ──────────────────────────────────────────────────
    num_ids = len(train_dataset.key_to_label)
    embed_dim = model.embed_dim
    arcface_loss = ArcFaceLoss(
        embed_dim=embed_dim,
        num_classes=num_ids,
        margin=loss_cfg["margin"],
        scale=loss_cfg["scale"],
    ).to(device)

    # ────────── optimizer & scheduler ──────────────────────────────────
    optimizer = AdamW(
        list(model.parameters()) + list(arcface_loss.parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )

    total_steps = training_cfg["epochs"] * len(train_loader)
    scheduler = make_polynomial_scheduler(
        optimizer,
        total_steps=total_steps,
        lr_start=opt_cfg["lr"],
        lr_min=sched_cfg["min_lr"],
        power=sched_cfg["power"],
    )

    # ────────── training ──────────────────────────────────────────────
    best_eer = float("inf")
    ckpt_interval = training_cfg["checkpoint_interval"]
    best_model_name = output_cfg["best_model_name"]
    global_step = 0

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    epoch_pbar = tqdm(
        range(1, training_cfg["epochs"] + 1), desc="Training", unit="epoch"
    )
    for epoch in epoch_pbar:
        avg_loss, global_step = train_one_epoch(
            model,
            arcface_loss,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            global_step,
            use_wandb,
        )

        eer, thr = evaluate(model, val_loader, device, epoch)

        epoch_pbar.set_postfix(
            loss=f"{avg_loss:.4f}", eer=f"{eer:.4f}", thr=f"{thr:.4f}"
        )
        tqdm.write(
            f"Epoch {epoch:03d} | avg loss: {avg_loss:.4f} | "
            f"val EER: {eer:.4f}  (threshold={thr:.4f})"
        )

        # ── per-epoch wandb log ─────────────────────────────────────────
        if use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/loss_epoch": avg_loss,
                    "val/eer": eer,
                    "val/threshold": thr,
                    "epoch": epoch,
                },
                step=global_step,
            )

        # Save best model
        if eer < best_eer:
            best_eer = eer
            save_best(
                ckpt_dir,
                best_model_name,
                epoch,
                model,
                arcface_loss,
                eer,
                use_wandb,
            )

        # Periodic checkpoint
        if epoch % ckpt_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:03d}.pth")
            save_checkpoint(
                ckpt_path,
                epoch,
                model,
                arcface_loss,
                optimizer,
                scheduler,
                best_eer,
                use_wandb,
            )

    print("=" * 60)
    print(f"Training complete. Best val EER: {best_eer:.4f}")
    print("=" * 60)

    if use_wandb and wandb.run is not None:
        wandb.finish()


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint Recognition Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config_recog.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, use_wandb=not args.no_wandb)
