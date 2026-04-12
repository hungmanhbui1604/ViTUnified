import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import PADDataset
from model import ViTUnified
from schedulers import polynomial_scheduler
from transforms import get_transforms


def setup_ddp() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)
    return local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _unwrap(module):
    return module.module if isinstance(module, DDP) else module


def load_pretrained_recog(
    ckpt_path: str,
    model_cfg: dict,
    device: torch.device,
) -> ViTUnified:
    teacher = ViTUnified(
        model_name=model_cfg["model_name"],
        pretrained=False,
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(ckpt["model"])

    teacher.eval()
    if is_main():
        tqdm.write(f"[teacher] loaded from '{ckpt_path}'")

    return teacher


def pad_ce_loss(pad_outputs: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    loss = sum(F.cross_entropy(logits, labels) for logits in pad_outputs)
    return loss / len(pad_outputs)


def recog_mse_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
) -> torch.Tensor:
    student_norm = F.normalize(student_emb, p=2, dim=1)
    teacher_norm = F.normalize(teacher_emb, p=2, dim=1)
    return F.mse_loss(student_norm, teacher_norm)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:03d} [val]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        _, pad_outputs = model(images)

        ensemble_logits = torch.stack(pad_outputs, dim=0).mean(dim=0)  # (B, C)

        total_loss += pad_ce_loss(pad_outputs, labels).item()

        preds = ensemble_logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(val_loader)

    spoof_mask = all_labels == 1
    apcer = (all_preds[spoof_mask] == 0).mean() if spoof_mask.any() else 0.0

    live_mask = all_labels == 0
    bpcer = (all_preds[live_mask] == 1).mean() if live_mask.any() else 0.0

    ace = (apcer + bpcer) / 2.0

    return {
        "val/loss": avg_loss,
        "val/apcer": float(apcer),
        "val/bpcer": float(bpcer),
        "val/ace": float(ace),
    }


def load_checkpoint(
    path: str,
    model: DDP,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float]:
    start_epoch = 1
    best_ace = float("inf")

    if os.path.isfile(path):
        if is_main():
            print(f"=> Loading checkpoint '{path}'")
        ckpt_dict = torch.load(path, map_location="cpu")
        _unwrap(model).load_state_dict(ckpt_dict["model"])
        if "optimizer" in ckpt_dict:
            optimizer.load_state_dict(ckpt_dict["optimizer"])
        if "scheduler" in ckpt_dict:
            scheduler.load_state_dict(ckpt_dict["scheduler"])
        if "scaler" in ckpt_dict:
            scaler.load_state_dict(ckpt_dict["scaler"])
        if "epoch" in ckpt_dict:
            start_epoch = ckpt_dict["epoch"] + 1
        if "ace" in ckpt_dict:
            best_ace = ckpt_dict["ace"]

        if is_main():
            print(f"=> Loaded checkpoint (epoch {start_epoch - 1})")
    else:
        if is_main():
            print(f"=> No checkpoint found at '{path}'")

    return start_epoch, best_ace


def save_checkpoint(
    path: str,
    epoch: int,
    model: DDP,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    ace: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ace": ace,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"checkpoint-epoch{epoch:03d}",
            type="checkpoint",
            metadata={"epoch": epoch, "ace": ace},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        tqdm.write("  [wandb] checkpoint artifact logged")


def save_best(
    ckpt_dir: str, best_name: str, epoch: int, model: DDP, metrics: dict
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "metrics": metrics,
        },
        path,
    )
    tqdm.write(f"  [best model] ACE={metrics['val/ace']:.4f} saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name="best-model",
            type="model",
            metadata={"epoch": epoch, **metrics},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.run.summary["best_val_ace"] = metrics["val/ace"]
        wandb.run.summary["best_val_ace_epoch"] = epoch
        tqdm.write("  [wandb] best-model artifact logged")


def train_one_epoch(
    model: DDP,
    teacher: ViTUnified,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    lambda_mse: float,
) -> tuple[float, float, float]:
    model.train()

    train_sampler.set_epoch(epoch)

    total_loss_sum = ce_loss_sum = mse_loss_sum = 0.0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:03d} [train]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward passes & losses inside autocast for proper AMP ───────────
        with torch.autocast(device_type="cuda"):
            student_emb, pad_outputs = model(images)

            with torch.no_grad():
                teacher_emb, _ = teacher(images)

            # ── PAD cross-entropy loss (all heads) ───────────────────────────
            ce = pad_ce_loss(pad_outputs, labels)

            # ── Recog distillation: MSE vs frozen teacher embedding ──────────
            mse = recog_mse_loss(student_emb, teacher_emb)
            loss = ce + lambda_mse * mse

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss_sum += loss.item()
        ce_loss_sum += ce.item()
        mse_loss_sum += mse.item()

        lr_val = scheduler.get_last_lr()[0]
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{ce.item():.4f}",
            mse=f"{mse.item():.4f}",
            lr=f"{lr_val:.2e}",
        )

    n = len(train_loader)
    return total_loss_sum / n, ce_loss_sum / n, mse_loss_sum / n


def main(cfg: dict, no_wandb: bool = False, checkpoint: str = None) -> None:
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    output_cfg = cfg["output"]
    wandb_cfg = cfg["wandb"]
    loss_cfg = cfg["loss"]

    # ── DDP init ────────────────────────────────────────────────────────────
    local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    set_seed(general_cfg["seed"] + local_rank)

    if is_main():
        print(f"Device: {device}  |  world_size: {world_size}")

    # ── Wandb ──────────────────────────────────────────────────
    if is_main() and not no_wandb and wandb_cfg.get("api_key"):
        wandb.login(key=wandb_cfg["api_key"])
        wandb.init(
            project=wandb_cfg.get("project", "ViTUnified-Recognition"), config=cfg
        )

    # ── transforms ──────────────────────────────────────────────────────────
    train_transform, eval_transform = get_transforms("all")

    # ── datasets ─────────────────────────────────────────────────────────────
    train_dataset = PADDataset(
        split_path=data_cfg["split_path"],
        split="train",
        transform=train_transform,
    )
    val_dataset = PADDataset(
        split_path=data_cfg["split_path"],
        split="val",
        transform=eval_transform,
    )

    if is_main():
        print(f"\n{train_dataset}")
        print(f"{val_dataset}")

    # ── dataloaders ──────────────────────────────────────────────────────────
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=general_cfg["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        sampler=train_sampler,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    # ── student model ────────────────────────────────────────────────────────
    model = ViTUnified(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"],
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── teacher model ────────────────────────────────
    teacher = load_pretrained_recog(model_cfg["recog_ckpt"], model_cfg, device)

    # ── optimizer & scheduler & scaler ────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )
    total_iters = training_cfg["epochs"] * len(train_loader)
    scheduler = polynomial_scheduler(
        optimizer,
        total_iters=total_iters,
        base_lr=opt_cfg["lr"],
        lr_min=sched_cfg["min_lr"],
        power=sched_cfg["power"],
    )

    lambda_mse = loss_cfg["lambda_mse"]

    scaler = torch.amp.GradScaler("cuda")

    # ── training loop ────────────────────────────────────────────────────────
    start_epoch = 1
    best_ace = float("inf")

    if checkpoint is not None:
        start_epoch, best_ace = load_checkpoint(checkpoint, model, optimizer, scheduler, scaler)

    if is_main():
        if not wandb.run:
            history = {"epoch": [], "train_loss": [], "val_ace": []}

        os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

        print("\n" + "=" * 60)
        print(
            f"Starting PAD training  (GPUs: {world_size}  |  "
            f"effective batch: {training_cfg['batch_size'] * world_size})"
        )
        print("=" * 60)

    epoch_pbar = tqdm(
        range(start_epoch, training_cfg["epochs"] + 1),
        desc="Training",
        unit="epoch",
        disable=not is_main(),
    )

    for epoch in epoch_pbar:
        avg_loss, avg_ce, avg_mse = train_one_epoch(
            model,
            teacher,
            train_loader,
            train_sampler,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            lambda_mse=lambda_mse,
        )

        dist.barrier()

        if is_main():
            metrics = evaluate(_unwrap(model), val_loader, device, epoch)

            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                ace=f"{metrics['val/ace']:.4f}",
            )
            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"loss: {avg_loss:.4f} (ce={avg_ce:.4f}, mse={avg_mse:.4f}) | "
                f"val ACE: {metrics['val/ace']:.4f} "
                f"(APCER={metrics['val/apcer']:.4f}, BPCER={metrics['val/bpcer']:.4f})"
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss_epoch": avg_loss,
                        "train/ce_loss_epoch": avg_ce,
                        "train/mse_loss_epoch": avg_mse,
                        "epoch": epoch,
                        **metrics,
                    }
                )
            else:
                history["epoch"].append(epoch)
                history["train_loss"].append(avg_loss)
                history["val_ace"].append(metrics["val/ace"])

            if metrics["val/ace"] < best_ace:
                best_ace = metrics["val/ace"]
                save_best(
                    output_cfg["checkpoint_dir"],
                    output_cfg["best_model_name"],
                    epoch,
                    model,
                    metrics,
                )

            if epoch % training_cfg["checkpoint_interval"] == 0:
                ckpt_path = os.path.join(
                    output_cfg["checkpoint_dir"], f"checkpoint_epoch{epoch:03d}.pth"
                )
                save_checkpoint(
                    ckpt_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    metrics["val/ace"],
                )

        dist.barrier()

    if is_main():
        print("=" * 60)
        print(f"Training complete. Best val ACE: {best_ace:.4f}")
        print("=" * 60)

        if wandb.run is not None:
            wandb.finish()
        else:
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

            ax1.plot(history["epoch"], history["train_loss"], "g-", label="Train Loss")
            ax2.plot(history["epoch"], history["val_ace"], "b-", label="Val ACE")

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Train Loss", color="g")
            ax2.set_ylabel("Val ACE", color="b")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            plt.title("Training History")
            plot_path = os.path.join(
                output_cfg["checkpoint_dir"], "training_history.png"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved training history plot to {plot_path}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint PAD Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config_pad.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, no_wandb=args.no_wandb, checkpoint=args.checkpoint)
