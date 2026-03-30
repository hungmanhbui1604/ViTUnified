import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from data import PADDataset, create_LivDet_PAD_splits, create_LivDet_recog_splits
from model import ViTUnified
from schedulers import polynomial_scheduler
from transforms import get_transforms

# ────────────────────────────────────────────────────────────
#  DDP helpers
# ────────────────────────────────────────────────────────────


def setup_ddp() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# ────────────────────────────────────────────────────────────
#  Misc helpers
# ────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────
#  Load pretrained recognition model (teacher, frozen)
# ────────────────────────────────────────────────────────────


def load_pretrained_recog(
    ckpt_path: str,
    model_cfg: dict,
    device: torch.device,
) -> ViTUnified:
    """
    Loads the best recognition checkpoint and returns a fully frozen model
    that serves as teacher for the MSE distillation loss on the final embedding.
    """
    teacher = ViTUnified(
        model_name=model_cfg["model_name"],
        pretrained=False,  # weights come from checkpoint
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(ckpt["model"])

    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    if is_main():
        tqdm.write(
            f"[teacher] loaded from '{ckpt_path}' (epoch {ckpt.get('epoch', '?')})"
        )

    return teacher


# ────────────────────────────────────────────────────────────
#  Loss helpers
# ────────────────────────────────────────────────────────────


def pad_ce_loss(pad_outputs: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """
    Average cross-entropy loss over all PAD head outputs.
    pad_outputs: list of (B, num_classes) tensors, one per transformer block.
    labels:      (B,) int64 tensor  — 0=live, 1=spoof.
    """
    loss = sum(F.cross_entropy(logits, labels) for logits in pad_outputs)
    return loss / len(pad_outputs)


def recog_mse_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between L2-normalised student and teacher final embeddings.
    Penalises forgetting recognition structure while learning PAD.
    """
    student_norm = F.normalize(student_emb, p=2, dim=1)
    teacher_norm = F.normalize(teacher_emb, p=2, dim=1)
    return F.mse_loss(student_norm, teacher_norm)


# ────────────────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: DDP,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """
    Returns average CE loss and ACE (Average Classification Error).
    ACE = (APCER + BPCER) / 2, where:
      APCER = spoof samples misclassified as live / total spoof samples
      BPCER = live  samples misclassified as spoof / total live  samples
    Logits are ensembled by averaging across all PAD heads.
    """
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

        # Ensemble: average logits across all PAD heads
        ensemble_logits = torch.stack(pad_outputs, dim=0).mean(dim=0)  # (B, C)

        total_loss += pad_ce_loss(pad_outputs, labels).item()

        preds = ensemble_logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(val_loader)

    # APCER: spoof (label=1) misclassified as live (pred=0)
    spoof_mask = all_labels == 1
    apcer = (all_preds[spoof_mask] == 0).mean() if spoof_mask.any() else 0.0

    # BPCER: live (label=0) misclassified as spoof (pred=1)
    live_mask = all_labels == 0
    bpcer = (all_preds[live_mask] == 1).mean() if live_mask.any() else 0.0

    ace = (apcer + bpcer) / 2.0

    return {
        "val/loss": avg_loss,
        "val/apcer": float(apcer),
        "val/bpcer": float(bpcer),
        "val/ace": float(ace),
    }


# ────────────────────────────────────────────────────────────
#  Checkpoint helpers  (rank-0 only)
# ────────────────────────────────────────────────────────────


def save_checkpoint(
    path: str,
    epoch: int,
    model: DDP,
    optimizer: AdamW,
    scheduler: LambdaLR,
    best_ace: float,
    use_wandb: bool = True,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_ace": best_ace,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")

    if use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"pad-checkpoint-epoch{epoch:03d}",
            type="checkpoint",
            metadata={"epoch": epoch, "best_ace": best_ace},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)


def save_best(
    ckpt_dir: str,
    best_name: str,
    epoch: int,
    model: DDP,
    metrics: dict,
    use_wandb: bool = True,
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
    tqdm.write(f"  [best model] ACE={metrics['val/ace']:.4f} → saved → {path}")

    if use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(
            name="pad-best-model",
            type="model",
            metadata={"epoch": epoch, **metrics},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.run.summary["best_val_ace"] = metrics["val/ace"]
        wandb.run.summary["best_val_ace_epoch"] = epoch


# ────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────


def train_one_epoch(
    model: DDP,
    teacher: ViTUnified,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
    global_step: int,
    lambda_mse: float = 1.0,
    use_wandb: bool = True,
) -> tuple[float, float, float, int]:
    """
    Returns (avg_total_loss, avg_ce_loss, avg_mse_loss, global_step).
    """
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

        # ── Student forward ──────────────────────────────────────────────────
        student_emb, pad_outputs = model(images)

        # ── PAD cross-entropy loss (all heads) ───────────────────────────────
        ce = pad_ce_loss(pad_outputs, labels)

        # ── Recog distillation: MSE vs frozen teacher embedding ──────────────
        with torch.no_grad():
            teacher_emb, _ = teacher(images)

        mse = recog_mse_loss(student_emb, teacher_emb)

        loss = ce + lambda_mse * mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss_sum += loss.item()
        ce_loss_sum += ce.item()
        mse_loss_sum += mse.item()
        global_step += 1

        lr_val = scheduler.get_last_lr()[0]
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{ce.item():.4f}",
            mse=f"{mse.item():.4f}",
            lr=f"{lr_val:.2e}",
        )

        if use_wandb and is_main() and wandb.run is not None:
            wandb.log(
                {
                    "train/loss_step": loss.item(),
                    "train/ce_loss_step": ce.item(),
                    "train/mse_loss_step": mse.item(),
                    "train/lr": lr_val,
                },
                step=global_step,
            )

    n = len(train_loader)
    return total_loss_sum / n, ce_loss_sum / n, mse_loss_sum / n, global_step


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────


def main(cfg: dict, use_wandb: bool = True) -> None:
    local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    output_cfg = cfg["output"]
    wandb_cfg = cfg.get("wandb", {})

    set_seed(general_cfg["seed"] + local_rank)

    # ── wandb (rank-0 only) ──────────────────────────────────────────────────
    if use_wandb and is_main():
        api_key = wandb_cfg.get("api_key")
        if not api_key or api_key == "your_wandb_api_key_here":
            raise ValueError(
                "wandb.api_key is not set. "
                "Set it in the config or pass --no-wandb to skip."
            )
        wandb.login(key=api_key)
        wandb.init(
            project=wandb_cfg.get("project", "fingerprint-pad"),
            tags=["pad"],
            config=cfg,
            save_code=True,
        )
        tqdm.write(f"[wandb] run: {wandb.run.url}")

    if is_main():
        print(f"Device: {device}  |  world_size: {world_size}")

    ckpt_dir = output_cfg["checkpoint_dir"]
    if is_main():
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── transforms ──────────────────────────────────────────────────────────
    train_transform, eval_transform = get_transforms("all")

    # ── splits (rank-0 creates them if needed, others wait) ─────────────────
    pad_json = data_cfg["pad_splits"]
    if is_main() and not os.path.exists(pad_json):
        if not os.path.exists(data_cfg["livdet_recog_splits"]):
            create_LivDet_recog_splits(
                data_root=data_cfg["data_root"],
                output_path=data_cfg["livdet_recog_splits"],
                seed=general_cfg["seed"]
            )
        create_LivDet_PAD_splits(
            data_root=data_cfg["data_root"],
            recog_json_path=data_cfg["livdet_recog_splits"],
            output_path=pad_json,
            seed=general_cfg["seed"]
        )
    dist.barrier()

    # ── datasets ─────────────────────────────────────────────────────────────
    train_dataset = PADDataset(
        json_path=pad_json,
        split="train",
        transform=train_transform,
    )
    val_dataset = PADDataset(
        json_path=pad_json,
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

    if use_wandb and is_main() and wandb.run is not None:
        wandb.run.summary.update(
            {
                "dataset/train_samples": len(train_dataset),
                "dataset/val_samples": len(val_dataset),
                "training/world_size": world_size,
                "training/per_gpu_batch": training_cfg["batch_size"],
                "training/effective_batch": training_cfg["batch_size"] * world_size,
            }
        )

    # ── student model ────────────────────────────────────────────────────────
    model = ViTUnified(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"],
    ).to(device)

    # Optionally warm-start the backbone from a recog checkpoint
    recog_ckpt = output_cfg.get("recog_checkpoint")
    if recog_ckpt and os.path.exists(recog_ckpt):
        ckpt = torch.load(recog_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if is_main():
            tqdm.write(
                f"[student] warm-started from '{recog_ckpt}'  "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── teacher model (frozen recog backbone) ────────────────────────────────
    teacher_ckpt = output_cfg["teacher_checkpoint"]
    teacher = load_pretrained_recog(teacher_ckpt, model_cfg, device)

    if use_wandb and is_main() and wandb.run is not None:
        wandb.watch(model, log="parameters", log_freq=100)

    # ── optimizer & scheduler ────────────────────────────────────────────────
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

    lambda_mse = training_cfg.get("lambda_mse", 1.0)

    # ── training loop ────────────────────────────────────────────────────────
    best_ace = float("inf")
    ckpt_interval = training_cfg["checkpoint_interval"]
    best_name = output_cfg["best_model_name"]
    global_step = 0

    if is_main():
        print("\n" + "=" * 60)
        print(
            f"Starting PAD training  (GPUs: {world_size}  |  "
            f"effective batch: {training_cfg['batch_size'] * world_size})"
        )
        print(f"  lambda_mse = {lambda_mse}")
        print("=" * 60)

    epoch_pbar = tqdm(
        range(1, training_cfg["epochs"] + 1),
        desc="Training",
        unit="epoch",
        disable=not is_main(),
    )

    for epoch in epoch_pbar:
        avg_loss, avg_ce, avg_mse, global_step = train_one_epoch(
            model,
            teacher,
            train_loader,
            train_sampler,
            optimizer,
            scheduler,
            device,
            epoch,
            global_step,
            lambda_mse=lambda_mse,
            use_wandb=use_wandb,
        )

        dist.barrier()

        metrics = evaluate(model, val_loader, device, epoch)

        if is_main():
            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                ace=f"{metrics['val/ace']:.4f}",
            )
            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"loss: {avg_loss:.4f} (ce={avg_ce:.4f}, mse={avg_mse:.4f}) | "
                f"val ACE: {metrics['val/ace']:.4f}  "
                f"(APCER={metrics['val/apcer']:.4f}  BPCER={metrics['val/bpcer']:.4f})"
            )

            if use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "train/loss_epoch": avg_loss,
                        "train/ce_loss_epoch": avg_ce,
                        "train/mse_loss_epoch": avg_mse,
                        "epoch": epoch,
                        **metrics,
                    },
                    step=global_step,
                )

            if metrics["val/ace"] < best_ace:
                best_ace = metrics["val/ace"]
                save_best(ckpt_dir, best_name, epoch, model, metrics, use_wandb)

            if epoch % ckpt_interval == 0:
                ckpt_path = os.path.join(
                    ckpt_dir, f"pad_checkpoint_epoch{epoch:03d}.pth"
                )
                save_checkpoint(
                    ckpt_path, epoch, model, optimizer, scheduler, best_ace, use_wandb
                )

        dist.barrier()

    if is_main():
        print("=" * 60)
        print(f"Training complete. Best val ACE: {best_ace:.4f}")
        print("=" * 60)

        if use_wandb and wandb.run is not None:
            wandb.finish()

    cleanup_ddp()


# ────────────────────────────────────────────────────────────

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
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, use_wandb=not args.no_wandb)
