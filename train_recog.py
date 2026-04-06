import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from data import RecogEvaluationDataset, RecogTrainingDataset
from loss import ArcFaceLoss
from model import ViTUnified
from schedulers import polynomial_scheduler
from transforms import get_transforms

# ────────────────────────────────────────────────────────────
#  DDP helpers
# ────────────────────────────────────────────────────────────


def setup_ddp() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)
    return local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


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


# ──────────────────────────────────────────────────────────────
#  Evaluation loop
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_eer_gpu(pos_hist: torch.Tensor, neg_hist: torch.Tensor, bins: torch.Tensor):

    # cumulative
    pos_cum = torch.cumsum(pos_hist.flip(0), dim=0).flip(0)
    neg_cum = torch.cumsum(neg_hist, dim=0)

    pos_total = pos_hist.sum()
    neg_total = neg_hist.sum()

    fnr = pos_cum / pos_total
    fpr = neg_cum / neg_total

    diff = torch.abs(fpr - fnr)
    idx = torch.argmin(diff)

    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = bins[idx]

    return eer.item(), thr.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.eval()

    n_bins = 10000
    min_val, max_val = -1.0, 1.0

    # 🔥 histogram trên GPU
    pos_hist = torch.zeros(n_bins, device=device)
    neg_hist = torch.zeros(n_bins, device=device)

    bin_edges = torch.linspace(min_val, max_val, steps=n_bins + 1, device=device)

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:03d} [val]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for img_a, img_b, labels in pbar:
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        imgs = torch.cat([img_a, img_b], dim=0)

        with torch.autocast(device_type="cuda"):
            emb, _ = model(imgs)

        emb = F.normalize(emb, dim=1)
        emb_a, emb_b = torch.chunk(emb, 2, dim=0)

        cos_sim = (emb_a * emb_b).sum(dim=1)

        pos_scores = cos_sim[labels == 1]
        neg_scores = cos_sim[labels == 0]

        if pos_scores.numel() > 0:
            pos_hist += torch.histc(pos_scores, bins=n_bins, min=min_val, max=max_val)

        if neg_scores.numel() > 0:
            neg_hist += torch.histc(neg_scores, bins=n_bins, min=min_val, max=max_val)
    
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(pos_hist, op=dist.ReduceOp.SUM)
        dist.all_reduce(neg_hist, op=dist.ReduceOp.SUM)

    eer, thr = compute_eer_gpu(pos_hist, neg_hist, bin_edges)

    return eer, thr


# ──────────────────────────────────────────────────────────────
#  Checkpoint helpers  (rank-0 only)
# ──────────────────────────────────────────────────────────────


def _unwrap(module):
    """Return the underlying module regardless of DDP wrapping."""
    return module.module if isinstance(module, DDP) else module


def save_checkpoint(
    path: str,
    epoch: int,
    model: DDP,
    arcface_loss: DDP,
    optimizer: AdamW,
    scheduler: LambdaLR,
    best_eer: float,
    use_wandb: bool = True,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "arcface": _unwrap(arcface_loss).state_dict(),
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
    model: DDP,
    arcface_loss: DDP,
    eer: float,
    use_wandb: bool = True,
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "arcface": _unwrap(arcface_loss).state_dict(),
            "eer": eer,
        },
        path,
    )
    tqdm.write(f"  [best model] EER={eer:.4f} → saved → {path}")

    if use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(
            name="recog-best-model",
            type="model",
            metadata={"epoch": epoch, "val_eer": eer},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.run.summary["best_val_eer"] = eer
        wandb.run.summary["best_val_eer_epoch"] = epoch
        tqdm.write("  [wandb] best-model artifact logged")


# ──────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────


def train_one_epoch(
    model: DDP,
    arcface_loss: DDP,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    use_wandb: bool = True,
) -> tuple[float, int]:
    model.train()
    arcface_loss.train()

    # DistributedSampler must be re-seeded each epoch for proper shuffling
    train_sampler.set_epoch(epoch)

    total_loss = 0.0
    all_params = list(model.parameters()) + list(arcface_loss.parameters())

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

        with torch.autocast(device_type="cuda"):
            embeddings, _ = model(images)
            loss, _ = arcface_loss(embeddings, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            all_params,
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        lr_val = scheduler.get_last_lr()[0]
        total_loss += loss_val
        global_step += 1

        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_val:.2e}")

        if use_wandb and is_main() and wandb.run is not None:
            wandb.log(
                {"train/loss_step": loss_val, "train/lr": lr_val},
                step=global_step,
            )

    return total_loss / len(train_loader), global_step


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────


def main(cfg: dict, use_wandb: bool = True) -> None:
    # ── DDP init ────────────────────────────────────────────────────────────
    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)

    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    output_cfg = cfg["output"]
    wandb_cfg = cfg.get("wandb", {})
    evaluation_cfg = cfg["evaluation"]

    set_seed(general_cfg["seed"] + local_rank)  # unique seed per rank

    # ── wandb (rank-0 only) ─────────────────────────────────────────────────
    if use_wandb and is_main():
        api_key = wandb_cfg.get("api_key")
        if not api_key or api_key == "your_wandb_api_key_here":
            raise ValueError(
                "wandb.api_key is not set in the config. "
                "Add your key to config_recog.yaml or pass --no-wandb to skip."
            )
        wandb.login(key=api_key)
        wandb.init(
            project=wandb_cfg.get("project", "fingerprint-recognition"),
            tags=["recog"],
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

    # ── datasets ─────────────────────────────────────────────────────────────
    train_dataset = RecogTrainingDataset(
        splits_path=data_cfg["splits_path"],
        transform=train_transform,
    )
    val_dataset = RecogEvaluationDataset(
        splits_path=data_cfg["splits_path"],
        split="val",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=data_cfg["n_impostor_subset"],
        transform=eval_transform,
        seed=general_cfg["seed"],
    )

    if is_main():
        print(f"\n{train_dataset}")
        print(f"{val_dataset}")

    # ── dataloaders ───────────────────────────────────────────────────────────
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

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=evaluation_cfg["batch_size"],
        sampler=val_sampler,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    if use_wandb and is_main() and wandb.run is not None:
        wandb.run.summary.update(
            {
                "dataset/train_samples": len(train_dataset),
                "dataset/train_ids": len(train_dataset.id_to_label),
                "dataset/val_pairs": len(val_dataset),
                "training/world_size": world_size,
                "training/per_gpu_batch": training_cfg["batch_size"],
                "training/effective_batch": training_cfg["batch_size"] * world_size,
            }
        )

    # ── model ─────────────────────────────────────────────────────────────────
    model = ViTUnified(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
        pad_dropout=model_cfg["pad_dropout"],
    ).to(device)
    model._freeze_pad_heads()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if use_wandb and is_main() and wandb.run is not None:
        wandb.watch(model, log="parameters", log_freq=100)

    # ── loss ──────────────────────────────────────────────────────────────────
    num_ids = len(train_dataset.id_to_label)
    embed_dim = _unwrap(model).embed_dim
    arcface_loss = ArcFaceLoss(
        embed_dim=embed_dim,
        num_classes=num_ids,
        margin=loss_cfg["margin"],
        scale=loss_cfg["scale"],
    ).to(device)
    arcface_loss = DDP(arcface_loss, device_ids=[local_rank], output_device=local_rank)

    # ── optimizer & scheduler& scaler ─────────────────────────────────────────────────
    optimizer = AdamW(
        list(model.parameters()) + list(arcface_loss.parameters()),
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

    scaler = torch.amp.GradScaler("cuda")

    # ── training loop ─────────────────────────────────────────────────────────
    best_eer = float("inf")
    ckpt_interval = training_cfg["checkpoint_interval"]
    best_model_name = output_cfg["best_model_name"]
    global_step = 0

    if is_main():
        print("\n" + "=" * 60)
        print(
            f"Starting training  (GPUs: {world_size}  |  "
            f"effective batch: {training_cfg['batch_size'] * world_size})"
        )
        print("=" * 60)

    epoch_pbar = tqdm(
        range(1, training_cfg["epochs"] + 1),
        desc="Training",
        unit="epoch",
        disable=not is_main(),
    )
    for epoch in epoch_pbar:
        avg_loss, global_step = train_one_epoch(
            model,
            arcface_loss,
            train_loader,
            train_sampler,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            global_step,
            use_wandb,
        )

        dist.barrier()

        eer, thr = evaluate(_unwrap(model), val_loader, device, epoch)

        if is_main():
            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}", eer=f"{eer:.4f}", thr=f"{thr:.4f}"
            )
            tqdm.write(
                f"Epoch {epoch:03d} | avg loss: {avg_loss:.4f} | "
                f"val EER: {eer:.4f}  (threshold={thr:.4f})"
            )

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

        dist.barrier()

    if is_main():
        print("=" * 60)
        print(f"Training complete. Best val EER: {best_eer:.4f}")
        print("=" * 60)

        if use_wandb and wandb.run is not None:
            wandb.finish()

    cleanup_ddp()


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
