"""
FakeInversion - Training Script
=================================
Train the ResNet classifier on 9-channel DDIM inversion features.
Supports mixed precision, checkpointing, and TensorBoard logging.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

import config
from utils import setup_logger, set_seed, save_checkpoint, load_checkpoint, AverageMeter
from classifier.model import build_model
from classifier.dataset import create_dataloaders


logger = setup_logger("train")


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
) -> dict:
    """Train for one epoch.

    Returns:
        Dict with loss, accuracy, auc.
    """
    model.train()

    loss_meter = AverageMeter("Loss")
    all_labels = []
    all_probs = []

    for batch_idx, (features, labels, _model_names) in enumerate(dataloader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Collect metrics
        loss_meter.update(loss.item(), features.size(0))
        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # Compute epoch metrics
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, (all_probs > 0.5).astype(int))

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # If only one class present

    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy,
        "auc": auc,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Validate the model.

    Returns:
        Dict with loss, accuracy, auc.
    """
    model.eval()

    loss_meter = AverageMeter("Val Loss")
    all_labels = []
    all_probs = []

    for features, labels, _model_names in dataloader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        loss_meter.update(loss.item(), features.size(0))
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, (all_probs > 0.5).astype(int))

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy,
        "auc": auc,
    }


def train(
    num_epochs: int = None,
    learning_rate: float = None,
    batch_size: int = None,
    resume_from: str = None,
    subset: int = None,
):
    """Full training pipeline.

    Args:
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        batch_size: Batch size.
        resume_from: Path to checkpoint to resume from.
        subset: If set, only use this many training samples.
    """
    # Setup
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    set_seed(config.SEED)
    config.ensure_dirs()
    device = config.get_device()

    logger.info("🚀 FakeInversion Training")
    logger.info(f"   Device: {device}")
    logger.info(f"   Epochs: {num_epochs}")
    logger.info(f"   LR: {learning_rate}")
    logger.info(f"   Batch Size: {batch_size}")

    # Create DataLoaders
    logger.info("\n📦 Loading datasets...")
    loaders = create_dataloaders(batch_size=batch_size)

    if "train" not in loaders:
        logger.error("No training data found! Run feature extraction first.")
        return

    # Build model
    logger.info("\n🏗️  Building model...")
    model = build_model(pretrained=True).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )

    # Mixed precision
    scaler = GradScaler(enabled=config.USE_AMP)

    # Resume from checkpoint
    start_epoch = 0
    best_auc = 0.0

    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from: {resume_from}")
        ckpt = load_checkpoint(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", 0.0)
        logger.info(f"  Resumed at epoch {start_epoch}, best_auc={best_auc:.4f}")

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=config.LOGS_DIR)
    except ImportError:
        tb_writer = None
        logger.warning("TensorBoard not available.")

    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info(
        f"{'Epoch':>6s} | {'Train Loss':>10s} {'Train Acc':>10s} {'Train AUC':>10s} | "
        f"{'Val Loss':>10s} {'Val Acc':>10s} {'Val AUC':>10s} | {'LR':>10s}"
    )
    logger.info("-" * 70)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler,
            device, epoch, use_amp=config.USE_AMP,
        )

        # Validate
        val_metrics = {"loss": 0, "accuracy": 0, "auc": 0}
        if "val" in loaders:
            val_metrics = validate(
                model, loaders["val"], criterion, device,
                use_amp=config.USE_AMP,
            )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        elapsed = time.time() - epoch_start
        logger.info(
            f"{epoch+1:>6d} | "
            f"{train_metrics['loss']:>10.4f} {train_metrics['accuracy']:>10.4f} {train_metrics['auc']:>10.4f} | "
            f"{val_metrics['loss']:>10.4f} {val_metrics['accuracy']:>10.4f} {val_metrics['auc']:>10.4f} | "
            f"{current_lr:>10.6f}  ({elapsed:.1f}s)"
        )

        # TensorBoard
        if tb_writer:
            tb_writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            tb_writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            tb_writer.add_scalar("train/auc", train_metrics["auc"], epoch)
            tb_writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            tb_writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            tb_writer.add_scalar("val/auc", val_metrics["auc"], epoch)
            tb_writer.add_scalar("lr", current_lr, epoch)

        # Save best model
        val_auc = val_metrics["auc"]
        is_best = val_auc > best_auc

        if is_best:
            best_auc = val_auc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_auc": best_auc,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                os.path.join(config.CHECKPOINTS_DIR, "best_model.pt"),
            )
            logger.info(f"  ★ New best model saved (AUC: {best_auc:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_auc": best_auc,
                },
                os.path.join(config.CHECKPOINTS_DIR, f"checkpoint_epoch_{epoch+1}.pt"),
            )

    logger.info("=" * 70)
    logger.info(f"\n✅ Training complete! Best Val AUC: {best_auc:.4f}")

    if tb_writer:
        tb_writer.close()

    return model, best_auc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train FakeInversion classifier")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        resume_from=args.resume,
        subset=args.subset,
    )
