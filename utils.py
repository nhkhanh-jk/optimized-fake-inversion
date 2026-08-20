"""
FakeInversion - Shared Utilities
=================================
Common helper functions used across all modules.
"""

import os
import random
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

import torch


# =============================================================================
# Logging Setup
# =============================================================================
def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Seed Management
# =============================================================================
def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Image I/O
# =============================================================================
def load_image(path: str, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Load an image and optionally resize it.

    Args:
        path: Path to the image file.
        size: Optional (width, height) to resize to.

    Returns:
        PIL Image in RGB mode.
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def save_image(img: Image.Image, path: str):
    """Save a PIL image to disk, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def image_to_tensor(img: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL Image to torch tensor.

    Args:
        img: PIL Image (RGB).
        normalize: If True, normalize to [-1, 1]. If False, normalize to [0, 1].

    Returns:
        Tensor of shape (3, H, W).
    """
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    if normalize:
        tensor = tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    return tensor


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
    """Convert torch tensor back to PIL Image.

    Args:
        tensor: Tensor of shape (3, H, W) or (1, 3, H, W).
        denormalize: If True, map from [-1, 1] to [0, 255].

    Returns:
        PIL Image in RGB mode.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().float()
    if denormalize:
        tensor = (tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    tensor = tensor.clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# =============================================================================
# Checkpointing
# =============================================================================
def save_checkpoint(state: dict, filepath: str):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, map_location=None) -> dict:
    """Load a training checkpoint."""
    return torch.load(filepath, map_location=map_location, weights_only=False)


# =============================================================================
# Progress & Metrics
# =============================================================================
class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# =============================================================================
# Google Drive Utilities (for Colab)
# =============================================================================
def mount_google_drive(mount_point: str = "/content/drive"):
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive
        drive.mount(mount_point)
        print(f"✅ Google Drive mounted at {mount_point}")
        return True
    except ImportError:
        print("⚠️  Not running in Colab, skipping Drive mount.")
        return False
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False


def get_gpu_info() -> str:
    """Get GPU information string."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        return f"{gpu_name} ({gpu_mem:.1f} GB)"
    return "No GPU available"


# =============================================================================
# File Utilities
# =============================================================================
def count_files(directory: str, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> int:
    """Count image files in a directory recursively."""
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extensions):
                count += 1
    return count


def list_image_files(directory: str, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> list:
    """List all image files in a directory recursively."""
    files = []
    for root, _, filenames in os.walk(directory):
        for f in sorted(filenames):
            if f.lower().endswith(extensions):
                files.append(os.path.join(root, f))
    return files
