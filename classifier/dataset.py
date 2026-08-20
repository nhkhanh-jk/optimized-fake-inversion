"""
FakeInversion - Custom Dataset
================================
PyTorch Dataset for loading pre-extracted 9-channel features.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Optional, Tuple

import config


class FakeInversionDataset(Dataset):
    """Dataset for pre-extracted 9-channel FakeInversion features.

    Each .pt file contains:
        - features: (9, H, W) tensor
        - label: 0 (real) or 1 (fake)
        - model_name: source model string
        - image_path: original image path
    """

    def __init__(
        self,
        features_dir: str,
        transform: Optional[callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        filter_model: Optional[str] = None,
    ):
        """Initialize the dataset.

        Args:
            features_dir: Directory containing .pt feature files.
            transform: Optional transform to apply to features.
            target_size: Resize features to this (H, W) for the classifier.
            filter_model: If set, only include entries from this model.
        """
        self.features_dir = features_dir
        self.transform = transform
        self.target_size = target_size
        self.filter_model = filter_model

        # Collect all feature files
        self.entries = []
        if os.path.exists(features_dir):
            for f in sorted(os.listdir(features_dir)):
                if f.endswith(".pt"):
                    filepath = os.path.join(features_dir, f)
                    self.entries.append(filepath)

        # Pre-filter by model if needed (lazy: read metadata on first access)
        if filter_model is not None:
            self._filter_by_model()

    def _filter_by_model(self):
        """Filter entries to only include a specific model."""
        filtered = []
        for filepath in self.entries:
            try:
                data = torch.load(filepath, map_location="cpu", weights_only=False)
                if data.get("model_name") == self.filter_model:
                    filtered.append(filepath)
            except Exception:
                continue
        self.entries = filtered

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get a single item.

        Returns:
            Tuple of (features, label, model_name):
                - features: (9, H, W) tensor
                - label: int (0=real, 1=fake)
                - model_name: str
        """
        filepath = self.entries[idx]
        data = torch.load(filepath, map_location="cpu", weights_only=False)

        features = data["features"]  # (9, H, W)
        label = int(data["label"])
        model_name = data.get("model_name", "unknown")

        # Resize to target size
        if self.target_size is not None:
            features = T.functional.resize(
                features,
                self.target_size,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )

        # Apply transform
        if self.transform is not None:
            features = self.transform(features)

        return features, label, model_name


class TrainTransform:
    """Data augmentation transform for training."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to 9-channel features.

        Args:
            x: Tensor of shape (9, H, W).

        Returns:
            Augmented tensor of shape (9, H, W).
        """
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = T.functional.hflip(x)

        # Random crop and resize
        if torch.rand(1).item() > 0.3:
            i, j, h, w = T.RandomResizedCrop.get_params(
                x, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            x = T.functional.resized_crop(
                x, i, j, h, w,
                size=(self.target_size, self.target_size),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )

        # Normalize each group of 3 channels with ImageNet stats
        # Channels 0-2: original image
        # Channels 3-5: decoded noise map
        # Channels 6-8: decoded reconstruction
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        for offset in range(0, 9, 3):
            for c in range(3):
                x[offset + c] = (x[offset + c] - mean[c]) / std[c]

        return x


class EvalTransform:
    """Transform for evaluation (no augmentation)."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to 9-channel features.

        Args:
            x: Tensor of shape (9, H, W).

        Returns:
            Normalized tensor of shape (9, H, W).
        """
        # Normalize each group of 3 channels with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        for offset in range(0, 9, 3):
            for c in range(3):
                x[offset + c] = (x[offset + c] - mean[c]) / std[c]

        return x


def create_dataloaders(
    train_dir: str = None,
    val_dir: str = None,
    test_dir: str = None,
    batch_size: int = None,
    num_workers: int = 2,
) -> dict:
    """Create DataLoaders for train/val/test splits.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to a DataLoader.
    """
    from torch.utils.data import DataLoader

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if train_dir is None:
        train_dir = os.path.join(config.FEATURES_DIR, "train")
    if val_dir is None:
        val_dir = os.path.join(config.FEATURES_DIR, "val")
    if test_dir is None:
        test_dir = os.path.join(config.FEATURES_DIR, "test")

    loaders = {}

    # Training
    if os.path.exists(train_dir):
        train_dataset = FakeInversionDataset(
            train_dir, transform=TrainTransform()
        )
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"  Train: {len(train_dataset)} samples")

    # Validation
    if os.path.exists(val_dir):
        val_dataset = FakeInversionDataset(
            val_dir, transform=EvalTransform()
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"  Val:   {len(val_dataset)} samples")

    # Test
    if os.path.exists(test_dir):
        test_dataset = FakeInversionDataset(
            test_dir, transform=EvalTransform()
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"  Test:  {len(test_dataset)} samples")

    return loaders
