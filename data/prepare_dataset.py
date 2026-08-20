"""
FakeInversion - Prepare Dataset
=================================
Organize images into train/val/test splits and create manifest CSV files.
Training only uses SD-1.5 fakes + corresponding reals.
Testing uses ALL models for generalization evaluation.
"""

import os
import sys
import csv
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import setup_logger, set_seed, list_image_files


logger = setup_logger("prepare_dataset")


def find_paired_images(
    fake_dir: str,
    real_dir: str,
    model_name: str,
    max_pairs: int = None,
) -> list:
    """Find paired (fake, real) images based on matching indices.

    Args:
        fake_dir: Directory with fake images.
        real_dir: Directory with real images.
        model_name: Name of the source model.
        max_pairs: Maximum number of pairs.

    Returns:
        List of dicts: {fake_path, real_path, model_name, prompt_index}
    """
    fake_files = {}
    real_files = {}

    # Index fake images by their numeric ID
    if os.path.exists(fake_dir):
        for f in os.listdir(fake_dir):
            if f.endswith((".png", ".jpg", ".jpeg")):
                idx = os.path.splitext(f)[0]
                fake_files[idx] = os.path.join(fake_dir, f)

    # Index real images by their numeric ID
    if os.path.exists(real_dir):
        for f in os.listdir(real_dir):
            if f.endswith((".png", ".jpg", ".jpeg")):
                idx = os.path.splitext(f)[0]
                real_files[idx] = os.path.join(real_dir, f)

    # Find matching pairs
    common_indices = sorted(set(fake_files.keys()) & set(real_files.keys()))

    if max_pairs is not None:
        common_indices = common_indices[:max_pairs]

    pairs = []
    for idx in common_indices:
        pairs.append({
            "fake_path": fake_files[idx],
            "real_path": real_files[idx],
            "model_name": model_name,
            "prompt_index": idx,
        })

    return pairs


def collect_unpaired_images(
    image_dir: str,
    label: int,
    model_name: str,
    max_images: int = None,
) -> list:
    """Collect images from a directory without requiring pairs.

    Args:
        image_dir: Directory with images.
        label: 0 for real, 1 for fake.
        model_name: Source model name.
        max_images: Maximum number of images.

    Returns:
        List of dicts: {path, label, model_name}
    """
    entries = []
    if not os.path.exists(image_dir):
        return entries

    files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])

    if max_images is not None:
        files = files[:max_images]

    for f in files:
        entries.append({
            "path": os.path.join(image_dir, f),
            "label": label,
            "model_name": model_name,
        })

    return entries


def create_splits(entries: list, seed: int = 42) -> dict:
    """Split entries into train/val/test sets.

    Args:
        entries: List of entry dicts.
        seed: Random seed.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to a list.
    """
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    return splits


def save_manifest(entries: list, filepath: str):
    """Save a manifest CSV file.

    Args:
        entries: List of entry dicts with keys: path, label, model_name.
        filepath: Output CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "model_name"])
        writer.writeheader()
        writer.writerows(entries)

    logger.info(f"  Saved {len(entries)} entries to {filepath}")


def prepare_training_data(max_per_class: int = None):
    """Prepare training data: ONLY SD-1.5 fakes + corresponding reals.

    Args:
        max_per_class: Maximum images per class (real/fake).
    """
    if max_per_class is None:
        max_per_class = config.SUBSET_SIZE

    logger.info("📦 Preparing TRAINING data (SD-1.5 only)...")

    train_model = config.TRAIN_SOURCE_MODEL
    fake_dir = os.path.join(config.FAKE_IMAGES_DIR, train_model)
    real_dir = os.path.join(config.REAL_IMAGES_DIR, train_model)

    # Collect fake images
    fake_entries = collect_unpaired_images(
        fake_dir, label=1, model_name=train_model, max_images=max_per_class
    )
    logger.info(f"  Fake images (SD-1.5): {len(fake_entries)}")

    # Collect real images
    real_entries = collect_unpaired_images(
        real_dir, label=0, model_name=train_model, max_images=max_per_class
    )
    logger.info(f"  Real images: {len(real_entries)}")

    # Balance classes
    min_count = min(len(fake_entries), len(real_entries))
    if min_count == 0:
        logger.warning("⚠️  No images found! Check data directories.")
        return

    fake_entries = fake_entries[:min_count]
    real_entries = real_entries[:min_count]

    # Combine and split
    all_entries = fake_entries + real_entries
    splits = create_splits(all_entries, seed=config.SEED)

    # Save manifests
    for split_name, entries in splits.items():
        filepath = os.path.join(config.MANIFESTS_DIR, f"train_{split_name}.csv")
        save_manifest(entries, filepath)

    logger.info(
        f"  ✅ Training splits: "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )


def prepare_evaluation_data(max_per_model: int = None):
    """Prepare evaluation data: ALL models (for generalization testing).

    Args:
        max_per_model: Maximum images per model.
    """
    if max_per_model is None:
        max_per_model = config.SUBSET_SIZE

    logger.info("📦 Preparing EVALUATION data (all models)...")

    all_entries = []

    for model_name in config.SYNRIS_MODELS:
        fake_dir = os.path.join(config.FAKE_IMAGES_DIR, model_name)
        real_dir = os.path.join(config.REAL_IMAGES_DIR, model_name)

        # Fake images for this model
        fake_entries = collect_unpaired_images(
            fake_dir, label=1, model_name=model_name,
            max_images=max_per_model,
        )

        # Real images for this model
        real_entries = collect_unpaired_images(
            real_dir, label=0, model_name=model_name,
            max_images=max_per_model,
        )

        if fake_entries or real_entries:
            logger.info(
                f"  {model_name:20s}: "
                f"{len(fake_entries)} fake, {len(real_entries)} real"
            )

        all_entries.extend(fake_entries)
        all_entries.extend(real_entries)

    # Save full evaluation manifest
    filepath = os.path.join(config.MANIFESTS_DIR, "eval_all_models.csv")
    save_manifest(all_entries, filepath)

    # Also save per-model manifests
    for model_name in config.SYNRIS_MODELS:
        model_entries = [e for e in all_entries if e["model_name"] == model_name]
        if model_entries:
            filepath = os.path.join(config.MANIFESTS_DIR, f"eval_{model_name}.csv")
            save_manifest(model_entries, filepath)

    logger.info(f"  ✅ Total evaluation entries: {len(all_entries)}")


def prepare_all(subset: bool = True):
    """Prepare all datasets.

    Args:
        subset: If True, use SUBSET_SIZE.
    """
    set_seed(config.SEED)
    config.ensure_dirs()

    max_images = config.SUBSET_SIZE if subset else config.FULL_SIZE

    prepare_training_data(max_per_class=max_images)
    prepare_evaluation_data(max_per_model=max_images)

    logger.info("\n✅ All datasets prepared!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    if args.max_images:
        prepare_training_data(max_per_class=args.max_images)
        prepare_evaluation_data(max_per_model=args.max_images)
    else:
        prepare_all(subset=not args.full)
