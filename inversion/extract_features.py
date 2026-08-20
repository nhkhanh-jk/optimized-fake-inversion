"""
FakeInversion - Batch Feature Extraction
==========================================
Extract 9-channel DDIM inversion features for all images in the dataset.
Supports checkpointing for Colab session recovery.
"""

import os
import sys
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from tqdm import tqdm

import config
from utils import setup_logger, set_seed
from inversion.ddim_inverter import DDIMInverter


logger = setup_logger("extract_features")


def get_completed_features(features_dir: str) -> set:
    """Get set of already extracted feature file basenames."""
    completed = set()
    if os.path.exists(features_dir):
        for f in os.listdir(features_dir):
            if f.endswith(".pt"):
                completed.add(f.replace(".pt", ""))
    return completed


def extract_features_from_manifest(
    manifest_path: str,
    output_dir: str,
    num_samples: int = None,
    debug: bool = False,
):
    """Extract features for all images in a manifest CSV.

    Args:
        manifest_path: Path to manifest CSV (path, label, model_name).
        output_dir: Directory to save .pt feature files.
        num_samples: If set, only process this many samples.
        debug: If True, show extra debug info.
    """
    # Read manifest
    entries = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)

    if num_samples is not None:
        entries = entries[:num_samples]

    logger.info(f"📋 Manifest: {manifest_path}")
    logger.info(f"   Total entries: {len(entries)}")

    os.makedirs(output_dir, exist_ok=True)

    # Check which features are already extracted
    completed = get_completed_features(output_dir)
    remaining = []
    for entry in entries:
        # Create unique key from path
        key = os.path.splitext(os.path.basename(entry["path"]))[0]
        model = entry["model_name"]
        label = entry["label"]
        feature_key = f"{model}_{label}_{key}"

        if feature_key not in completed:
            remaining.append((entry, feature_key))

    logger.info(f"   Already done: {len(completed)}")
    logger.info(f"   Remaining: {len(remaining)}")

    if not remaining:
        logger.info("✅ All features already extracted!")
        return

    # Initialize inverter
    inverter = DDIMInverter()
    inverter.load_models()

    # Extract features
    success = 0
    failed = 0
    start_time = time.time()

    pbar = tqdm(remaining, desc="Extracting features")
    for entry, feature_key in pbar:
        image_path = entry["path"]
        label = int(entry["label"])
        model_name = entry["model_name"]

        try:
            # Load image
            if not os.path.exists(image_path):
                failed += 1
                continue

            image = Image.open(image_path).convert("RGB")

            # Extract 9-channel features
            features = inverter.extract_features(image)

            # Save feature tensor with metadata
            save_data = {
                "features": features,  # (9, H, W)
                "label": label,
                "model_name": model_name,
                "image_path": image_path,
            }
            save_path = os.path.join(output_dir, f"{feature_key}.pt")
            torch.save(save_data, save_path)

            success += 1

            if debug and success <= 3:
                logger.info(
                    f"  ✓ {feature_key}: features shape={features.shape}, "
                    f"label={label}, model={model_name}"
                )

        except Exception as e:
            failed += 1
            if debug:
                logger.warning(f"  ✗ {feature_key}: {e}")

        # Update progress bar
        elapsed = time.time() - start_time
        rate = success / max(elapsed, 1)
        pbar.set_postfix({
            "ok": success,
            "fail": failed,
            "rate": f"{rate:.1f}/s",
        })

    # Cleanup
    inverter.unload_models()

    elapsed = time.time() - start_time
    logger.info(
        f"\n✅ Feature extraction complete!\n"
        f"   Success: {success}, Failed: {failed}\n"
        f"   Time: {elapsed:.1f}s ({elapsed/max(success,1):.1f}s/image)\n"
        f"   Output: {output_dir}"
    )


def extract_all_features(subset: bool = True, num_samples: int = None):
    """Extract features for training and evaluation data.

    Args:
        subset: If True, only process subset.
        num_samples: Override number of samples.
    """
    set_seed(config.SEED)
    config.ensure_dirs()

    # Training features
    train_manifest = os.path.join(config.MANIFESTS_DIR, "train_train.csv")
    if os.path.exists(train_manifest):
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTING TRAINING FEATURES")
        logger.info("=" * 60)
        extract_features_from_manifest(
            manifest_path=train_manifest,
            output_dir=os.path.join(config.FEATURES_DIR, "train"),
            num_samples=num_samples,
        )

    # Validation features
    val_manifest = os.path.join(config.MANIFESTS_DIR, "train_val.csv")
    if os.path.exists(val_manifest):
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTING VALIDATION FEATURES")
        logger.info("=" * 60)
        extract_features_from_manifest(
            manifest_path=val_manifest,
            output_dir=os.path.join(config.FEATURES_DIR, "val"),
            num_samples=num_samples,
        )

    # Test features
    test_manifest = os.path.join(config.MANIFESTS_DIR, "train_test.csv")
    if os.path.exists(test_manifest):
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTING TEST FEATURES")
        logger.info("=" * 60)
        extract_features_from_manifest(
            manifest_path=test_manifest,
            output_dir=os.path.join(config.FEATURES_DIR, "test"),
            num_samples=num_samples,
        )

    # Evaluation features (all models)
    eval_manifest = os.path.join(config.MANIFESTS_DIR, "eval_all_models.csv")
    if os.path.exists(eval_manifest):
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTING EVALUATION FEATURES (ALL MODELS)")
        logger.info("=" * 60)
        extract_features_from_manifest(
            manifest_path=eval_manifest,
            output_dir=os.path.join(config.FEATURES_DIR, "eval"),
            num_samples=num_samples,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract DDIM inversion features")
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to specific manifest CSV"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for features"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to process (for testing)"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        output_dir = args.output_dir or os.path.join(config.FEATURES_DIR, "custom")
        extract_features_from_manifest(
            manifest_path=args.manifest,
            output_dir=output_dir,
            num_samples=args.num_samples,
            debug=args.debug,
        )
    else:
        extract_all_features(subset=not args.full, num_samples=args.num_samples)
