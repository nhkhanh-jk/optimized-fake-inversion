"""
FakeInversion - Fetch Real Images
===================================
Fetch real images from URLs in the JSONL files and from HuggingFace datasets.
"""

import os
import sys
import json
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from PIL import Image
from tqdm import tqdm

import config
from utils import setup_logger


logger = setup_logger("fetch_real_images")

# Request timeout and retry settings
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 2
NUM_WORKERS = 8  # Parallel download threads


def fetch_single_image(url: str, save_path: str, timeout: int = REQUEST_TIMEOUT) -> bool:
    """Download a single image from URL.

    Args:
        url: Image URL.
        save_path: Path to save the image.
        timeout: Request timeout in seconds.

    Returns:
        True if successful, False otherwise.
    """
    if os.path.exists(save_path):
        return True

    for attempt in range(MAX_RETRIES):
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()

            # Verify it's a valid image
            img_bytes = response.content
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()

            # Re-open (verify closes the image) and save
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
            return True

        except Exception:
            if attempt < MAX_RETRIES - 1:
                continue
            return False


def fetch_from_jsonl(
    jsonl_path: str,
    save_dir: str,
    max_images: int = None,
) -> dict:
    """Fetch real images from a JSONL file containing URLs.

    Args:
        jsonl_path: Path to the JSONL file.
        save_dir: Directory to save images.
        max_images: Maximum number of images to fetch.

    Returns:
        Dict with stats: {total, success, failed}.
    """
    dataset_name = os.path.basename(jsonl_path).replace(".jsonl", "")
    logger.info(f"Fetching real images for: {dataset_name}")

    os.makedirs(save_dir, exist_ok=True)

    # Read JSONL
    with open(jsonl_path, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Sort by prompt_index
    entries.sort(key=lambda x: int(x["prompt_index"]))

    if max_images is not None:
        entries = entries[:max_images]

    # Prepare download tasks
    tasks = []
    for entry in entries:
        prompt_index = int(entry["prompt_index"])
        url = entry["url"]
        save_path = os.path.join(save_dir, f"{prompt_index:05d}.png")
        tasks.append((url, save_path))

    # Parallel download
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(fetch_single_image, url, path): (url, path)
            for url, path in tasks
        }

        pbar = tqdm(total=len(futures), desc=f"Fetching [{dataset_name}]")
        for future in as_completed(futures):
            result = future.result()
            if result:
                success += 1
            else:
                failed += 1
            pbar.update(1)
        pbar.close()

    stats = {"total": len(tasks), "success": success, "failed": failed}
    fail_rate = failed / max(len(tasks), 1) * 100

    logger.info(
        f"  {dataset_name}: {success}/{len(tasks)} downloaded "
        f"({fail_rate:.1f}% failure rate)"
    )
    return stats


def fetch_dalle3_images(save_dir: str, num_images: int = None):
    """Fetch DALL-E 3 images from HuggingFace dataset.

    Args:
        save_dir: Directory to save images.
        num_images: Number of images to fetch (default: config value).
    """
    import datasets

    if num_images is None:
        num_images = min(config.DALLE3_NUM_IMAGES, config.SUBSET_SIZE)

    logger.info(f"Fetching {num_images} DALL-E 3 images from HuggingFace...")
    os.makedirs(save_dir, exist_ok=True)

    dataset = datasets.load_dataset(
        config.DALLE3_DATASET_ID,
        ignore_verifications=True,
        revision=config.DALLE3_DATASET_REVISION,
    )

    ds = dataset["train"].shard(num_shards=5, index=0)

    for i, row in tqdm(enumerate(ds.select(range(num_images))), total=num_images, desc="Fetching [dalle3]"):
        save_path = os.path.join(save_dir, f"{i:05d}.png")
        if os.path.exists(save_path):
            continue
        try:
            img = row["image"]
            if img is not None:
                img.save(save_path)
        except Exception as e:
            logger.warning(f"Failed to save DALL-E 3 image {i}: {e}")

    num_saved = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
    logger.info(f"✅ DALL-E 3: {num_saved} images saved to {save_dir}")


def fetch_midjourney_images(save_dir: str, num_images: int = None):
    """Fetch Midjourney images from HuggingFace dataset.

    Args:
        save_dir: Directory to save images.
        num_images: Number of images to fetch.
    """
    import datasets

    if num_images is None:
        num_images = config.SUBSET_SIZE

    logger.info(f"Fetching {num_images} Midjourney images from HuggingFace...")
    os.makedirs(save_dir, exist_ok=True)

    try:
        dataset = datasets.load_dataset(
            "ehristoforu/midjourney-images",
            split="train",
        )

        # Filter non-thumbnail images
        count = 0
        for i, row in tqdm(enumerate(dataset), desc="Fetching [midjourney]"):
            if count >= num_images:
                break
            try:
                img = row.get("image")
                if img is not None:
                    save_path = os.path.join(save_dir, f"{count:05d}.png")
                    if not os.path.exists(save_path):
                        img.save(save_path)
                    count += 1
            except Exception:
                continue

    except Exception as e:
        logger.warning(f"Failed to fetch Midjourney dataset: {e}")
        logger.info("Falling back to URL-based fetching from JSONL...")
        jsonl_path = os.path.join(config.URL_FILES_DIR, "midjourney.jsonl")
        if os.path.exists(jsonl_path):
            fetch_from_jsonl(jsonl_path, save_dir, max_images=num_images)

    num_saved = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
    logger.info(f"✅ Midjourney: {num_saved} images saved to {save_dir}")


def fetch_all_real_images(
    max_images_per_source: int = None,
    subset: bool = True,
):
    """Fetch real images from all sources.

    Args:
        max_images_per_source: Maximum images per source.
        subset: If True, use SUBSET_SIZE.
    """
    if max_images_per_source is None:
        max_images_per_source = config.SUBSET_SIZE if subset else config.FULL_SIZE

    all_stats = {}

    # 1. Fetch from JSONL URL files
    if os.path.exists(config.URL_FILES_DIR):
        jsonl_files = sorted([
            f for f in os.listdir(config.URL_FILES_DIR)
            if f.endswith(".jsonl")
        ])

        for jsonl_file in jsonl_files:
            source_name = jsonl_file.replace(".jsonl", "")
            jsonl_path = os.path.join(config.URL_FILES_DIR, jsonl_file)
            save_dir = os.path.join(config.REAL_IMAGES_DIR, source_name)

            stats = fetch_from_jsonl(
                jsonl_path, save_dir,
                max_images=max_images_per_source,
            )
            all_stats[source_name] = stats
    else:
        logger.warning(f"URL files directory not found: {config.URL_FILES_DIR}")

    # 2. Fetch DALL-E 3 images (from HuggingFace)
    dalle3_dir = os.path.join(config.REAL_IMAGES_DIR, "dalle3_hf")
    fetch_dalle3_images(dalle3_dir, num_images=max_images_per_source)

    # 3. Fetch Midjourney images (from HuggingFace)
    mj_dir = os.path.join(config.REAL_IMAGES_DIR, "midjourney_hf")
    fetch_midjourney_images(mj_dir, num_images=max_images_per_source)

    # Summary
    logger.info("\n📊 Fetch Summary:")
    logger.info("-" * 50)
    total_success = 0
    total_failed = 0
    for source, stats in all_stats.items():
        logger.info(
            f"  {source:20s}: {stats['success']:5d} / {stats['total']:5d} "
            f"({stats['failed']} failed)"
        )
        total_success += stats["success"]
        total_failed += stats["failed"]

    logger.info("-" * 50)
    logger.info(f"  {'TOTAL':20s}: {total_success:5d} success, {total_failed:5d} failed")
    logger.info("✅ All real images fetched!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch real images")
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Max images per source"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Fetch full dataset instead of subset"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Fetch only from a specific JSONL source"
    )
    args = parser.parse_args()

    if args.source:
        jsonl_path = os.path.join(config.URL_FILES_DIR, f"{args.source}.jsonl")
        save_dir = os.path.join(config.REAL_IMAGES_DIR, args.source)
        fetch_from_jsonl(jsonl_path, save_dir, max_images=args.max_images)
    else:
        fetch_all_real_images(
            max_images_per_source=args.max_images,
            subset=not args.full,
        )
