"""
FakeInversion - Build Prompts
==============================
Download and prepare 5,000 prompts from the Midjourney dataset on HuggingFace.
Subset mode: only take SUBSET_SIZE prompts for quick prototyping.
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import setup_logger, set_seed


logger = setup_logger("build_prompts")


def build_prompts(num_prompts: int = None, subset: bool = True):
    """Build prompt list from HuggingFace Midjourney dataset.

    Args:
        num_prompts: Number of prompts to generate. If None, uses config.
        subset: If True, limit to SUBSET_SIZE for quick testing.
    """
    import datasets

    if num_prompts is None:
        num_prompts = config.SUBSET_SIZE if subset else config.NUM_PROMPTS

    set_seed(config.PROMPT_SHUFFLE_SEED)

    logger.info(f"Loading dataset: {config.PROMPTS_DATASET_ID}")
    dataset = datasets.load_dataset(
        config.PROMPTS_DATASET_ID,
        split=config.PROMPTS_DATASET_SPLIT,
    )

    # Filter for upscaled images and shuffle
    logger.info("Filtering upscaled images and shuffling...")
    dataset = dataset.filter(lambda example: example["upscaled"])
    dataset = dataset.shuffle(seed=config.PROMPT_SHUFFLE_SEED)

    # Select prompts
    actual_count = min(num_prompts, len(dataset))
    selected = dataset.select(range(actual_count))
    prompts = [row["clean_prompts"] for row in selected]

    # Save to file
    os.makedirs(os.path.dirname(config.PROMPTS_FILE), exist_ok=True)
    with open(config.PROMPTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(prompts))

    logger.info(f"✅ Saved {len(prompts)} prompts to {config.PROMPTS_FILE}")
    return prompts


def load_prompts() -> list:
    """Load prompts from the saved file."""
    if not os.path.exists(config.PROMPTS_FILE):
        raise FileNotFoundError(
            f"Prompts file not found: {config.PROMPTS_FILE}\n"
            "Run build_prompts() first."
        )

    with open(config.PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(prompts)} prompts from {config.PROMPTS_FILE}")
    return prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build prompts for FakeInversion")
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Use full dataset instead of subset")
    args = parser.parse_args()

    build_prompts(num_prompts=args.num_prompts, subset=not args.full)
