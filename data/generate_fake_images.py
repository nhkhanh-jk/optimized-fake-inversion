"""
FakeInversion - Generate Fake Images
======================================
Generate synthetic images using multiple text-to-image models.
Uses the easy-diffusion-generation repo for batch generation.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from utils import setup_logger, set_seed
from data.build_prompts import load_prompts


logger = setup_logger("generate_fake_images")

# Map of model names to their HuggingFace IDs for direct generation
MODEL_HF_IDS = {
    "sd-15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-21": "stabilityai/stable-diffusion-2-1",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-dpo": "mhdang/dpo-sdxl-text2image-v1",
    "ssd1b": "segmind/SSD-1B",
    "kandinsky2": "kandinsky-community/kandinsky-2-2-decoder",
    "kandinsky3": "kandinsky-community/kandinsky-3",
    "pixart-alpha": "PixArt-alpha/PixArt-XL-2-1024-MS",
    "playground-25": "playgroundai/playground-v2.5-1024px-aesthetic",
    "stable-cascade": "stabilityai/stable-cascade",
    "vega": "stabilityai/stable-diffusion-xl-base-1.0",  # Placeholder
    "wurstchen2": "warp-diffusion/wuerstchen",
}


def generate_with_sd_pipeline(
    model_name: str,
    prompts: list,
    save_dir: str,
    num_images: int = None,
):
    """Generate images using a Stable Diffusion-family model.

    Args:
        model_name: Name of the model (key in MODEL_HF_IDS).
        prompts: List of text prompts.
        save_dir: Directory to save generated images.
        num_images: Number of images to generate (default: len(prompts)).
    """
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        AutoPipelineForText2Image,
    )
    from tqdm import tqdm

    set_seed(config.GENERATION_SEED)

    if num_images is None:
        num_images = len(prompts)
    num_images = min(num_images, len(prompts))

    model_id = MODEL_HF_IDS.get(model_name)
    if model_id is None:
        logger.error(f"Unknown model: {model_name}")
        return

    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Loading model: {model_name} ({model_id})")

    # Choose pipeline based on model type
    try:
        if "xl" in model_name.lower() or "sdxl" in model_name.lower() or "playground" in model_name.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        elif "kandinsky" in model_name.lower():
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )

        pipe = pipe.to("cuda")

        # Enable memory optimizations
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return

    # Generate images
    logger.info(f"Generating {num_images} images with {model_name}...")

    generator = torch.Generator(device="cuda").manual_seed(config.GENERATION_SEED)

    for i in tqdm(range(num_images), desc=f"Generating [{model_name}]"):
        save_path = os.path.join(save_dir, f"{i:05d}.png")

        # Skip if already exists (for resume support)
        if os.path.exists(save_path):
            continue

        prompt = prompts[i]
        try:
            result = pipe(
                prompt=prompt,
                num_inference_steps=30,
                generator=generator,
            )
            image = result.images[0]
            image.save(save_path)
        except Exception as e:
            logger.warning(f"Failed to generate image {i}: {e}")
            continue

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    num_generated = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
    logger.info(f"✅ {model_name}: {num_generated} images saved to {save_dir}")


def generate_all_models(
    models: list = None,
    num_images: int = None,
    subset: bool = True,
):
    """Generate fake images for all specified models.

    Args:
        models: List of model names. If None, uses all models.
        num_images: Number of images per model. If None, uses config.
        subset: If True, generate only SUBSET_SIZE images.
    """
    if models is None:
        models = config.ALL_GENERATOR_MODELS

    if num_images is None:
        num_images = config.SUBSET_SIZE if subset else config.FULL_SIZE

    prompts = load_prompts()

    if len(prompts) < num_images:
        logger.warning(
            f"Only {len(prompts)} prompts available, but {num_images} requested. "
            f"Using {len(prompts)} prompts."
        )
        num_images = len(prompts)

    for model_name in models:
        save_dir = os.path.join(config.FAKE_IMAGES_DIR, model_name)
        generate_with_sd_pipeline(
            model_name=model_name,
            prompts=prompts,
            save_dir=save_dir,
            num_images=num_images,
        )

    logger.info("✅ All models generated!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate fake images")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to generate with (default: all)"
    )
    parser.add_argument(
        "--num_images", type=int, default=None,
        help="Number of images per model"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Generate full dataset instead of subset"
    )
    args = parser.parse_args()

    generate_all_models(
        models=args.models,
        num_images=args.num_images,
        subset=not args.full,
    )
