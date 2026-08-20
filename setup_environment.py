"""
FakeInversion - Environment Setup for Google Colab
====================================================
Run this script first to set up the Colab environment.
"""

import subprocess
import sys


def install_dependencies():
    """Install all required packages."""
    print("📦 Installing dependencies...")

    packages = [
        "torch", "torchvision",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "datasets>=2.16.0",
        "Pillow",
        "numpy", "scipy", "scikit-learn", "pandas",
        "matplotlib", "seaborn",
        "tqdm", "tensorboard",
        "requests",
        "safetensors",
    ]

    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print("✅ All dependencies installed.")


def check_environment():
    """Verify the environment is correctly set up."""
    import torch

    print("\n🔍 Environment Check")
    print("=" * 50)
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  CUDA:         {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"  GPU Memory:   {mem:.1f} GB")
    else:
        print("  ⚠️  No GPU detected! DDIM Inversion will be very slow.")

    import diffusers
    import transformers
    print(f"  Diffusers:    {diffusers.__version__}")
    print(f"  Transformers: {transformers.__version__}")
    print("=" * 50)


def setup_google_drive():
    """Mount Google Drive and create project directory."""
    from utils import mount_google_drive
    import config

    if config.USE_GOOGLE_DRIVE:
        success = mount_google_drive()
        if success:
            config.ensure_dirs()
            print(f"📁 Project directory: {config.BASE_DIR}")
        return success
    return False


def download_sd_model():
    """Pre-download Stable Diffusion v1.5 model to cache."""
    import torch
    from diffusers import StableDiffusionPipeline

    import config

    print("\n🔄 Downloading Stable Diffusion v1.5...")
    print("   (This may take a few minutes on first run)")

    dtype = torch.float16 if config.SD_DTYPE == "float16" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        config.SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    print("✅ Stable Diffusion v1.5 downloaded and cached.")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def download_blip_model():
    """Pre-download BLIP captioning model."""
    from transformers import BlipProcessor, BlipForConditionalGeneration

    import config

    print("\n🔄 Downloading BLIP captioning model...")

    processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL_ID)

    print("✅ BLIP model downloaded and cached.")
    del processor, model


def main():
    """Full environment setup."""
    print("🚀 FakeInversion - Environment Setup")
    print("=" * 50)

    # Step 1: Install packages
    install_dependencies()

    # Step 2: Check environment
    check_environment()

    # Step 3: Mount Google Drive
    setup_google_drive()

    # Step 4: Download models
    download_sd_model()
    download_blip_model()

    print("\n✅ Setup complete! Ready to start.")


if __name__ == "__main__":
    main()
