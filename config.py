"""
FakeInversion - Centralized Configuration
==========================================
All paths, hyperparameters, and model configurations in one place.
Designed for Google Colab Pro with Google Drive storage.
"""

import os
from pathlib import Path


# =============================================================================
# Google Drive / Base Paths
# =============================================================================
USE_GOOGLE_DRIVE = True
DRIVE_MOUNT_POINT = "/content/drive"
DRIVE_PROJECT_DIR = os.path.join(DRIVE_MOUNT_POINT, "MyDrive", "fake_inversion")

# Base directory: use Google Drive if available, else local
if USE_GOOGLE_DRIVE and os.path.exists(DRIVE_MOUNT_POINT):
    BASE_DIR = DRIVE_PROJECT_DIR
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Data Paths
# =============================================================================
DATA_DIR = os.path.join(BASE_DIR, "data")
PROMPTS_FILE = os.path.join(DATA_DIR, "fake_inversion_prompts.txt")
FAKE_IMAGES_DIR = os.path.join(DATA_DIR, "fake_images")
REAL_IMAGES_DIR = os.path.join(DATA_DIR, "real_images")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MANIFESTS_DIR = os.path.join(DATA_DIR, "manifests")

# URL files from the original dataset
URL_FILES_DIR = os.path.join(BASE_DIR, "fakeinversion_data", "url_files")

# =============================================================================
# Dataset Configuration
# =============================================================================
SUBSET_SIZE = 500  # Number of images per model for subset mode
FULL_SIZE = 5000   # Full dataset size per model

# Train/Val/Test split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Training model (detector trained ONLY on SD-1.5 fakes + corresponding reals)
TRAIN_SOURCE_MODEL = "sd-15"

# All available generator models
ALL_GENERATOR_MODELS = [
    "sd-15",
    "sd-21",
    "sdxl",
    "sdxl-dpo",
    "ssd1b",
    "kandinsky2",
    "kandinsky3",
    "pixart-alpha",
    "playground-25",
    "stable-cascade",
    "vega",
    "wurstchen2",
]

# Models for which we also have real images via URL files (for SynRIS evaluation)
SYNRIS_MODELS = [
    "dalle3",
    "midjourney",
    "segmoe",
] + ALL_GENERATOR_MODELS

# =============================================================================
# Stable Diffusion Model Configuration
# =============================================================================
SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_REVISION = "main"
SD_DTYPE = "float16"  # Use fp16 for memory efficiency on Colab

# =============================================================================
# BLIP Captioning Model Configuration
# =============================================================================
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"

# =============================================================================
# DDIM Inversion Parameters
# =============================================================================
DDIM_NUM_INFERENCE_STEPS = 50   # Number of DDIM steps
DDIM_GUIDANCE_SCALE = 1.0       # CFG scale for inversion (1.0 = no guidance)
IMAGE_SIZE = 512                 # Resize images to this size before processing

# =============================================================================
# Classifier Training Hyperparameters
# =============================================================================
CLASSIFIER_BACKBONE = "resnet50"   # ResNet variant to use
INPUT_CHANNELS = 9                  # 3 (original) + 3 (noise map) + 3 (reconstruction)
NUM_CLASSES = 2                     # Binary: real vs fake

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2

# Mixed precision training
USE_AMP = True

# =============================================================================
# Checkpointing & Logging
# =============================================================================
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =============================================================================
# Random Seeds
# =============================================================================
SEED = 42
PROMPT_SHUFFLE_SEED = 42
GENERATION_SEED = 42

# =============================================================================
# Prompts Dataset (HuggingFace)
# =============================================================================
PROMPTS_DATASET_ID = "wanng/midjourney-v5-202304-clean"
PROMPTS_DATASET_SPLIT = "train"
NUM_PROMPTS = 5000

# DALL-E 3 Dataset (HuggingFace)
DALLE3_DATASET_ID = "OpenDatasets/dalle-3-dataset"
DALLE3_DATASET_REVISION = "22a1f7dc2ea1137ec5608bf791e70937b6a4df78"
DALLE3_NUM_IMAGES = 3000

# Midjourney Dataset (HuggingFace)
MJ_DATASET_URL = "https://huggingface.co/datasets/ehristoforu/midjourney-images/resolve/main/Midjourney-dataset.zip?download=true"


# =============================================================================
# Helper Functions
# =============================================================================
def ensure_dirs():
    """Create all necessary directories."""
    dirs = [
        DATA_DIR, FAKE_IMAGES_DIR, REAL_IMAGES_DIR,
        FEATURES_DIR, MANIFESTS_DIR,
        CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_config():
    """Print current configuration summary."""
    print("=" * 60)
    print("FakeInversion Configuration")
    print("=" * 60)
    print(f"  Base Directory:     {BASE_DIR}")
    print(f"  Google Drive:       {USE_GOOGLE_DRIVE}")
    print(f"  Subset Size:        {SUBSET_SIZE} images/model")
    print(f"  SD Model:           {SD_MODEL_ID}")
    print(f"  DDIM Steps:         {DDIM_NUM_INFERENCE_STEPS}")
    print(f"  Image Size:         {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Classifier:         {CLASSIFIER_BACKBONE} ({INPUT_CHANNELS}ch → {NUM_CLASSES} classes)")
    print(f"  Training:           lr={LEARNING_RATE}, bs={BATCH_SIZE}, epochs={NUM_EPOCHS}")
    print(f"  Device:             {get_device()}")
    print("=" * 60)
