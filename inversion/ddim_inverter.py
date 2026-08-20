"""
FakeInversion - Core DDIM Inverter
====================================
Implements DDIM Inversion using Stable Diffusion v1.5 to extract
9-channel features: [original_image, decoded_noise_map, decoded_reconstruction].

This is the heart of the FakeInversion method.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple

import config
from utils import setup_logger


logger = setup_logger("ddim_inverter")


class DDIMInverter:
    """DDIM Inversion feature extractor using Stable Diffusion v1.5.

    Pipeline:
        1. Caption image with BLIP → text embedding via CLIP
        2. Encode image → latent z₀ via VAE encoder
        3. DDIM Inversion: z₀ → ẑ_T (inverted noise)
        4. DDIM Denoise: ẑ_T → ẑ₀ (reconstructed latent)
        5. Decode ẑ_T → D(ẑ_T) (decoded noise map)
        6. Decode ẑ₀ → D(ẑ₀) (reconstructed image)
        7. Output: concat[x, D(ẑ_T), D(ẑ₀)] → 9 channels
    """

    def __init__(
        self,
        sd_model_id: str = None,
        blip_model_id: str = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
    ):
        """Initialize the DDIM Inverter.

        Args:
            sd_model_id: HuggingFace model ID for Stable Diffusion.
            blip_model_id: HuggingFace model ID for BLIP.
            device: Compute device (cuda/cpu).
            dtype: Model dtype (float16/float32).
            num_inference_steps: Number of DDIM steps.
            guidance_scale: Classifier-free guidance scale.
        """
        self.sd_model_id = sd_model_id or config.SD_MODEL_ID
        self.blip_model_id = blip_model_id or config.BLIP_MODEL_ID
        self.device = device or config.get_device()
        self.dtype = dtype or (torch.float16 if config.SD_DTYPE == "float16" else torch.float32)
        self.num_inference_steps = num_inference_steps or config.DDIM_NUM_INFERENCE_STEPS
        self.guidance_scale = guidance_scale or config.DDIM_GUIDANCE_SCALE
        self.image_size = config.IMAGE_SIZE

        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.blip_model = None
        self.blip_processor = None

        self._loaded = False

    def load_models(self):
        """Load all models into memory."""
        if self._loaded:
            return

        from diffusers import StableDiffusionPipeline, DDIMScheduler
        from transformers import BlipProcessor, BlipForConditionalGeneration

        logger.info(f"Loading Stable Diffusion: {self.sd_model_id}")

        # Load SD pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            self.sd_model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Extract components
        self.vae = pipe.vae.to(self.device)
        self.unet = pipe.unet.to(self.device)
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.tokenizer = pipe.tokenizer

        # Set up DDIM scheduler
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        # Set models to eval mode
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        del pipe

        # Load BLIP
        logger.info(f"Loading BLIP: {self.blip_model_id}")
        self.blip_processor = BlipProcessor.from_pretrained(self.blip_model_id)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            self.blip_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.blip_model.eval()

        self._loaded = True
        logger.info("✅ All models loaded.")

    def unload_models(self):
        """Free GPU memory by unloading models."""
        del self.vae, self.unet, self.text_encoder, self.tokenizer
        del self.blip_model, self.blip_processor, self.scheduler
        self.vae = self.unet = self.text_encoder = self.tokenizer = None
        self.blip_model = self.blip_processor = self.scheduler = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Models unloaded.")

    # -------------------------------------------------------------------------
    # Step 1: Image Captioning (BLIP)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def caption_image(self, image: Image.Image) -> str:
        """Generate a caption for the input image using BLIP.

        Args:
            image: PIL Image (RGB).

        Returns:
            Caption string.
        """
        inputs = self.blip_processor(images=image, return_tensors="pt").to(
            self.device, self.dtype
        )
        output_ids = self.blip_model.generate(**inputs, max_new_tokens=77)
        caption = self.blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return caption

    # -------------------------------------------------------------------------
    # Step 2: Text → CLIP Embedding
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to CLIP text embedding.

        Args:
            text: Input text string.

        Returns:
            Text embedding tensor of shape (1, 77, 768).
        """
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return text_embeddings

    @torch.no_grad()
    def get_unconditional_embedding(self) -> torch.Tensor:
        """Get unconditional (empty prompt) embedding for CFG."""
        return self.encode_text("")

    # -------------------------------------------------------------------------
    # Step 3: Image → Latent (VAE Encode)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def encode_image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to VAE latent space.

        Args:
            image: PIL Image (RGB), will be resized to IMAGE_SIZE.

        Returns:
            Latent tensor z₀ of shape (1, 4, H/8, W/8).
        """
        # Resize and convert to tensor
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor * 2.0 - 1.0).to(self.device, self.dtype)  # [0,1] → [-1,1]

        # VAE encode
        posterior = self.vae.encode(img_tensor)
        z0 = posterior.latent_dist.sample()
        z0 = z0 * self.vae.config.scaling_factor

        return z0

    # -------------------------------------------------------------------------
    # Step 4: Latent → Image (VAE Decode)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor back to pixel space.

        Args:
            latent: Latent tensor of shape (1, 4, H/8, W/8).

        Returns:
            Image tensor of shape (1, 3, H, W) in range [-1, 1].
        """
        latent = latent / self.vae.config.scaling_factor
        decoded = self.vae.decode(latent).sample
        return decoded

    # -------------------------------------------------------------------------
    # Step 5: DDIM Inversion (z₀ → ẑ_T)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def ddim_inversion(
        self,
        z0: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Perform DDIM inversion: map z₀ to inverted noise ẑ_T.

        This reverses the diffusion process: starting from clean latent z₀,
        iteratively add noise following the DDIM schedule in reverse.

        Args:
            z0: Clean latent tensor (1, 4, H/8, W/8).
            text_embedding: CLIP text embedding (1, 77, 768).

        Returns:
            Inverted noise tensor ẑ_T (1, 4, H/8, W/8).
        """
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Start from z₀
        latent = z0.clone()

        # DDIM inversion: iterate forward through timesteps (reversed order)
        # We go from t=0 → t=T, adding noise at each step
        reversed_timesteps = torch.flip(timesteps, [0])

        for i in range(len(reversed_timesteps) - 1):
            t = reversed_timesteps[i]
            t_next = reversed_timesteps[i + 1]

            # Predict noise using UNet
            noise_pred = self.unet(
                latent, t,
                encoder_hidden_states=text_embedding,
            ).sample

            # Apply classifier-free guidance if scale > 1
            if self.guidance_scale > 1.0:
                uncond_embedding = self.get_unconditional_embedding()
                noise_pred_uncond = self.unet(
                    latent, t,
                    encoder_hidden_states=uncond_embedding,
                ).sample
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred - noise_pred_uncond
                )

            # DDIM inversion step: x_t → x_{t+1}
            alpha_t = self.scheduler.alphas_cumprod[t]
            alpha_t_next = self.scheduler.alphas_cumprod[t_next]

            # Predicted x₀ from current latent
            pred_x0 = (latent - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Compute x_{t+1} (adding noise)
            latent = (
                torch.sqrt(alpha_t_next) * pred_x0
                + torch.sqrt(1 - alpha_t_next) * noise_pred
            )

        return latent

    # -------------------------------------------------------------------------
    # Step 6: DDIM Denoising (ẑ_T → ẑ₀)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def ddim_denoise(
        self,
        zT: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Perform DDIM denoising: map ẑ_T back to reconstructed ẑ₀.

        Standard DDIM sampling from noise to clean latent.

        Args:
            zT: Noise tensor (1, 4, H/8, W/8).
            text_embedding: CLIP text embedding (1, 77, 768).

        Returns:
            Reconstructed latent ẑ₀ (1, 4, H/8, W/8).
        """
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latent = zT.clone()

        for t in timesteps:
            # Predict noise
            noise_pred = self.unet(
                latent, t,
                encoder_hidden_states=text_embedding,
            ).sample

            # Apply classifier-free guidance
            if self.guidance_scale > 1.0:
                uncond_embedding = self.get_unconditional_embedding()
                noise_pred_uncond = self.unet(
                    latent, t,
                    encoder_hidden_states=uncond_embedding,
                ).sample
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred - noise_pred_uncond
                )

            # DDIM step
            alpha_t = self.scheduler.alphas_cumprod[t]
            alpha_prev = (
                self.scheduler.alphas_cumprod[timesteps[list(timesteps).index(t) + 1]]
                if t != timesteps[-1]
                else self.scheduler.final_alpha_cumprod
            )

            pred_x0 = (latent - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            latent = (
                torch.sqrt(alpha_prev) * pred_x0
                + torch.sqrt(1 - alpha_prev) * noise_pred
            )

        return latent

    # -------------------------------------------------------------------------
    # Step 7: Full Feature Extraction Pipeline
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract 9-channel FakeInversion features from an image.

        Pipeline:
            1. Caption image (BLIP) → text embedding (CLIP)
            2. Encode image → z₀ (VAE)
            3. DDIM Inversion: z₀ → ẑ_T
            4. DDIM Denoise: ẑ_T → ẑ₀
            5. Decode ẑ_T → D(ẑ_T) (noise map visualization)
            6. Decode ẑ₀ → D(ẑ₀) (reconstructed image)
            7. Concat: [x, D(ẑ_T), D(ẑ₀)] → 9 channels

        Args:
            image: Input PIL Image (RGB).

        Returns:
            Feature tensor of shape (9, H, W) in range [0, 1].
        """
        self.load_models()

        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # 1. Caption → text embedding
        caption = self.caption_image(image)
        text_embedding = self.encode_text(caption)

        # 2. Encode → z₀
        z0 = self.encode_image_to_latent(image)

        # 3. DDIM Inversion: z₀ → ẑ_T
        zT = self.ddim_inversion(z0, text_embedding)

        # 4. DDIM Denoise: ẑ_T → ẑ₀
        z0_reconstructed = self.ddim_denoise(zT, text_embedding)

        # 5. Decode ẑ_T → D(ẑ_T) (noise map)
        decoded_noise = self.decode_latent(zT)  # (1, 3, H, W), [-1, 1]

        # 6. Decode ẑ₀ → D(ẑ₀) (reconstruction)
        decoded_recon = self.decode_latent(z0_reconstructed)  # (1, 3, H, W), [-1, 1]

        # 7. Original image tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device, self.dtype)  # (1, 3, H, W), [0, 1]

        # Normalize all to [0, 1]
        decoded_noise = (decoded_noise + 1.0) / 2.0
        decoded_recon = (decoded_recon + 1.0) / 2.0

        # Clamp
        decoded_noise = decoded_noise.clamp(0, 1)
        decoded_recon = decoded_recon.clamp(0, 1)
        img_tensor = img_tensor.clamp(0, 1)

        # Concatenate: [x, D(ẑ_T), D(ẑ₀)] → (1, 9, H, W)
        features = torch.cat([img_tensor, decoded_noise, decoded_recon], dim=1)

        return features.squeeze(0).cpu().float()  # (9, H, W)

    def extract_components(self, image: Image.Image) -> dict:
        """Extract individual components for visualization.

        Returns:
            Dict with keys: original, noise_map, reconstruction, caption
        """
        self.load_models()

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        caption = self.caption_image(image)
        text_embedding = self.encode_text(caption)
        z0 = self.encode_image_to_latent(image)
        zT = self.ddim_inversion(z0, text_embedding)
        z0_recon = self.ddim_denoise(zT, text_embedding)

        decoded_noise = self.decode_latent(zT)
        decoded_recon = self.decode_latent(z0_recon)

        # Convert to PIL images
        from utils import tensor_to_image

        return {
            "original": image,
            "noise_map": tensor_to_image(decoded_noise),
            "reconstruction": tensor_to_image(decoded_recon),
            "caption": caption,
        }
