import torch
from typing import Any
from streamdiffusion.preprocessing.processors import PipelineAwareProcessor


class LatentNoisePreprocessor(PipelineAwareProcessor):
    """
    Latent noise injection processor with pipeline-aware noise schedule.

    Operates AFTER VAE encode (in latent space) to inject noise directly into latents.
    This is much more efficient than image-space noise and has different artistic effects.

    Key Features:
    - Respects pipeline noise schedule (alpha/beta) for proper noise levels
    - Supports seed travel via noise_seed_mix (blend pipeline seed noise with random)
    - Multiple noise modes (gaussian, uniform)
    - Efficient operation in latent space (8x8 smaller than image space)

    Research-backed design:
    - Maintains Gaussian noise properties
    - Respects noise schedule by using alpha/beta from t_index_list
    - Supports seed travel for coherent variations
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "stage": "latent_pre",
            "display_name": "Latent Noise (Pipeline-Aware)",
            "description": "Inject noise directly into latent space with proper noise schedule and seed travel support. Very efficient and creates unique artistic effects.",
            "parameters": {
                "noise_strength": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Re-noise strength (0.0 = no noise, 1.0 = full re-noise). Respects pipeline noise schedule."
                },
                "noise_seed_mix": {
                    "type": "float",
                    "default": 0.8,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Blend pipeline seed noise vs random noise (0.0 = pure random, 1.0 = use pipeline seed travel noise). Higher = more consistent, lower = more chaotic"
                },
                "noise_mode": {
                    "type": "string",
                    "default": "gaussian",
                    "options": ["gaussian", "uniform"],
                    "description": "Type of noise distribution (only affects random component)"
                }
            },
            "use_cases": [
                "Latent-space texture injection",
                "Creating variation in diffusion",
                "Artistic noise effects with seed travel",
                "Breaking up patterns in latents",
                "Coherent noise for animations"
            ]
        }

    def __init__(self,
                 pipeline_ref: Any,
                 noise_strength: float = 0.0,
                 noise_seed_mix: float = 0.8,
                 noise_mode: str = "gaussian",
                 **kwargs):
        """
        Initialize LatentNoise preprocessor

        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required for noise schedule)
            noise_strength: Noise injection strength (respects noise schedule)
            noise_seed_mix: Blend pipeline seed vs random noise (0.0 = random, 1.0 = seeded)
            noise_mode: Type of noise ('gaussian' or 'uniform')
            **kwargs: Additional parameters
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            noise_strength=noise_strength,
            noise_seed_mix=noise_seed_mix,
            noise_mode=noise_mode,
            **kwargs
        )

        self.noise_strength = max(0.0, min(1.0, noise_strength))  # Clamp [0, 1]
        self.noise_seed_mix = max(0.0, min(1.0, noise_seed_mix))  # Clamp [0, 1]
        self.noise_mode = noise_mode

    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """Preserve latent dimensions - DON'T resize!"""
        return latent_tensor.to(device=self.device, dtype=self.dtype)

    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Override resize - latent tensors should NOT be resized"""
        return tensor

    def _process_core(self, image):
        """Not used for latent processors"""
        raise NotImplementedError("Use _process_tensor_core for latent processors")

    def _apply_noise_injection(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Re-noise latent to maintain proper noise schedule

        Research-backed approach:
        - Prevents "too clean" latents that break diffusion
        - Maintains Gaussian noise properties
        - Respects the noise schedule (alpha/beta from t_index_list)
        - Supports blending pipeline seed noise with random noise

        Formula: noisy_latent = alpha * latent + beta * noise
        where alpha and beta come from the first denoising step (t_index_list[0])

        Args:
            latent: Input latent tensor [B, C, H, W]

        Returns:
            Re-noised latent tensor
        """
        if self.noise_strength <= 0.0:
            return latent

        # Check if pipeline has noise schedule
        if self.pipeline_ref is None or not hasattr(self.pipeline_ref, 'alpha_prod_t_sqrt') or not hasattr(self.pipeline_ref, 'beta_prod_t_sqrt'):
            # Fallback: simple noise injection without schedule respect
            if self.noise_mode == "gaussian":
                noise = torch.randn_like(latent, device=latent.device, dtype=latent.dtype)
            elif self.noise_mode == "uniform":
                noise = (torch.rand_like(latent, device=latent.device, dtype=latent.dtype) * 2 - 1)
            else:
                noise = torch.randn_like(latent, device=latent.device, dtype=latent.dtype)

            return (1.0 - self.noise_strength) * latent + self.noise_strength * noise

        # Get alpha and beta for first denoising step
        # Shape: alpha_prod_t_sqrt[0] is typically a scalar or [1, 1, 1, 1] tensor
        alpha = self.pipeline_ref.alpha_prod_t_sqrt[0]
        beta = self.pipeline_ref.beta_prod_t_sqrt[0]

        # Generate base noise based on mode
        if self.noise_mode == "gaussian":
            random_noise = torch.randn_like(latent, device=latent.device, dtype=latent.dtype)
        elif self.noise_mode == "uniform":
            random_noise = (torch.rand_like(latent, device=latent.device, dtype=latent.dtype) * 2 - 1)
        else:
            random_noise = torch.randn_like(latent, device=latent.device, dtype=latent.dtype)

        # Blend pipeline seed noise with random noise
        if self.noise_seed_mix > 0.0 and hasattr(self.pipeline_ref, 'init_noise'):
            # Get pipeline's blended seed noise (updated by seed_list via OSC)
            # init_noise contains the weighted blend from seed travel
            pipeline_noise = self.pipeline_ref.init_noise

            # Match batch size if needed
            if pipeline_noise.shape[0] == 1 and latent.shape[0] > 1:
                # Expand single batch to match latent batch
                pipeline_noise = pipeline_noise.expand(latent.shape[0], -1, -1, -1)
            elif pipeline_noise.shape[0] > latent.shape[0]:
                # Take first N elements to match latent batch
                pipeline_noise = pipeline_noise[:latent.shape[0]]

            # Check spatial dimensions match
            if pipeline_noise.shape[2:] != latent.shape[2:]:
                # Size mismatch - fallback to random
                pipeline_noise = random_noise
            else:
                # Ensure correct device/dtype
                pipeline_noise = pipeline_noise.to(latent.device, dtype=latent.dtype)

            # Blend seeded noise with random noise
            # noise_seed_mix=1.0 → pure seed-traveled noise (coherent)
            # noise_seed_mix=0.0 → pure random noise (chaotic)
            noise = (1.0 - self.noise_seed_mix) * random_noise + self.noise_seed_mix * pipeline_noise
        else:
            noise = random_noise

        # Re-noise formula: x_t = alpha * x_0 + beta * noise
        # We blend this with the original latent based on noise_strength
        noised_latent = alpha * latent + beta * noise

        # Blend based on noise_strength
        # 0.0 = no re-noise (pure latent)
        # 1.0 = full re-noise (respect noise schedule completely)
        result = (1.0 - self.noise_strength) * latent + self.noise_strength * noised_latent

        return result

    def _process_tensor_core(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inject noise into latent tensor with proper noise schedule

        Args:
            latent_tensor: [B, 4, H/8, W/8] latent tensor

        Returns:
            Latent with noise injected (respects pipeline noise schedule)
        """
        if self.noise_strength <= 0:
            return latent_tensor

        # Ensure batch dimension
        if latent_tensor.dim() == 3:
            latent_tensor = latent_tensor.unsqueeze(0)

        # Ensure correct device and dtype
        latent_tensor = latent_tensor.to(device=self.device, dtype=self.dtype)

        # Apply sophisticated noise injection
        result = self._apply_noise_injection(latent_tensor)

        # Clamp to safe latent range
        result = torch.clamp(result, -10.0, 10.0)

        return result
