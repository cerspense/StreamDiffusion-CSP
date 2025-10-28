import torch
from .base import BasePreprocessor


class LatentNoisePreprocessor(BasePreprocessor):
    """
    Latent noise injection processor.

    Operates AFTER VAE encode (in latent space) to inject noise directly into latents.
    This is much more efficient than image-space noise and has different artistic effects.

    Latent space is 8x8 smaller than image space (64x64 vs 512x512),
    so operations are extremely fast.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Latent Noise",
            "description": "Inject noise directly into latent space after VAE encoding. Very efficient and creates unique artistic effects.",
            "parameters": {
                "noise_strength": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 2.0],
                    "step": 0.01,
                    "description": "Noise injection strength (0 = off, 0.1 = subtle, 0.5 = moderate, 2.0 = extreme)"
                },
                "noise_mode": {
                    "type": "string",
                    "default": "gaussian",
                    "options": ["gaussian", "uniform"],
                    "description": "Type of noise distribution"
                }
            },
            "use_cases": [
                "Latent-space texture injection",
                "Creating variation in diffusion",
                "Artistic noise effects",
                "Breaking up patterns in latents"
            ]
        }

    def __init__(self,
                 noise_strength: float = 0.0,
                 noise_mode: str = "gaussian",
                 **kwargs):
        """
        Initialize LatentNoise preprocessor

        Args:
            noise_strength: Noise injection strength
            noise_mode: Type of noise ('gaussian' or 'uniform')
            **kwargs: Additional parameters
        """
        super().__init__(
            noise_strength=noise_strength,
            noise_mode=noise_mode,
            **kwargs
        )

        self.noise_strength = noise_strength
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

    def _process_tensor_core(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inject noise into latent tensor

        Args:
            latent_tensor: [B, 4, H/8, W/8] latent tensor

        Returns:
            Latent with noise injected
        """
        if self.noise_strength <= 0:
            return latent_tensor

        # Ensure batch dimension
        if latent_tensor.dim() == 3:
            latent_tensor = latent_tensor.unsqueeze(0)

        # Ensure correct device and dtype
        latent_tensor = latent_tensor.to(device=self.device, dtype=self.dtype)

        # Generate noise matching latent shape
        if self.noise_mode == "gaussian":
            noise = torch.randn_like(latent_tensor)
        elif self.noise_mode == "uniform":
            noise = torch.rand_like(latent_tensor) * 2 - 1  # Range [-1, 1]
        else:
            noise = torch.randn_like(latent_tensor)

        # Inject noise
        result = latent_tensor + noise * self.noise_strength

        # Clamp to safe latent range
        return torch.clamp(result, -10.0, 10.0)
