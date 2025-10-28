"""
Post-Process Sharpen & Noise Processor

Applies sharpening and noise injection in image space after VAE decode and color correction.

Features:
- Unsharp mask sharpening with configurable radius and amount
- Fine noise injection for film grain / texture effects
- Operates in image postprocessing stage (after color correction)
- GPU-accelerated processing

Author: StreamDiffusion Team
Date: 2025-10-28
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict
from .base import BasePreprocessor


class PostProcessSharpenNoisePreprocessor(BasePreprocessor):
    """
    Post-processing stage sharpening and noise injection.

    This processor operates in image space after VAE decode and color correction,
    applying unsharp mask sharpening and fine noise for texture enhancement.

    Processing Order (in image postprocessing stage):
    1. Unsharp mask sharpening (if sharpen_amount > 0)
    2. Noise injection (if noise_amount > 0)
    3. Final clamping to [0, 1]

    Sharpen Parameters:
    - sharpen_radius: Gaussian blur radius for unsharp mask (larger = broader sharpening)
    - sharpen_amount: Sharpening strength (0 = off, 1 = moderate, 2 = strong)

    Noise Parameters:
    - noise_amount: Noise injection strength (0 = off, 0.1 = subtle grain, 0.5 = heavy)
    """

    @classmethod
    def get_preprocessor_metadata(cls) -> Dict[str, Any]:
        """Get metadata for TouchDesigner integration"""
        return {
            "display_name": "Post-Process Sharpen & Noise",
            "description": "Image postprocessing with sharpening and noise injection",
            "parameters": {
                "sharpen_radius": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.1, 5.0],
                    "step": 0.1,
                    "description": "Sharpen blur radius (larger = broader sharpening)"
                },
                "sharpen_amount": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 3.0],
                    "step": 0.01,
                    "description": "Sharpen strength (0 = off, 1 = moderate, 2 = strong)"
                },
                "noise_amount": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Noise injection amount (0 = off, 0.1 = subtle grain)"
                },
            },
            "use_cases": [
                "Post-decode sharpening for detail enhancement",
                "Film grain / texture injection",
                "Final image refinement after color grading"
            ]
        }

    def __init__(
        self,
        sharpen_radius: float = 1.0,
        sharpen_amount: float = 0.0,
        noise_amount: float = 0.0,
        **kwargs
    ):
        """
        Initialize post-process sharpen & noise processor.

        Args:
            sharpen_radius: Gaussian blur radius for unsharp mask (0.1-5.0)
            sharpen_amount: Sharpening strength (0.0-3.0, 0=off)
            noise_amount: Noise injection strength (0.0-1.0, 0=off)
        """
        super().__init__(
            sharpen_radius=sharpen_radius,
            sharpen_amount=sharpen_amount,
            noise_amount=noise_amount,
            **kwargs
        )
        self.sharpen_radius = sharpen_radius
        self.sharpen_amount = sharpen_amount
        self.noise_amount = noise_amount

    def _process_core(self, image):
        """Not used - we use GPU tensor processing"""
        raise NotImplementedError("Use _process_tensor_core for GPU processing")

    def _apply_gaussian_blur(self, tensor: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Apply Gaussian blur for unsharp mask.

        Args:
            tensor: [B, C, H, W] image tensor
            radius: Blur radius (sigma for Gaussian kernel)

        Returns:
            Blurred tensor
        """
        # Calculate kernel size from radius (must be odd)
        kernel_size = int(2 * round(radius * 3) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)  # Minimum size 3

        # Create Gaussian kernel
        sigma = radius
        x = torch.arange(kernel_size, dtype=self.dtype, device=self.device) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # Create 2D kernel (separable Gaussian)
        kernel_1d = gauss.view(1, 1, -1)

        # Apply separable convolution (horizontal then vertical)
        padding = kernel_size // 2
        B, C, H, W = tensor.shape

        # Horizontal blur
        tensor_padded = F.pad(tensor, (padding, padding, 0, 0), mode='reflect')
        blurred_h = F.conv1d(
            tensor_padded.reshape(B * C, 1, H, W + 2 * padding).transpose(2, 3).reshape(B * C * (W + 2 * padding), 1, H),
            kernel_1d,
            padding=0
        ).reshape(B * C, W + 2 * padding, 1, H).transpose(2, 3).reshape(B, C, H, W + 2 * padding)

        # Vertical blur
        blurred_padded = F.pad(blurred_h, (0, 0, padding, padding), mode='reflect')
        blurred = F.conv1d(
            blurred_padded.reshape(B * C, 1, H + 2 * padding, W).transpose(2, 3).reshape(B * C * W, 1, H + 2 * padding),
            kernel_1d,
            padding=0
        ).reshape(B * C, W, 1, H + 2 * padding).transpose(2, 3).reshape(B, C, H + 2 * padding, W)[:, :, padding:-padding, :]

        return blurred

    def _apply_unsharp_mask(self, tensor: torch.Tensor, radius: float, amount: float) -> torch.Tensor:
        """
        Apply unsharp mask sharpening.

        Formula: sharpened = original + amount * (original - blurred)

        Args:
            tensor: [B, C, H, W] image tensor in [0, 1]
            radius: Blur radius for unsharp mask
            amount: Sharpening strength

        Returns:
            Sharpened tensor
        """
        if amount <= 1e-6:
            return tensor

        # Apply Gaussian blur
        blurred = self._apply_gaussian_blur(tensor, radius)

        # Unsharp mask: original + amount * (original - blurred)
        sharpened = tensor + amount * (tensor - blurred)

        return sharpened

    def _inject_noise(self, tensor: torch.Tensor, amount: float) -> torch.Tensor:
        """
        Inject fine noise for film grain / texture effect.

        Args:
            tensor: [B, C, H, W] image tensor in [0, 1]
            amount: Noise strength (0.0-1.0)

        Returns:
            Tensor with noise injected
        """
        if amount <= 1e-6:
            return tensor

        # Generate fine-grained noise (per-pixel)
        noise = torch.randn_like(tensor) * amount

        # Add noise
        noisy = tensor + noise

        return noisy

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated sharpen and noise processing.

        Args:
            tensor: [B, C, H, W] image tensor in [0, 1] range

        Returns:
            Processed tensor with sharpening and noise, clamped to [0, 1]
        """
        result = tensor

        # 1. Apply sharpening (if enabled)
        if abs(self.sharpen_amount) > 1e-6:
            result = self._apply_unsharp_mask(result, self.sharpen_radius, self.sharpen_amount)

        # 2. Inject noise (if enabled)
        if abs(self.noise_amount) > 1e-6:
            result = self._inject_noise(result, self.noise_amount)

        # 3. Final clamp to [0, 1]
        result = result.clamp(0, 1)

        return result
