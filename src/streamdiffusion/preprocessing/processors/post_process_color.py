import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Optional, Any
from .base import BasePreprocessor


class PostProcessColorPreprocessor(BasePreprocessor):
    """
    Simple post-processing color correction without transforms or feedback

    Applies color grading (brightness, saturation, contrast, black level, gamma) to
    the final output after VAE decode. This is a stateless processor designed to work
    alongside latent_transform for motion effects.

    Key Features:
    - Brightness: Adjust overall luminance (-1.0 to 1.0)
    - Saturation: Control color intensity (0.0 = grayscale, 2.0 = hyper-saturated)
    - Contrast: Adjust tonal range around midpoint (0.5 = flat, 2.0 = high)
    - Black Level: Lift shadows/adjust minimum luminance (0.0 to 0.3)
    - Gamma: Power curve adjustment (0.5 = brighter mids, 2.0 = darker mids)
    - Temperature: Color temperature shift (-1.0 = cooler/blue, 1.0 = warmer/orange)

    Processing Pipeline:
    1. Take decoded image from VAE
    2. Apply color correction operations in sequence
    3. Clamp to [0, 1] to prevent drift
    4. Return corrected image

    This processor is stateless and does NOT use feedback. It's designed to be paired
    with latent_transform for motion in the latent stage, and this handles color in
    the post-processing stage.

    Examples:
    - brightness=0.1, saturation=1.2: Brighter, more saturated output
    - black_level=0.05, contrast=1.1: Lifted shadows with more punch
    - gamma=0.8, temperature=0.2: Brighter mids with warm tone
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Post-Process Color Correction",
            "description": "Stateless color grading for final output (pairs with latent_transform)",
            "parameters": {
                "brightness": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-1.0, 1.0],
                    "step": 0.01,
                    "description": "Brightness adjustment (-1.0 = black, 0.0 = neutral, 1.0 = white)"
                },
                "saturation": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.0, 2.0],
                    "step": 0.01,
                    "description": "Saturation multiplier (0.0 = grayscale, 1.0 = neutral, 2.0 = hyper-saturated)"
                },
                "contrast": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.5, 2.0],
                    "step": 0.01,
                    "description": "Contrast multiplier (0.5 = flat, 1.0 = neutral, 2.0 = high contrast)"
                },
                "black_level": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 0.3],
                    "step": 0.001,
                    "description": "Black level lift (raises minimum luminance)"
                },
                "gamma": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.5, 2.0],
                    "step": 0.01,
                    "description": "Gamma correction (0.5 = brighter mids, 1.0 = neutral, 2.0 = darker mids)"
                },
                "temperature": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-1.0, 1.0],
                    "step": 0.01,
                    "description": "Color temperature (-1.0 = cooler/blue, 0.0 = neutral, 1.0 = warmer/orange)"
                }
            },
            "use_cases": [
                "Final output color grading",
                "Pair with latent_transform for motion + color",
                "Stateless color correction",
                "Live color adjustment without feedback"
            ]
        }

    def __init__(self,
                 image_resolution: int = 512,
                 brightness: float = 0.0,
                 saturation: float = 1.0,
                 contrast: float = 1.0,
                 black_level: float = 0.0,
                 gamma: float = 1.0,
                 temperature: float = 0.0,
                 **kwargs):
        """
        Initialize post-process color correction preprocessor

        Args:
            image_resolution: Output image resolution
            brightness: Brightness adjustment (-1.0 to 1.0)
            saturation: Saturation multiplier (0.0 to 2.0)
            contrast: Contrast multiplier (0.5 to 2.0)
            black_level: Black level lift (0.0 to 0.3)
            gamma: Gamma correction (0.5 to 2.0)
            temperature: Color temperature (-1.0 to 1.0)
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            image_resolution=image_resolution,
            brightness=brightness,
            saturation=saturation,
            contrast=contrast,
            black_level=black_level,
            gamma=gamma,
            temperature=temperature,
            **kwargs
        )
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.black_level = black_level
        self.gamma = gamma
        self.temperature = temperature

    def _apply_color_correction_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply color correction to tensor (GPU-accelerated)

        Args:
            tensor: Input tensor [C, H, W] or [B, C, H, W] in range [0, 1]

        Returns:
            Color-corrected tensor [0, 1]
        """
        # Ensure batch dimension
        original_shape = tensor.shape
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # [B, C, H, W]

        result = tensor.clone()

        # 1. Black Level (lift shadows) - applied first to establish new floor
        if abs(self.black_level) > 1e-6:
            # Compress range: [0, 1] -> [black_level, 1]
            result = result * (1.0 - self.black_level) + self.black_level

        # 2. Brightness (additive) - shift all values
        if abs(self.brightness) > 1e-6:
            result = result + self.brightness

        # 3. Contrast (around 0.5 midpoint) - adjust tonal range
        if abs(self.contrast - 1.0) > 1e-6:
            # Contrast around midpoint 0.5
            result = (result - 0.5) * self.contrast + 0.5

        # 4. Gamma correction - adjust midtones
        if abs(self.gamma - 1.0) > 1e-6:
            # Clamp before gamma to avoid negative values
            result = result.clamp(0, 1)
            result = torch.pow(result, 1.0 / self.gamma)

        # 5. Saturation (desaturate/saturate)
        if abs(self.saturation - 1.0) > 1e-6:
            # Convert to grayscale using luminance weights (Rec. 709)
            # Y = 0.2126*R + 0.7152*G + 0.0722*B
            weights = torch.tensor([0.2126, 0.7152, 0.0722],
                                  device=result.device,
                                  dtype=result.dtype).view(1, 3, 1, 1)
            grayscale = (result * weights).sum(dim=1, keepdim=True)

            # Blend between grayscale and original based on saturation
            # saturation=0.0 -> grayscale, saturation=1.0 -> original, saturation=2.0 -> hyper
            result = grayscale + self.saturation * (result - grayscale)

        # 6. Temperature (color temperature shift)
        if abs(self.temperature) > 1e-6:
            # Temperature adjustment: shift blue/orange balance
            # Positive = warmer (more orange), Negative = cooler (more blue)
            temp_adjustment = torch.zeros_like(result)
            if self.temperature > 0:
                # Warmer: boost red, reduce blue
                temp_adjustment[:, 0, :, :] = self.temperature * 0.1  # Red boost
                temp_adjustment[:, 2, :, :] = -self.temperature * 0.1  # Blue reduction
            else:
                # Cooler: reduce red, boost blue
                temp_adjustment[:, 0, :, :] = self.temperature * 0.1  # Red reduction
                temp_adjustment[:, 2, :, :] = -self.temperature * 0.1  # Blue boost

            result = result + temp_adjustment

        # Final clamp to [0, 1] to prevent color space drift
        result = result.clamp(0, 1)

        # Restore original shape
        if len(original_shape) == 3:
            result = result.squeeze(0)

        return result

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Process PIL image with color correction

        Args:
            image: Input PIL image

        Returns:
            Color-corrected PIL image
        """
        # Convert to tensor
        tensor = self.pil_to_tensor(image).squeeze(0)  # [C, H, W]

        # Apply color correction
        corrected = self._apply_color_correction_tensor(tensor)

        # Convert back to PIL
        return self.tensor_to_pil(corrected)

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized path for tensor processing

        Args:
            tensor: Input tensor [B, C, H, W] or [C, H, W] in range [0, 1] or [0, 255]

        Returns:
            Color-corrected tensor [B, C, H, W] in range [0, 1]
        """
        # Normalize input to [0, 1] if needed
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        # Ensure batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Apply color correction
        result = self._apply_color_correction_tensor(tensor)

        # Ensure correct device and dtype
        result = result.to(device=self.device, dtype=self.dtype)

        return result
