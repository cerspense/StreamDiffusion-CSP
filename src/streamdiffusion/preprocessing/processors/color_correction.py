import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
from typing import Union
from .base import BasePreprocessor


class ColorCorrectionPreprocessor(BasePreprocessor):
    """
    GPU-accelerated color correction preprocessor for feedback loop enhancement

    Provides real-time color grading with brightness, saturation, and black level controls.
    Designed to work in preprocessing chains with feedback processors for enhanced visual control.

    Key Features:
    - Brightness: Additive luminance adjustment
    - Saturation: Color intensity multiplication
    - Black Level: Shadow lifting (crushes blacks)
    - Contrast: Optional midpoint-based contrast control
    - GPU-accelerated tensor operations for performance

    Processing Pipeline:
    1. Apply black level lift (raises shadows)
    2. Apply brightness adjustment (shifts all values)
    3. Convert RGB to HSV for saturation control
    4. Adjust saturation in HSV space
    5. Convert back to RGB
    6. Apply optional contrast adjustment
    7. Clamp to valid [0,1] range
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Color Correction",
            "description": "Real-time color grading with brightness, saturation, and black level controls. Perfect for enhancing feedback loops.",
            "parameters": {
                "brightness": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-0.5, 0.5],
                    "step": 0.01,
                    "description": "Brightness adjustment (-0.5 = darker, 0.0 = normal, 0.5 = brighter)"
                },
                "saturation": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.0, 2.0],
                    "step": 0.01,
                    "description": "Saturation multiplier (0.0 = grayscale, 1.0 = normal, 2.0 = hyper-saturated)"
                },
                "black_level": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 0.3],
                    "step": 0.01,
                    "description": "Black level lift (0.0 = normal, 0.3 = lifted shadows, crushes blacks)"
                },
                "contrast": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.5, 2.0],
                    "step": 0.01,
                    "description": "Contrast multiplier (0.5 = low contrast, 1.0 = normal, 2.0 = high contrast)"
                }
            },
            "use_cases": [
                "Feedback loop color enhancement",
                "Real-time color grading",
                "Shadow and highlight control",
                "Video aesthetic adjustment"
            ]
        }

    def __init__(self,
                 brightness: float = 0.0,
                 saturation: float = 1.0,
                 black_level: float = 0.0,
                 contrast: float = 1.0,
                 **kwargs):
        """
        Initialize Color Correction preprocessor

        Args:
            brightness: Brightness adjustment (-0.5 to 0.5, 0.0 = no change)
            saturation: Saturation multiplier (0.0 = grayscale, 1.0 = normal, 2.0 = hyper)
            black_level: Black level lift (0.0 = normal, 0.3 = lifted shadows)
            contrast: Contrast multiplier (0.5 = low, 1.0 = normal, 2.0 = high)
            **kwargs: Additional parameters
        """
        super().__init__(
            brightness=brightness,
            saturation=saturation,
            black_level=black_level,
            contrast=contrast,
            **kwargs
        )

    def _rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB to HSV color space (GPU-accelerated)

        Args:
            rgb: RGB tensor in [0,1] range, shape [C, H, W] or [B, C, H, W]

        Returns:
            HSV tensor in [0,1] range
        """
        # Handle batch dimension
        input_shape = rgb.shape
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)  # Add batch dimension

        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

        max_rgb, argmax_rgb = rgb.max(1)
        min_rgb = rgb.min(1)[0]
        delta = max_rgb - min_rgb

        # Hue calculation
        hue = torch.zeros_like(max_rgb)

        # Red is max
        mask = (argmax_rgb == 0)
        hue[mask] = (((g - b) / (delta + 1e-10)) % 6)[mask]

        # Green is max
        mask = (argmax_rgb == 1)
        hue[mask] = (((b - r) / (delta + 1e-10)) + 2)[mask]

        # Blue is max
        mask = (argmax_rgb == 2)
        hue[mask] = (((r - g) / (delta + 1e-10)) + 4)[mask]

        hue = hue / 6.0  # Normalize to [0, 1]

        # Saturation calculation
        saturation = torch.where(max_rgb > 1e-10, delta / (max_rgb + 1e-10), torch.zeros_like(max_rgb))

        # Value is just the max
        value = max_rgb

        hsv = torch.stack([hue, saturation, value], dim=1)

        # Remove batch dimension if input didn't have it
        if len(input_shape) == 3:
            hsv = hsv.squeeze(0)

        return hsv

    def _hsv_to_rgb(self, hsv: torch.Tensor) -> torch.Tensor:
        """
        Convert HSV to RGB color space (GPU-accelerated)

        Args:
            hsv: HSV tensor in [0,1] range, shape [C, H, W] or [B, C, H, W]

        Returns:
            RGB tensor in [0,1] range
        """
        # Handle batch dimension
        input_shape = hsv.shape
        if hsv.dim() == 3:
            hsv = hsv.unsqueeze(0)  # Add batch dimension

        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

        h = h * 6.0  # Convert to [0, 6] range

        c = v * s
        x = c * (1.0 - torch.abs((h % 2.0) - 1.0))
        m = v - c

        # Initialize RGB
        rgb = torch.zeros_like(hsv)

        # Different formulas for different hue ranges
        mask = (h >= 0) & (h < 1)
        rgb[:, 0, :, :][mask] = c[mask]
        rgb[:, 1, :, :][mask] = x[mask]

        mask = (h >= 1) & (h < 2)
        rgb[:, 0, :, :][mask] = x[mask]
        rgb[:, 1, :, :][mask] = c[mask]

        mask = (h >= 2) & (h < 3)
        rgb[:, 1, :, :][mask] = c[mask]
        rgb[:, 2, :, :][mask] = x[mask]

        mask = (h >= 3) & (h < 4)
        rgb[:, 1, :, :][mask] = x[mask]
        rgb[:, 2, :, :][mask] = c[mask]

        mask = (h >= 4) & (h < 5)
        rgb[:, 0, :, :][mask] = x[mask]
        rgb[:, 2, :, :][mask] = c[mask]

        mask = (h >= 5) & (h < 6)
        rgb[:, 0, :, :][mask] = c[mask]
        rgb[:, 2, :, :][mask] = x[mask]

        # Add m to all channels
        rgb = rgb + m.unsqueeze(1)

        # Remove batch dimension if input didn't have it
        if len(input_shape) == 3:
            rgb = rgb.squeeze(0)

        return rgb

    def _apply_color_correction_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply color correction operations on GPU tensor

        Args:
            image_tensor: Input tensor in [0,1] range

        Returns:
            Color-corrected tensor
        """
        # Get parameters
        brightness = self.params.get('brightness', 0.0)
        saturation = self.params.get('saturation', 1.0)
        black_level = self.params.get('black_level', 0.0)
        contrast = self.params.get('contrast', 1.0)

        result = image_tensor.clone()

        # Step 1: Apply black level (lift shadows)
        if black_level > 0:
            # Scale the range: [0, 1] -> [black_level, 1]
            result = result * (1.0 - black_level) + black_level

        # Step 2: Apply brightness (additive)
        if brightness != 0:
            result = result + brightness

        # Step 3: Apply saturation adjustment (requires HSV conversion)
        if saturation != 1.0:
            # Convert to HSV
            hsv = self._rgb_to_hsv(result)

            # Adjust saturation channel
            if hsv.dim() == 4:
                hsv[:, 1, :, :] = torch.clamp(hsv[:, 1, :, :] * saturation, 0, 1)
            else:
                hsv[1, :, :] = torch.clamp(hsv[1, :, :] * saturation, 0, 1)

            # Convert back to RGB
            result = self._hsv_to_rgb(hsv)

        # Step 4: Apply contrast (around midpoint 0.5)
        if contrast != 1.0:
            # Contrast adjustment: (pixel - 0.5) * contrast + 0.5
            result = (result - 0.5) * contrast + 0.5

        # Final clamp to ensure valid range
        result = torch.clamp(result, 0, 1)

        return result

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply color correction using PIL fallback

        Args:
            image: Input PIL Image

        Returns:
            Color-corrected PIL Image
        """
        # Get parameters
        brightness = self.params.get('brightness', 0.0)
        saturation = self.params.get('saturation', 1.0)
        black_level = self.params.get('black_level', 0.0)
        contrast = self.params.get('contrast', 1.0)

        result = image

        # Convert to numpy for black level adjustment
        if black_level > 0:
            img_array = np.array(result).astype(np.float32) / 255.0
            img_array = img_array * (1.0 - black_level) + black_level
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
            result = Image.fromarray(img_array)

        # Apply brightness using PIL
        if brightness != 0:
            enhancer = ImageEnhance.Brightness(result)
            # Convert from additive (-0.5 to 0.5) to multiplicative (0.5 to 1.5)
            brightness_factor = 1.0 + brightness
            result = enhancer.enhance(brightness_factor)

        # Apply contrast using PIL
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(contrast)

        # Apply saturation using PIL
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(saturation)

        return result

    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated color correction processing

        Args:
            image_tensor: Input tensor

        Returns:
            Color-corrected tensor
        """
        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Ensure correct device and dtype
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)

        # Ensure [0, 1] range
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        # Apply color correction
        result = self._apply_color_correction_tensor(image_tensor)

        return result
