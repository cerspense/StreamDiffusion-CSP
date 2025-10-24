import torch
import torch.nn.functional as F
from PIL import Image
from typing import Any, Optional
from .base import PipelineAwareProcessor


class ColorCorrectionFeedbackPreprocessor(PipelineAwareProcessor):
    """
    Image-space color correction with feedback loop integration

    Applies color adjustments (brightness, saturation, contrast, black level) to the
    feedback loop, accumulating color effects over time for creative color grading.

    Key Features:
    - Brightness: Adjust overall luminance (-1.0 to 1.0)
    - Saturation: Control color intensity (0.0 = grayscale, 2.0 = hyper-saturated)
    - Black Level: Lift shadows/adjust minimum luminance (0.0 to 1.0)
    - Contrast: Adjust tonal range (0.5 = low contrast, 2.0 = high contrast)
    - Feedback Strength: How much color correction accumulates (0.0 = pure input, 1.0 = pure feedback)

    Processing Pipeline:
    1. Get previous frame's output image (prev_image_result)
    2. Apply color correction to previous output
    3. Blend corrected previous with current input (feedback_strength controls mix)

    This creates accumulative color effects over time, similar to how LatentTransform
    creates accumulative motion.

    Examples:
    - brightness=0.1, saturation=1.2, feedback_strength=0.8: Gradually brighten and saturate
    - black_level=0.1, contrast=1.2, feedback_strength=0.5: Lift shadows with 50% blend

    CRITICAL: Requires requires_sync_processing=True to avoid 1-frame delay!
    """

    requires_sync_processing = True

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Color Correction Feedback",
            "description": "Real-time color grading with feedback loop for accumulative color effects",
            "parameters": {
                "feedback_strength": {
                    "type": "float",
                    "default": 0.8,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Feedback blend strength (0.0 = pure input, 1.0 = pure corrected feedback)"
                },
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
                    "description": "Gamma correction (0.5 = brighter mids, 2.0 = darker mids)"
                }
            },
            "use_cases": [
                "Real-time color grading",
                "Accumulative color effects",
                "Creative color feedback loops",
                "Live VJ color manipulation"
            ]
        }

    def __init__(self,
                 pipeline_ref: Any,
                 feedback_strength: float = 0.8,
                 brightness: float = 0.0,
                 saturation: float = 1.0,
                 contrast: float = 1.0,
                 black_level: float = 0.0,
                 gamma: float = 1.0,
                 **kwargs):
        """
        Initialize color correction feedback preprocessor

        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            feedback_strength: Feedback blend strength (0.0 = pure input, 1.0 = pure feedback)
            brightness: Brightness adjustment (-1.0 to 1.0)
            saturation: Saturation multiplier (0.0 to 2.0)
            contrast: Contrast multiplier (0.5 to 2.0)
            black_level: Black level lift (0.0 to 0.3)
            gamma: Gamma correction (0.5 to 2.0)
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            feedback_strength=feedback_strength,
            brightness=brightness,
            saturation=saturation,
            contrast=contrast,
            black_level=black_level,
            gamma=gamma,
            **kwargs
        )
        self.feedback_strength = max(0.0, min(1.0, feedback_strength))
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.black_level = black_level
        self.gamma = gamma
        self._first_frame = True

    def reset(self):
        """Reset the processor state (useful for new sequences)"""
        self._first_frame = True

    def _get_previous_data(self):
        """Get previous frame image data from pipeline"""
        if self.pipeline_ref is not None:
            if hasattr(self.pipeline_ref, 'prev_image_result'):
                if self.pipeline_ref.prev_image_result is not None and not self._first_frame:
                    return self.pipeline_ref.prev_image_result
        return None

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

        # 1. Black Level (lift shadows)
        if abs(self.black_level) > 1e-6:
            # Compress range: [0, 1] -> [black_level, 1]
            result = result * (1.0 - self.black_level) + self.black_level

        # 2. Brightness (additive)
        if abs(self.brightness) > 1e-6:
            result = result + self.brightness

        # 3. Contrast (around 0.5 midpoint)
        if abs(self.contrast - 1.0) > 1e-6:
            # contrast around midpoint 0.5
            result = (result - 0.5) * self.contrast + 0.5

        # 4. Saturation (desaturate/saturate)
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

        # 5. Gamma correction
        if abs(self.gamma - 1.0) > 1e-6:
            # Clamp before gamma to avoid negative values
            result = result.clamp(0, 1)
            result = torch.pow(result, 1.0 / self.gamma)

        # Final clamp to [0, 1] to prevent color space drift
        result = result.clamp(0, 1)

        # Restore original shape
        if len(original_shape) == 3:
            result = result.squeeze(0)

        return result

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Process using configurable blend of input image + color-corrected previous frame

        Args:
            image: Current input image

        Returns:
            Blended PIL Image with color correction feedback
        """
        # Get previous frame output
        prev_output_tensor = self._get_previous_data()

        if prev_output_tensor is None:
            # First frame - no feedback yet
            self._first_frame = False
            return image

        # Convert previous output tensor to [0, 1] range
        if prev_output_tensor.dim() == 4:
            prev_output_tensor = prev_output_tensor[0]  # Remove batch

        # Convert from [-1, 1] (VAE output) to [0, 1] (image processing range)
        prev_output_tensor = (prev_output_tensor / 2.0 + 0.5).clamp(0, 1)

        # Apply color correction to PREVIOUS output (accumulative effect)
        corrected_prev = self._apply_color_correction_tensor(prev_output_tensor)

        # Convert input image to tensor
        input_tensor = self.pil_to_tensor(image).squeeze(0)  # [C, H, W]

        # Ensure matching shapes
        if corrected_prev.shape != input_tensor.shape:
            target_size = input_tensor.shape[-2:]
            corrected_prev = corrected_prev.unsqueeze(0)
            corrected_prev = F.interpolate(
                corrected_prev, size=target_size, mode='bilinear', align_corners=False
            )
            corrected_prev = corrected_prev.squeeze(0)

        # Blend with feedback strength
        blended = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * corrected_prev

        # Safety clamp
        blended = blended.clamp(0, 1)

        # Convert back to PIL
        result = self.tensor_to_pil(blended)

        self._first_frame = False
        return result

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized path for tensor processing

        Args:
            tensor: Current input tensor

        Returns:
            Blended tensor with color correction feedback
        """
        # Get previous output
        prev_output = self._get_previous_data()

        if prev_output is None:
            # First frame
            self._first_frame = False
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(device=self.device, dtype=self.dtype)

        # Convert from [-1, 1] to [0, 1]
        prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)

        # Normalize input
        input_tensor = tensor
        if input_tensor.max() > 1.0:
            input_tensor = input_tensor / 255.0

        # Remove batch dims for processing
        if prev_output.dim() == 4 and prev_output.shape[0] == 1:
            prev_output = prev_output[0]
        if input_tensor.dim() == 4 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor[0]

        # Apply color correction to previous output
        corrected_prev = self._apply_color_correction_tensor(prev_output)

        # Match shapes
        if corrected_prev.shape != input_tensor.shape:
            if input_tensor.dim() == 3:
                target_size = input_tensor.shape[-2:]
                if corrected_prev.dim() == 3:
                    corrected_prev = corrected_prev.unsqueeze(0)
                corrected_prev = F.interpolate(
                    corrected_prev, size=target_size, mode='bilinear', align_corners=False
                )
                if corrected_prev.shape[0] == 1:
                    corrected_prev = corrected_prev.squeeze(0)

        # Blend
        blended = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * corrected_prev

        # Safety clamp
        blended = blended.clamp(0, 1)

        # Restore batch dimension
        if blended.dim() == 3:
            blended = blended.unsqueeze(0)

        blended = blended.to(device=self.device, dtype=self.dtype)

        self._first_frame = False
        return blended
