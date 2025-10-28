import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Optional, Any
from .base import PipelineAwareProcessor


class PreprocessColorPreprocessor(PipelineAwareProcessor):
    """
    Image preprocessing color correction with optional feedback

    Applies color grading in Stage 1 (before VAE encode) so color-corrected
    output participates in the image feedback loop. Works with prev_image_result
    from previous frame's final output.

    Key Features:
    - Brightness: Adjust overall luminance (-1.0 to 1.0)
    - Saturation: Control color intensity (0.0 = grayscale, 2.0 = hyper-saturated)
    - Contrast: Adjust tonal range around midpoint (0.5 = flat, 2.0 = high)
    - Black Level: Lift shadows/adjust minimum luminance (0.0 to 0.3)
    - Gamma: Power curve adjustment (0.5 = brighter mids, 2.0 = darker mids)
    - Temperature: Color temperature shift (-1.0 = cooler/blue, 1.0 = warmer/orange)
    - Feedback: Temporal smoothing with previous output (0.0 = off, 1.0 = full)

    Processing Pipeline:
    1. Get current input image
    2. Get previous output image (if feedback enabled)
    3. Apply color correction to current input
    4. Apply color correction to previous output (if available)
    5. Blend based on feedback_strength
    6. Pass to VAE encode → diffusion

    This processor operates in IMAGE PREPROCESSING (Stage 1), so it:
    - ✅ Color-corrected output stored in prev_image_result
    - ✅ Participates in image feedback loop
    - ✅ Can combine with other Stage 1 processors
    - ❌ Doesn't influence latent_transform (separate feedback path)

    Use Cases:
    - Stateless color grading before diffusion (feedback_strength=0.0)
    - Temporal color consistency with feedback (feedback_strength>0.0)
    - Combine with feedback_transform for color-aware motion
    - Apply color correction that feeds into VAE encoder

    Examples:
    - brightness=0.1, saturation=1.2, feedback_strength=0.0: Stateless color grading
    - brightness=0.1, saturation=1.2, feedback_strength=0.3: Color + temporal smoothing
    - Pair with feedback_transform for color-aware zoom/pan/rotate!

    CRITICAL: Uses requires_sync_processing=True to avoid 1-frame delay!
    """

    # CRITICAL: Force synchronous processing to avoid 1-frame delay
    requires_sync_processing = True

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Preprocess Color Correction (Stage 1 Feedback)",
            "description": "Color grading before VAE encode - participates in image feedback loop",
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
                },
                "feedback_strength": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Temporal feedback strength (0.0 = stateless, 1.0 = full smoothing)"
                }
            },
            "use_cases": [
                "Color grading before VAE encode (participates in feedback)",
                "Temporal color consistency with feedback",
                "Combine with feedback_transform for color-aware motion",
                "Stateless color correction (feedback_strength=0.0)"
            ]
        }

    def __init__(self,
                 pipeline_ref: Any,
                 image_resolution: int = 512,
                 brightness: float = 0.0,
                 saturation: float = 1.0,
                 contrast: float = 1.0,
                 black_level: float = 0.0,
                 gamma: float = 1.0,
                 temperature: float = 0.0,
                 feedback_strength: float = 0.0,
                 **kwargs):
        """
        Initialize preprocessing color correction with optional feedback

        Args:
            pipeline_ref: Reference to pipeline for accessing previous output
            image_resolution: Output image resolution
            brightness: Brightness adjustment (-1.0 to 1.0)
            saturation: Saturation multiplier (0.0 to 2.0)
            contrast: Contrast multiplier (0.5 to 2.0)
            black_level: Black level lift (0.0 to 0.3)
            gamma: Gamma correction (0.5 to 2.0)
            temperature: Color temperature (-1.0 to 1.0)
            feedback_strength: Temporal feedback (0.0 = off, 1.0 = full)
            **kwargs: Additional parameters passed to PipelineAwareProcessor
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            image_resolution=image_resolution,
            brightness=brightness,
            saturation=saturation,
            contrast=contrast,
            black_level=black_level,
            gamma=gamma,
            temperature=temperature,
            feedback_strength=feedback_strength,
            **kwargs
        )
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.black_level = black_level
        self.gamma = gamma
        self.temperature = temperature
        self.feedback_strength = feedback_strength
        self._first_frame = True

    def _get_previous_data(self):
        """Get previous frame output from pipeline (if available)"""
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
        Process PIL image with color correction + optional feedback

        Args:
            image: Input PIL image

        Returns:
            Color-corrected PIL image (optionally blended with previous output)
        """
        # Convert to tensor
        tensor = self.pil_to_tensor(image).squeeze(0)  # [C, H, W]

        # Apply color correction to current input
        corrected_input = self._apply_color_correction_tensor(tensor)

        # Get previous output (if available)
        prev_output = self._get_previous_data()

        if prev_output is None or abs(self.feedback_strength) < 1e-6:
            # No feedback - just return corrected input
            self._first_frame = False
            return self.tensor_to_pil(corrected_input)

        # CRITICAL: Convert VAE output range from [-1, 1] to [0, 1]
        prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)

        # Handle batch size mismatch
        if prev_output.shape[0] != corrected_input.shape[0]:
            if prev_output.shape[0] < corrected_input.shape[0]:
                prev_output = prev_output.repeat(corrected_input.shape[0], 1, 1, 1)
            else:
                prev_output = prev_output[:corrected_input.shape[0]]

        # Apply color correction to previous output as well
        corrected_prev = self._apply_color_correction_tensor(prev_output)

        # Blend: (1 - feedback_strength) * corrected_input + feedback_strength * corrected_prev
        blended = (1.0 - self.feedback_strength) * corrected_input + self.feedback_strength * corrected_prev

        # CRITICAL: Clamp to prevent color drift
        blended = blended.clamp(0, 1)

        self._first_frame = False
        return self.tensor_to_pil(blended.squeeze(0))

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

        # Apply color correction to current input
        corrected_input = self._apply_color_correction_tensor(tensor)

        # Get previous output (if available)
        prev_output = self._get_previous_data()

        if prev_output is None or abs(self.feedback_strength) < 1e-6:
            # No feedback - just return corrected input
            self._first_frame = False
            result = corrected_input.to(device=self.device, dtype=self.dtype)
            return result

        # CRITICAL: Convert VAE output range from [-1, 1] to [0, 1]
        prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)

        # Handle batch size mismatch
        if prev_output.shape[0] != corrected_input.shape[0]:
            if prev_output.shape[0] < corrected_input.shape[0]:
                prev_output = prev_output.repeat(corrected_input.shape[0], 1, 1, 1)
            else:
                prev_output = prev_output[:corrected_input.shape[0]]

        # Apply color correction to previous output as well
        corrected_prev = self._apply_color_correction_tensor(prev_output)

        # Blend: (1 - feedback_strength) * corrected_input + feedback_strength * corrected_prev
        blended = (1.0 - self.feedback_strength) * corrected_input + self.feedback_strength * corrected_prev

        # CRITICAL: Clamp to prevent color drift
        blended = blended.clamp(0, 1)

        self._first_frame = False

        # Ensure correct device and dtype
        result = blended.to(device=self.device, dtype=self.dtype)
        return result
