import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Optional, Any, Literal
from .base import PipelineAwareProcessor


class PostProcessTransformCCPreprocessor(PipelineAwareProcessor):
    """
    Image-space postprocessing with geometric transforms and color correction

    Combines feedback loop with geometric transforms (zoom, pan, rotate) and color grading
    (brightness, saturation, black level). Operates in image space AFTER VAE decoding,
    providing final output control with Deforum-style motion and color manipulation.

    Key Features:
    - Feedback Loop: Blend input with previous output image
    - Zoom: Scale previous output (interpolate + crop/pad)
    - Pan: Translate previous output in X/Y
    - Rotate: Rotate previous output around center
    - Color Correction: Brightness, saturation, black level adjustments
    - Border Handling: Configurable edge treatment (zeros, border, reflection)

    Processing Pipeline:
    1. Get previous frame's output image (prev_image_result)
    2. Transform the PREVIOUS output (zoom/pan/rotate creates accumulative motion)
    3. Apply color correction to transformed previous
    4. Blend corrected+transformed previous with current input (feedback_strength controls mix)

    Examples:
    - zoom=1.05, feedback_strength=0.8, saturation=1.2: Zoom with color boost
    - pan_x=0.01, brightness=0.1, feedback_strength=0.5: Pan with brightness lift
    - rotation=2.0, black_level=0.05, feedback_strength=0.7: Rotate with shadow lift

    CRITICAL: Requires requires_sync_processing=True to avoid 1-frame delay!
    """

    # CRITICAL: Force synchronous processing to avoid 1-frame delay from pipelined orchestrator
    requires_sync_processing = True

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Post-Process Transform + Color Correction",
            "description": "Image-space postprocessing with geometric transforms and color grading",
            "parameters": {
                "feedback_strength": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Feedback blend strength (0.0 = pure input, 1.0 = pure transformed+corrected feedback)"
                },
                "zoom": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.75, 1.5],
                    "step": 0.01,
                    "description": "Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)"
                },
                "pan_x": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-0.15, 0.15],
                    "step": 0.001,
                    "description": "Pan in X direction (normalized: -1.0 to 1.0 is full width)"
                },
                "pan_y": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-0.15, 0.15],
                    "step": 0.001,
                    "description": "Pan in Y direction (normalized: -1.0 to 1.0 is full height)"
                },
                "rotation": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-10.0, 10.0],
                    "step": 0.1,
                    "description": "Rotation angle in degrees (positive = clockwise)"
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
                "black_level": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 0.3],
                    "step": 0.001,
                    "description": "Black level lift (raises minimum luminance)"
                },
                "border_mode": {
                    "type": "string",
                    "default": "zeros",
                    "options": ["zeros", "border", "reflection"],
                    "description": "How to handle borders: zeros (black), border (edge repeat), reflection (mirror)"
                }
            },
            "use_cases": [
                "Final output motion and color grading",
                "Deforum-style effects with color correction",
                "Feedback loops with geometric transforms and color",
                "All-in-one postprocessing control"
            ]
        }

    def __init__(self,
                 pipeline_ref: Any,
                 image_resolution: int = 512,
                 feedback_strength: float = 0.5,
                 zoom: float = 1.0,
                 pan_x: float = 0.0,
                 pan_y: float = 0.0,
                 rotation: float = 0.0,
                 brightness: float = 0.0,
                 saturation: float = 1.0,
                 black_level: float = 0.0,
                 border_mode: Literal["zeros", "border", "reflection"] = "zeros",
                 **kwargs):
        """
        Initialize post-process transform + color correction preprocessor

        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            image_resolution: Output image resolution
            feedback_strength: Feedback blend strength (0.0 = pure input, 1.0 = pure feedback)
            zoom: Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
            pan_x: Pan in X direction (normalized: -1.0 to 1.0 is full width)
            pan_y: Pan in Y direction (normalized: -1.0 to 1.0 is full height)
            rotation: Rotation angle in degrees (positive = clockwise)
            brightness: Brightness adjustment (-1.0 to 1.0)
            saturation: Saturation multiplier (0.0 to 2.0)
            black_level: Black level lift (0.0 to 0.3)
            border_mode: How to handle borders ("zeros", "border", "reflection")
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            image_resolution=image_resolution,
            feedback_strength=feedback_strength,
            zoom=zoom,
            pan_x=pan_x,
            pan_y=pan_y,
            rotation=rotation,
            brightness=brightness,
            saturation=saturation,
            black_level=black_level,
            border_mode=border_mode,
            **kwargs
        )
        self.feedback_strength = max(0.0, min(1.0, feedback_strength))  # Clamp to [0, 1]
        self.zoom = zoom
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.rotation = rotation
        self.brightness = brightness
        self.saturation = saturation
        self.black_level = black_level
        self.border_mode = border_mode
        self._first_frame = True

        # Map border_mode to grid_sample padding mode
        self._padding_mode_map = {
            "zeros": "zeros",
            "border": "border",
            "reflection": "reflection"
        }
        self._padding_mode = self._padding_mode_map.get(border_mode, "zeros")

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

    def _create_transform_grid(self, batch_size: int, channels: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Create affine transformation grid for grid_sample

        Combines zoom, pan, and rotation into a single affine matrix.

        Args:
            batch_size: Batch size
            channels: Number of channels (typically 3 for RGB)
            height: Image height
            width: Image width
            device: Device to create grid on

        Returns:
            Affine grid for F.grid_sample [B, H, W, 2]
        """
        # Convert rotation to radians
        angle_rad = torch.tensor(self.rotation * 3.14159265 / 180.0, device=device, dtype=self.dtype)
        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)

        # Build transformation matrix components
        # Zoom: scale = 1/zoom (because we're transforming sampling grid, not image)
        scale = 1.0 / self.zoom

        # Rotation matrix (around center)
        a = scale * cos_theta
        b = scale * -sin_theta
        c = scale * sin_theta
        d = scale * cos_theta

        # Translation (pan)
        # Normalize pan to grid coordinates [-1, 1]
        tx = -self.pan_x * 2.0  # Negative because we're transforming the grid
        ty = -self.pan_y * 2.0

        # Construct theta matrix [B, 2, 3]
        theta = torch.zeros(batch_size, 2, 3, device=device, dtype=self.dtype)
        theta[:, 0, 0] = a
        theta[:, 0, 1] = b
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = c
        theta[:, 1, 1] = d
        theta[:, 1, 2] = ty

        # Generate affine grid
        grid = F.affine_grid(theta, [batch_size, channels, height, width], align_corners=False)

        return grid

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

        # 3. Saturation (desaturate/saturate)
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

        # Final clamp to [0, 1] to prevent color space drift
        result = result.clamp(0, 1)

        # Restore original shape
        if len(original_shape) == 3:
            result = result.squeeze(0)

        return result

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Process using configurable blend of input + transformed+color-corrected previous frame

        Args:
            image: Current input image

        Returns:
            Blended PIL Image with transform + color correction
        """
        # Check if we have a pipeline reference and previous output
        prev_output_tensor = self._get_previous_data()

        if prev_output_tensor is None:
            # First frame or no previous output available
            self._first_frame = False
            return image

        # Convert previous output tensor to [0, 1] range
        if prev_output_tensor.dim() == 4:
            prev_output_tensor = prev_output_tensor[0]  # Remove batch dimension

        # CRITICAL FIX: Convert from [-1, 1] (VAE output) to [0, 1] (image processing range)
        prev_output_tensor = (prev_output_tensor / 2.0 + 0.5).clamp(0, 1)

        # Convert input image to tensor
        input_tensor = self.pil_to_tensor(image).squeeze(0)  # Remove batch dim [C, H, W]

        # ======================================================================
        # STEP 1: Transform the PREVIOUS output (accumulative motion)
        # ======================================================================

        # Check if any transform is active
        needs_transform = (
            abs(self.zoom - 1.0) > 1e-6 or
            abs(self.pan_x) > 1e-6 or
            abs(self.pan_y) > 1e-6 or
            abs(self.rotation) > 1e-3
        )

        if needs_transform:
            # Transform the PREVIOUS output (accumulative motion)
            # Add batch dimension for grid_sample
            prev_output_batch = prev_output_tensor.unsqueeze(0)  # [1, C, H, W]

            batch_size, channels, height, width = prev_output_batch.shape

            # Create transformation grid
            grid = self._create_transform_grid(
                batch_size, channels, height, width, prev_output_batch.device
            )

            # Apply transform
            transformed_prev = F.grid_sample(
                prev_output_batch,
                grid,
                mode='bilinear',
                padding_mode=self._padding_mode,
                align_corners=False
            )

            # Remove batch dimension
            transformed_prev = transformed_prev.squeeze(0)  # [C, H, W]
        else:
            transformed_prev = prev_output_tensor

        # ======================================================================
        # STEP 2: Apply color correction to transformed previous
        # ======================================================================

        corrected_prev = self._apply_color_correction_tensor(transformed_prev)

        # ======================================================================
        # STEP 3: Ensure matching shapes and blend
        # ======================================================================

        # Ensure both tensors have same shape for blending
        if corrected_prev.shape != input_tensor.shape:
            # Resize corrected_prev to match input
            target_size = input_tensor.shape[-2:]
            corrected_prev = corrected_prev.unsqueeze(0)  # Add batch
            corrected_prev = F.interpolate(
                corrected_prev, size=target_size, mode='bilinear', align_corners=False
            )
            corrected_prev = corrected_prev.squeeze(0)  # Remove batch

        # Blend with configurable strength
        blended_tensor = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * corrected_prev

        # CRITICAL: Clamp to [0, 1] to prevent color space drift/accumulation
        blended_tensor = blended_tensor.clamp(0, 1)

        # Convert back to PIL
        blended_pil = self.tensor_to_pil(blended_tensor)

        self._first_frame = False
        return blended_pil

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process using configurable blend of input + transformed+color-corrected previous frame
        (GPU-optimized path)

        Args:
            tensor: Current input tensor

        Returns:
            Blended tensor with transform + color correction
        """
        # Check if we have a pipeline reference and previous output
        prev_output = self._get_previous_data()

        if prev_output is None:
            # First frame or no previous output available - use input tensor
            self._first_frame = False
            # Ensure input tensor has correct format
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(device=self.device, dtype=self.dtype)

        # CRITICAL FIX: Convert from [-1, 1] (VAE output) to [0, 1] (image processing range)
        prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)

        # Normalize input tensor to [0, 1] if needed
        input_tensor = tensor
        if input_tensor.max() > 1.0:
            input_tensor = input_tensor / 255.0

        # Ensure both tensors have same format for blending
        if prev_output.dim() == 4 and prev_output.shape[0] == 1:
            prev_output = prev_output[0]  # Remove batch dimension
        if input_tensor.dim() == 4 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor[0]  # Remove batch dimension

        # ======================================================================
        # STEP 1: Transform the PREVIOUS output (accumulative motion)
        # ======================================================================

        # Check if any transform is active
        needs_transform = (
            abs(self.zoom - 1.0) > 1e-6 or
            abs(self.pan_x) > 1e-6 or
            abs(self.pan_y) > 1e-6 or
            abs(self.rotation) > 1e-3
        )

        if needs_transform:
            # Transform the PREVIOUS output (accumulative motion)
            # Add batch dimension for grid_sample
            if prev_output.dim() == 3:
                prev_output_batch = prev_output.unsqueeze(0)  # [1, C, H, W]
            else:
                prev_output_batch = prev_output

            batch_size, channels, height, width = prev_output_batch.shape

            # Create transformation grid
            grid = self._create_transform_grid(
                batch_size, channels, height, width, prev_output_batch.device
            )

            # Apply transform
            transformed_prev = F.grid_sample(
                prev_output_batch,
                grid,
                mode='bilinear',
                padding_mode=self._padding_mode,
                align_corners=False
            )

            # Remove batch dimension if needed
            if transformed_prev.shape[0] == 1:
                transformed_prev = transformed_prev.squeeze(0)  # [C, H, W]
        else:
            transformed_prev = prev_output

        # ======================================================================
        # STEP 2: Apply color correction to transformed previous
        # ======================================================================

        corrected_prev = self._apply_color_correction_tensor(transformed_prev)

        # ======================================================================
        # STEP 3: Resize if dimensions don't match
        # ======================================================================

        if corrected_prev.shape != input_tensor.shape:
            # Use the input tensor's shape as target
            if input_tensor.dim() == 3:
                target_size = input_tensor.shape[-2:]
                if corrected_prev.dim() == 3:
                    corrected_prev = corrected_prev.unsqueeze(0)
                corrected_prev = F.interpolate(
                    corrected_prev, size=target_size, mode='bilinear', align_corners=False
                )
                if corrected_prev.shape[0] == 1:
                    corrected_prev = corrected_prev.squeeze(0)

        # ======================================================================
        # STEP 4: Blend with configurable strength
        # ======================================================================

        blended_tensor = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * corrected_prev

        # CRITICAL: Clamp to [0, 1] to prevent color space drift/accumulation
        blended_tensor = blended_tensor.clamp(0, 1)

        # Ensure correct output format
        if blended_tensor.dim() == 3:
            blended_tensor = blended_tensor.unsqueeze(0)  # Add batch dimension back

        # Ensure correct device and dtype
        blended_tensor = blended_tensor.to(device=self.device, dtype=self.dtype)

        self._first_frame = False
        return blended_tensor
