import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Optional, Any, Literal
from .base import PipelineAwareProcessor


class FeedbackTransformPreprocessor(PipelineAwareProcessor):
    """
    Image-space feedback preprocessor with color correction and geometric transformations

    Combines color grading with geometric transforms and feedback loop. Operates in image
    space BEFORE VAE encoding, replicating traditional external feedback workflows.

    Key Features:
    - Color Correction: Brightness, saturation, contrast, gamma, black level, temperature
    - Geometric Transforms: Zoom, pan, rotate on previous output
    - Feedback Loop: Blend corrected+transformed previous with current input
    - Border Handling: Configurable edge treatment (zeros, border, reflection)

    Advantages over LatentTransformPreprocessor:
    - Operates in image space (RGB domain) - more intuitive visual control
    - No VAE encode/decode artifacts from latent transforms
    - True image-space temporal feedback (not latent-space approximation)
    - Color correction participates in feedback loop to prevent VAE brightness drift

    Processing Pipeline:
    1. Get previous frame's output image (prev_image_result) from VAE decode
    2. Convert from [-1, 1] to [0, 1] range
    3. Apply color correction to previous output (brightness/contrast/saturation/etc)
    4. Apply geometric transforms to color-corrected previous (zoom/pan/rotate)
    5. Blend corrected+transformed previous with current input (feedback_strength controls mix)
    6. Clamp to [0, 1] and send to VAE encode

    Examples:
    - brightness=-0.1, feedback_strength=0.8: Correct VAE brightness bias with strong feedback
    - zoom=1.05, saturation=1.2, feedback_strength=0.8: Zoom with saturated feedback
    - pan_x=0.01, contrast=1.1, feedback_strength=0.5: Panning with punchy blend

    CRITICAL: Requires requires_sync_processing=true to avoid 1-frame delay!
    """

    # CRITICAL: Force synchronous processing to avoid 1-frame delay from pipelined orchestrator
    requires_sync_processing = True

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Feedback Transform (Zoom/Pan/Rotate + Color + Feedback)",
            "description": "Image-space feedback with color correction and geometric transformations",
            "parameters": {
                "feedback_strength": {
                    "type": "float",
                    "default": 0.8,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Feedback blend strength (0.0 = pure input, 1.0 = pure corrected+transformed feedback)"
                },
                "brightness": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-1.0, 1.0],
                    "step": 0.01,
                    "description": "Brightness adjustment applied to feedback (-1.0 = black, 0.0 = neutral, 1.0 = white)"
                },
                "saturation": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.0, 2.0],
                    "step": 0.01,
                    "description": "Saturation multiplier applied to feedback (0.0 = grayscale, 1.0 = neutral, 2.0 = hyper-saturated)"
                },
                "contrast": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.5, 2.0],
                    "step": 0.01,
                    "description": "Contrast multiplier applied to feedback (0.5 = flat, 1.0 = neutral, 2.0 = high contrast)"
                },
                "black_level": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 0.3],
                    "step": 0.001,
                    "description": "Black level lift applied to feedback (raises minimum luminance)"
                },
                "gamma": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.5, 2.0],
                    "step": 0.01,
                    "description": "Gamma correction applied to feedback (0.5 = brighter mids, 1.0 = neutral, 2.0 = darker mids)"
                },
                "temperature": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-1.0, 1.0],
                    "step": 0.01,
                    "description": "Color temperature applied to feedback (-1.0 = cooler/blue, 0.0 = neutral, 1.0 = warmer/orange)"
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
                "border_mode": {
                    "type": "string",
                    "default": "zeros",
                    "options": ["zeros", "border", "reflection"],
                    "description": "How to handle borders: zeros (black), border (edge repeat), reflection (mirror)"
                }
            },
            "use_cases": [
                "Deforum-style camera motion with color correction",
                "Feedback loops with color grading and geometric transforms",
                "Smooth zoom/pan effects with brightness/contrast control",
                "Image-space animation with temporal color consistency"
            ]
        }

    def __init__(self,
                 pipeline_ref: Any,
                 image_resolution: int = 512,
                 feedback_strength: float = 0.8,
                 brightness: float = 0.0,
                 saturation: float = 1.0,
                 contrast: float = 1.0,
                 black_level: float = 0.0,
                 gamma: float = 1.0,
                 temperature: float = 0.0,
                 zoom: float = 1.0,
                 pan_x: float = 0.0,
                 pan_y: float = 0.0,
                 rotation: float = 0.0,
                 border_mode: Literal["zeros", "border", "reflection"] = "zeros",
                 **kwargs):
        """
        Initialize feedback transform preprocessor

        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            image_resolution: Output image resolution
            feedback_strength: Feedback blend strength (0.0 = pure input, 1.0 = pure corrected+transformed feedback)
            brightness: Brightness adjustment applied to feedback (-1.0 to 1.0)
            saturation: Saturation multiplier applied to feedback (0.0 to 2.0)
            contrast: Contrast multiplier applied to feedback (0.5 to 2.0)
            black_level: Black level lift applied to feedback (0.0 to 0.3)
            gamma: Gamma correction applied to feedback (0.5 to 2.0)
            temperature: Color temperature applied to feedback (-1.0 to 1.0)
            zoom: Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
            pan_x: Pan in X direction (normalized: -1.0 to 1.0 is full width)
            pan_y: Pan in Y direction (normalized: -1.0 to 1.0 is full height)
            rotation: Rotation angle in degrees (positive = clockwise)
            border_mode: How to handle borders ("zeros", "border", "reflection")
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            image_resolution=image_resolution,
            feedback_strength=feedback_strength,
            brightness=brightness,
            saturation=saturation,
            contrast=contrast,
            black_level=black_level,
            gamma=gamma,
            temperature=temperature,
            zoom=zoom,
            pan_x=pan_x,
            pan_y=pan_y,
            rotation=rotation,
            border_mode=border_mode,
            **kwargs
        )
        self.feedback_strength = max(0.0, min(1.0, feedback_strength))  # Clamp to [0, 1]
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.black_level = black_level
        self.gamma = gamma
        self.temperature = temperature
        self.zoom = zoom
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.rotation = rotation
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
        Process using configurable blend of input image + transformed previous frame output

        Args:
            image: Current input image

        Returns:
            Blended PIL Image (blend strength controlled by feedback_strength), or input image for first frame
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

        # CRITICAL FIX: prev_image_result is ALWAYS from VAE decode (always [-1, 1] range)
        # But VAE doesn't actually output the full [-1, 1] range - it has a gray floor
        # We need to REMAP the actual VAE range to [0, 1] to prevent gray accumulation

        # Get actual min/max from VAE output
        actual_min = prev_output_tensor.min()
        actual_max = prev_output_tensor.max()

        # Remap actual range to [0, 1] to force blacks to be black
        # This prevents gray floor accumulation in the feedback loop
        if actual_max > actual_min:
            prev_output_tensor = (prev_output_tensor - actual_min) / (actual_max - actual_min)
        else:
            # Fallback if min == max (uniform color)
            prev_output_tensor = (prev_output_tensor / 2.0 + 0.5).clamp(0, 1)

        # STEP 1: Apply color correction to prev_output FIRST
        prev_output_tensor = self._apply_color_correction_tensor(prev_output_tensor)

        # Convert input image to tensor
        input_tensor = self.pil_to_tensor(image).squeeze(0)  # Remove batch dim [C, H, W]

        # Normalize input tensor to [0, 1] if needed (same as _process_tensor_core)
        # Input may come from VaeImageProcessor which outputs [-1, 1]
        if input_tensor.min() < 0.0:
            # [-1, 1] range (from VAE preprocessor) - convert to [0, 1]
            input_tensor = (input_tensor / 2.0 + 0.5).clamp(0, 1)
        elif input_tensor.max() > 1.0:
            # [0, 255] range
            input_tensor = input_tensor / 255.0
        # else: already in [0, 1]

        # STEP 2: Check if any transform is active
        needs_transform = (
            abs(self.zoom - 1.0) > 1e-6 or
            abs(self.pan_x) > 1e-6 or
            abs(self.pan_y) > 1e-6 or
            abs(self.rotation) > 1e-3
        )

        if needs_transform:
            # STEP 3: Transform the color-corrected PREVIOUS output (accumulative motion)
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

        # Ensure both tensors have same shape for blending
        if transformed_prev.shape != input_tensor.shape:
            # Resize transformed_prev to match input
            target_size = input_tensor.shape[-2:]
            transformed_prev = transformed_prev.unsqueeze(0)  # Add batch
            transformed_prev = F.interpolate(
                transformed_prev, size=target_size, mode='bilinear', align_corners=False
            )
            transformed_prev = transformed_prev.squeeze(0)  # Remove batch

        # STEP 4: Blend color-corrected+transformed previous with input (feedback_strength controls mix)
        blended_tensor = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * transformed_prev

        # CRITICAL: Clamp to [0, 1] to prevent color space drift/accumulation
        blended_tensor = blended_tensor.clamp(0, 1)

        # Convert back to PIL
        blended_pil = self.tensor_to_pil(blended_tensor)

        self._first_frame = False
        return blended_pil

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process using configurable blend of input tensor + transformed previous frame output (GPU-optimized path)

        Args:
            tensor: Current input tensor

        Returns:
            Blended tensor (blend strength controlled by feedback_strength), or input tensor for first frame
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

        # CRITICAL FIX: prev_image_result is ALWAYS from VAE decode (always [-1, 1] range)
        # But VAE doesn't actually output the full [-1, 1] range - it has a gray floor
        # We need to REMAP the actual VAE range to [0, 1] to prevent gray accumulation
        print(f"[FEEDBACK DEBUG] prev_output BEFORE conversion: min={prev_output.min().item():.4f}, max={prev_output.max().item():.4f}")

        # Get actual min/max from VAE output
        actual_min = prev_output.min()
        actual_max = prev_output.max()

        # Remap actual range to [0, 1] to force blacks to be black
        # This prevents gray floor accumulation in the feedback loop
        if actual_max > actual_min:
            prev_output = (prev_output - actual_min) / (actual_max - actual_min)
        else:
            # Fallback if min == max (uniform color)
            prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)

        print(f"[FEEDBACK DEBUG] prev_output AFTER remapping: min={prev_output.min().item():.4f}, max={prev_output.max().item():.4f}")

        # STEP 1: Apply color correction to prev_output FIRST
        prev_output = self._apply_color_correction_tensor(prev_output)
        print(f"[FEEDBACK DEBUG] prev_output AFTER color correction: min={prev_output.min().item():.4f}, max={prev_output.max().item():.4f}")

        # Normalize input tensor to [0, 1] if needed
        # Input comes from image_processor.preprocess() which outputs [-1, 1]
        input_tensor = tensor
        print(f"[FEEDBACK DEBUG] input_tensor BEFORE normalization: min={input_tensor.min().item():.4f}, max={input_tensor.max().item():.4f}")
        if input_tensor.min() < 0.0:
            # [-1, 1] range (from VAE preprocessor) - convert to [0, 1]
            input_tensor = (input_tensor / 2.0 + 0.5).clamp(0, 1)
            print(f"[FEEDBACK DEBUG] input_tensor converted from [-1,1] to [0,1]")
        elif input_tensor.max() > 1.0:
            # [0, 255] range
            input_tensor = input_tensor / 255.0
            print(f"[FEEDBACK DEBUG] input_tensor converted from [0,255] to [0,1]")
        else:
            print(f"[FEEDBACK DEBUG] input_tensor already in [0,1] range")
        print(f"[FEEDBACK DEBUG] input_tensor AFTER normalization: min={input_tensor.min().item():.4f}, max={input_tensor.max().item():.4f}")

        # Ensure both tensors have same format for blending
        if prev_output.dim() == 4 and prev_output.shape[0] == 1:
            prev_output = prev_output[0]  # Remove batch dimension
        if input_tensor.dim() == 4 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor[0]  # Remove batch dimension

        # STEP 2: Check if any transform is active
        needs_transform = (
            abs(self.zoom - 1.0) > 1e-6 or
            abs(self.pan_x) > 1e-6 or
            abs(self.pan_y) > 1e-6 or
            abs(self.rotation) > 1e-3
        )

        if needs_transform:
            # STEP 3: Transform the color-corrected PREVIOUS output (accumulative motion)
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

        # Resize if dimensions don't match
        if transformed_prev.shape != input_tensor.shape:
            # Use the input tensor's shape as target
            if input_tensor.dim() == 3:
                target_size = input_tensor.shape[-2:]
                if transformed_prev.dim() == 3:
                    transformed_prev = transformed_prev.unsqueeze(0)
                transformed_prev = F.interpolate(
                    transformed_prev, size=target_size, mode='bilinear', align_corners=False
                )
                if transformed_prev.shape[0] == 1:
                    transformed_prev = transformed_prev.squeeze(0)

        # STEP 4: Blend color-corrected+transformed previous with input (feedback_strength controls mix)
        blended_tensor = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * transformed_prev

        # CRITICAL: Clamp to [0, 1] to prevent color space drift/accumulation
        blended_tensor = blended_tensor.clamp(0, 1)

        # Ensure correct output format
        if blended_tensor.dim() == 3:
            blended_tensor = blended_tensor.unsqueeze(0)  # Add batch dimension back

        # Ensure correct device and dtype
        blended_tensor = blended_tensor.to(device=self.device, dtype=self.dtype)

        self._first_frame = False
        return blended_tensor
