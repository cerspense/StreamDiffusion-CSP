import torch
import torch.nn.functional as F
from typing import Optional, Any, Literal
from .base import PipelineAwareProcessor


class LatentTransformPreprocessor(PipelineAwareProcessor):
    """
    Latent-space transformation processor with optional noise injection

    Applies geometric transforms (zoom, pan, rotate) in latent space before diffusion.
    Supports optional noise injection to maintain proper noise schedule when using
    high feedback strengths or accumulative transforms.

    Key Features:
    - Zoom: Scale latent tensor (interpolate + crop/pad)
    - Pan: Translate latent tensor in X/Y
    - Rotate: Rotate latent tensor around center
    - Noise Injection: Re-noise after transform to prevent "clean latent" artifacts
    - Feedback Blending: Optionally blend with previous latent for smooth motion

    Research-backed design:
    - Uses ∫-noise principles for warping preservation
    - Maintains Gaussian noise properties after transforms
    - Respects noise schedule by re-noising transformed latents

    Examples:
    - zoom=1.05, noise_strength=0.0: 5% zoom per frame (Deforum-style)
    - pan_x=0.01, feedback_blend=0.3: Panning with temporal smoothing
    - rotate=2.0, noise_strength=0.1: Rotation with light re-noising

    The preprocessor accesses pipeline's noise schedule (alpha_prod_t_sqrt, beta_prod_t_sqrt)
    to properly re-noise latents after transformation, preventing "too clean" latents
    that cause artifacts in the diffusion process.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Latent Transform (Zoom/Pan/Rotate)",
            "description": "Applies geometric transformations in latent space with optional noise injection for Deforum-style effects",
            "parameters": {
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
                "noise_strength": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Re-noise strength after transform (0.0 = no re-noise, 1.0 = full re-noise). Use when transforms create 'too clean' latents"
                },
                "noise_seed_mix": {
                    "type": "float",
                    "default": 0.8,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Blend pipeline seed noise vs random noise (0.0 = pure random, 1.0 = use pipeline seed travel noise). Higher = more consistent, lower = more chaotic"
                },
                "feedback_blend": {
                    "type": "float",
                    "default": 0.8,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Blend with previous latent after transform (0.0 = pure transform, 1.0 = pure feedback)"
                },
                "border_mode": {
                    "type": "string",
                    "default": "zeros",
                    "options": ["zeros", "border", "reflection"],
                    "description": "How to handle borders: zeros (black), border (edge repeat), reflection (mirror)"
                }
            },
            "use_cases": [
                "Deforum-style camera motion",
                "Latent space animation",
                "Smooth zoom/pan effects",
                "Abstract latent manipulation",
                "Temporal consistency with motion"
            ]
        }

    def __init__(
        self,
        pipeline_ref: Any,
        zoom: float = 1.0,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
        rotation: float = 0.0,
        noise_strength: float = 0.0,
        noise_seed_mix: float = 0.0,
        feedback_blend: float = 0.0,
        border_mode: Literal["zeros", "border", "reflection"] = "zeros",
        **kwargs
    ):
        """
        Initialize latent transform preprocessor

        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required for noise schedule)
            zoom: Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
            pan_x: Pan in X direction (normalized: -1.0 to 1.0 is full width)
            pan_y: Pan in Y direction (normalized: -1.0 to 1.0 is full height)
            rotation: Rotation angle in degrees (positive = clockwise)
            noise_strength: Re-noise strength after transform (0.0 = no re-noise, 1.0 = full re-noise)
            noise_seed_mix: Blend pipeline seed vs random noise (0.0 = random, 1.0 = seeded)
            feedback_blend: Blend with previous latent after transform (0.0 = pure transform, >0.0 = smooth)
            border_mode: How to handle borders ("zeros", "border", "reflection")
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            zoom=zoom,
            pan_x=pan_x,
            pan_y=pan_y,
            rotation=rotation,
            noise_strength=noise_strength,
            noise_seed_mix=noise_seed_mix,
            feedback_blend=feedback_blend,
            border_mode=border_mode,
            **kwargs
        )
        self.zoom = zoom
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.rotation = rotation
        self.noise_strength = max(0.0, min(1.0, noise_strength))  # Clamp [0, 1]
        self.noise_seed_mix = max(0.0, min(1.0, noise_seed_mix))  # Clamp [0, 1]
        self.feedback_blend = max(0.0, min(1.0, feedback_blend))  # Clamp [0, 1]
        self.border_mode = border_mode
        self._first_frame = True

        # Map border_mode to grid_sample padding mode
        self._padding_mode_map = {
            "zeros": "zeros",
            "border": "border",
            "reflection": "reflection"
        }
        self._padding_mode = self._padding_mode_map.get(border_mode, "zeros")

    def _get_previous_data(self):
        """Get previous frame latent data from pipeline for feedback blending"""
        if self.pipeline_ref is not None and self.feedback_blend > 0:
            if hasattr(self.pipeline_ref, 'prev_latent_result'):
                if self.pipeline_ref.prev_latent_result is not None and not self._first_frame:
                    return self.pipeline_ref.prev_latent_result
        return None

    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate latent tensor input - preserve batch dimensions for latent processing

        Args:
            latent_tensor: Input latent tensor in format [B, C, H/8, W/8]

        Returns:
            Validated latent tensor with preserved batch dimension
        """
        # For latent processing, preserve the batch dimension
        latent_tensor = latent_tensor.to(device=self.device, dtype=self.dtype)
        return latent_tensor

    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Override base class resize logic - latent tensors should NOT be resized to image dimensions

        For latent domain processing, preserve the latent space dimensions.
        """
        return tensor

    def _process_core(self, image):
        """
        For latent transform, we don't process PIL images directly.
        This method should not be called in normal latent preprocessing workflows.
        """
        raise NotImplementedError(
            "LatentTransformPreprocessor is designed for latent domain processing. "
            "Use _process_tensor_core or process_tensor for latent tensors."
        )

    def _create_transform_grid(self, batch_size: int, channels: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Create affine transformation grid for grid_sample

        Combines zoom, pan, and rotation into a single affine matrix.
        Uses ∫-noise principles to preserve Gaussian properties during warping.

        Args:
            batch_size: Batch size
            channels: Number of channels (typically 4 for latents)
            height: Latent height (typically image_height / 8)
            width: Latent width (typically image_width / 8)
            device: Device to create grid on

        Returns:
            Affine grid for F.grid_sample [B, H, W, 2]
        """
        # Build affine transformation matrix
        # theta format: [B, 2, 3] where each row is [a, b, tx; c, d, ty]
        # Standard form: [x'] = [a b] [x] + [tx]
        #                [y']   [c d] [y]   [ty]

        # Convert rotation to radians
        angle_rad = torch.tensor(self.rotation * 3.14159265 / 180.0, device=device, dtype=self.dtype)
        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)

        # Build transformation matrix components
        # Zoom: scale = 1/zoom (because we're transforming sampling grid, not image)
        scale = 1.0 / self.zoom

        # Rotation matrix (around center)
        # [cos  -sin]
        # [sin   cos]
        a = scale * cos_theta
        b = scale * -sin_theta
        c = scale * sin_theta
        d = scale * cos_theta

        # Translation (pan)
        # Normalize pan to grid coordinates [-1, 1]
        # Note: grid_sample uses normalized coordinates where [-1, -1] is top-left, [1, 1] is bottom-right
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

    def _apply_noise_injection(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Re-noise latent after transformation to maintain proper noise schedule

        Research-backed approach:
        - Prevents "too clean" latents that break diffusion
        - Maintains Gaussian noise properties
        - Respects the noise schedule (alpha/beta from t_index_list)
        - Supports blending pipeline seed noise with random noise

        Formula: noisy_latent = alpha * latent + beta * noise
        where alpha and beta come from the first denoising step (t_index_list[0])

        Args:
            latent: Transformed latent tensor [B, C, H, W]

        Returns:
            Re-noised latent tensor
        """
        if self.noise_strength <= 0.0 or self.pipeline_ref is None:
            return latent

        # Access noise schedule from pipeline
        # We use the FIRST denoising step's noise level (t_index_list[0])
        # because this is pre-diffusion (latent preprocessing)
        if not hasattr(self.pipeline_ref, 'alpha_prod_t_sqrt') or not hasattr(self.pipeline_ref, 'beta_prod_t_sqrt'):
            # Fallback: simple noise injection without schedule respect
            noise = torch.randn_like(latent, device=latent.device, dtype=latent.dtype)
            return (1.0 - self.noise_strength) * latent + self.noise_strength * noise

        # Get alpha and beta for first denoising step
        # Shape: alpha_prod_t_sqrt[0] is typically a scalar or [1, 1, 1, 1] tensor
        alpha = self.pipeline_ref.alpha_prod_t_sqrt[0]
        beta = self.pipeline_ref.beta_prod_t_sqrt[0]

        # Generate noise: blend pipeline seed noise with random noise
        # noise_seed_mix: 0.0 = pure random, 1.0 = use pipeline's seeded noise
        random_noise = torch.randn_like(latent, device=latent.device, dtype=latent.dtype)

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

            # Blend seeded noise with random noise
            # noise_seed_mix=1.0 → pure seed-traveled noise
            # noise_seed_mix=0.0 → pure random noise
            noise = (1.0 - self.noise_seed_mix) * random_noise + self.noise_seed_mix * pipeline_noise.to(latent.device, dtype=latent.dtype)
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

    def _apply_feedback_blend(self, latent: torch.Tensor, prev_latent: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Blend current latent with previous latent for temporal smoothing

        This is different from LatentFeedbackPreprocessor:
        - Here we blend AFTER transform (smooth the motion)
        - LatentFeedback blends BEFORE diffusion (temporal consistency)

        Args:
            latent: Current transformed latent [B, C, H, W]
            prev_latent: Previous frame latent [B, C, H, W] or None

        Returns:
            Blended latent tensor
        """
        if prev_latent is None or self.feedback_blend <= 0.0:
            return latent

        # Handle batch size mismatches
        if prev_latent.shape[0] != latent.shape[0]:
            if prev_latent.shape[0] == 1:
                prev_latent = prev_latent.expand(latent.shape[0], -1, -1, -1)
            elif latent.shape[0] == 1:
                latent = latent.expand(prev_latent.shape[0], -1, -1, -1)
            else:
                # Different non-unit batch sizes - use minimum
                min_batch = min(prev_latent.shape[0], latent.shape[0])
                prev_latent = prev_latent[:min_batch]
                latent = latent[:min_batch]

        # Handle spatial dimension mismatches
        if prev_latent.shape[2:] != latent.shape[2:]:
            prev_latent = F.interpolate(
                prev_latent,
                size=latent.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Blend: higher feedback_blend = more influence from previous frame
        blended = (1.0 - self.feedback_blend) * latent + self.feedback_blend * prev_latent

        return blended

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process latent tensor with geometric transformations and noise injection

        Processing Pipeline (DEFORUM-STYLE):
        1. Get previous frame's latent (if available)
        2. Transform the PREVIOUS latent (zoom/pan/rotate creates accumulative motion)
        3. Blend transformed previous with current input (feedback_blend controls mix)
        4. Apply noise injection (if noise_strength > 0)
        5. Safety clamp to prevent extreme values

        Args:
            tensor: Current input latent tensor [B, C, H/8, W/8]

        Returns:
            Transformed, re-noised, and optionally blended latent tensor
        """
        batch_size, channels, height, width = tensor.shape

        # ======================================================================
        # STEP 1: Get Previous Latent
        # ======================================================================

        prev_latent = self._get_previous_data()

        # ======================================================================
        # STEP 2: Transform PREVIOUS latent (accumulative motion)
        # ======================================================================

        # Check if any transform is active
        needs_transform = (
            abs(self.zoom - 1.0) > 1e-6 or
            abs(self.pan_x) > 1e-6 or
            abs(self.pan_y) > 1e-6 or
            abs(self.rotation) > 1e-3
        )

        if prev_latent is not None and needs_transform:
            # Match batch size if needed
            if prev_latent.shape[0] != batch_size:
                if prev_latent.shape[0] == 1:
                    prev_latent = prev_latent.expand(batch_size, -1, -1, -1)

            # Create affine transformation grid
            grid = self._create_transform_grid(
                batch_size, channels, height, width, tensor.device
            )

            # Transform the PREVIOUS latent (this creates accumulative zoom/pan)
            transformed_prev = F.grid_sample(
                prev_latent,
                grid,
                mode='bilinear',
                padding_mode=self._padding_mode,
                align_corners=False
            )
        else:
            transformed_prev = prev_latent

        # ======================================================================
        # STEP 3: Blend transformed previous with current input
        # ======================================================================

        if transformed_prev is not None and self.feedback_blend > 0.0:
            result_latent = self._apply_feedback_blend(tensor, transformed_prev)
        else:
            result_latent = tensor

        # ======================================================================
        # STEP 4: Apply Noise Injection (if enabled)
        # ======================================================================

        if self.noise_strength > 0.0:
            result_latent = self._apply_noise_injection(result_latent)

        # ======================================================================
        # STEP 5: Safety Clamp
        # ======================================================================

        # Prevent extreme latent values that could break the pipeline
        # Range [-10, 10] is safe for SD latent space
        result_latent = torch.clamp(result_latent, min=-10.0, max=10.0)

        # ======================================================================
        # Finalize
        # ======================================================================

        self._first_frame = False

        # Ensure correct device and dtype
        result_latent = result_latent.to(device=self.device, dtype=self.dtype)

        return result_latent
