import torch
import torch.nn.functional as F
from PIL import Image
from .base import PipelineAwareProcessor


class FeedbackMorphologyPreprocessor(PipelineAwareProcessor):
    """
    Morphological operations applied to PREVIOUS FRAME before feedback mixing.

    Operates BEFORE VAE encode, applying dilate/erode to the previous output
    BEFORE it gets mixed with the current input. Perfect for creating temporal
    glow effects that persist across frames.

    Requires sync processing to avoid 1-frame delay.
    """

    # CRITICAL: Force sync processing to avoid 1-frame delay from pipelined orchestrator
    requires_sync_processing = True

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Feedback Morphology (Pre-Mix)",
            "description": "Morphological dilate/erode on PREVIOUS FRAME before mixing with current input. Creates temporal glow effects.",
            "parameters": {
                "dilate_amount": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 10.0],
                    "step": 0.1,
                    "description": "Dilation on previous frame (0 = off, expands bright areas before mixing)"
                },
                "erode_amount": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 10.0],
                    "step": 0.1,
                    "description": "Erosion on previous frame (0 = off, contracts bright areas before mixing)"
                },
                "kernel_size": {
                    "type": "int",
                    "default": 3,
                    "range": [3, 15],
                    "step": 2,
                    "description": "Kernel size for morphology (must be odd, larger = stronger effect)"
                },
                "feedback_strength": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Mix amount (0 = only input, 1 = only morphed previous frame)"
                }
            },
            "use_cases": [
                "Temporal glow that persists and grows",
                "Expanding halos across frames",
                "Edge bleeding effects",
                "Vintage film temporal artifacts"
            ]
        }

    def __init__(self,
                 pipeline_ref,
                 dilate_amount: float = 0.0,
                 erode_amount: float = 0.0,
                 kernel_size: int = 3,
                 feedback_strength: float = 0.0,
                 **kwargs):
        """
        Initialize FeedbackMorphology preprocessor

        Args:
            pipeline_ref: Pipeline reference for accessing previous frames
            dilate_amount: Dilation strength on previous frame
            erode_amount: Erosion strength on previous frame
            kernel_size: Size of morphological kernel (must be odd)
            feedback_strength: Mix amount with current input
            **kwargs: Additional parameters
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            dilate_amount=dilate_amount,
            erode_amount=erode_amount,
            kernel_size=kernel_size,
            feedback_strength=feedback_strength,
            **kwargs
        )

        self.dilate_amount = dilate_amount
        self.erode_amount = erode_amount
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd
        self.feedback_strength = feedback_strength

        self._first_frame = True

    def _create_morphology_kernel(self, size: int) -> torch.Tensor:
        """Create circular morphological kernel"""
        kernel = torch.zeros(size, size, dtype=self.dtype, device=self.device)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= center ** 2:
                    kernel[i, j] = 1.0
        return kernel / kernel.sum()

    def _apply_morphology_kernel(self, image: torch.Tensor, kernel: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply morphological operation"""
        num_channels = image.shape[1]
        padding = kernel.shape[-1] // 2

        # Expand kernel for all channels
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)

        if mode == 'dilate':
            # Dilation = max pooling with kernel
            result = F.conv2d(image, kernel_conv, padding=padding, groups=num_channels)
            result = torch.clamp(result * 2.0, 0, 1)
        elif mode == 'erode':
            # Erosion = min pooling with kernel
            inverted = 1.0 - image
            result = F.conv2d(inverted, kernel_conv, padding=padding, groups=num_channels)
            result = torch.clamp(result * 2.0, 0, 1)
            result = 1.0 - result
        else:
            result = image

        return result

    def _dilate(self, image: torch.Tensor, iterations: int, kernel_size: int) -> torch.Tensor:
        """Apply dilation multiple times"""
        if iterations <= 0:
            return image

        kernel = self._create_morphology_kernel(kernel_size)
        result = image.clone()

        # Apply dilation iterations
        num_iterations = int(iterations)
        for _ in range(num_iterations):
            result = self._apply_morphology_kernel(result, kernel, 'dilate')

        # Fractional iteration (blend)
        frac = iterations - num_iterations
        if frac > 0:
            dilated = self._apply_morphology_kernel(result, kernel, 'dilate')
            result = (1 - frac) * result + frac * dilated

        return torch.clamp(result, 0, 1)

    def _erode(self, image: torch.Tensor, iterations: int, kernel_size: int) -> torch.Tensor:
        """Apply erosion multiple times"""
        if iterations <= 0:
            return image

        kernel = self._create_morphology_kernel(kernel_size)
        result = image.clone()

        # Apply erosion iterations
        num_iterations = int(iterations)
        for _ in range(num_iterations):
            result = self._apply_morphology_kernel(result, kernel, 'erode')

        # Fractional iteration (blend)
        frac = iterations - num_iterations
        if frac > 0:
            eroded = self._apply_morphology_kernel(result, kernel, 'erode')
            result = (1 - frac) * result + frac * eroded

        return torch.clamp(result, 0, 1)

    def _get_previous_output(self):
        """Get previous frame output from pipeline"""
        if self.pipeline_ref is not None:
            if hasattr(self.pipeline_ref, 'prev_image_result'):
                if self.pipeline_ref.prev_image_result is not None and not self._first_frame:
                    return self.pipeline_ref.prev_image_result
        return None

    def _process_core(self, image: Image.Image) -> Image.Image:
        """Process PIL image (fallback path)"""
        # Convert to tensor
        tensor = self.pil_to_tensor(image)
        tensor = tensor.squeeze(0)  # Remove batch dimension if present

        # Process on GPU
        processed = self._process_tensor_core(tensor)

        # Convert back to PIL
        return self.tensor_to_pil(processed)

    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized processing: dilate/erode previous frame, then mix with current

        Args:
            image_tensor: [B, C, H, W] or [C, H, W] in range [0, 1]

        Returns:
            Processed tensor, same shape
        """
        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Ensure correct device and dtype
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)

        # Get previous frame
        prev_output = self._get_previous_output()

        if prev_output is None:
            # First frame - passthrough
            self._first_frame = False
            return image_tensor

        # CRITICAL: Convert from VAE output range [-1, 1] to image range [0, 1]
        prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)

        # Handle batch size mismatch
        if prev_output.shape[0] != image_tensor.shape[0]:
            if prev_output.shape[0] < image_tensor.shape[0]:
                prev_output = prev_output.repeat(image_tensor.shape[0], 1, 1, 1)
            else:
                prev_output = prev_output[:image_tensor.shape[0]]

        # Apply morphology to PREVIOUS FRAME
        morphed_prev = prev_output.clone()

        # Dilate previous frame first (expands bright areas)
        if self.dilate_amount > 0:
            morphed_prev = self._dilate(morphed_prev, self.dilate_amount, self.kernel_size)

        # Then erode (contracts bright areas)
        if self.erode_amount > 0:
            morphed_prev = self._erode(morphed_prev, self.erode_amount, self.kernel_size)

        # Mix morphed previous frame with current input
        if self.feedback_strength > 0:
            result = (1 - self.feedback_strength) * image_tensor + self.feedback_strength * morphed_prev
        else:
            result = image_tensor

        # Final clamp to ensure valid range
        result = torch.clamp(result, 0, 1)

        self._first_frame = False
        return result
