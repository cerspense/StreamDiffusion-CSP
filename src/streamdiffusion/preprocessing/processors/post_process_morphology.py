import torch
import torch.nn.functional as F
from PIL import Image
from .base import BasePreprocessor


class PostProcessMorphologyPreprocessor(BasePreprocessor):
    """
    Morphological operations (dilate/erode) for image postprocessing.

    Operates AFTER VAE decode on the final output image.
    Great for final glow/halo effects, edge expansion/contraction, and aesthetic finishing touches.
    Does not affect the diffusion process itself.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Morphology (Post-VAE)",
            "description": "Morphological dilate/erode operations. Applied after VAE decode to final output.",
            "parameters": {
                "dilate_amount": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 10.0],
                    "step": 0.1,
                    "description": "Dilation strength (0 = off, expands bright areas)"
                },
                "erode_amount": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 10.0],
                    "step": 0.1,
                    "description": "Erosion strength (0 = off, contracts bright areas)"
                },
                "kernel_size": {
                    "type": "int",
                    "default": 3,
                    "range": [3, 15],
                    "step": 2,
                    "description": "Kernel size for morphology (must be odd, larger = stronger effect)"
                }
            },
            "use_cases": [
                "Final output glow/halo effects",
                "Edge expansion without affecting diffusion",
                "Aesthetic finishing touches",
                "Clean up VAE decode artifacts"
            ]
        }

    def __init__(self,
                 dilate_amount: float = 0.0,
                 erode_amount: float = 0.0,
                 kernel_size: int = 3,
                 **kwargs):
        """
        Initialize PostProcessMorphology preprocessor

        Args:
            dilate_amount: Dilation strength (iterations)
            erode_amount: Erosion strength (iterations)
            kernel_size: Size of morphological kernel (must be odd)
            **kwargs: Additional parameters
        """
        super().__init__(
            dilate_amount=dilate_amount,
            erode_amount=erode_amount,
            kernel_size=kernel_size,
            **kwargs
        )

        self.dilate_amount = dilate_amount
        self.erode_amount = erode_amount
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd

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
        """
        Apply morphological operation

        Args:
            image: Input tensor [B, C, H, W]
            kernel: Morphological kernel
            mode: 'dilate' or 'erode'

        Returns:
            Processed tensor
        """
        num_channels = image.shape[1]
        padding = kernel.shape[-1] // 2

        # Expand kernel for all channels
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)

        if mode == 'dilate':
            # Dilation = max pooling with kernel
            # We'll use convolution approximation: convolve and threshold
            result = F.conv2d(image, kernel_conv, padding=padding, groups=num_channels)
            # Boost the result to approximate max operation
            result = torch.clamp(result * 2.0, 0, 1)
        elif mode == 'erode':
            # Erosion = min pooling with kernel
            # Approximate by inverting, dilating, then inverting back
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
        GPU-optimized processing: dilate and/or erode

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

        result = image_tensor.clone()

        # Apply dilate first (expands bright areas)
        if self.dilate_amount > 0:
            result = self._dilate(result, self.dilate_amount, self.kernel_size)

        # Then apply erode (contracts bright areas)
        if self.erode_amount > 0:
            result = self._erode(result, self.erode_amount, self.kernel_size)

        # Final clamp to ensure valid range
        result = torch.clamp(result, 0, 1)

        return result
