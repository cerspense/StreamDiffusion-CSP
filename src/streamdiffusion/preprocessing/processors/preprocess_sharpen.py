import torch
import torch.nn.functional as F
from PIL import Image
from .base import BasePreprocessor


class PreprocessSharpenPreprocessor(BasePreprocessor):
    """
    Image preprocessing with sharpening only.

    Operates BEFORE VAE encode to sharpen input details.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Sharpen (Pre-VAE)",
            "description": "Image preprocessing with sharpening. Applied before VAE encode.",
            "parameters": {
                "sharpen_amount": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 2.0],
                    "step": 0.01,
                    "description": "Sharpening strength (0 = no sharpen, 1 = normal, 2 = extreme)"
                },
                "sharpen_radius": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.1, 5.0],
                    "step": 0.1,
                    "description": "Sharpening radius in pixels (affects detail scale)"
                }
            },
            "use_cases": [
                "Detail enhancement pre-VAE",
                "Input sharpening before encoding"
            ]
        }

    def __init__(self,
                 sharpen_amount: float = 0.5,
                 sharpen_radius: float = 1.0,
                 **kwargs):
        """
        Initialize PreprocessSharpen preprocessor

        Args:
            sharpen_amount: Sharpening strength
            sharpen_radius: Blur radius for unsharp masking
            **kwargs: Additional parameters
        """
        super().__init__(
            sharpen_amount=sharpen_amount,
            sharpen_radius=sharpen_radius,
            **kwargs
        )

        self.sharpen_amount = sharpen_amount
        self.sharpen_radius = sharpen_radius

        # Cache for efficiency
        self._cached_gaussian_kernels = {}

    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel for blurring"""
        coords = torch.arange(size, dtype=self.dtype, device=self.device)
        coords = coords - (size - 1) / 2
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')
        gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        return gaussian / gaussian.sum()

    def _get_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        """Get cached Gaussian kernel"""
        size = max(3, int(6 * sigma + 1))
        if size % 2 == 0:
            size += 1

        key = (size, sigma)
        if key not in self._cached_gaussian_kernels:
            self._cached_gaussian_kernels[key] = self._create_gaussian_kernel(size, sigma)

        return self._cached_gaussian_kernels[key]

    def _apply_kernel(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply convolution kernel to image"""
        num_channels = image.shape[1]
        padding = kernel.shape[-1] // 2

        # Expand kernel for all channels
        kernel_conv = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)

        return F.conv2d(image, kernel_conv, padding=padding, groups=num_channels)

    def _gaussian_blur(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur"""
        kernel = self._get_gaussian_kernel(sigma)
        return self._apply_kernel(image, kernel)

    def _unsharp_mask(self, image: torch.Tensor, radius: float, amount: float) -> torch.Tensor:
        """
        Apply unsharp masking for sharpening

        Args:
            image: Input tensor [B, C, H, W]
            radius: Blur radius (sigma)
            amount: Sharpening strength

        Returns:
            Sharpened image
        """
        if amount <= 0:
            return image

        # Create blurred version
        blurred = self._gaussian_blur(image, radius)

        # Create mask (original - blurred)
        mask = image - blurred

        # Apply sharpening: original + amount * mask
        sharpened = image + amount * mask

        return torch.clamp(sharpened, 0, 1)

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
        GPU-optimized processing: sharpen only

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

        # Sharpen
        if self.sharpen_amount > 0:
            result = self._unsharp_mask(result, self.sharpen_radius, self.sharpen_amount)

        # Final clamp to ensure valid range
        result = torch.clamp(result, 0, 1)

        return result
