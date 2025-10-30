import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union
from streamdiffusion.preprocessing.processors import BasePreprocessor


class PostProcessSharpenNoisePreprocessor(BasePreprocessor):
    """
    Post-processing processor combining image sharpening with fractal noise injection.

    Operates after VAE decode to:
    - Sharpen image details with configurable radius and amount
    - Add multi-octave fractal noise with adjustable period and strength

    Perfect for adding texture and enhancing details in the final output.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "stage": "image_post",
            "display_name": "Sharpen + Noise",
            "description": "Post-processing that combines sharpening with fractal noise injection. Enhances details and adds texture to final output.",
            "parameters": {
                # Sharpen controls
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
                },
                # Noise controls
                "noise_strength": {
                    "type": "float",
                    "default": 0.0,
                    "range": [0.0, 0.3],
                    "step": 0.001,
                    "description": "Fractal noise strength (0 = no noise, 0.1 = subtle, 0.3 = strong)"
                },
                "noise_period": {
                    "type": "float",
                    "default": 4.0,
                    "range": [1.0, 64.0],
                    "step": 0.5,
                    "description": "Noise period/frequency (lower = finer grain, higher = coarser)"
                },
                "noise_octaves": {
                    "type": "int",
                    "default": 3,
                    "range": [1, 6],
                    "step": 1,
                    "description": "Number of noise octaves for fractal detail (more = richer texture)"
                },
                "noise_persistence": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.1, 1.0],
                    "step": 0.05,
                    "description": "How much each octave contributes (lower = smoother, higher = rougher)"
                },
                "noise_lacunarity": {
                    "type": "float",
                    "default": 2.0,
                    "range": [1.0, 4.0],
                    "step": 0.1,
                    "description": "Frequency multiplier between octaves (controls texture complexity)"
                }
            },
            "use_cases": [
                "Detail enhancement with texture",
                "Film grain simulation",
                "Organic texture addition",
                "Final output sharpening"
            ]
        }

    def __init__(self,
                 sharpen_amount: float = 0.5,
                 sharpen_radius: float = 1.0,
                 noise_strength: float = 0.0,
                 noise_period: float = 4.0,
                 noise_octaves: int = 3,
                 noise_persistence: float = 0.5,
                 noise_lacunarity: float = 2.0,
                 **kwargs):
        """
        Initialize PostProcessSharpenNoise preprocessor

        Args:
            sharpen_amount: Sharpening strength
            sharpen_radius: Blur radius for unsharp masking
            noise_strength: Fractal noise injection strength
            noise_period: Base noise frequency/period
            noise_octaves: Number of fractal octaves
            noise_persistence: Amplitude falloff for each octave
            noise_lacunarity: Frequency multiplier between octaves
            **kwargs: Additional parameters
        """
        super().__init__(
            sharpen_amount=sharpen_amount,
            sharpen_radius=sharpen_radius,
            noise_strength=noise_strength,
            noise_period=noise_period,
            noise_octaves=noise_octaves,
            noise_persistence=noise_persistence,
            noise_lacunarity=noise_lacunarity,
            **kwargs
        )

        # Store parameters as instance attributes for OSC live updates
        self.sharpen_amount = sharpen_amount
        self.sharpen_radius = sharpen_radius
        self.noise_strength = noise_strength
        self.noise_period = noise_period
        self.noise_octaves = noise_octaves
        self.noise_persistence = noise_persistence
        self.noise_lacunarity = noise_lacunarity

        # Cache for efficiency
        self._cached_gaussian_kernels = {}
        self._noise_offset = torch.rand(2, device=self.device, dtype=self.dtype) * 1000.0

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

    def _generate_perlin_noise_2d(self, shape: tuple, period: float) -> torch.Tensor:
        """
        Generate 2D Perlin-style gradient noise

        Args:
            shape: (height, width)
            period: Base frequency/period of the noise

        Returns:
            Noise tensor [H, W]
        """
        height, width = shape

        # Create coordinate grid
        y = torch.linspace(0, height / period, height, device=self.device, dtype=self.dtype)
        x = torch.linspace(0, width / period, width, device=self.device, dtype=self.dtype)

        # Add random offset to vary noise pattern each time
        y = y + self._noise_offset[0]
        x = x + self._noise_offset[1]

        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Simple gradient noise using sine waves
        # This creates a smooth, organic noise pattern
        noise = torch.sin(xx * 2 * np.pi) * torch.cos(yy * 2 * np.pi)
        noise += torch.sin(xx * 4 * np.pi + 1.5) * torch.cos(yy * 4 * np.pi + 1.5) * 0.5
        noise += torch.sin(xx * 8 * np.pi + 3.0) * torch.cos(yy * 8 * np.pi + 3.0) * 0.25

        # Normalize to [-1, 1]
        noise = noise / (1.0 + 0.5 + 0.25)

        return noise

    def _generate_fractal_noise(self, shape: tuple, base_period: float,
                                octaves: int, persistence: float,
                                lacunarity: float) -> torch.Tensor:
        """
        Generate multi-octave fractal noise

        Args:
            shape: (height, width)
            base_period: Base frequency/period
            octaves: Number of octaves to combine
            persistence: Amplitude multiplier for each octave (typically 0.5)
            lacunarity: Frequency multiplier for each octave (typically 2.0)

        Returns:
            Fractal noise tensor [H, W] in range [-1, 1]
        """
        height, width = shape
        noise = torch.zeros(height, width, device=self.device, dtype=self.dtype)

        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0

        for octave in range(octaves):
            # Generate noise at this octave's frequency
            octave_noise = self._generate_perlin_noise_2d(
                shape,
                base_period / frequency
            )

            # Add to accumulator with current amplitude
            noise += octave_noise * amplitude

            # Track max amplitude for normalization
            max_amplitude += amplitude

            # Update for next octave
            amplitude *= persistence
            frequency *= lacunarity

        # Normalize to [-1, 1]
        if max_amplitude > 0:
            noise = noise / max_amplitude

        return noise

    def _add_fractal_noise(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add fractal noise to image

        Args:
            image: Input tensor [B, C, H, W]

        Returns:
            Image with noise added
        """
        if self.noise_strength <= 0:
            return image

        batch_size, channels, height, width = image.shape

        # Generate fractal noise
        noise = self._generate_fractal_noise(
            (height, width),
            self.noise_period,
            self.noise_octaves,
            self.noise_persistence,
            self.noise_lacunarity
        )

        # Expand noise to match image dimensions [B, C, H, W]
        noise = noise.unsqueeze(0).unsqueeze(0)
        noise = noise.repeat(batch_size, channels, 1, 1)

        # Add noise to image (noise is in [-1, 1], scale by strength)
        noisy_image = image + noise * self.noise_strength

        return torch.clamp(noisy_image, 0, 1)

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
        GPU-optimized processing: sharpen + noise

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

        # Step 1: Sharpen
        if self.sharpen_amount > 0:
            result = self._unsharp_mask(result, self.sharpen_radius, self.sharpen_amount)

        # Step 2: Add fractal noise
        result = self._add_fractal_noise(result)

        # Final clamp to ensure valid range
        result = torch.clamp(result, 0, 1)

        return result
