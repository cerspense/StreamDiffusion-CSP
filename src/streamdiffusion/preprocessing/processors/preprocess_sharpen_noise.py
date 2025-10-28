import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union
from .base import BasePreprocessor


class PreprocessSharpenNoisePreprocessor(BasePreprocessor):
    """
    Image preprocessing combining sharpening with high-quality fractal noise.

    Operates BEFORE VAE encode so the diffusion process can integrate the noise naturally.
    Uses proper Perlin-style gradient noise with bilinear interpolation for smooth, organic results.

    Perfect for adding film grain or texture that gets processed through diffusion.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Sharpen + Noise (Pre-Diffusion)",
            "description": "Image preprocessing with sharpening and high-quality fractal noise. Applied before VAE encode so diffusion integrates the noise organically.",
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
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Fractal noise strength (0 = no noise, 0.1 = subtle, 0.3 = moderate, 1.0 = extreme)"
                },
                "noise_scale": {
                    "type": "float",
                    "default": 8.0,
                    "range": [1.0, 64.0],
                    "step": 0.5,
                    "description": "Noise scale/frequency (lower = finer grain, higher = coarser)"
                },
                "noise_octaves": {
                    "type": "int",
                    "default": 4,
                    "range": [1, 8],
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
                },
                "noise_temporal_coherence": {
                    "type": "float",
                    "default": 0.95,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Temporal smoothing of noise (0 = new noise every frame, 1 = static)"
                }
            },
            "use_cases": [
                "Film grain that gets diffused naturally",
                "Organic texture before encoding",
                "Detail enhancement pre-VAE",
                "Vintage film look with grain"
            ]
        }

    def __init__(self,
                 sharpen_amount: float = 0.5,
                 sharpen_radius: float = 1.0,
                 noise_strength: float = 0.0,
                 noise_scale: float = 8.0,
                 noise_octaves: int = 4,
                 noise_persistence: float = 0.5,
                 noise_lacunarity: float = 2.0,
                 noise_temporal_coherence: float = 0.95,
                 **kwargs):
        """
        Initialize PreprocessSharpenNoise preprocessor

        Args:
            sharpen_amount: Sharpening strength
            sharpen_radius: Blur radius for unsharp masking
            noise_strength: Fractal noise injection strength
            noise_scale: Base noise frequency/scale
            noise_octaves: Number of fractal octaves
            noise_persistence: Amplitude falloff for each octave
            noise_lacunarity: Frequency multiplier between octaves
            noise_temporal_coherence: Temporal smoothing (0=new every frame, 1=static)
            **kwargs: Additional parameters
        """
        super().__init__(
            sharpen_amount=sharpen_amount,
            sharpen_radius=sharpen_radius,
            noise_strength=noise_strength,
            noise_scale=noise_scale,
            noise_octaves=noise_octaves,
            noise_persistence=noise_persistence,
            noise_lacunarity=noise_lacunarity,
            noise_temporal_coherence=noise_temporal_coherence,
            **kwargs
        )

        # Store parameters as instance attributes for OSC live updates
        self.sharpen_amount = sharpen_amount
        self.sharpen_radius = sharpen_radius
        self.noise_strength = noise_strength
        self.noise_scale = noise_scale
        self.noise_octaves = noise_octaves
        self.noise_persistence = noise_persistence
        self.noise_lacunarity = noise_lacunarity
        self.noise_temporal_coherence = noise_temporal_coherence

        # Cache for efficiency
        self._cached_gaussian_kernels = {}

        # Persistent noise state for temporal coherence
        self._noise_offset_x = torch.rand(1, device=self.device, dtype=self.dtype) * 1000.0
        self._noise_offset_y = torch.rand(1, device=self.device, dtype=self.dtype) * 1000.0
        self._noise_time = torch.zeros(1, device=self.device, dtype=self.dtype)
        self._cached_noise = None

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

    def _fade(self, t: torch.Tensor) -> torch.Tensor:
        """Perlin fade function: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation"""
        return a + t * (b - a)

    def _gradient_2d(self, hash_val: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate gradient vectors for 2D Perlin noise
        Uses hash to select gradient direction
        """
        # Use hash to select gradient direction (4 possibilities)
        h = hash_val & 3

        # Create gradient components based on hash
        u = torch.where(h < 2, x, -x)
        v = torch.where((h == 0) | (h == 2), y, -y)

        return u + v

    def _generate_perlin_noise_2d(self, shape: tuple, scale: float, offset_x: float = 0.0, offset_y: float = 0.0) -> torch.Tensor:
        """
        Generate high-quality 2D Perlin noise with bilinear interpolation

        Args:
            shape: (height, width)
            scale: Frequency scale
            offset_x, offset_y: Spatial offsets for variation

        Returns:
            Noise tensor [H, W] in range approximately [-1, 1]
        """
        height, width = shape

        # Create coordinate grids scaled by frequency
        y_coords = torch.linspace(0, height / scale, height, device=self.device, dtype=self.dtype) + offset_y
        x_coords = torch.linspace(0, width / scale, width, device=self.device, dtype=self.dtype) + offset_x

        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Integer parts (grid cell)
        xi = xx.floor().long()
        yi = yy.floor().long()

        # Fractional parts (position within cell)
        xf = xx - xi.float()
        yf = yy - yi.float()

        # Apply fade curves for smooth interpolation
        u = self._fade(xf)
        v = self._fade(yf)

        # Generate pseudo-random hash for each corner
        # Using a simple but effective hash function
        def hash_coords(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Simple hash that produces different values for each coordinate
            h = (x * 374761393 + y * 668265263) & 0x7FFFFFFF
            h = (h ^ (h >> 13)) * 1274126177
            return (h ^ (h >> 16)) & 0xFF

        # Get gradients at four corners
        n00 = self._gradient_2d(hash_coords(xi, yi), xf, yf)
        n10 = self._gradient_2d(hash_coords(xi + 1, yi), xf - 1, yf)
        n01 = self._gradient_2d(hash_coords(xi, yi + 1), xf, yf - 1)
        n11 = self._gradient_2d(hash_coords(xi + 1, yi + 1), xf - 1, yf - 1)

        # Bilinear interpolation
        x1 = self._lerp(n00, n10, u)
        x2 = self._lerp(n01, n11, u)
        result = self._lerp(x1, x2, v)

        return result

    def _generate_fractal_noise(self, shape: tuple, base_scale: float,
                                octaves: int, persistence: float,
                                lacunarity: float) -> torch.Tensor:
        """
        Generate multi-octave fractal noise (fBm - fractional Brownian motion)

        Args:
            shape: (height, width)
            base_scale: Base frequency scale
            octaves: Number of octaves to combine
            persistence: Amplitude multiplier for each octave (typically 0.5)
            lacunarity: Frequency multiplier for each octave (typically 2.0)

        Returns:
            Fractal noise tensor [H, W] in range approximately [-1, 1]
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
                base_scale / frequency,
                self._noise_offset_x.item() * frequency,
                self._noise_offset_y.item() * frequency
            )

            # Add to accumulator with current amplitude
            noise += octave_noise * amplitude

            # Track max amplitude for normalization
            max_amplitude += amplitude

            # Update for next octave
            amplitude *= persistence
            frequency *= lacunarity

        # Normalize to approximately [-1, 1]
        if max_amplitude > 0:
            noise = noise / max_amplitude

        return noise

    def _add_fractal_noise(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add high-quality fractal noise to image with temporal coherence

        Args:
            image: Input tensor [B, C, H, W]

        Returns:
            Image with noise added
        """
        if self.noise_strength <= 0:
            return image

        batch_size, channels, height, width = image.shape

        # Update noise time for evolution (if not using full temporal coherence)
        if self.noise_temporal_coherence < 1.0:
            self._noise_time += (1.0 - self.noise_temporal_coherence) * 0.1

        # Generate or update cached noise
        if self._cached_noise is None or self._cached_noise.shape != (height, width):
            # Generate fresh noise
            noise = self._generate_fractal_noise(
                (height, width),
                self.noise_scale,
                self.noise_octaves,
                self.noise_persistence,
                self.noise_lacunarity
            )
            self._cached_noise = noise
        elif self.noise_temporal_coherence < 1.0:
            # Blend with new noise for temporal evolution
            new_noise = self._generate_fractal_noise(
                (height, width),
                self.noise_scale,
                self.noise_octaves,
                self.noise_persistence,
                self.noise_lacunarity
            )
            self._cached_noise = (
                self.noise_temporal_coherence * self._cached_noise +
                (1.0 - self.noise_temporal_coherence) * new_noise
            )
        else:
            # Use cached noise (fully static)
            noise = self._cached_noise

        noise = self._cached_noise

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

        # Step 1: Add fractal noise FIRST (before sharpening)
        # This way sharpening can enhance the noise texture
        result = self._add_fractal_noise(result)

        # Step 2: Sharpen (enhances both image detail and noise texture)
        if self.sharpen_amount > 0:
            result = self._unsharp_mask(result, self.sharpen_radius, self.sharpen_amount)

        # Final clamp to ensure valid range
        result = torch.clamp(result, 0, 1)

        return result
