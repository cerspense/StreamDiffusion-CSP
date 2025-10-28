"""
Synthetic Input Generator for Frame Capture Testing

Generates animated fractal noise patterns to feed into the pipeline
instead of requiring TouchDesigner shared memory input.
"""

import numpy as np
import torch
from typing import Tuple


class SyntheticInputGenerator:
    """
    Generates animated fractal noise for testing frame capture.

    Creates a slowly moving, high-frequency noise pattern that's
    easy to track through the pipeline stages.
    """

    def __init__(
        self,
        width: int = 576,
        height: int = 448,
        frequency: float = 4.0,
        speed: float = 0.02,
        seed: int = 42
    ):
        """
        Initialize synthetic input generator.

        Args:
            width: Frame width
            height: Frame height
            frequency: Noise frequency (higher = more detail)
            speed: Animation speed (offset per frame)
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.frequency = frequency
        self.speed = speed
        self.frame_count = 0

        np.random.seed(seed)

        # Pre-generate coordinate grids for fractal noise
        x = np.linspace(0, frequency, width)
        y = np.linspace(0, frequency, height)
        self.xx, self.yy = np.meshgrid(x, y)

        print(f"[SYNTHETIC INPUT] Initialized {width}x{height} fractal noise generator")
        print(f"   Frequency: {frequency} | Speed: {speed}")

    def generate_fractal_noise(self, offset: float) -> np.ndarray:
        """
        Generate fractal noise pattern using multiple octaves.

        Args:
            offset: Time offset for animation

        Returns:
            Noise array [H, W] in range [0, 1]
        """
        # Start with base noise layer
        noise = np.zeros((self.height, self.width), dtype=np.float32)

        # Add multiple octaves for fractal appearance
        amplitude = 1.0
        frequency = 1.0

        for octave in range(4):  # 4 octaves for nice detail
            # Animated offset
            x_offset = offset * frequency
            y_offset = offset * frequency * 0.7  # Different speed for Y

            # Perlin-style noise approximation using sine waves
            octave_noise = (
                np.sin(self.xx * frequency + x_offset) *
                np.cos(self.yy * frequency + y_offset) +
                np.sin((self.xx + self.yy) * frequency * 0.5 + x_offset * 0.5)
            )

            noise += octave_noise * amplitude

            # Next octave: double frequency, half amplitude
            frequency *= 2.0
            amplitude *= 0.5

        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        return noise

    def generate_frame(self) -> np.ndarray:
        """
        Generate next frame in the sequence.

        Returns:
            RGB frame [H, W, 3] in uint8 format [0, 255]
        """
        # Calculate animated offset
        offset = self.frame_count * self.speed

        # Generate grayscale fractal noise
        noise = self.generate_fractal_noise(offset)

        # Convert to RGB (colorize for easier tracking)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Color channels with phase shifts for visual variety
        rgb[:, :, 0] = noise  # Red
        rgb[:, :, 1] = self.generate_fractal_noise(offset + 1.0)  # Green (shifted)
        rgb[:, :, 2] = self.generate_fractal_noise(offset + 2.0)  # Blue (shifted more)

        # Convert to uint8 [0, 255]
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        self.frame_count += 1

        if self.frame_count % 10 == 0:
            print(f"[SYNTHETIC INPUT] Generated frame {self.frame_count}")

        return rgb_uint8

    def reset(self):
        """Reset frame counter."""
        self.frame_count = 0
        print("[SYNTHETIC INPUT] Frame counter reset")


def test_generator():
    """Test the synthetic input generator."""
    from PIL import Image
    import os

    gen = SyntheticInputGenerator(width=576, height=448, frequency=4.0, speed=0.05)

    # Generate test frames
    os.makedirs("test_synthetic_frames", exist_ok=True)

    for i in range(5):
        frame = gen.generate_frame()
        img = Image.fromarray(frame, mode='RGB')
        img.save(f"test_synthetic_frames/frame_{i:03d}.png")
        print(f"Saved test frame {i}")

    print("Test frames saved to test_synthetic_frames/")


if __name__ == "__main__":
    test_generator()
