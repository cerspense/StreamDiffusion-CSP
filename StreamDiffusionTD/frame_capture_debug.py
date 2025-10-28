"""
Frame Capture Debug Utility

Captures frames at multiple pipeline stages for debugging timing/sync issues.
Skips the first 30 warmup frames, then captures 4 sequential frames.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from PIL import Image


class FrameCaptureDebug:
    """
    Manages frame capture for debugging pipeline stages.

    Usage:
        capturer = FrameCaptureDebug(enabled=True, skip_frames=30, capture_count=4)

        # In streaming loop
        if capturer.should_capture():
            capturer.capture_stage("input", input_frame)
            capturer.capture_stage("output", output_frame)
            capturer.increment_frame()
    """

    def __init__(
        self,
        enabled: bool = False,
        skip_frames: int = 30,
        capture_count: int = 4,
        output_dir: Optional[str] = None
    ):
        """
        Initialize frame capture.

        Args:
            enabled: Enable frame capture
            skip_frames: Number of warmup frames to skip before capturing
            capture_count: Number of frames to capture
            output_dir: Output directory (auto-generated if None)
        """
        self.enabled = enabled
        self.skip_frames = skip_frames
        self.capture_count = capture_count

        self.current_frame = 0
        self.captured_frames = 0
        self.is_complete = False

        if self.enabled:
            # Create timestamped output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = Path(f"debug_frames/capture_{timestamp}")
            else:
                self.output_dir = Path(output_dir)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            print("[FRAME CAPTURE] Debug Enabled")
            print(f"   Skip: {skip_frames} frames | Capture: {capture_count} frames")
            print(f"   Output: {self.output_dir.absolute()}")

            # Create info file
            self._write_capture_info()
        else:
            self.output_dir = None

    def should_capture(self) -> bool:
        """Check if current frame should be captured."""
        if not self.enabled or self.is_complete:
            return False

        # Skip warmup frames
        if self.current_frame < self.skip_frames:
            return False

        # Check if we've captured enough frames
        if self.captured_frames >= self.capture_count:
            self.is_complete = True
            print(f"\n[FRAME CAPTURE] Complete! {self.captured_frames} frames saved to:")
            print(f"   {self.output_dir.absolute()}")
            return False

        return True

    def increment_frame(self) -> None:
        """Increment frame counter (call once per frame in main loop)."""
        if not self.enabled or self.is_complete:
            return

        self.current_frame += 1

        # Track when we start capturing
        if self.current_frame > self.skip_frames:
            self.captured_frames += 1
            if self.captured_frames <= self.capture_count:
                print(f"[FRAME CAPTURE] Captured frame {self.captured_frames}/{self.capture_count} (total frame {self.current_frame})")

    def capture_stage(
        self,
        stage_name: str,
        data: Union[torch.Tensor, np.ndarray, Image.Image],
        stage_order: Optional[int] = None
    ) -> None:
        """
        Capture data from a specific pipeline stage.

        Args:
            stage_name: Name of the stage (e.g., "input", "after_vae_encode")
            data: Image data (torch.Tensor, np.ndarray, or PIL Image)
            stage_order: Optional ordering number for filename sorting
        """
        if not self.should_capture():
            return

        try:
            # Convert data to numpy array [0, 255] uint8
            if isinstance(data, torch.Tensor):
                # Handle different tensor formats
                if data.ndim == 4:  # [B, C, H, W]
                    data = data[0]  # Take first batch item

                if data.shape[0] in [1, 3, 4]:  # Channel-first [C, H, W]
                    data = data.permute(1, 2, 0)  # -> [H, W, C]

                data = data.detach().cpu().float().numpy()

                # Normalize to [0, 1]
                if data.min() < 0 or data.max() > 1:
                    # Handle latent space or [-1, 1] range
                    if data.min() < -0.5:
                        # Likely [-1, 1] range
                        data = (data + 1.0) / 2.0
                    else:
                        # Clamp to safe range
                        data = np.clip(data, 0, 1)

                # Handle latent visualization (4 channels)
                if data.shape[-1] == 4:
                    # Visualize latent channels as RGB
                    # Use first 3 channels, normalize each independently
                    latent_vis = np.zeros((*data.shape[:2], 3), dtype=np.float32)
                    for i in range(3):
                        channel = data[:, :, i]
                        channel_min = channel.min()
                        channel_max = channel.max()
                        if channel_max > channel_min:
                            latent_vis[:, :, i] = (channel - channel_min) / (channel_max - channel_min)
                    data = latent_vis
                    stage_name = f"{stage_name}_latent_vis"

                # Convert to uint8
                data = (data * 255).astype(np.uint8)

            elif isinstance(data, np.ndarray):
                # Numpy array
                if data.dtype == np.float32 or data.dtype == np.float64:
                    data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
                elif data.dtype != np.uint8:
                    data = data.astype(np.uint8)

            elif isinstance(data, Image.Image):
                # PIL Image - convert to numpy
                data = np.array(data)

            else:
                print(f"[FRAME CAPTURE] Unsupported data type for capture: {type(data)}")
                return

            # Ensure 3-channel RGB
            if data.ndim == 2:
                # Grayscale -> RGB
                data = np.stack([data] * 3, axis=-1)
            elif data.shape[-1] == 1:
                # Single channel -> RGB
                data = np.repeat(data, 3, axis=-1)

            # Create filename
            frame_num = self.skip_frames + self.captured_frames
            if stage_order is not None:
                filename = f"frame_{frame_num:03d}_{stage_order}_{stage_name}.png"
            else:
                filename = f"frame_{frame_num:03d}_{stage_name}.png"

            filepath = self.output_dir / filename

            # Save as PNG
            img = Image.fromarray(data, mode='RGB')
            img.save(filepath)

        except Exception as e:
            print(f"[FRAME CAPTURE] Error capturing stage '{stage_name}': {e}")

    def _write_capture_info(self) -> None:
        """Write capture metadata to info file."""
        info_path = self.output_dir / "capture_info.txt"
        with open(info_path, 'w') as f:
            f.write("StreamDiffusion Frame Capture Debug\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Skip Frames: {self.skip_frames}\n")
            f.write(f"Capture Count: {self.capture_count}\n")
            f.write(f"Output Directory: {self.output_dir.absolute()}\n\n")
            f.write("Pipeline Stages:\n")
            f.write("  0. Input (from shared memory)\n")
            f.write("  1. After image preprocessing\n")
            f.write("  2. After VAE encode (latent visualization)\n")
            f.write("  3. After diffusion (latent visualization)\n")
            f.write("  4. After latent postprocessing (latent visualization)\n")
            f.write("  5. After VAE decode\n")
            f.write("  6. After image postprocessing\n")
            f.write("  7. Final output\n")

    def get_capture_folder(self) -> Optional[Path]:
        """Get the output folder path."""
        return self.output_dir if self.enabled else None
