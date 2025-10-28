# Frame Capture Debug System

## Overview

A debugging tool that captures frames at every stage of the StreamDiffusion pipeline for visual inspection of timing/synchronization issues.

## Features

- **Automatic Warmup Skip:** Skips first 30 frames to avoid unstable initialization frames
- **Multi-Stage Capture:** Captures at 8 different pipeline stages per frame
- **Latent Visualization:** Automatically visualizes 4-channel latent tensors as RGB
- **Auto-Shutdown:** Stops after capturing 4 frames
- **Zero Performance Impact:** When disabled, only a single boolean check per frame

## Usage

### Quick Start

Simply run the debug capture script:

```bash
Start_StreamDiffusion_DebugCapture.bat
```

The system will:
1. Start StreamDiffusion normally
2. Skip the first 30 frames
3. Capture the next 4 frames at all pipeline stages
4. Save to `debug_frames/capture_YYYYMMDD_HHMMSS/`
5. Auto-shutdown

### Manual Usage

If you want to use the flag directly:

```bash
python streamdiffusionTD\td_main.py --debug-capture-frames
```

## Output Structure

```
debug_frames/capture_20251027_143022/
├── capture_info.txt                      # Metadata about capture
├── frame_034_0_input.png                 # Raw input from shared memory
├── frame_034_1_after_img_preprocess.png  # After image preprocessing hooks
├── frame_034_2_after_vae_encode_latent_vis.png  # VAE latent (visualized)
├── frame_034_3_after_diffusion_latent_vis.png   # After UNet diffusion
├── frame_034_4_after_latent_postprocess_latent_vis.png  # After latent hooks
├── frame_034_5_after_vae_decode.png      # After VAE decode
├── frame_034_6_after_img_postprocess.png # After image postprocessing hooks
├── frame_034_7_output.png                # Final output to TouchDesigner
├── frame_035_*.png                       # Frame 35 (all stages)
├── frame_036_*.png                       # Frame 36 (all stages)
└── frame_037_*.png                       # Frame 37 (all stages)
```

## Pipeline Stages Captured

### Stage 0: Input
- **Location:** `td_manager.py` streaming loop
- **Format:** Raw input from shared memory (uint8 [0, 255])
- **Purpose:** Verify input frame is correct

### Stage 1: After Image Preprocessing
- **Location:** `pipeline.py:__call__()` after `_apply_image_preprocessing_hooks()`
- **Format:** Float tensor [0, 1]
- **Purpose:** Check feedback transform, optical flow, etc.

### Stage 2: After VAE Encode (Latent)
- **Location:** `pipeline.py:__call__()` after `encode_image()`
- **Format:** Latent tensor [B, 4, H/8, W/8] (visualized as RGB)
- **Purpose:** Verify VAE encoding is correct

### Stage 3: After Diffusion (Latent)
- **Location:** `pipeline.py:__call__()` after `predict_x0_batch()`
- **Format:** Latent tensor [B, 4, H/8, W/8] (visualized as RGB)
- **Purpose:** Check UNet diffusion output

### Stage 4: After Latent Postprocessing (Latent)
- **Location:** `pipeline.py:__call__()` after `_apply_latent_postprocessing_hooks()`
- **Format:** Latent tensor [B, 4, H/8, W/8] (visualized as RGB)
- **Purpose:** Check color correction feedback, latent effects

### Stage 5: After VAE Decode
- **Location:** `pipeline.py:__call__()` after `decode_image()`
- **Format:** Float tensor [0, 1] (converted from [-1, 1])
- **Purpose:** Verify VAE decoding is correct

### Stage 6: After Image Postprocessing
- **Location:** `pipeline.py:__call__()` after `_apply_image_postprocessing_hooks()`
- **Format:** Float tensor [0, 1]
- **Purpose:** Check sharpening, upscaling, etc.

### Stage 7: Final Output
- **Location:** `td_manager.py` streaming loop before `_send_output_frame()`
- **Format:** Postprocessed output (numpy array)
- **Purpose:** Verify final output matches what TD receives

## Latent Visualization

Latent tensors (4 channels) are automatically visualized as RGB by:
1. Taking the first 3 latent channels
2. Normalizing each channel independently to [0, 1]
3. Saving as RGB image

This allows visual inspection of latent-space transformations.

## Troubleshooting Sync Issues

### Expected Patterns

**Normal Operation:**
- `input.png` → should be from TouchDesigner
- `after_img_preprocess.png` → may have feedback/transform applied
- `after_vae_encode_latent_vis.png` → colorful latent representation
- `after_diffusion_latent_vis.png` → should differ from encode
- `after_vae_decode.png` → diffused image
- `output.png` → final result

**Common Issues to Check:**

1. **Frame Delay (1-frame lag):**
   - Compare `input.png` across frames 34, 35, 36, 37
   - Check if they match expected input sequence
   - If delayed, check `requires_sync_processing` on image preprocessors

2. **Feedback Oscillation:**
   - Compare `after_img_preprocess.png` to `output.png` from previous frame
   - If wildly different, feedback strength may be too high
   - Or `requires_sync_processing = True` may be needed

3. **Latent Transform Accumulation:**
   - Compare `after_latent_postprocess_latent_vis.png` across frames
   - Check for runaway zoom/pan (visual drift)
   - Verify blend strength is < 1.0

4. **Color Drift:**
   - Compare color balance across all frames
   - Check `after_latent_postprocess_latent_vis.png` for color shift
   - May need to adjust color correction feedback

## Implementation Details

### Files Modified

1. **`streamdiffusionTD/frame_capture_debug.py`** (NEW)
   - Core capture utility class
   - Handles tensor conversion, latent visualization
   - Auto-creates timestamped output folders

2. **`streamdiffusionTD/td_main.py`**
   - Added `--debug-capture-frames` CLI argument
   - Passes flag to config dict

3. **`streamdiffusionTD/td_manager.py`**
   - Initialize `FrameCaptureDebug` from config
   - Capture input/output in streaming loop
   - Wire capturer to pipeline on startup
   - Auto-shutdown when capture completes

4. **`src/streamdiffusion/pipeline.py`**
   - Added `self.frame_capturer` attribute
   - Capture hooks at all 6 internal pipeline stages

5. **`Start_StreamDiffusion_DebugCapture.bat`** (NEW)
   - Convenience script with clear instructions
   - Automatically adds `--debug-capture-frames` flag

### Performance Impact

**When Disabled (Normal Operation):**
- Zero overhead (capturer is `None`, early exit)

**When Enabled (Debug Mode):**
- ~5-10ms per frame for saving PNGs
- Negligible for debugging (only 4 frames captured)

## Example Workflow

1. **Run Capture:**
   ```bash
   Start_StreamDiffusion_DebugCapture.bat
   ```

2. **Wait for Completion:**
   ```
   🎥 Frame Capture Debug Enabled
      Skip: 30 frames | Capture: 4 frames
      Output: C:\...\debug_frames\capture_20251027_143022
   📸 Captured frame 1/4 (total frame 31)
   📸 Captured frame 2/4 (total frame 32)
   📸 Captured frame 3/4 (total frame 33)
   📸 Captured frame 4/4 (total frame 34)
   ✅ Frame capture complete!
   🎬 Frame capture complete! Stopping streaming...
   ```

3. **Inspect Frames:**
   - Open `debug_frames/capture_YYYYMMDD_HHMMSS/`
   - Sort by filename (automatically ordered by stage)
   - Compare across frames 34, 35, 36, 37
   - Look for unexpected differences or delays

4. **Share Results:**
   - Folder contains all frames + metadata
   - Easy to zip and share for debugging

## Tips

- **Compare Input Sequence:** Look at `frame_034_0_input.png`, `frame_035_0_input.png`, etc. to verify input timing
- **Check Feedback Loop:** Compare `frame_034_6_after_img_postprocess.png` to `frame_035_1_after_img_preprocess.png` (should show feedback)
- **Latent Transforms:** Watch `*_latent_vis.png` files for visual evidence of latent zoom/pan/rotation
- **Color Consistency:** Check if colors stay consistent or drift across frames

## Future Enhancements

Potential improvements:
- [ ] Configurable skip/capture counts via CLI args
- [ ] Capture only specific stages (e.g., `--stages=input,output`)
- [ ] Side-by-side comparison HTML viewer
- [ ] Diff images between consecutive frames
- [ ] Optional: capture every Nth frame instead of first 4

---

**Version:** 1.0
**Created:** 2025-10-27
**Maintainer:** Claude (Anthropic)
