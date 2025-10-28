# CLAUDE.md - StreamDiffusion Fx Processor Development Guide

**Project:** StreamDiffusion LivePeer Fork - TouchDesigner Integration
**Focus:** Building Dynamic Pipeline Processors (Fx) for Real-Time Image Generation
**Date:** 2025-10-27

---

## Executive Summary

This project implements a **metadata-driven Fx (Dynamic Pipeline Processors) system** that enables real-time control of latent-domain and image-domain processors through TouchDesigner. The system automatically generates TouchDesigner parameters from processor metadata, supports live OSC updates, and seamlessly integrates with the StreamDiffusion pipeline.

**Key Architecture:**
- **4-Stage Hook System:** Image Pre/Post, Latent Pre/Post processing stages
- **Metadata-Driven:** Single source of truth in `get_preprocessor_metadata()`
- **TouchDesigner Integration:** Auto-generates UI parameters, OSC handlers, and YAML config
- **Zero-Code Extension:** Add new processors by implementing one class - everything else is automatic

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [The Four Processing Stages](#the-four-processing-stages)
3. [Creating New Processors](#creating-new-processors)
4. [TouchDesigner Integration Pattern](#touchdesigner-integration-pattern)
5. [Processor Examples](#processor-examples)
6. [Development Workflow](#development-workflow)
7. [Critical Design Patterns](#critical-design-patterns)

---

## System Architecture Overview

### Data Flow: TouchDesigner ↔ Python Backend

```
TouchDesigner Parameter Change (e.g., Fxlatenttransformzoom = 1.05)
    ↓
parexec_fx: onValueChange(par, prev)
    ↓
StreamDiffusionExt.Fxparameterupdate(par)
    ↓
Build OSC address from table_preprocessors metadata
    ↓
oscout1.sendOSC('/fx/latent_transform/zoom', [1.05])
    ↓
[OSC over network to Python backend]
    ↓
td_osc_handler._handle_fx_parameter(address, *args)
    ↓
Parse processor_type and param_name
    ↓
Find processor in stream._latent_preprocessing_module
    ↓
setattr(processor, 'zoom', 1.05)
    ↓
[Next frame uses updated value - NO pipeline rebuild!]
```

### Pipeline Processing Flow

```
SharedMemory: input_mem_name (from TouchDesigner)
    ↓
┌─────────────────────────────────────────────────┐
│ IMAGE PREPROCESSING HOOKS                       │
│ - FeedbackPreprocessor (image-space V2V)        │
│ - FeedbackTransformPreprocessor (zoom/pan/rot)  │
│ - TemporalNetTensorRT (optical flow)            │
└─────────────────────────────────────────────────┘
    ↓
VAE Encode → x_t_latent [B, 4, H/8, W/8]
    ↓
┌─────────────────────────────────────────────────┐
│ LATENT PREPROCESSING HOOKS ⭐                    │
│ - LatentFeedbackPreprocessor                    │
│ - LatentTransformPreprocessor (Deforum-style)   │
└─────────────────────────────────────────────────┘
    ↓
UNet Diffusion (with ControlNet + IPAdapter)
    ↓
┌─────────────────────────────────────────────────┐
│ LATENT POSTPROCESSING HOOKS 🔮                   │
│ - ColorCorrectionFeedbackProcessor              │
│ - (OPEN for custom processors!)                 │
└─────────────────────────────────────────────────┘
    ↓
VAE Decode → x_output [B, 3, H, W]
    ↓
┌─────────────────────────────────────────────────┐
│ IMAGE POSTPROCESSING HOOKS                       │
│ - Sharpen, Blur, Upscale, etc.                  │
└─────────────────────────────────────────────────┘
    ↓
SharedMemory: output_mem_name (back to TouchDesigner)
```

---

## The Four Processing Stages

### 1. Image Preprocessing (Before VAE Encode)

**Location:** `stream.image_preprocessing_hooks`
**File:** `src/streamdiffusion/preprocessing/processors/`
**Use Cases:**
- Input image filtering/enhancement
- Image-space temporal feedback (V2V)
- Geometric transforms in image space
- Optical flow warping

**Advantages:**
- Intuitive visual control (RGB domain)
- No VAE encode/decode artifacts
- True image-space temporal feedback

**Example Processors:**
- `FeedbackTransformPreprocessor` - Zoom/pan/rotate + feedback in image space
- `TemporalNetTensorRTPreprocessor` - Optical flow warping

**CRITICAL:** Use `requires_sync_processing = True` to avoid 1-frame delay from pipelined orchestrator!

---

### 2. Latent Preprocessing (After VAE Encode, Before Diffusion) ⭐

**Location:** `stream.latent_preprocessing_hooks`
**File:** `src/streamdiffusion/preprocessing/processors/`
**Use Cases:**
- Temporal consistency in latent space (most efficient!)
- Latent-space geometric transforms
- Latent noise injection
- Pre-diffusion latent effects

**Advantages:**
- 8x faster than image operations (64x64 vs 512x512)
- Operates directly on diffusion input
- Temporal feedback without VAE artifacts

**Example Processors:**
- `LatentFeedbackPreprocessor` - Temporal blending in latent space
- `LatentTransformPreprocessor` - Deforum-style zoom/pan/rotate

**Key Pattern:**
```python
class LatentTransformPreprocessor(PipelineAwareProcessor):
    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """Preserve latent dimensions - DON'T resize!"""
        return latent_tensor.to(device=self.device, dtype=self.dtype)

    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Override resize - latent tensors should NOT be resized"""
        return tensor
```

---

### 3. Latent Postprocessing (After Diffusion, Before VAE Decode) 🔮

**Location:** `stream.latent_postprocessing_hooks`
**File:** `src/streamdiffusion/preprocessing/processors/`
**Use Cases:**
- Latent-space color grading
- Latent detail enhancement
- Post-diffusion latent feedback
- Latent filtering/denoise

**Advantages:**
- Modify final latent before decode
- Efficient (latent space operations)
- Can create accumulative effects over time

**Example Processors:**
- `ColorCorrectionFeedbackProcessor` - Color correction with temporal smoothing

**Opportunity:** This stage is relatively unexplored - ideal for custom processors!

---

### 4. Image Postprocessing (After VAE Decode)

**Location:** `stream.image_postprocessing_hooks`
**File:** `src/streamdiffusion/preprocessing/processors/`
**Use Cases:**
- Output sharpening
- Upscaling
- Color grading in image space
- Final aesthetic effects

**Example Processors:**
- `SharpenPreprocessor`
- `RealESRGANProcessor`

---

## Creating New Processors

### Step 1: Choose Your Base Class

```python
from .base import BasePreprocessor, PipelineAwareProcessor

# Use PipelineAwareProcessor if you need access to previous frame data
# Use BasePreprocessor for stateless processing
```

### Step 2: Implement Metadata (CRITICAL - This is Your Single Source of Truth!)

```python
@classmethod
def get_preprocessor_metadata(cls):
    return {
        "display_name": "My Custom Processor",
        "description": "Short description of what this processor does",
        "parameters": {
            "strength": {
                "type": "float",
                "default": 0.5,
                "range": [0.0, 1.0],
                "step": 0.01,
                "description": "Effect strength (0 = off, 1 = full)"
            },
            "mode": {
                "type": "string",
                "default": "linear",
                "options": ["linear", "exponential", "sigmoid"],
                "description": "Blending mode"
            },
            # Add more parameters...
        },
        "use_cases": [
            "Temporal consistency",
            "Visual effects",
            "Your use case here"
        ]
    }
```

**Parameter Types:**
- `float`: Slider (uses `range`, `step`, `default`)
- `int`: Integer slider
- `string` with `options`: Dropdown menu in TouchDesigner
- `string` without `options`: Text field

**Clamping Behavior:**
- **Transform params** (`zoom`, `pan_x`, `pan_y`, `rotation`): NOT clamped - allows typing values outside slider range
- **Blend params** (`strength`, `feedback_blend`, etc.): CLAMPED to `[0, 1]`

### Step 3: Implement Constructor

```python
def __init__(self,
             pipeline_ref: Any,  # Required for PipelineAwareProcessor
             strength: float = 0.5,
             mode: str = "linear",
             **kwargs):
    super().__init__(
        pipeline_ref=pipeline_ref,
        strength=strength,
        mode=mode,
        **kwargs
    )
    self.strength = strength
    self.mode = mode
    self._first_frame = True  # Track first frame if using feedback
```

### Step 4: Implement Processing Logic

**For Image-Domain Processors:**
```python
def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
    """
    GPU-optimized processing path

    Args:
        tensor: [B, C, H, W] in image space (typically [0, 1] range)

    Returns:
        Processed tensor, same shape
    """
    # Your processing logic here
    processed = tensor * self.strength  # Example
    return processed.clamp(0, 1)  # CRITICAL: Always clamp!
```

**For Latent-Domain Processors:**
```python
def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
    """Preserve latent dimensions - DON'T resize!"""
    return latent_tensor.to(device=self.device, dtype=self.dtype)

def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
    """Override resize - latent tensors should NOT be resized"""
    return tensor

def _process_core(self, image):
    """Not used for latent processors"""
    raise NotImplementedError("Use _process_tensor_core for latent processors")

def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
    """
    Process latent tensor [B, 4, H/8, W/8]

    Latent channels (SD 1.5/SDXL):
    - Channel 0: Low frequency detail
    - Channel 1: Mid frequency detail
    - Channel 2: High frequency detail
    - Channel 3: Color/tone information
    """
    # Your latent processing logic here
    processed = tensor  # Example
    return processed.clamp(-10.0, 10.0)  # CRITICAL: Clamp to safe range!
```

**For Pipeline-Aware Processors (Feedback):**
```python
def _get_previous_data(self):
    """Get previous frame data from pipeline"""
    if self.pipeline_ref is not None:
        # For latent preprocessing
        if hasattr(self.pipeline_ref, 'prev_latent_result'):
            if self.pipeline_ref.prev_latent_result is not None and not self._first_frame:
                return self.pipeline_ref.prev_latent_result
        # For image preprocessing/postprocessing
        if hasattr(self.pipeline_ref, 'prev_image_result'):
            if self.pipeline_ref.prev_image_result is not None and not self._first_frame:
                return self.pipeline_ref.prev_image_result
    return None

def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
    prev_data = self._get_previous_data()

    if prev_data is None:
        # First frame - passthrough
        self._first_frame = False
        return tensor

    # Process with feedback
    blended = (1 - self.strength) * tensor + self.strength * prev_data
    self._first_frame = False
    return blended.clamp(0, 1)  # Or clamp(-10, 10) for latents
```

### Step 5: Register Your Processor

**File:** `src/streamdiffusion/preprocessing/processors/__init__.py`

```python
from .my_custom_processor import MyCustomProcessor

_preprocessor_registry = {
    # ... existing processors ...
    "my_custom": MyCustomProcessor,
}
```

### Step 6: Add TouchDesigner Toggle

**In TouchDesigner (StreamDiffusionExt.py):**

1. Add toggle parameter: `Usemycustom` (Pulse, page `Fx`)
2. Add to `update_fx_dynamic_parameters()` (around line 6496):
```python
if hasattr(self.ownerComp.par, 'Usemycustom') and self.ownerComp.par.Usemycustom.eval():
    active_fx.append('my_custom')
```

3. Add to latent-only filter if applicable (around line 6006):
```python
latent_only_processors = ['latent_feedback', 'latent_transform', 'my_custom']
```

**That's it!** The system will automatically:
- ✅ Generate `Fx*` parameters on the Fx page
- ✅ Create OSC handlers for live updates
- ✅ Include in YAML config generation
- ✅ Filter from ControlNet dropdown (if latent-only)

---

## TouchDesigner Integration Pattern

### Auto-Generated Parameters

**Naming Convention:**
- TouchDesigner param: `Fx` + lowercase alphanumeric
- Example: `Fxlatenttransformzoom`, `Fxmycustomstrength`

**Parameter Creation:**
```python
# In update_fx_dynamic_parameters()
for proc_name in active_fx:
    metadata = get metadata from table_preprocessors
    for param_name, param_info in metadata['parameters'].items():
        # Auto-generate parameter
        td_param_name = 'Fx' + normalize(proc_name + '_' + param_name)
        self.create_parameter(
            name=td_param_name,
            label=param_name.title(),
            type=param_info['type'],
            default=param_info['default'],
            range=param_info['range'],
            ...
        )
```

### Live Parameter Updates

**OSC Flow:**
```python
# TouchDesigner callback
def Fxparameterupdate(self, par):
    # Extract processor type and param name from metadata
    osc_address = f'/fx/{processor_type}/{param_name}'
    osc_out.sendOSC(osc_address, [par.eval()])

# Python backend handler
def _handle_fx_parameter(self, address, *args):
    # Parse: /fx/latent_transform/zoom -> processor='latent_transform', param='zoom'
    setattr(processor, param, args[0])
```

### YAML Configuration

```python
# In generate_td_config_yaml()
if use_my_custom:
    params = self.gather_fx_parameters_for_processor('my_custom')
    yaml_content += f'''
    - type: "my_custom"
      order: {order}
      enabled: true
      params:
'''
    for param_name, param_value in params.items():
        yaml_content += f'        {param_name}: {param_value}\n'
```

**Generated YAML:**
```yaml
latent_preprocessing:
  enabled: true
  processors:
    - type: "my_custom"
      order: 1
      enabled: true
      params:
        strength: 0.5
        mode: "linear"
```

---

## Processor Examples

### Example 1: Latent Transform (Deforum-Style)

**File:** `src/streamdiffusion/preprocessing/processors/latent_transform.py`

**Key Features:**
- Transforms PREVIOUS frame's latent (accumulative motion)
- Blends with current input to prevent runaway accumulation
- Respects pipeline noise schedule for coherent seed travel
- Configurable border modes (zeros, border, reflection)

**Processing Pipeline:**
```python
1. Get previous frame latent
2. Transform PREVIOUS latent (zoom/pan/rotate)
3. Blend transformed previous with current input
4. Optional: Add noise injection (respects seed_list)
5. Clamp to safe range [-10, 10]
```

**Critical Pattern - Accumulative Transform:**
```python
# WRONG: Transform current input
transformed = transform(current_input)
result = blend(transformed, previous)

# CORRECT: Transform previous output (accumulative)
transformed_prev = transform(previous)
result = blend(current_input, transformed_prev)
```

### Example 2: Feedback Transform (Image-Space)

**File:** `src/streamdiffusion/preprocessing/processors/feedback_transform.py`

**Key Features:**
- Operates in image space BEFORE VAE encoding
- Combines feedback loop with geometric transforms
- `requires_sync_processing = True` to avoid 1-frame delay
- Converts VAE output from `[-1, 1]` to `[0, 1]` for processing

**Critical Pattern - Prevent Feedback Oscillation:**
```python
# WRONG: Process in async pipeline - creates 1-frame delay
class MyProcessor(BasePreprocessor):
    pass  # Default: requires_sync_processing = False

# CORRECT: Force synchronous processing
class FeedbackTransformPreprocessor(PipelineAwareProcessor):
    requires_sync_processing = True  # CRITICAL!
```

**Critical Pattern - Color Space Handling:**
```python
# CRITICAL: Convert from VAE output range [-1, 1] to image range [0, 1]
prev_output = self._get_previous_data()
prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)

# Blend in image space [0, 1]
blended = (1 - self.feedback_strength) * input + self.feedback_strength * prev_output

# CRITICAL: Clamp to prevent color drift
blended = blended.clamp(0, 1)
```

### Example 3: Color Correction Feedback

**File:** `src/streamdiffusion/preprocessing/processors/color_correction_feedback.py`

**Key Features:**
- Latent postprocessing (after diffusion)
- Temporal smoothing of color corrections
- Avoids flicker from per-frame corrections

**Use Case Pattern:**
```python
# Apply color correction to current frame
corrected = apply_color_correction(latent)

# Smooth with previous corrected latent
if has_previous:
    smoothed = (1 - temporal_smooth) * corrected + temporal_smooth * prev_corrected

# Store for next frame
pipeline.prev_corrected_latent = smoothed
```

---

## Development Workflow

### 1. Create Processor
```bash
# Create new processor file
touch src/streamdiffusion/preprocessing/processors/my_processor.py
```

### 2. Implement Processor Class
```python
# Follow patterns from examples above
# CRITICAL: Implement get_preprocessor_metadata()
```

### 3. Register Processor
```python
# In __init__.py
from .my_processor import MyProcessor
_preprocessor_registry['my_processor'] = MyProcessor
```

### 4. Refresh Metadata Table
```python
# In TouchDesigner, run:
op('table_preprocessors').par.execute.pulse()
```

### 5. Add Toggle & Test
```python
# 1. Add Usemyprocessor toggle in TD
# 2. Add to update_fx_dynamic_parameters()
# 3. Toggle ON
# 4. Verify Fx* parameters appear
# 5. Test OSC updates
```

### 6. Verify YAML Generation
```python
# Start stream and check generated YAML includes your processor
# Check logs for config output
```

---

## Critical Design Patterns

### Pattern 1: Latent Dimension Preservation

**Problem:** Base processor tries to resize latent tensors to image dimensions.

**Solution:**
```python
class MyLatentProcessor(PipelineAwareProcessor):
    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """Preserve latent dimensions - DON'T resize!"""
        return latent_tensor.to(device=self.device, dtype=self.dtype)

    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Override resize - latent tensors should NOT be resized"""
        return tensor
```

### Pattern 2: First Frame Handling

**Problem:** `prev_*_result` is `None` on first frame.

**Solution:**
```python
def __init__(self, ...):
    self._first_frame = True

def _process_tensor_core(self, tensor):
    prev_data = self._get_previous_data()

    if prev_data is None:
        self._first_frame = False
        return tensor  # Passthrough on first frame

    # Process with feedback
    ...
    self._first_frame = False
    return result
```

### Pattern 3: Clamping for Stability

**Image Space:**
```python
# ALWAYS clamp image tensors to [0, 1]
result = process(tensor)
return result.clamp(0, 1)
```

**Latent Space:**
```python
# ALWAYS clamp latent tensors to safe range
result = process(latent)
return result.clamp(-10.0, 10.0)
```

### Pattern 4: Batch Size Mismatch Handling

**Problem:** Previous frame may have different batch size.

**Solution:**
```python
if prev_data.shape[0] != tensor.shape[0]:
    if prev_data.shape[0] < tensor.shape[0]:
        # Expand: repeat previous data
        prev_data = prev_data.repeat(tensor.shape[0], 1, 1, 1)
    else:
        # Slice: use first item
        prev_data = prev_data[:tensor.shape[0]]
```

### Pattern 5: Synchronous Processing for Feedback

**Problem:** Async pipeline creates 1-frame delay in feedback loops.

**Solution:**
```python
class FeedbackProcessor(PipelineAwareProcessor):
    # CRITICAL: Force synchronous processing
    requires_sync_processing = True
```

**When to use:**
- Image preprocessing with feedback
- Any processor that reads `prev_image_result` before VAE encode
- Processors that must see "current" frame, not buffered frame

### Pattern 6: Noise Integration with Seed Travel

**Problem:** Need to inject noise that respects seed travel.

**Solution:**
```python
# Get pipeline's blended noise (from seed_list)
pipeline_noise = self.pipeline_ref.init_noise

# Blend with random noise
random_noise = torch.randn_like(latent)
noise = (1 - self.noise_seed_mix) * random_noise + self.noise_seed_mix * pipeline_noise

# Inject into latent
result = latent + self.noise_strength * noise
```

### Pattern 7: Affine Transforms in Latent Space

**Problem:** Need geometric transforms without resampling artifacts.

**Solution:**
```python
def _create_transform_grid(self, batch_size, channels, height, width, device):
    """Create affine transformation grid for F.grid_sample"""
    # Build theta matrix [B, 2, 3]
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=self.dtype)

    # Zoom (scale = 1/zoom because we're transforming the grid)
    scale = 1.0 / self.zoom
    theta[:, 0, 0] = scale  # x scale
    theta[:, 1, 1] = scale  # y scale

    # Pan (normalized to [-1, 1])
    theta[:, 0, 2] = -self.pan_x * 2.0
    theta[:, 1, 2] = -self.pan_y * 2.0

    # Generate grid
    grid = F.affine_grid(theta, [batch_size, channels, height, width], align_corners=False)
    return grid

# Apply transform
transformed = F.grid_sample(
    tensor,
    grid,
    mode='bilinear',
    padding_mode='zeros',  # or 'border', 'reflection'
    align_corners=False
)
```

---

## Best Practices Checklist

### Code Quality
- ✅ Implement comprehensive `get_preprocessor_metadata()`
- ✅ Add docstrings to class and methods
- ✅ Use type hints (`torch.Tensor`, `float`, etc.)
- ✅ Handle first frame edge case
- ✅ Clamp outputs to safe ranges
- ✅ Handle batch size mismatches

### Performance
- ✅ Use GPU operations (`_process_tensor_core`)
- ✅ Avoid CPU-GPU transfers
- ✅ Prefer latent operations over image operations (8x faster)
- ✅ Early exit when parameters are at identity values

### Integration
- ✅ Register in `__init__.py`
- ✅ Add to TouchDesigner toggle list
- ✅ Test OSC parameter updates
- ✅ Verify YAML generation
- ✅ Add to latent-only filter if applicable

### Testing
- ✅ Test first frame (no previous data)
- ✅ Test parameter ranges (min, max, default)
- ✅ Test edge cases (zero strength, max strength)
- ✅ Test batch size variations
- ✅ Monitor for memory leaks over time
- ✅ Use frame capture debug mode to visually inspect pipeline stages

---

## Frame Capture Debug System

### Overview

The frame capture system allows you to capture and save frames at all 8 pipeline stages for visual inspection and debugging. This is essential for diagnosing timing/sync issues, feedback oscillation, color drift, and other pipeline problems.

### Quick Start

```bash
# Run frame capture debug mode
Start_StreamDiffusion_DebugCapture.bat

# Or manually:
python streamdiffusionTD/td_main.py --debug-capture-frames
```

### What It Does

1. **Skips 30 warmup frames** - Avoids unstable initialization
2. **Captures 4 sequential frames** - Enough to see temporal effects
3. **Saves 8 stages per frame** - Complete pipeline visibility
4. **Auto-shutdown** - Stops after capture completes
5. **Generates synthetic input** - Animated fractal noise (no TouchDesigner required)

### Output Structure

```
debug_frames/capture_YYYYMMDD_HHMMSS/
├── capture_info.txt                    # Metadata
├── frame_030_0_input.png               # Raw input
├── frame_030_1_after_img_preprocess.png
├── frame_030_2_after_vae_encode_latent_vis.png
├── frame_030_3_after_diffusion_latent_vis.png
├── frame_030_4_after_latent_postprocess_latent_vis.png
├── frame_030_5_after_vae_decode.png
├── frame_030_6_after_img_postprocess.png
├── frame_030_7_output.png              # Final output
├── frame_031_*.png                     # Frame 31 (all 8 stages)
├── frame_032_*.png                     # Frame 32 (all 8 stages)
└── frame_033_*.png                     # Frame 33 (all 8 stages)

Total: 32 PNG files (4 frames × 8 stages)
```

### Pipeline Stages Captured

| Stage | Filename Suffix | Description |
|-------|----------------|-------------|
| 0 | `_0_input.png` | Raw input (TouchDesigner or synthetic) |
| 1 | `_1_after_img_preprocess.png` | After image preprocessing hooks |
| 2 | `_2_after_vae_encode_latent_vis.png` | VAE latent (visualized as RGB) |
| 3 | `_3_after_diffusion_latent_vis.png` | After UNet diffusion (latent vis) |
| 4 | `_4_after_latent_postprocess_latent_vis.png` | After latent postprocessing |
| 5 | `_5_after_vae_decode.png` | Raw VAE decode output |
| 6 | `_6_after_img_postprocess.png` | After image postprocessing |
| 7 | `_7_output.png` | Final output |

### How to Use for Debugging

#### Checking Temporal Consistency
```bash
# Open frames 030-033 side by side
# Compare _0_input.png - should show smooth animation
# Compare _7_output.png - should reflect input after diffusion
```

#### Detecting Feedback Oscillation
```bash
# Compare frame_N_6 to frame_N+1_1
# If feedback is enabled, check for unexpected changes
# Look for oscillating patterns or instability
```

#### Analyzing Latent Operations
```bash
# Stage 2: VAE latent encoding
# Stage 3: After diffusion (UNet output)
# Stage 4: After latent preprocessing/postprocessing
# Watch for accumulation or drift in latent visualizations
```

#### Identifying Color Drift
```bash
# Compare frame_*_5_after_vae_decode.png sequence
# Check for progressive color shift across frames
# Indicates need for color correction feedback
```

#### Finding Processing Artifacts
```bash
# Compare stage 5 (raw VAE) to stage 7 (final output)
# Stage 5 often has black holes, oversaturation
# Stage 7 should clean these up via postprocessing
```

### Synthetic Input Generator

When running in debug mode, synthetic input is automatically enabled:

```python
# Generates animated fractal noise
# High frequency (4.0) for visible detail
# Slow movement (0.02 speed) for tracking
# Multi-octave fractal for rich texture
```

**Advantages:**
- No TouchDesigner dependency
- Deterministic (reproducible results)
- Easy to track through pipeline
- Controllable frequency and speed

### Debug Mode Features

**Alternate OSC Ports:**
- Listen: 9999 (vs 8247 in normal mode)
- Transmit: 9998 (vs 8248 in normal mode)
- Prevents conflicts with running TouchDesigner

**CUDA Graphs Disabled:**
- Temporary workaround for CUDA 12+ API incompatibility
- ~10-20% performance reduction
- Acceptable for debugging purposes

**Auto-Shutdown:**
- Exits gracefully after capturing 4 frames
- No manual intervention needed

### Integration with Processor Development

**When creating new processors:**

1. Run frame capture BEFORE implementing processor
2. Implement processor with metadata
3. Run frame capture AFTER implementing processor
4. Compare before/after frames at relevant stages
5. Verify processor effect is visible and correct

**Example workflow:**

```bash
# 1. Baseline capture (no processor)
Start_StreamDiffusion_DebugCapture.bat

# 2. Implement LatentNoiseProcessor
# ... code ...

# 3. Enable processor in td_config.yaml
latent_preprocessing:
  processors:
    - type: "latent_noise"
      params:
        strength: 0.3

# 4. Capture with processor
Start_StreamDiffusion_DebugCapture.bat

# 5. Compare stage 2 and stage 4 before/after
# Should see noise injection in latent space
```

### Performance Notes

- Frame capture adds ~5-10% overhead per captured frame
- Skipping 30 frames ensures stable performance metrics
- Synthetic input runs at ~4-5 FPS without CUDA graphs
- Real TouchDesigner input typically runs at 8-15 FPS

### Troubleshooting

**No frames captured:**
- Check `debug_frames/` directory exists
- Verify write permissions
- Check console for errors

**Frames look identical:**
- Increase synthetic input speed (modify `synthetic_input_generator.py`)
- Reduce skip_frames count to see earlier animation
- Verify processors are actually enabled in config

**Latent visualizations are noise:**
- This is normal! Latent space is 4-channel compressed representation
- Look for patterns/structure changes between stages 2, 3, 4
- Don't expect latent vis to look like final image

**Performance too slow:**
- Disable unused processors
- Reduce image resolution in config
- Skip more warmup frames (increase skip_frames)
- Re-enable CUDA graphs when API fixed

---

## Common Pitfalls & Solutions

### ❌ Pitfall 1: Resizing Latent Tensors
**Problem:** Base processor tries to resize latents to image dimensions.
**Solution:** Override `validate_tensor_input()` and `_ensure_target_size_tensor()`.

### ❌ Pitfall 2: Forgetting First Frame Handling
**Problem:** `prev_*_result` is `None`, causing crashes.
**Solution:** Always check `if prev_data is None` and use `_first_frame` flag.

### ❌ Pitfall 3: Color Space Mismatch
**Problem:** VAE outputs `[-1, 1]`, image processors expect `[0, 1]`.
**Solution:** Convert: `prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)`

### ❌ Pitfall 4: Feedback Oscillation
**Problem:** 1-frame delay in async pipeline causes feedback to oscillate.
**Solution:** Set `requires_sync_processing = True` for image preprocessing feedback.

### ❌ Pitfall 5: Forgetting to Clamp
**Problem:** Values accumulate beyond safe range, breaking pipeline.
**Solution:** Always clamp: `[0, 1]` for images, `[-10, 10]` for latents.

### ❌ Pitfall 6: Recreating Processors on Parameter Updates
**Problem:** OSC handler recreates processor instead of updating attribute.
**Solution:** Use `setattr(processor, param_name, value)` - direct attribute update!

---

## Quick Reference

### File Locations
- **Processors:** `src/streamdiffusion/preprocessing/processors/`
- **Registration:** `src/streamdiffusion/preprocessing/processors/__init__.py`
- **TD Extension:** `StreamDiffusionTD/StreamDiffusionExt.py`
- **OSC Handler:** `StreamDiffusionTD/td_osc_handler.py`
- **Pipeline:** `src/streamdiffusion/pipeline.py`

### Key Functions
- **TD Parameter Generation:** `update_fx_dynamic_parameters()` (line 6492)
- **TD Parameter Callback:** `Fxparameterupdate(par)` (line 5958)
- **OSC Handler:** `_handle_fx_parameter(address, *args)`
- **YAML Generation:** `generate_td_config_yaml()`

### Metadata Example
```python
{
    "display_name": "My Processor",
    "description": "What it does",
    "parameters": {
        "param_name": {
            "type": "float|int|string",
            "default": value,
            "range": [min, max],  # for float/int
            "options": [...],      # for string dropdown
            "step": 0.01,
            "description": "What this param controls"
        }
    },
    "use_cases": ["Use case 1", "Use case 2"]
}
```

### OSC Address Format
```
/fx/{processor_type}/{param_name}
Example: /fx/latent_transform/zoom
```

### Parameter Naming
```python
# Processor: latent_transform
# Metadata param: zoom
# TD param name: Fxlatenttransformzoom
# OSC address: /fx/latent_transform/zoom
```

---

## Advanced Topics

### Multi-Stage Temporal Effects

Chain processors across stages:
```yaml
image_preprocessing:
  - feedback_transform (image space)
latent_preprocessing:
  - latent_feedback (latent space)
latent_postprocessing:
  - color_correction_feedback (post-diffusion)
```

### Conditional Processing

Adapt processing based on input:
```python
def _process_tensor_core(self, tensor):
    # Detect motion
    if has_motion(tensor, self.prev_data):
        strength = self.motion_strength
    else:
        strength = self.static_strength

    # Apply adaptive processing
    ...
```

### Cross-Processor Communication

Access other processors via pipeline:
```python
# In processor A
self.pipeline_ref.custom_state['my_data'] = data

# In processor B (later in pipeline)
if 'my_data' in self.pipeline_ref.custom_state:
    data = self.pipeline_ref.custom_state['my_data']
```

---

## Conclusion

The Fx Dynamic Pipeline Processors system provides a **powerful, extensible architecture** for real-time image and latent manipulation in StreamDiffusion. By following the patterns in this guide, you can create sophisticated processors that integrate seamlessly with TouchDesigner.

**Key Takeaways:**

1. ✅ **Metadata is your single source of truth** - everything auto-generates from it
2. ✅ **Four hook stages** with distinct use cases - choose the right one
3. ✅ **Latent operations are 8x faster** - prefer when possible
4. ✅ **Pipeline-aware processors** can access previous frame data
5. ✅ **Live OSC updates** enable real-time experimentation without rebuilds
6. ✅ **Zero-code extension** - implement class, add toggle, done!

**Next Steps:**

1. Read the architecture docs: `multistage_processing_system_architecture.md`
2. Study example processors: `feedback_transform.py`, `latent_transform.py`
3. Create your first processor following this guide
4. Experiment with parameter ranges in TouchDesigner
5. Share your processors with the community!

---

**Document Version:** 1.0
**Last Updated:** 2025-10-27
**Maintainer:** Claude (Anthropic)
**Related Documents:**
- `multistage_processing_system_architecture.md`
- `fx_dynamic_pipeline_processors_implementation_2025-10-23.md`
