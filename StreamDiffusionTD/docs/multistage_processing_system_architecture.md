# Multistage Processing System Architecture
**StreamDiffusion LivePeer Fork - TouchDesigner Integration**

**Date:** 2025-01-23
**Session Focus:** Complete architectural overview of the hook-based multistage processing system

---

## Executive Summary

The LivePeer StreamDiffusion fork implements a **sophisticated hook-based processing pipeline** that allows custom processors to intercept and modify data at four critical stages:

1. **Image Preprocessing** - Before VAE encode (ControlNet input processing)
2. **Latent Preprocessing** - After VAE encode, before diffusion
3. **Latent Postprocessing** - After diffusion, before VAE decode
4. **Image Postprocessing** - After VAE decode (output enhancement)

This document provides a complete reference for understanding and extending this system, including shared memory architecture, processor creation patterns, and integration with TouchDesigner.

---

## Table of Contents

1. [Pipeline Flow Overview](#pipeline-flow-overview)
2. [Hook System Architecture](#hook-system-architecture)
3. [Shared Memory Architecture](#shared-memory-architecture)
4. [The Four Processing Stages](#the-four-processing-stages)
5. [Creating Custom Processors](#creating-custom-processors)
6. [TouchDesigner Integration](#touchdesigner-integration)
7. [Complete Data Flow](#complete-data-flow)
8. [Reference Implementation: LatentFeedbackPreprocessor](#reference-implementation)

---

## Pipeline Flow Overview

### Complete img2img Pipeline

```
TouchDesigner
    ↓
SharedMemory: input_mem_name (e.g., "StreamDiffusionTD_512-512")
    ↓
┌─────────────────────────────────────────────────────────────┐
│ INPUT IMAGE [H, W, 3] uint8 [0-255]                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ IMAGE PREPROCESSING HOOKS                                    │
│ (_apply_image_preprocessing_hooks)                          │
│                                                              │
│ Processors available:                                        │
│ - FeedbackPreprocessor (image-space V2V)                    │
│ - TemporalNetTensorRT (optical flow warping)               │
│ - Sharpen, Blur, Upscale, etc.                              │
│                                                              │
│ Use case: Process input before ControlNet/VAE encoding      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ CONTROLNET MODULE (if enabled)                              │
│ - Reads from: SharedMemory control_mem_name                 │
│   (e.g., "StreamDiffusionTD_512-512-cn")                    │
│ - Preprocesses with: Canny, Depth, Pose, etc.              │
│ - Outputs to: SharedMemory control_processed_mem_name       │
│   (e.g., "StreamDiffusionTD_512-512_out-cn-processed")      │
│ - Extracts features for UNet conditioning                   │
└─────────────────────────────────────────────────────────────┘
    ↓
VAE Encode → x_t_latent [B, 4, H/8, W/8] float16
    ↓
┌─────────────────────────────────────────────────────────────┐
│ LATENT PREPROCESSING HOOKS ⭐                                │
│ (_apply_latent_preprocessing_hooks)                         │
│                                                              │
│ Processors available:                                        │
│ - LatentFeedbackPreprocessor (temporal latent blending)     │
│                                                              │
│ Use case: Modify latents before diffusion                   │
│ - Temporal consistency                                       │
│ - Latent noise injection                                    │
│ - Latent interpolation                                      │
└─────────────────────────────────────────────────────────────┘
    ↓
UNet Diffusion (predict_x0_batch)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ LATENT POSTPROCESSING HOOKS 🔮                               │
│ (_apply_latent_postprocessing_hooks)                        │
│                                                              │
│ Processors available:                                        │
│ - (NONE CURRENTLY IMPLEMENTED)                              │
│                                                              │
│ Use case: Modify latents after diffusion                    │
│ - Latent color grading                                      │
│ - Latent sharpening                                         │
│ - Latent feedback loops (different from pre-diffusion)      │
└─────────────────────────────────────────────────────────────┘
    ↓
Store → stream.prev_latent_result (for next frame)
    ↓
VAE Decode → x_output [B, 3, H, W] float32 [-1, 1]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ IMAGE POSTPROCESSING HOOKS                                   │
│ (_apply_image_postprocessing_hooks)                         │
│                                                              │
│ Processors available:                                        │
│ - Sharpen, Blur, Upscale, etc. (reusable)                  │
│                                                              │
│ Use case: Enhance final output                              │
│ - Sharpening, color grading                                 │
│ - Upscaling, denoise                                        │
└─────────────────────────────────────────────────────────────┘
    ↓
Store → stream.prev_image_result (for next frame)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT IMAGE [H, W, 3] uint8 [0-255]                        │
└─────────────────────────────────────────────────────────────┘
    ↓
SharedMemory: output_mem_name (e.g., "StreamDiffusionTD_512-512_out")
    ↓
TouchDesigner
```

### txt2img Pipeline (Simplified)

```
Generate noise from seed
    ↓
LATENT PREPROCESSING HOOKS (LatentFeedback)
    ↓
UNet Diffusion
    ↓
LATENT POSTPROCESSING HOOKS
    ↓
VAE Decode
    ↓
IMAGE POSTPROCESSING HOOKS
    ↓
Output to SharedMemory
```

---

## Hook System Architecture

### Hook Types and Context Objects

The hook system is defined in [src/streamdiffusion/hooks.py](../src/streamdiffusion/hooks.py):

```python
# Hook context dataclasses
@dataclass
class ImageCtx:
    """Context for image processing hooks"""
    image: torch.Tensor      # [B, C, H, W] in image space
    width: int
    height: int
    step_index: Optional[int] = None

@dataclass
class LatentCtx:
    """Context for latent processing hooks"""
    latent: torch.Tensor     # [B, C, H/8, W/8] in latent space
    timestep: Optional[torch.Tensor] = None
    step_index: Optional[int] = None

# Hook type aliases
ImageHook = Callable[[ImageCtx], ImageCtx]
LatentHook = Callable[[LatentCtx], LatentCtx]
```

### Hook Registration

Hooks are registered in the pipeline at initialization:

**File:** [src/streamdiffusion/pipeline.py](../src/streamdiffusion/pipeline.py)

```python
# Line 113-118
self.image_preprocessing_hooks: List[ImageHook] = []
self.latent_preprocessing_hooks: List[LatentHook] = []
self.latent_postprocessing_hooks: List[LatentHook] = []
self.image_postprocessing_hooks: List[ImageHook] = []
```

### Hook Execution

Hooks are called at specific points in the pipeline:

```python
# Image preprocessing (line 876)
x = self._apply_image_preprocessing_hooks(x)

# Latent preprocessing (line 887)
x_t_latent = self._apply_latent_preprocessing_hooks(x_t_latent)

# Latent postprocessing (line 897)
x_0_pred_out = self._apply_latent_postprocessing_hooks(x_0_pred_out)

# Image postprocessing (line 906)
x_output = self._apply_image_postprocessing_hooks(x_output)
```

**Performance Note:** Hook execution is **zero overhead when no hooks are registered** (early exit check):

```python
def _apply_latent_preprocessing_hooks(self, latent: torch.Tensor) -> torch.Tensor:
    # Early exit - zero overhead when no hooks registered
    if not self.latent_preprocessing_hooks:
        return latent

    # Process hooks
    latent_ctx = LatentCtx(latent=latent)
    for hook in self.latent_preprocessing_hooks:
        latent_ctx = hook(latent_ctx)

    return latent_ctx.latent
```

---

## Shared Memory Architecture

### Overview

TouchDesigner communicates with the Python backend via **SharedMemory** (Windows/Linux) or **Syphon** (macOS). This provides zero-copy, high-performance frame transfer.

### Memory Buffers

**File:** [StreamDiffusionTD/td_manager.py](../StreamDiffusionTD/td_manager.py)

| Buffer Name | Direction | Purpose | Format |
|-------------|-----------|---------|--------|
| `input_mem_name` | TD → Python | Main input image | [H, W, 3] uint8 |
| `output_mem_name` | Python → TD | Final output image | [H, W, 3] uint8 |
| `control_mem_name` | TD → Python | ControlNet input (per-frame) | [H, W, 3] uint8 |
| `control_processed_mem_name` | Python → TD | Pre-processed ControlNet | [H, W, 3] uint8 |
| `ipadapter_mem_name` | TD → Python | IPAdapter style image (OSC-triggered) | [H, W, 3] uint8 |

**Example naming:**
- Input: `StreamDiffusionTD_512-512`
- Output: `StreamDiffusionTD_512-512_out`
- ControlNet input: `StreamDiffusionTD_512-512-cn`
- ControlNet processed: `StreamDiffusionTD_512-512_out-cn-processed`
- IPAdapter: `StreamDiffusionTD_512-512-ip`

### ControlNet Preprocessing Flow

**Key Insight:** ControlNet preprocessing happens **INSIDE the ControlNet module**, not in image preprocessing hooks.

```
TouchDesigner ControlNet TOP
    ↓
SharedMemory: control_mem_name
    ↓
td_manager._process_controlnet_frame()
    ↓
wrapper.update_control_image(0, control_frame)
    ↓
ControlNet Module:
  - Applies preprocessor (Canny, Depth, etc.)
  - Stores in controlnet_images[0]
    ↓
td_manager extracts processed image
    ↓
SharedMemory: control_processed_mem_name
    ↓
TouchDesigner (for visualization)
```

**Code:** [td_manager.py:604-651](../StreamDiffusionTD/td_manager.py#L604-L651)

```python
def _process_controlnet_frame(self) -> None:
    # Read from shared memory
    control_frame = np.ndarray(
        (height, width, 3),
        dtype=np.uint8,
        buffer=self.control_memory.buf
    )

    # Update ControlNet (triggers preprocessing internally)
    self.wrapper.update_control_image(0, control_frame)

    # Extract preprocessed output
    if hasattr(self.wrapper.stream, '_controlnet_module'):
        controlnet_module = self.wrapper.stream._controlnet_module
        processed_tensor = controlnet_module.controlnet_images[0]

        # Send back to TouchDesigner for visualization
        self._send_processed_controlnet_frame(processed_tensor)
```

### IPAdapter Image Flow

**Key Insight:** IPAdapter updates are **OSC-triggered**, not per-frame.

```
TouchDesigner IPAdapter TOP
    ↓
SharedMemory: ipadapter_mem_name (idle)
    ↓
User clicks "Update IPAdapter" button
    ↓
TD sends OSC: /ipadapter_update
    ↓
td_manager.request_ipadapter_update() → sets flag
    ↓
Next frame: _process_ipadapter_frame()
  - Reads from ipadapter_mem_name
  - Calls wrapper.update_style_image()
  - Triggers prompt re-blending (concatenates embeddings)
    ↓
IPAdapter embeddings cached for all future frames
```

**Code:** [td_manager.py:653-713](../StreamDiffusionTD/td_manager.py#L653-L713)

---

## The Four Processing Stages

### 1. Image Preprocessing Hooks

**Location:** Before VAE encode
**Hook List:** `stream.image_preprocessing_hooks`
**Module:** `ImagePreprocessingModule`
**File:** [src/streamdiffusion/modules/image_processing_module.py](../src/streamdiffusion/modules/image_processing_module.py)

**Purpose:**
- Process input image before ControlNet and VAE encoding
- Temporal consistency in image space (feedback loops)
- Image enhancement/filtering

**Available Processors:**
- **FeedbackPreprocessor** - Blends current input with previous output (image-space V2V)
- **TemporalNetTensorRTPreprocessor** - Optical flow warping using RAFT
- **SharpenPreprocessor** - Sharpen images
- **BlurPreprocessor** - Blur images
- **UpscalePreprocessor** - Upscale images
- **PassthroughPreprocessor** - No-op (testing)
- **ExternalPreprocessor** - External processing hook

**Configuration:**
```yaml
image_preprocessing:
  enabled: true
  processors:
    - type: "feedback"
      order: 1
      enabled: true
      params:
        feedback_strength: 0.5  # 50% blend with previous output
```

**IMPORTANT:** ControlNet preprocessors (Canny, Depth, Pose, etc.) are **NOT** in this list. They are managed by the ControlNet module separately.

---

### 2. Latent Preprocessing Hooks ⭐

**Location:** After VAE encode, before diffusion
**Hook List:** `stream.latent_preprocessing_hooks`
**Module:** `LatentPreprocessingModule`
**File:** [src/streamdiffusion/modules/latent_processing_module.py](../src/streamdiffusion/modules/latent_processing_module.py)

**Purpose:**
- Modify latents before they go through diffusion
- Temporal consistency in latent space (most efficient!)
- Latent-space effects (noise, interpolation, morphing)

**Available Processors:**
- **LatentFeedbackPreprocessor** - Blends current latent with previous latent output

**Configuration:**
```yaml
latent_preprocessing:
  enabled: true
  processors:
    - type: "latent_feedback"
      order: 1
      enabled: true
      params:
        feedback_strength: 0.13  # 13% blend with previous latent
```

**Why Only One Processor?**

Latent preprocessing is **highly specialized** - most effects are better suited for image space or latent postprocessing. The latent feedback processor is unique because it needs access to `prev_latent_result` which is only available at this exact point in the pipeline.

**Opportunities for Extension:**
- Latent noise injection (controlled randomness)
- Latent interpolation (between cached states)
- Latent channel manipulation (color grading in latent space)
- Latent morphing (optical flow in latent space)

---

### 3. Latent Postprocessing Hooks 🔮

**Location:** After diffusion, before VAE decode
**Hook List:** `stream.latent_postprocessing_hooks`
**Module:** `LatentPostprocessingModule`
**File:** [src/streamdiffusion/modules/latent_processing_module.py](../src/streamdiffusion/modules/latent_processing_module.py)

**Purpose:**
- Modify latents after diffusion but before decoding
- Latent-space color grading
- Latent-space sharpening/filtering
- Different feedback loops (accumulation, trails)

**Available Processors:**
- **NONE CURRENTLY IMPLEMENTED!**

**Configuration:**
```yaml
latent_postprocessing:
  enabled: true
  processors:
    - type: "latent_sharpen"  # Example (not implemented)
      order: 1
      enabled: true
      params:
        strength: 0.5
```

**Why Empty?**

This is an **untapped opportunity** for custom processors. Most post-diffusion effects are done in image space (after VAE decode), but latent postprocessing could be more efficient for certain operations.

**Opportunities for Extension:**
- Latent color grading (modify latent channels for color shifts)
- Latent detail enhancement (sharpen in latent space before decode)
- Latent feedback accumulation (different from preprocessing - accumulates over time)
- Latent denoise/smooth (reduce artifacts before decode)

---

### 4. Image Postprocessing Hooks

**Location:** After VAE decode, before output
**Hook List:** `stream.image_postprocessing_hooks`
**Module:** `ImagePostprocessingModule`
**File:** [src/streamdiffusion/modules/image_processing_module.py](../src/streamdiffusion/modules/image_processing_module.py)

**Purpose:**
- Enhance final output image
- Color grading, sharpening, upscaling
- Aesthetic effects

**Available Processors:**
- **SharpenPreprocessor** - Sharpen output
- **BlurPreprocessor** - Blur output
- **UpscalePreprocessor** - Upscale output
- **RealESRGANProcessor** - Real-ESRGAN upscaling (TensorRT)

**Configuration:**
```yaml
image_postprocessing:
  enabled: true
  processors:
    - type: "sharpen"
      order: 1
      enabled: true
      params:
        amount: 0.5
```

---

## Creating Custom Processors

### Base Processor Pattern

All processors inherit from `BasePreprocessor` or `PipelineAwareProcessor`:

**File:** [src/streamdiffusion/preprocessing/processors/base.py](../src/streamdiffusion/preprocessing/processors/base.py)

```python
from .base import BasePreprocessor

class MyCustomProcessor(BasePreprocessor):
    """
    Custom processor for [purpose].
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        """Metadata for UI/registration"""
        return {
            "display_name": "My Custom Processor",
            "description": "Does something cool",
            "parameters": {
                "strength": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Effect strength"
                }
            },
            "use_cases": ["Cool effects", "Temporal consistency"]
        }

    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(strength=strength, **kwargs)
        self.strength = strength

    def _process_core(self, image: Image.Image) -> Image.Image:
        """Process PIL Image"""
        # Your processing logic here
        return image

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process torch.Tensor (GPU-optimized path)"""
        # Your processing logic here
        return tensor
```

### Pipeline-Aware Processor Pattern

For processors that need access to previous frame data:

```python
from .base import PipelineAwareProcessor

class MyFeedbackProcessor(PipelineAwareProcessor):
    """
    Processor that accesses previous frame data.
    """

    def __init__(self, pipeline_ref: Any, strength: float = 0.5, **kwargs):
        super().__init__(pipeline_ref=pipeline_ref, strength=strength, **kwargs)
        self.strength = strength
        self._first_frame = True

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        # Access previous frame data
        if self.pipeline_ref and hasattr(self.pipeline_ref, 'prev_image_result'):
            prev_output = self.pipeline_ref.prev_image_result

            # Blend current with previous
            if prev_output is not None and not self._first_frame:
                blended = (1 - self.strength) * tensor + self.strength * prev_output
                return blended

        self._first_frame = False
        return tensor
```

### Latent Domain Processor Pattern

For processors that work on latents:

```python
from .base import PipelineAwareProcessor

class MyLatentProcessor(PipelineAwareProcessor):
    """
    Processor for latent domain.
    """

    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """Preserve latent dimensions - don't resize"""
        latent_tensor = latent_tensor.to(device=self.device, dtype=self.dtype)
        return latent_tensor

    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Override resize - latent tensors should NOT be resized"""
        return tensor

    def _process_core(self, image):
        """Not used for latent processors"""
        raise NotImplementedError("Use _process_tensor_core for latent processors")

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process latent tensor [B, 4, H/8, W/8]"""
        # Your latent processing logic here
        return tensor
```

### Registration

Add to [src/streamdiffusion/preprocessing/processors/__init__.py](../src/streamdiffusion/preprocessing/processors/__init__.py):

```python
from .my_custom_processor import MyCustomProcessor

_preprocessor_registry = {
    # ... existing processors ...
    "my_custom": MyCustomProcessor,
}
```

---

## TouchDesigner Integration

### YAML Config Generation

**File:** [StreamDiffusionExt.py:4704-4728](../StreamDiffusionExt.py#L4704-L4728)

The YAML config is **dynamically generated** from TouchDesigner parameters every time you click "Start Stream":

```python
def generate_td_config_yaml(self):
    # ... other config ...

    # Add latent preprocessing if enabled
    if self.ownerComp.par.Uselatentfeedback.eval():
        feedback_strength = self.ownerComp.par.Latentfeedbackstrength.eval()

        yaml_content += """
latent_preprocessing:
  enabled: true
  processors:
    - type: "latent_feedback"
      order: 1
      enabled: true
      params:
        feedback_strength: {feedback_strength}
"""
```

### Live Parameter Updates via OSC

**File:** [StreamDiffusionTD/td_osc_handler.py:295-312](../StreamDiffusionTD/td_osc_handler.py#L295-L312)

Parameters can be updated in real-time via OSC:

```python
def _handle_latent_feedback_strength(self, address, *args):
    """Handle latent feedback strength updates"""
    feedback_strength = float(args[0])

    # Direct attribute update (avoids processor replacement)
    if hasattr(self.manager, 'wrapper') and self.manager.wrapper:
        stream = self.manager.wrapper.stream

        if hasattr(stream, '_latent_preprocessing_module'):
            module = stream._latent_preprocessing_module
            for processor in module.processors:
                if processor.__class__.__name__ == 'LatentFeedbackPreprocessor':
                    processor.feedback_strength = feedback_strength
                    return
```

**TouchDesigner Callback:**

```python
# In StreamDiffusionExt.py
def Latentfeedbackstrength(self):
    """Called when parameter changes"""
    if not self.ownerComp.par.Streamactive:
        return

    feedback_strength = self.ownerComp.par.Latentfeedbackstrength.eval()

    osc_out = op('oscout1')
    if osc_out:
        osc_out.sendOSC('/latent_feedback_strength', [feedback_strength])
```

### Filtering ControlNet Preprocessors

**File:** [StreamDiffusionExt.py:6006-6009](../StreamDiffusionExt.py#L6006-L6009)

Latent-domain processors are filtered out of ControlNet dropdown:

```python
# Filter out latent-domain processors (they're not for ControlNet)
latent_only_processors = ['latent_feedback']
processors = [p for p in processors if p not in latent_only_processors]
```

---

## Complete Data Flow

### Memory Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     TOUCHDESIGNER                            │
│                                                              │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │ Input TOP  │  │ControlNet   │  │ IPAdapter    │         │
│  │ (camera)   │  │ TOP         │  │ TOP          │         │
│  └──────┬─────┘  └──────┬──────┘  └──────┬───────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          │ SharedMemory   │ SharedMemory   │ SharedMemory
          │ (per-frame)    │ (per-frame)    │ (on OSC trigger)
          ↓                ↓                ↓
┌──────────────────────────────────────────────────────────────┐
│                  PYTHON BACKEND (td_manager.py)              │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Main Input                                        │     │
│  │  input_mem_name: "StreamDiffusionTD_512-512"     │     │
│  └──────┬─────────────────────────────────────────────┘     │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Image Preprocessing Hooks                         │     │
│  │  - FeedbackPreprocessor                            │     │
│  │  - TemporalNetTensorRT                             │     │
│  └──────┬─────────────────────────────────────────────┘     │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  ControlNet Module                                 │     │
│  │  Input: control_mem_name                           │     │
│  │         "StreamDiffusionTD_512-512-cn"             │     │
│  │  Preprocesses: Canny, Depth, Pose, etc.           │     │
│  │  Output: control_processed_mem_name                │     │
│  │          "StreamDiffusionTD_512-512_out-cn-        │     │
│  │           processed"                                │     │
│  └──────┬─────────────────────────────────────────────┘     │
│         ↓                                                    │
│  VAE Encode → latent [B, 4, H/8, W/8]                       │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Latent Preprocessing Hooks ⭐                      │     │
│  │  - LatentFeedbackPreprocessor                      │     │
│  └──────┬─────────────────────────────────────────────┘     │
│         ↓                                                    │
│  UNet Diffusion (with ControlNet + IPAdapter)               │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Latent Postprocessing Hooks 🔮                     │     │
│  │  - (EMPTY - opportunity for custom processors)     │     │
│  └──────┬─────────────────────────────────────────────┘     │
│         ↓                                                    │
│  VAE Decode → image [B, 3, H, W]                            │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Image Postprocessing Hooks                         │     │
│  │  - Sharpen, Upscale, etc.                          │     │
│  └──────┬─────────────────────────────────────────────┘     │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Output Memory                                      │     │
│  │  output_mem_name: "StreamDiffusionTD_512-512_out" │     │
│  └──────┬─────────────────────────────────────────────┘     │
└─────────┼────────────────────────────────────────────────────┘
          │ SharedMemory
          ↓
┌──────────────────────────────────────────────────────────────┐
│                     TOUCHDESIGNER                            │
│                                                              │
│  ┌────────────┐  ┌─────────────────────┐                   │
│  │ Output TOP │  │ ControlNet          │                   │
│  │            │  │ Processed TOP       │                   │
│  │            │  │ (visualization)     │                   │
│  └────────────┘  └─────────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

### OSC Command Flow

```
TouchDesigner Parameter Change
    ↓
Callback Function (e.g., Latentfeedbackstrength())
    ↓
OSC Message: /latent_feedback_strength [0.13]
    ↓
td_osc_handler._handle_latent_feedback_strength()
    ↓
Direct Processor Attribute Update
    processor.feedback_strength = 0.13
    ↓
✅ Next frame uses new value (no engine rebuild!)
```

---

## Reference Implementation

### LatentFeedbackPreprocessor

**File:** [src/streamdiffusion/preprocessing/processors/latent_feedback.py](../src/streamdiffusion/preprocessing/processors/latent_feedback.py)

This is the **complete, production-ready implementation** of a latent domain processor. Use it as a reference for creating your own processors.

**Key Features:**
1. ✅ Inherits from `PipelineAwareProcessor` (needs `prev_latent_result`)
2. ✅ Overrides `validate_tensor_input()` to preserve latent dimensions
3. ✅ Overrides `_ensure_target_size_tensor()` to prevent resizing
4. ✅ Implements `_process_tensor_core()` for latent processing
5. ✅ Raises `NotImplementedError` in `_process_core()` (not for PIL images)
6. ✅ Provides metadata via `get_preprocessor_metadata()`
7. ✅ Handles first frame edge case
8. ✅ Clamps output to safe range (`[-10.0, 10.0]`)
9. ✅ Handles batch size mismatches gracefully

**Core Processing Logic:**

```python
def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
    # Get previous frame latent
    prev_latent = self._get_previous_data()

    if prev_latent is not None:
        # Handle batch size mismatches
        if prev_latent.shape[0] != tensor.shape[0]:
            # Expand or slice to match
            # ...

        # Blend current with previous
        blended_latent = (1 - self.feedback_strength) * tensor + \
                        self.feedback_strength * prev_latent

        # Safety clamp
        blended_latent = torch.clamp(blended_latent, min=-10.0, max=10.0)

        return blended_latent
    else:
        # First frame - passthrough
        self._first_frame = False
        return tensor
```

---

## Extending the System

### Adding a New Latent Postprocessor

**Example: Latent Color Grading**

```python
# File: src/streamdiffusion/preprocessing/processors/latent_color_grade.py

import torch
from .base import PipelineAwareProcessor

class LatentColorGradeProcessor(PipelineAwareProcessor):
    """
    Latent-space color grading processor.
    Modifies latent channels to shift colors after diffusion.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Latent Color Grading",
            "description": "Color grading in latent space (post-diffusion)",
            "parameters": {
                "red_shift": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-1.0, 1.0],
                    "step": 0.01,
                    "description": "Red channel shift"
                },
                "blue_shift": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-1.0, 1.0],
                    "step": 0.01,
                    "description": "Blue channel shift"
                },
                "contrast": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.5, 2.0],
                    "step": 0.1,
                    "description": "Latent contrast"
                }
            },
            "use_cases": ["Color grading", "Latent manipulation", "Post-diffusion effects"]
        }

    def __init__(self, pipeline_ref,
                 red_shift: float = 0.0,
                 blue_shift: float = 0.0,
                 contrast: float = 1.0,
                 **kwargs):
        super().__init__(pipeline_ref=pipeline_ref, **kwargs)
        self.red_shift = red_shift
        self.blue_shift = blue_shift
        self.contrast = contrast

    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """Preserve latent dimensions"""
        return latent_tensor.to(device=self.device, dtype=self.dtype)

    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Don't resize latents"""
        return tensor

    def _process_core(self, image):
        raise NotImplementedError("LatentColorGradeProcessor is for latent domain only")

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process latent tensor [B, 4, H/8, W/8]

        Latent channels (SD 1.5/SDXL):
        - Channel 0: Low frequency detail
        - Channel 1: Mid frequency detail
        - Channel 2: High frequency detail
        - Channel 3: Color/tone information
        """
        latent = tensor.clone()

        # Apply contrast (scale from mean)
        mean = latent.mean(dim=(2, 3), keepdim=True)
        latent = mean + (latent - mean) * self.contrast

        # Apply color shifts (modify specific channels)
        # This is simplified - real color grading would be more sophisticated
        if self.red_shift != 0.0:
            latent[:, 0, :, :] += self.red_shift

        if self.blue_shift != 0.0:
            latent[:, 2, :, :] += self.blue_shift

        # Clamp to safe range
        latent = torch.clamp(latent, min=-10.0, max=10.0)

        return latent
```

**Register:**

```python
# In __init__.py
from .latent_color_grade import LatentColorGradeProcessor

_preprocessor_registry = {
    # ...
    "latent_color_grade": LatentColorGradeProcessor,
}
```

**Use in YAML:**

```yaml
latent_postprocessing:
  enabled: true
  processors:
    - type: "latent_color_grade"
      order: 1
      enabled: true
      params:
        red_shift: 0.2
        blue_shift: -0.1
        contrast: 1.2
```

**Add TouchDesigner Integration:**

1. Add parameters: `Uselatentcolorgrade`, `Latentredshift`, `Latentblueshift`, `Latentcontrast`
2. Add to YAML generation
3. Add OSC handlers for live updates
4. Add callbacks for parameter changes

---

## Best Practices

### When to Use Each Stage

| Stage | Use For | Don't Use For |
|-------|---------|---------------|
| **Image Preprocessing** | Input filtering, temporal effects in image space, ControlNet input enhancement | Latent manipulation, output enhancement |
| **Latent Preprocessing** | Temporal consistency in latent space, latent noise injection, pre-diffusion latent effects | Post-diffusion effects, color grading |
| **Latent Postprocessing** | Latent color grading, latent sharpening, post-diffusion latent effects | Image-space effects (use image postprocessing instead) |
| **Image Postprocessing** | Output sharpening, upscaling, color grading in image space | Input processing, latent effects |

### Performance Considerations

1. **Latent operations are faster** than image operations (8x smaller resolution)
2. **Preprocessing is better than postprocessing** for temporal effects (feedback loops)
3. **Direct attribute updates** are faster than config replacement (avoid processor recreation)
4. **Early exit checks** ensure zero overhead when hooks are disabled
5. **GPU operations** (`_process_tensor_core`) are much faster than CPU (`_process_core`)

### Common Pitfalls

1. ❌ **Don't resize latent tensors** - they're in latent space, not image space
2. ❌ **Don't use image processors for latents** - they'll try to resize to image dimensions
3. ❌ **Don't recreate processors on parameter updates** - update attributes directly
4. ❌ **Don't forget first-frame handling** - `prev_result` will be `None`
5. ❌ **Don't forget to clamp outputs** - prevent extreme values that break the pipeline

---

## Future Opportunities

### Unexplored Hook Slots

1. **Latent Postprocessing** - Currently empty, huge potential for:
   - Latent color grading
   - Latent detail enhancement
   - Latent feedback accumulation
   - Latent denoise/filtering

2. **Advanced Latent Preprocessing** - Could add:
   - Multi-frame latent interpolation
   - Latent noise scheduling
   - Latent morphing (optical flow in latent space)
   - Latent caching/blending systems

3. **Cross-Domain Effects** - Combine stages:
   - Image feedback → latent preprocessing → latent postprocessing
   - ControlNet influence on latent processing
   - IPAdapter-aware latent manipulation

### Advanced Integration Ideas

1. **Multi-stage Temporal Effects** - Chain processors:
   ```yaml
   image_preprocessing:
     - feedback (image space)
   latent_preprocessing:
     - latent_feedback (latent space)
   latent_postprocessing:
     - latent_accumulator (trails effect)
   ```

2. **Conditional Processing** - Processors that adapt based on input:
   - Motion-adaptive latent feedback
   - Content-aware color grading
   - Dynamic strength based on scene change

3. **TouchDesigner UI Integration** - Could add:
   - Visual editor for processor chains
   - Real-time parameter mapping (CHOP → OSC → processor)
   - Processor preset system

---

## Conclusion

The multistage processing system in the LivePeer StreamDiffusion fork provides a **powerful, extensible architecture** for custom image and latent manipulation. By understanding the four hook stages, shared memory architecture, and processor patterns, you can create sophisticated real-time effects that integrate seamlessly with TouchDesigner.

**Key Takeaways:**

1. ✅ **Four distinct hook stages** with zero overhead when unused
2. ✅ **Shared memory architecture** provides zero-copy frame transfer
3. ✅ **ControlNet preprocessing is separate** from image preprocessing hooks
4. ✅ **Latent postprocessing is completely unexplored** - huge opportunity!
5. ✅ **Live parameter updates via OSC** enable real-time experimentation
6. ✅ **LatentFeedbackPreprocessor is the reference implementation** for custom processors

---

**Document Version:** 1.0
**Last Updated:** 2025-01-23
**Maintainer:** Claude (Anthropic)
**Related Documents:**
- [txt2img_implementation_architecture_2025-01-22.md](./txt2img_implementation_architecture_2025-01-22.md)
- [ControlNet & IPAdapter Implementation Review](../claude_session_notes/controlnet_ipadapter_implementation_review.md)
