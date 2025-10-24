# StreamDiffusion Processor Development Guide

**Complete guide to creating real-time image processors for StreamDiffusion + TouchDesigner**

Last Updated: 2025-10-24

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Processor Types](#processor-types)
3. [Creating a New Processor](#creating-a-new-processor)
4. [TouchDesigner Integration](#touchdesigner-integration)
5. [Testing & Debugging](#testing--debugging)
6. [Common Patterns](#common-patterns)
7. [Known Issues & Solutions](#known-issues--solutions)

---

## Architecture Overview

### The Pipeline Flow

```
TouchDesigner (UI)
    ↓ OSC Parameters
Python Backend (StreamDiffusion)
    ↓
┌─────────────────────────────────────┐
│  1. Image Input (Webcam/TOP)        │
│  2. Image Preprocessing (Pre-VAE)   │ ← Color Correction, Feedback, etc.
│  3. VAE Encode → Latent Space       │
│  4. Latent Preprocessing            │ ← Transform, Zoom, Pan, Rotate
│  5. Diffusion (UNet + Scheduler)    │
│  6. Latent Postprocessing           │
│  7. VAE Decode → Image Space        │
│  8. Image Postprocessing (Post-VAE) │ ← Sharpen, Upscale, etc.
│  9. Output (Shared Memory → TD)     │
└─────────────────────────────────────┘
    ↑ OSC Feedback
TouchDesigner (Display)
```

### Component Locations

**Python Side:**
- Processors: `src/streamdiffusion/preprocessing/processors/`
- Registration: `src/streamdiffusion/preprocessing/processors/__init__.py`
- Orchestrators: `src/streamdiffusion/preprocessing/`
- OSC Handler: `streamdiffusionTD/td_osc_handler.py`

**TouchDesigner Side:**
- Extension: `StreamDiffusionTD/StreamDiffusionExt.py`
- Parameter callbacks
- YAML generation
- OSC transmission

---

## Processor Types

### 1. Simple Processors (BasePreprocessor)

**Use when:** Processing single frames independently, no temporal effects

**Examples:** Blur, Sharpen, Canny Edge, Depth Map

**Characteristics:**
- Inherits from `BasePreprocessor`
- No access to previous frames
- Stateless (each frame processed independently)
- Can be used as ControlNet preprocessors

### 2. Pipeline-Aware Processors (PipelineAwareProcessor)

**Use when:** Need access to previous outputs, temporal effects, feedback loops

**Examples:** FeedbackTransform, LatentTransform, ColorCorrectionFeedback

**Characteristics:**
- Inherits from `PipelineAwareProcessor`
- Access to `pipeline_ref` (previous frames, latents, noise)
- **CRITICAL:** Set `requires_sync_processing = True` to avoid frame delays
- Stateful (tracks frame history)

### 3. Domain Types

**Image Domain (Pre-VAE):**
- Works with RGB images in `[0, 1]` range
- Best for: Color grading, feedback effects, geometric transforms
- Module: `_image_preprocessing_module`
- Applied before VAE encoding

**Latent Domain:**
- Works with compressed 4-channel latents
- Best for: Motion (zoom/pan), noise injection, latent-space effects
- Module: `_latent_preprocessing_module`
- Applied after VAE encoding

---

## Creating a New Processor

### Step 1: Create the Processor File

**Location:** `src/streamdiffusion/preprocessing/processors/your_processor.py`

#### Simple Processor Template

```python
from PIL import Image
import torch
import torch.nn.functional as F
from .base import BasePreprocessor


class YourEffectPreprocessor(BasePreprocessor):
    """
    Brief description of what this processor does.

    Parameters:
        intensity (float): Effect strength (0.0 to 1.0)
        mode (str): Processing mode ('fast' or 'quality')
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        """
        Metadata for TouchDesigner UI generation and documentation.
        This is CRITICAL - it defines the TD parameters!
        """
        return {
            "display_name": "Your Effect",
            "description": "What it does in one sentence",
            "parameters": {
                "intensity": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Effect strength"
                },
                "mode": {
                    "type": "string",
                    "default": "fast",
                    "options": ["fast", "quality"],
                    "description": "Processing quality vs speed"
                }
            },
            "use_cases": [
                "Use case 1",
                "Use case 2"
            ]
        }

    def __init__(self, intensity: float = 0.5, mode: str = "fast", **kwargs):
        """
        Initialize processor.
        All parameters from metadata should be accepted here.
        """
        super().__init__(intensity=intensity, mode=mode, **kwargs)
        self.intensity = intensity
        self.mode = mode

        # Cache expensive resources (kernels, models, etc.)
        self._cached_kernel = None

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        REQUIRED: PIL Image processing path.

        The base class handles:
        - Input validation
        - Format conversions (PIL ↔ Tensor ↔ NumPy)
        - Resizing to target dimensions

        You only implement the core algorithm.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Processed PIL Image in RGB mode
        """
        # Your PIL-based processing here
        # Simple example: adjust brightness
        import numpy as np

        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.clip(img_array * (1.0 + self.intensity), 0, 1)
        result = Image.fromarray((img_array * 255).astype(np.uint8))

        return result

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        OPTIONAL: GPU-accelerated tensor processing path.

        If not implemented, falls back to _process_core (PIL path).
        Implement this for performance-critical processors.

        Args:
            tensor: Input tensor [C, H, W] or [B, C, H, W] in range [0, 1]

        Returns:
            Processed tensor [0, 1]
        """
        # Ensure batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Ensure on correct device
        tensor = tensor.to(device=self.device, dtype=self.dtype)

        # Your GPU processing here
        result = tensor * (1.0 + self.intensity)
        result = result.clamp(0, 1)

        return result
```

#### Pipeline-Aware Processor Template

```python
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Any
from .base import PipelineAwareProcessor


class YourFeedbackPreprocessor(PipelineAwareProcessor):
    """
    Processor with access to previous frame outputs.

    Use for: Feedback loops, temporal effects, motion trails
    """

    # CRITICAL: Prevents 1-frame delay in feedback
    requires_sync_processing = True

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Your Feedback Effect",
            "description": "Temporal effect with feedback",
            "parameters": {
                "feedback_strength": {
                    "type": "float",
                    "default": 0.8,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Feedback blend (0=pure input, 1=pure feedback)"
                },
                "effect_amount": {
                    "type": "float",
                    "default": 0.1,
                    "range": [-1.0, 1.0],
                    "step": 0.01,
                    "description": "Effect intensity"
                }
            },
            "use_cases": [
                "Accumulative effects over time",
                "Feedback-based transitions"
            ]
        }

    def __init__(self,
                 pipeline_ref: Any,
                 feedback_strength: float = 0.8,
                 effect_amount: float = 0.1,
                 **kwargs):
        """
        REQUIRED: Must accept pipeline_ref as first parameter!
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            feedback_strength=feedback_strength,
            effect_amount=effect_amount,
            **kwargs
        )
        self.feedback_strength = feedback_strength
        self.effect_amount = effect_amount
        self._first_frame = True

    def reset(self):
        """Optional: Reset state for new sequences"""
        self._first_frame = True

    def _get_previous_data(self):
        """
        Get previous frame data from pipeline.

        Available data:
        - pipeline_ref.prev_image_result: Previous output image [B,C,H,W] in [-1,1]
        - pipeline_ref.prev_latent_result: Previous latent [B,4,H/8,W/8]
        - pipeline_ref.alpha_prod_t_sqrt: Noise schedule sqrt(alpha)
        - pipeline_ref.beta_prod_t_sqrt: Noise schedule sqrt(1-alpha)
        - pipeline_ref.init_noise: Seeded noise for temporal consistency
        """
        if self.pipeline_ref is not None:
            if hasattr(self.pipeline_ref, 'prev_image_result'):
                if self.pipeline_ref.prev_image_result is not None and not self._first_frame:
                    return self.pipeline_ref.prev_image_result
        return None

    def _process_core(self, image: Image.Image) -> Image.Image:
        """Process with feedback from previous frame"""
        prev_output_tensor = self._get_previous_data()

        if prev_output_tensor is None:
            # First frame - no feedback yet
            self._first_frame = False
            return image

        # Convert previous output from [-1, 1] (VAE range) to [0, 1] (image range)
        if prev_output_tensor.dim() == 4:
            prev_output_tensor = prev_output_tensor[0]  # Remove batch
        prev_output_tensor = (prev_output_tensor / 2.0 + 0.5).clamp(0, 1)

        # Apply your effect to PREVIOUS output (for accumulation)
        # Example: brightness adjustment
        modified_prev = prev_output_tensor + self.effect_amount
        modified_prev = modified_prev.clamp(0, 1)

        # Convert current input to tensor
        input_tensor = self.pil_to_tensor(image).squeeze(0)  # [C, H, W]

        # Ensure matching shapes
        if modified_prev.shape != input_tensor.shape:
            target_size = input_tensor.shape[-2:]
            modified_prev = modified_prev.unsqueeze(0)
            modified_prev = F.interpolate(
                modified_prev, size=target_size, mode='bilinear', align_corners=False
            )
            modified_prev = modified_prev.squeeze(0)

        # Blend: (1-strength)*input + strength*modified_previous
        blended = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * modified_prev
        blended = blended.clamp(0, 1)

        # Convert back to PIL
        result = self.tensor_to_pil(blended)

        self._first_frame = False
        return result
```

### Step 2: Register the Processor

**File:** `src/streamdiffusion/preprocessing/processors/__init__.py`

```python
# 1. Add import
from .your_processor import YourEffectPreprocessor

# 2. Add to registry (around line 71)
_preprocessor_registry = {
    # ... existing processors
    "your_effect": YourEffectPreprocessor,
}

# 3. Add to exports (around line 175)
__all__ = [
    # ... existing exports
    "YourEffectPreprocessor",
]
```

### Step 3: TouchDesigner Integration

#### A. Add Callback Function

**File:** `StreamDiffusionTD/StreamDiffusionExt.py`

Find the section with other `Use*` callbacks (around line 5930) and add:

```python
def Useyoureffect(self):
    self.logger.log('Useyoureffect changed', level='INFO')
    """Called when Useyoureffect toggle changes - updates Fx dynamic parameters"""
    self.update_fx_dynamic_parameters()
```

#### B. Add to Dynamic Parameter System

**File:** `StreamDiffusionTD/StreamDiffusionExt.py`

In `update_fx_dynamic_parameters()` function (around line 6475):

```python
def update_fx_dynamic_parameters(self):
    # ... existing code ...

    """Updates Fx* parameters for ... and your_effect on Fx page"""
    active_fx = []
    # ... existing checks ...

    # Add your processor check
    if hasattr(self.ownerComp.par, 'Useyoureffect') and self.ownerComp.par.Useyoureffect.eval():
        active_fx.append('your_effect')
```

#### C. Add to YAML Generation

**File:** `StreamDiffusionTD/StreamDiffusionExt.py`

Find the image or latent preprocessing section (around line 4683 or 4720) and add:

**For Image Preprocessing (Pre-VAE):**

```python
# Around line 4683
use_your_effect = (hasattr(self.ownerComp.par, 'Useyoureffect') and
                   self.ownerComp.par.Useyoureffect.eval())

if use_image_feedback or use_color_correction_feedback or use_your_effect:
    yaml_content += """# Multi-stage Image Preprocessing (Pre-Fx)
image_preprocessing:
  enabled: true
  processors:
"""
    processor_order = 1

    # ... existing processors ...

    if use_your_effect:
        params = self.gather_fx_parameters_for_processor('your_effect')
        yaml_content += f'    - type: "your_effect"\n'
        yaml_content += f'      order: {processor_order}\n'
        yaml_content += f'      enabled: true\n'
        yaml_content += f'      params:\n'
        if requires_sync:  # For pipeline-aware processors
            yaml_content += f'        requires_sync_processing: true\n'
        for param_name, param_value in params.items():
            yaml_content += f'        {param_name}: {param_value}\n'
```

**For Latent Preprocessing:**

```python
# Around line 4720 - similar pattern
if use_latent_feedback or use_latent_transform or use_your_latent_effect:
    yaml_content += """# Multi-stage Latent Preprocessing (Fx)
latent_preprocessing:
  enabled: true
  processors:
"""
    # ... add your processor ...
```

#### D. Create TouchDesigner Parameters

In TouchDesigner:

1. **Add Toggle Parameter:**
   - Name: `Useyoureffect`
   - Type: Toggle
   - Default: 0 (off)
   - Callback: Point to `Useyoureffect()` method

2. **Refresh Processor Table:**
   ```python
   op('your_component').Getpreprocessors()
   ```

3. **Enable Toggle:**
   - Set `Useyoureffect = 1`
   - Dynamic parameters (`Fxyoureffectintensity`, etc.) appear automatically

4. **Parameters Auto-Created:**
   - Naming: `Fx` + `processorname` + `paramname` (all lowercase, no separators)
   - Example: `Fxyoureffectintensity`, `Fxyoureffectmode`
   - These are created dynamically from metadata

---

## TouchDesigner Integration Deep Dive

### Parameter Flow

```
TD Parameter Change (e.g., Fxyoureffectintensity = 0.8)
    ↓
parexec_fx: onValueChange(par, prev)
    ↓
parent().Fxparameterupdate(par)
    ↓
[Match param to processor via table_preprocessors metadata]
    ↓
Build OSC address: /fx/your_effect/intensity
    ↓
oscout1.sendOSC(address, [0.8])
    ↓
[OSC over network → Python backend]
    ↓
td_osc_handler._handle_fx_parameter(address, *args)
    ↓
Parse: processor='your_effect', param='intensity', value=0.8
    ↓
Convert to class name: 'YourEffectPreprocessor'
    ↓
Find processor in stream._image_preprocessing_module
    ↓
setattr(processor, 'intensity', 0.8)
    ↓
✅ Parameter updated in real-time!
```

### Parameter Naming Convention

**TouchDesigner Parameter Name:**
```
Fx + processor_name + param_name (all lowercase, no separators)
```

**Examples:**
- `color_correction_feedback` + `brightness` → `Fxcolorcorrectionfeedbackbrightness`
- `your_effect` + `intensity` → `Fxyoureffectintensity`
- `latent_transform` + `zoom` → `Fxlatenttransformzoom`

**OSC Address:**
```
/fx/{processor_name}/{param_name}
```

**Examples:**
- `/fx/color_correction_feedback/brightness`
- `/fx/your_effect/intensity`
- `/fx/latent_transform/zoom`

### OSC Handler (Automatic)

The OSC handler in `streamdiffusionTD/td_osc_handler.py` is **fully dynamic**:

1. Receives: `/fx/your_effect/intensity 0.8`
2. Converts: `your_effect` → `YourEffectPreprocessor`
3. Searches: All preprocessing/postprocessing modules
4. Updates: `processor.intensity = 0.8`

**No hardcoding needed!** Just follow naming conventions.

---

## Testing & Debugging

### 1. Test Import

```python
# In Python console or script
from src.streamdiffusion.preprocessing.processors import YourEffectPreprocessor

# Check metadata
print(YourEffectPreprocessor.get_preprocessor_metadata())

# Verify class name
print(YourEffectPreprocessor.__name__)  # Should match expected format
```

### 2. Test in TouchDesigner

```python
# Refresh processor table
op('your_component').Getpreprocessors()

# Check if processor is in table
table = op('table_preprocessors')
for i in range(table.numRows):
    if 'your_effect' in str(table[i, 'name']):
        print(f"Found: {table[i, 'name']}")
        print(f"Params: {table[i, 'parameters_json']}")

# Enable processor
parent().par.Useyoureffect = 1

# Check dynamic parameters created
for p in parent().customPars:
    if 'youreffect' in p[0].name.lower():
        print(f"Parameter: {p[0].name} = {p[0].eval()}")
```

### 3. Debug OSC Communication

**Python Side (check logs):**
```
FX OSC received: /fx/your_effect/intensity = (0.8,)
Parsed FX: processor=your_effect, param=intensity, value=0.8
Looking for class names: ['YourEffectPreprocessor', 'YourEffectPostprocessor', 'YourEffect']
Found matching processor: YourEffectPreprocessor in _image_preprocessing_module
✅ FX UPDATE SUCCESS: YourEffectPreprocessor.intensity = 0.5 -> 0.8
```

**If not found:**
```
❌ Processor 'your_effect' (looking for ['YourEffectPreprocessor', ...]) not found in any module
```

**Causes:**
1. Processor not added to YAML (check YAML generation)
2. Wrong module (image vs latent)
3. Class name doesn't match convention

### 4. Check YAML Output

```python
# In TouchDesigner
yaml_content = op('your_component').Generateyaml()
print(yaml_content)

# Look for your processor:
# image_preprocessing:
#   enabled: true
#   processors:
#     - type: "your_effect"
#       order: 1
#       enabled: true
#       params:
#         intensity: 0.8
```

---

## Common Patterns

### Pattern 1: Caching Expensive Resources

```python
class MyProcessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None

    def _get_model(self):
        """Lazy-load expensive model"""
        if self._model is None:
            self._model = load_expensive_model()
            self._model.to(self.device)
        return self._model

    def _process_tensor_core(self, tensor):
        model = self._get_model()
        return model(tensor)
```

### Pattern 2: Smooth Parameter Transitions

```python
class MyProcessor(BasePreprocessor):
    def __init__(self, target_value: float = 1.0, **kwargs):
        super().__init__(target_value=target_value, **kwargs)
        self._current_value = target_value
        self._target_value = target_value
        self.smoothing = 0.1  # Lerp factor

    def _update_smooth_value(self):
        """Smooth parameter changes"""
        self._current_value += (self._target_value - self._current_value) * self.smoothing

    @property
    def target_value(self):
        return self._target_value

    @target_value.setter
    def target_value(self, value):
        self._target_value = value

    def _process_core(self, image):
        self._update_smooth_value()
        # Use self._current_value instead of self.target_value
        # ...
```

### Pattern 3: Multi-Scale Processing

```python
def _process_tensor_core(self, tensor):
    """Process at multiple scales for better quality"""
    original_size = tensor.shape[-2:]
    scales = [1.0, 0.5, 0.25]
    results = []

    for scale in scales:
        if scale != 1.0:
            h, w = int(original_size[0] * scale), int(original_size[1] * scale)
            scaled = F.interpolate(tensor, size=(h, w), mode='bilinear')
        else:
            scaled = tensor

        # Process at this scale
        processed = self._process_single_scale(scaled)

        # Resize back
        if scale != 1.0:
            processed = F.interpolate(processed, size=original_size, mode='bilinear')

        results.append(processed)

    # Blend multi-scale results
    return sum(results) / len(results)
```

### Pattern 4: Conditional Processing

```python
@classmethod
def get_preprocessor_metadata(cls):
    return {
        "parameters": {
            "mode": {
                "type": "string",
                "default": "auto",
                "options": ["auto", "edge", "smooth"],
                "description": "Processing mode"
            },
            "edge_threshold": {
                "type": "float",
                "default": 0.5,
                "range": [0.0, 1.0],
                "description": "Threshold for edge mode (ignored in smooth mode)"
            }
        }
    }

def _process_core(self, image):
    if self.mode == "edge":
        return self._process_edges(image, self.edge_threshold)
    elif self.mode == "smooth":
        return self._process_smooth(image)
    else:  # auto
        # Detect best mode based on content
        if self._is_high_frequency(image):
            return self._process_edges(image, self.edge_threshold)
        else:
            return self._process_smooth(image)
```

---

## Known Issues & Solutions

### Issue 1: Frame Oscillation (Ping-Pong Effect)

**Symptom:** Feedback loops oscillate between two frames instead of smooth accumulation

**Cause:** Frame delay in async pipeline - processor gets frame N-2 instead of N-1

**Solution:**
```python
class YourFeedbackProcessor(PipelineAwareProcessor):
    # CRITICAL: This prevents frame delay
    requires_sync_processing = True
```

And in YAML generation:
```python
yaml_content += f'        requires_sync_processing: true\n'
```

### Issue 2: Multiple Processors Conflict

**Symptom:** Processor works alone but fails when combined with others

**Possible Causes:**

1. **Both accessing same previous data:**
```python
# Problem: Both processors modify prev_image_result
# Solution: Processors should only READ prev data, not modify pipeline state
```

2. **Order dependency:**
```python
# Solution: Check processor_order in YAML
if use_processor_a:
    # ... order: 1
if use_processor_b:
    # ... order: 2  # Runs after A
```

3. **Domain mismatch:**
```python
# Image domain (pre-VAE) can't access latent domain data
# Latent domain (post-VAE) can't access original image
```

**Fix:** Ensure processors are in correct domain and order:
```python
# Image preprocessing: color, geometric, feedback
if use_image_feedback or use_color_correction:
    # image_preprocessing section

# Latent preprocessing: zoom, pan, latent effects
if use_latent_transform or use_latent_feedback:
    # latent_preprocessing section
```

### Issue 3: Parameters Not Updating

**Debug Checklist:**

```python
# 1. Check OSC is being sent (TD textport)
"Fx OSC: /fx/your_effect/intensity = 0.8"

# 2. Check Python receives OSC (Python logs)
"FX OSC received: /fx/your_effect/intensity = (0.8,)"

# 3. Check processor found (Python logs)
"Found matching processor: YourEffectPreprocessor in _image_preprocessing_module"

# 4. Check attribute exists (Python logs)
"✅ FX UPDATE SUCCESS: YourEffectPreprocessor.intensity = 0.5 -> 0.8"

# 5. If attribute not found:
"❌ YourEffectPreprocessor has no attribute 'intensity'"
# Fix: Make sure __init__ sets self.intensity

# 6. If processor not found:
"❌ Processor 'your_effect' not found in any module"
# Fix: Check YAML generation includes your processor
```

### Issue 4: Device/Dtype Errors

```python
# Problem
RuntimeError: Expected all tensors to be on the same device

# Solution: Always ensure correct device/dtype
def _process_tensor_core(self, tensor):
    # Force to correct device/dtype
    tensor = tensor.to(device=self.device, dtype=self.dtype)

    # Any tensors you create must also be on correct device
    weights = torch.tensor([0.2, 0.7, 0.1], device=self.device, dtype=self.dtype)

    return result.to(device=self.device, dtype=self.dtype)
```

### Issue 5: First Frame Handling

```python
class YourFeedbackProcessor(PipelineAwareProcessor):
    def __init__(self, pipeline_ref, **kwargs):
        super().__init__(pipeline_ref=pipeline_ref, **kwargs)
        self._first_frame = True  # Track first frame

    def _get_previous_data(self):
        if self.pipeline_ref is not None:
            if hasattr(self.pipeline_ref, 'prev_image_result'):
                # Check both exists AND not first frame
                if self.pipeline_ref.prev_image_result is not None and not self._first_frame:
                    return self.pipeline_ref.prev_image_result
        return None

    def _process_core(self, image):
        prev = self._get_previous_data()

        if prev is None:
            # First frame or no feedback - pass through
            self._first_frame = False
            return image

        # Process with feedback
        # ...
        self._first_frame = False  # Set at end
        return result
```

---

## Quick Reference

### Processor Checklist

- [ ] Create processor file in `processors/`
- [ ] Inherit from `BasePreprocessor` or `PipelineAwareProcessor`
- [ ] Implement `get_preprocessor_metadata()` with all parameters
- [ ] Implement `__init__()` accepting all parameters + **kwargs
- [ ] Implement `_process_core()` (PIL path) - REQUIRED
- [ ] Implement `_process_tensor_core()` (GPU path) - OPTIONAL
- [ ] Set `requires_sync_processing = True` if pipeline-aware
- [ ] Register in `__init__.py` (import, registry, __all__)
- [ ] Add `Use{processor}()` callback in StreamDiffusionExt.py
- [ ] Add to `active_fx` list in `update_fx_dynamic_parameters()`
- [ ] Add to YAML generation (image or latent preprocessing section)
- [ ] Test import in Python
- [ ] Run `Getpreprocessors()` in TouchDesigner
- [ ] Enable toggle and verify dynamic parameters appear
- [ ] Test OSC updates in real-time
- [ ] Check logs for errors

### File Locations Quick Map

```
Python Files:
├── src/streamdiffusion/preprocessing/processors/
│   ├── __init__.py                    ← Register here
│   ├── your_processor.py              ← Create here
│   ├── base.py                        ← Base classes (reference)
│   └── color_correction_feedback.py   ← Example (reference)
└── streamdiffusionTD/
    └── td_osc_handler.py              ← OSC receiver (auto-works)

TouchDesigner Files:
└── StreamDiffusionTD/
    └── StreamDiffusionExt.py          ← 3 places to update:
        ├── Line ~5930: Add Use{processor}() callback
        ├── Line ~6480: Add to active_fx list
        └── Line ~4683: Add to YAML generation
```

### Common Terminal Commands

```bash
# Test Python import
python -c "from src.streamdiffusion.preprocessing.processors import YourEffectPreprocessor; print('Success!')"

# Run StreamDiffusion server
Start_StreamDiffusion.bat

# Watch logs for OSC messages
python streamdiffusionTD/td_main.py 2>&1 | grep "FX OSC"

# Check for errors
python streamdiffusionTD/td_main.py 2>&1 | grep "ERROR\|❌"
```

### TouchDesigner Commands

```python
# Refresh processor table
op('your_component').Getpreprocessors()

# Generate and view YAML
yaml = op('your_component').Generateyaml()
print(yaml)

# List dynamic parameters
for p in parent().customPars:
    if p[0].name.startswith('Fx'):
        print(f"{p[0].name} = {p[0].eval()}")

# Enable processor
parent().par.Useyourprocessor = 1

# Update dynamic parameters (manual refresh)
parent().update_fx_dynamic_parameters()
```

---

## Advanced Topics

### GPU Optimization

```python
def _process_tensor_core(self, tensor):
    # 1. Minimize CPU-GPU transfers
    # BAD: Converting to numpy mid-processing
    # GOOD: Keep on GPU entire time

    # 2. Use torch operations, not loops
    # BAD:
    for i in range(tensor.shape[0]):
        tensor[i] = process_single(tensor[i])

    # GOOD:
    tensor = process_batch(tensor)  # Vectorized

    # 3. Cache kernels
    if self._cached_kernel is None:
        self._cached_kernel = self._create_kernel()
        self._cached_kernel = self._cached_kernel.to(self.device)

    # 4. Use in-place operations when safe
    tensor.mul_(self.intensity)  # In-place

    # 5. Prefer convolution for spatial operations
    result = F.conv2d(tensor, kernel, padding='same')
```

### Memory Management

```python
class MyProcessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = None

    def _ensure_buffer(self, shape):
        """Reuse buffers to avoid allocation overhead"""
        if self._buffer is None or self._buffer.shape != shape:
            self._buffer = torch.empty(shape, device=self.device, dtype=self.dtype)
        return self._buffer

    def _process_tensor_core(self, tensor):
        buffer = self._ensure_buffer(tensor.shape)
        # Use buffer for intermediate results
        torch.mul(tensor, self.intensity, out=buffer)
        return buffer
```

### Custom OSC Handlers

If you need custom OSC beyond standard parameters:

```python
# In td_osc_handler.py
def _setup_osc_handlers(self):
    # ... existing handlers ...

    # Custom handler for your processor
    self.dispatcher.map("/your_effect/reset", self._handle_your_effect_reset)

def _handle_your_effect_reset(self, address, *args):
    """Custom OSC command to reset processor state"""
    if hasattr(self.manager, 'wrapper') and self.manager.wrapper:
        stream = self.manager.wrapper.stream
        # Find your processor
        for processor in stream._image_preprocessing_module.processors:
            if isinstance(processor, YourEffectPreprocessor):
                processor.reset()
                logger.info("✅ YourEffect reset")
```

---

## Example: Complete Processor Creation

Let's create a "Vignette" effect from scratch:

### 1. Create Processor

**File:** `src/streamdiffusion/preprocessing/processors/vignette.py`

```python
import torch
import torch.nn.functional as F
from PIL import Image
from .base import BasePreprocessor


class VignettePreprocessor(BasePreprocessor):
    """
    Vignette effect - darkens corners of image

    Creates a radial gradient mask that darkens edges while keeping center bright.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Vignette",
            "description": "Darken image corners with radial gradient",
            "parameters": {
                "intensity": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Vignette strength (0=none, 1=maximum)"
                },
                "radius": {
                    "type": "float",
                    "default": 0.8,
                    "range": [0.1, 2.0],
                    "step": 0.01,
                    "description": "Vignette inner radius (smaller=tighter)"
                }
            },
            "use_cases": [
                "Cinematic look",
                "Focus attention on center",
                "Vintage photo effect"
            ]
        }

    def __init__(self, intensity: float = 0.5, radius: float = 0.8, **kwargs):
        super().__init__(intensity=intensity, radius=radius, **kwargs)
        self.intensity = intensity
        self.radius = radius
        self._mask_cache = None
        self._last_shape = None

    def _create_vignette_mask(self, h: int, w: int) -> torch.Tensor:
        """Create radial gradient mask"""
        # Create coordinate grid
        y = torch.linspace(-1, 1, h, device=self.device, dtype=self.dtype)
        x = torch.linspace(-1, 1, w, device=self.device, dtype=self.dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Distance from center
        distance = torch.sqrt(xx**2 + yy**2)

        # Apply radius and create smooth falloff
        mask = 1.0 - torch.clamp((distance - self.radius) / (1.0 - self.radius), 0, 1)

        # Apply intensity
        mask = 1.0 - (1.0 - mask) * self.intensity

        return mask

    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated processing"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device=self.device, dtype=self.dtype)

        _, _, h, w = tensor.shape

        # Cache mask if shape unchanged
        if self._mask_cache is None or self._last_shape != (h, w):
            self._mask_cache = self._create_vignette_mask(h, w)
            self._last_shape = (h, w)

        # Apply vignette: multiply by mask [H, W] → [1, 1, H, W]
        mask = self._mask_cache.view(1, 1, h, w)
        result = tensor * mask

        return result.clamp(0, 1)
```

### 2. Register

**File:** `src/streamdiffusion/preprocessing/processors/__init__.py`

```python
from .vignette import VignettePreprocessor

_preprocessor_registry = {
    # ...
    "vignette": VignettePreprocessor,
}

__all__ = [
    # ...
    "VignettePreprocessor",
]
```

### 3. Add TouchDesigner Integration

**File:** `StreamDiffusionTD/StreamDiffusionExt.py`

```python
# Add callback (~line 5930)
def Usevignette(self):
    self.logger.log('Usevignette changed', level='INFO')
    self.update_fx_dynamic_parameters()

# Add to active_fx (~line 6480)
if hasattr(self.ownerComp.par, 'Usevignette') and self.ownerComp.par.Usevignette.eval():
    active_fx.append('vignette')

# Add to YAML generation (~line 4687)
use_vignette = (hasattr(self.ownerComp.par, 'Usevignette') and
                self.ownerComp.par.Usevignette.eval())

if use_image_feedback or use_color_correction_feedback or use_vignette:
    # ...
    if use_vignette:
        params = self.gather_fx_parameters_for_processor('vignette')
        yaml_content += f'    - type: "vignette"\n'
        yaml_content += f'      order: {processor_order}\n'
        yaml_content += f'      enabled: true\n'
        yaml_content += f'      params:\n'
        for param_name, param_value in params.items():
            yaml_content += f'        {param_name}: {param_value}\n'
```

### 4. Use in TouchDesigner

```python
# Refresh
op('streamdiffusion').Getpreprocessors()

# Enable
parent().par.Usevignette = 1

# Parameters appear:
# - Fxvignetteintensity
# - Fxvignetteradius

# Adjust live
parent().par.Fxvignetteintensity = 0.7
parent().par.Fxvignetteradius = 0.6
```

**Done!** Real-time vignette effect with live parameter control.

---

## Resources

- **StreamDiffusion Docs:** https://github.com/cumulo-autumn/StreamDiffusion
- **Base Classes:** `src/streamdiffusion/preprocessing/processors/base.py`
- **Example Processors:** `src/streamdiffusion/preprocessing/processors/`
  - `blur.py` - Simple processor
  - `feedback_transform.py` - Pipeline-aware processor
  - `color_correction_feedback.py` - Complex feedback processor
- **OSC Handler:** `streamdiffusionTD/td_osc_handler.py`

---

**Last Updated:** 2025-10-24
**Version:** 1.0
**Author:** StreamDiffusion Development Team
