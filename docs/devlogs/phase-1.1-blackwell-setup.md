# Phase 1.1: Blackwell GPU Setup & ControlNet Integration

**Status**: 🚧 In Progress

**Date**: 2025-12-12
**Actual Time**: ~4 hours

**Commits**:
- `pending` - fix: add Blackwell GPU (RTX 5090) compatibility for CUDA 12.8 and PyTorch 2.9

## Overview

Setting up StreamDiffusion on a new RTX 5090 (Blackwell architecture) with CUDA 12.8 and PyTorch 2.9, then integrating ControlNet for use with TouchDesigner.

## What Was Built

### Blackwell Compatibility Fixes

- [x] Removed nvidia-pyindex installation requirement (deprecated, breaks on pip import)
- [x] Fixed PyTorch 2.9 ONNX export by adding `dynamo=False` parameter
- [x] Fixed Windows console Unicode error by using logging instead of print for emojis
- [x] Successfully built all TensorRT engines (VAE encoder/decoder, UNet)

### ControlNet Integration

- [x] Configured ControlNet in td_config.yaml with full HuggingFace path
- [x] Set preprocessor to "passthrough" for external edge detection from TouchDesigner
- [x] Rebuilt UNet engine with ControlNet inputs (`input_control_00` etc.)
- [x] Added ControlNet model ID alias system to translate short names like "canny" to full paths
- [x] Fixed ControlNet TensorRT model `get_sample_input` NoneType issue with defensive getattr

## Technical Decisions

### 1. Legacy ONNX Exporter (`dynamo=False`)

PyTorch 2.9+ uses a new `torch.export` based ONNX exporter by default which has incompatible requirements for `dynamic_shapes`. Solution was to force the legacy TorchScript-based exporter.

**File**: `src/streamdiffusion/acceleration/tensorrt/utilities.py:612`

```python
torch.onnx.export(
    wrapped_model,
    inputs,
    onnx_path,
    ...
    dynamo=False,  # Force legacy TorchScript-based exporter
)
```

### 2. ControlNet Model ID Aliases

TouchDesigner was sending short names like "canny" via OSC instead of full HuggingFace paths. Rather than requiring TD changes, added a Python-side alias system.

**File**: `src/streamdiffusion/stream_parameter_updater.py:1048-1076`

```python
CONTROLNET_MODEL_ALIASES = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "canny-sdxl": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    ...
}
```

### 3. Defensive Attribute Access in ControlNet Models

ControlNet TensorRT model `get_sample_input` was failing with NoneType error. Added defensive `getattr` with defaults.

**File**: `src/streamdiffusion/acceleration/tensorrt/models/controlnet_models.py:217-221`

```python
text_maxlen = getattr(self, 'text_maxlen', 77)
embedding_dim = getattr(self, 'embedding_dim', 2048)  # SDXL default
unet_dim = getattr(self, 'unet_dim', 4)
conditioning_channels = getattr(self, 'conditioning_channels', 3)
```

## Bugs Fixed

### Bug 1: nvidia-pyindex Installation Failure

- **Problem**: `ModuleNotFoundError: No module named 'pip'` during wheel building
- **Root Cause**: nvidia-pyindex tries to import pip internals during build
- **Fix**: Skip nvidia-pyindex installation (not needed for TensorRT 10.8+)
- **File**: `streamdiffusionTD/install_tensorrt.py:59-62`

### Bug 2: PyTorch 2.9 ONNX Export Incompatibility

- **Problem**: `torch._dynamo.exc.UserError: Detected mismatch between structure of inputs and dynamic_shapes`
- **Root Cause**: PyTorch 2.9+ torch.export-based ONNX exporter has different dynamic_shapes format
- **Fix**: Added `dynamo=False` to force legacy exporter
- **File**: `src/streamdiffusion/acceleration/tensorrt/utilities.py:612`

### Bug 3: Windows Console Unicode Error

- **Problem**: `UnicodeEncodeError: 'charmap' codec can't encode characters` for emojis
- **Root Cause**: Windows console uses cp1252 encoding
- **Fix**: Replaced print() with logging for non-ASCII characters
- **File**: `src/streamdiffusion/acceleration/tensorrt/models/models.py:57-61`

### Bug 4: UNet Engine Missing ControlNet Inputs

- **Problem**: `KeyError: 'input_control_00'` when running with ControlNet
- **Root Cause**: UNet engine was built without ControlNet support
- **Fix**: Deleted old engine, rebuilt with ControlNet enabled in config
- **File**: `engines/td/.../unet.engine` (deleted and rebuilt)

### Bug 5: TouchDesigner Sending Wrong ControlNet Model IDs

- **Problem**: OSC messages sending "canny" instead of "diffusers/controlnet-canny-sdxl-1.0"
- **Root Cause**: TouchDesigner configuration had short names
- **Fix**: Added alias system in Python + fixed TD config (user side)
- **File**: `src/streamdiffusion/stream_parameter_updater.py:1048-1089`

## Files Modified

1. `streamdiffusionTD/install_tensorrt.py` - Removed nvidia-pyindex installation
2. `src/streamdiffusion/acceleration/tensorrt/utilities.py` - Added `dynamo=False` to ONNX export
3. `src/streamdiffusion/acceleration/tensorrt/models/models.py` - Unicode-safe logging
4. `src/streamdiffusion/acceleration/tensorrt/models/controlnet_models.py` - Defensive getattr in get_sample_input
5. `src/streamdiffusion/stream_parameter_updater.py` - ControlNet model ID alias system
6. `StreamDiffusionTD/td_config.yaml` - ControlNet configuration with full paths and passthrough preprocessor

## Testing / Verification

- [x] TensorRT engines build successfully on RTX 5090
- [x] Stream runs at ~25 FPS with TensorRT acceleration
- [x] ControlNet shared memory connected (`StreamDiffusionTD12sdffdzsdfsdf-cn`)
- [x] ControlNet conditioning_scale updates via OSC work
- [ ] ControlNet active toggle needs testing
- [ ] ControlNet visual effect verification needed

## Known Issues

1. **ControlNet TensorRT Engine Not Building**: Falls back to PyTorch. Error: `randn(): argument 'size' failed to unpack`. Partial fix applied, needs verification.

2. **ControlNet Disable Toggle Error**: Reported by user when toggling ControlNet active off. Needs investigation.

## Next Steps

1. Test ControlNet disable/enable toggle and fix any errors
2. Verify ControlNet TensorRT engine compilation with fixes applied
3. Test full ControlNet visual effect (does edge detection actually guide output?)
4. Document final working configuration
