# Phase 1.1: ControlNet Integration

**Status**: 🚧 In Progress
**Date**: 2024-12-12 (Updated: 2024-12-16)

## Overview

Getting ControlNet working with StreamDiffusion on RTX 5090 (Blackwell) with CUDA 12.8 and PyTorch 2.9.

## What Was Done

### Blackwell GPU Compatibility (Previously Completed)
- Added `dynamo=False` to `torch.onnx.export()` for PyTorch 2.9 compatibility
- Fixed Windows console Unicode encoding errors in TensorRT model building
- Removed deprecated `nvidia-pyindex` dependency

### ControlNet Configuration Fixes
- Changed `model_id: "canny"` to full path `model_id: "diffusers/controlnet-canny-sdxl-1.0"`
- Set `preprocessor: "passthrough"` for external edge detection from TouchDesigner
- Commented out OpenPose ControlNet (no standard SDXL version available)

### UNet Engine Rebuild
- Deleted cached UNet engine to force rebuild with ControlNet inputs
- Successfully rebuilt 5.8GB UNet engine with ControlNet support

### ControlNet TensorRT Model Fixes
- Added defensive `getattr()` calls in `get_sample_input()` to prevent `NoneType` errors
- Fixed both `ControlNetTRT` and `ControlNetSDXLTRT` classes

### OSC Alias System
- Added `CONTROLNET_MODEL_ALIASES` dictionary for short name to full path translation
- Allows TouchDesigner to send "canny" instead of full HuggingFace path

### SDXL Lineart ControlNet Support (2024-12-16)
- Added SDXL lineart model aliases to `CONTROLNET_MODEL_ALIASES`:
  - `lineart` / `lineart-sdxl` → `TheMistoAI/MistoLine` (most versatile)
  - `lineart-anime-sdxl` → `kataragi/ControlNet-LineartXL`
  - `lineart-promeai` → `promeai/sdxl-controlnet-lineart-promeai`
- Fixed `_load_pytorch_controlnet_model()` to try fp16 variant first
  - Many community ControlNet models use `diffusion_pytorch_model.fp16.safetensors`
  - Falls back to standard loading if fp16 variant not found
- Successfully tested MistoLine loading with passthrough preprocessor

## Files Modified

1. `StreamDiffusionTD/td_config.yaml` - ControlNet config with full model paths
2. `src/streamdiffusion/acceleration/tensorrt/models/controlnet_models.py` - Defensive getattr fixes
3. `src/streamdiffusion/stream_parameter_updater.py` - ControlNet model ID alias system + lineart aliases
4. `src/streamdiffusion/modules/controlnet_module.py` - fp16 variant loading support

## Current State

- ControlNet loads and works with Canny and Lineart models
- MistoLine (lineart) working with passthrough preprocessor
- Using PyTorch fallback (TensorRT engine compilation still has issues)
- Disable/enable toggle causes errors (to be investigated)

## Next Steps

- Investigate ControlNet disable toggle error
- Fix ControlNet TensorRT engine compilation for full acceleration
- Test performance comparison: ControlNet-enabled vs non-ControlNet engines
