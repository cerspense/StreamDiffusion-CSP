# Phase 1: Blackwell GPU Compatibility

**Status**: 🚧 In Progress

**Time Estimate**: N/A (troubleshooting)
**Actual Time**: ~4 hours

**Devlogs**:
- [Phase 1.1: Blackwell Setup](../devlogs/phase-1.1-blackwell-setup.md)

## Overview

Getting StreamDiffusion running on RTX 5090 (Blackwell) with CUDA 12.8 and PyTorch 2.9.

## Sub-Phases

### 1.1 Base Compatibility ✅

- [x] Fix nvidia-pyindex installation failure
- [x] Fix wrong package installation directory
- [x] Fix PyTorch 2.9 ONNX export incompatibility (`dynamo=False`)
- [x] Fix Windows console Unicode encoding error
- [x] Build TensorRT engines for Blackwell GPU

### 1.2 ControlNet Integration 🚧

- [x] Configure ControlNet in td_config.yaml
- [x] Fix ControlNet model_id path (full HuggingFace path required)
- [x] Rebuild UNet engine with ControlNet inputs
- [x] Set preprocessor to passthrough for external edge detection
- [x] Add ControlNet model ID alias system for OSC
- [x] Fix ControlNet TensorRT sample input generation (NoneType fix)
- [ ] Fix ControlNet disable/enable toggle error
- [ ] Build ControlNet TensorRT engine (currently PyTorch fallback)

### 1.3 TouchDesigner OSC Integration 🚧

- [x] Basic OSC communication working
- [x] ControlNet conditioning_scale updates via OSC
- [x] Model ID alias translation for short names
- [ ] Fix ControlNet active toggle via OSC
- [ ] Verify all parameter updates work without errors

## Outstanding Items

- ControlNet TensorRT engine compilation fails (PyTorch 2.9 ONNX export issue in controlnet_models.py)
- ControlNet disable toggle causes error (needs investigation)

## Key Deliverables

1. StreamDiffusion running on RTX 5090 with TensorRT acceleration
2. ControlNet integration with external (TouchDesigner) edge detection
3. Live parameter updates via OSC without pipeline rebuild

## Lessons Learned

1. PyTorch 2.9+ requires `dynamo=False` for legacy ONNX export
2. Windows console encoding issues require using logging instead of print for non-ASCII
3. TensorRT 10.8+ works directly with PyPI without nvidia-pyindex
4. ControlNet model IDs must be full HuggingFace paths (or use alias system)
5. Always use the venv, never system Python
