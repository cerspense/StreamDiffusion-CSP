# StreamDiffusion Blackwell GPU Setup Report

## Overview
This report documents the issues encountered and solutions implemented to get StreamDiffusion running with TensorRT on an RTX 5090 (Blackwell) GPU with CUDA 12.8 and PyTorch 2.9.0.

## System Configuration
- **GPU**: NVIDIA RTX 5090 (Blackwell architecture)
- **CUDA Version**: 12.8
- **PyTorch Version**: 2.9.0+cu128
- **TensorRT Version**: 10.13.3.9
- **Python Version**: 3.11
- **OS**: Windows (cp1252 encoding)

## Issues Encountered and Solutions

### 1. Wrong Repository Branch
**Issue**: The system was initially on the wrong branch.

**Solution**: Switched to the `processors-dot-experimental` branch from the cerspense/StreamDiffusion-CSP repository.

```bash
git remote add cerspense https://github.com/cerspense/StreamDiffusion-CSP.git
git fetch cerspense
git checkout -b processors-dot-experimental cerspense/processors-dot-experimental
```

### 2. nvidia-pyindex Installation Failure
**Issue**: The `nvidia-pyindex` package failed to build during installation with the error:
```
ModuleNotFoundError: No module named 'pip'
```

**Root Cause**: `nvidia-pyindex` has a bug where it tries to import pip's internal modules during wheel building, which fails in certain environments.

**Solution**: Removed the `nvidia-pyindex` installation requirement as it's no longer needed for modern TensorRT installations (10.8+). Modified `streamdiffusionTD/install_tensorrt.py`:

```python
# Skip nvidia-pyindex - not needed for modern TensorRT installations
# The nvidia-pyindex package has known issues and is deprecated
# We can install TensorRT directly from PyPI without it
print("Skipping nvidia-pyindex (not required for TensorRT 10.8+)...")
```

**File**: `streamdiffusionTD/install_tensorrt.py:59-62`

### 3. Wrong Package Installation Directory
**Issue**: The Python environment was importing the `streamdiffusion` package from the old `StreamDiffusionBlackwell_4` directory instead of the current `StreamDiffusionBlackwell_5` directory.

**Solution**: Uninstalled and reinstalled the streamdiffusion package from the correct directory:

```bash
pip uninstall -y streamdiffusion
pip install -e .
```

### 4. PyTorch 2.9 ONNX Export Incompatibility
**Issue**: PyTorch 2.9's new ONNX exporter (using `torch.export.export` by default) was incompatible with the existing dynamic_axes format, causing the error:
```
torch._dynamo.exc.UserError: Detected mismatch between the structure of `inputs` and `dynamic_shapes`:
`inputs[0]` is a <class 'tuple'>, but `dynamic_shapes[0]` is a <class 'dict'>
```

**Root Cause**: PyTorch 2.9+ uses a new `torch.export` based ONNX exporter by default, which has different requirements for the `dynamic_shapes` parameter format.

**Solution**: Forced the use of the legacy TorchScript-based ONNX exporter by adding the `dynamo=False` parameter to `torch.onnx.export()`. Modified `src/streamdiffusion/acceleration/tensorrt/utilities.py`:

```python
# Export ONNX using legacy TorchScript-based exporter
# PyTorch 2.9+ uses torch.export by default which has compatibility issues
torch.onnx.export(
    wrapped_model,
    inputs,
    onnx_path,
    export_params=True,
    opset_version=onnx_opset,
    do_constant_folding=True,
    input_names=model_data.get_input_names(),
    output_names=model_data.get_output_names(),
    dynamic_axes=model_data.get_dynamic_axes(),
    dynamo=False,  # Force legacy TorchScript-based exporter
)
```

**File**: `src/streamdiffusion/acceleration/tensorrt/utilities.py:600-613`

### 5. Windows Console Unicode Encoding Error
**Issue**: When printing warning messages with emoji characters (⚠️, 🔧), the Windows console (cp1252 encoding) couldn't encode them, causing:
```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-1: character maps to <undefined>
```

**Root Cause**: Windows console uses cp1252 encoding by default, which doesn't support emoji characters. The code was using `print()` statements with emoji warnings.

**Solution**: Replaced `print()` statements with proper logging to avoid console encoding issues. Modified `src/streamdiffusion/acceleration/tensorrt/models/models.py`:

```python
def infer_shapes(self, return_onnx=False):
    onnx_graph = gs.export_onnx(self.graph)
    if onnx_graph.ByteSize() > 2147483648:
        # Use logger instead of print to avoid Windows console encoding issues
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Model size ({onnx_graph.ByteSize() / (1024**3):.2f} GB) exceeds 2GB - this is normal for SDXL models")
        logger.info("ONNX shape inference will be skipped for large models to avoid memory issues")
        # For large models like SDXL, skip shape inference to avoid memory/size issues
        # The model will still work with TensorRT's own shape inference during engine building
    else:
        onnx_graph = shape_inference.infer_shapes(onnx_graph)
```

**File**: `src/streamdiffusion/acceleration/tensorrt/models/models.py:54-65`

## Summary of Files Modified

1. **streamdiffusionTD/install_tensorrt.py**
   - Removed nvidia-pyindex installation (lines 59-62)

2. **src/streamdiffusion/acceleration/tensorrt/utilities.py**
   - Added `dynamo=False` parameter to torch.onnx.export() (line 612)

3. **src/streamdiffusion/acceleration/tensorrt/models/models.py**
   - Replaced print() with logger for Unicode safety (lines 57-61)

## Build Process
After implementing all fixes, the TensorRT engine build process completed successfully:

1. **VAE Decoder Engine**: Built in ~49 seconds
2. **VAE Encoder Engine**: Built in ~31 seconds
3. **UNet Engine**: Built in ~10-15 minutes (SDXL with IP-Adapter is large)

## Verification
The system successfully:
- Detected CUDA 12.8
- Installed TensorRT 10.13.3.9 for CUDA 12
- Loaded the SDXL-Turbo model
- Built all TensorRT engines for the Blackwell GPU
- Started StreamDiffusion with full TensorRT acceleration

## Notes for Future Reference

### CRITICAL: Virtual Environment and Engine Files

**⚠️ IMPORTANT RULES - READ CAREFULLY:**

1. **ALWAYS USE THE VENV (Virtual Environment)**
   - **NEVER** use the system Python installation for troubleshooting or running StreamDiffusion
   - **ALWAYS** use `venv/Scripts/python.exe` or activate the venv with `venv\Scripts\activate.bat`
   - The venv is located at: `C:\Users\cerspense\Documents\AI\StreamDiffusionBlackwell_5\StreamDiffusion\venv\`
   - All packages must be installed in the venv, not the system Python

2. **NEVER DELETE ENGINE FILES**
   - Engine files are located in `./engines/td/` directory
   - These files take 10-15 minutes to build and represent significant compilation work
   - **DO NOT** delete or modify engine files unless absolutely necessary
   - Engine files are automatically cached and reused
   - If engines need to be rebuilt, the system will do so automatically when needed

3. **Virtual Environment Setup**
   - To reinstall packages in venv: `venv/Scripts/pip.exe install <package>`
   - To run TensorRT install script: `venv/Scripts/python.exe streamdiffusionTD/install_tensorrt.py`
   - To run StreamDiffusion: `venv/Scripts/python.exe streamdiffusionTD/td_main.py`
   - The batch file `Start_StreamDiffusion.bat` automatically uses the venv

4. **Required venv Packages**
   - streamdiffusion (installed with `pip install -e .`)
   - tensorrt-cu12
   - nvidia-cudnn-cu12
   - polygraphy
   - onnx-graphsurgeon
   - All other dependencies from requirements

### onnxruntime Warning (Non-Critical)
Throughout the build process, warnings about missing `onnxruntime` appear:
```
[!] Module: 'onnxruntime.tools.symbolic_shape_infer' is required but could not be imported.
```
This is **non-critical** - it's only used for ONNX optimization and shape inference. TensorRT performs its own shape inference during engine building. The warning can be safely ignored or you can optionally install onnxruntime for potentially better optimization:
```bash
venv/Scripts/pip.exe install onnxruntime
```

### TensorRT Engine Build Times
- Small models (VAE): < 1 minute
- Large models (SDXL UNet with IP-Adapter): 10-15 minutes
- **Engines are cached in `./engines/td/` and only rebuilt when model or configuration changes**
- **NEVER delete engine files unless you want to wait 10-15 minutes for rebuild**

### Key Compatibility Notes
- PyTorch 2.9+ requires `dynamo=False` for legacy ONNX export
- Windows console encoding issues require using logging instead of print for non-ASCII characters
- TensorRT 10.8+ works directly with PyPI without nvidia-pyindex
- **Always work within the venv, never use system Python**

## Success Indicators
When everything is working correctly, you should see:
```
13:31:04 - streamdiffusion.acceleration.tensorrt.utilities - INFO - ONNX optimization complete with external data
13:31:05 - streamdiffusion.acceleration.tensorrt.utilities - INFO - Building TensorRT engine for [...]/unet.engine
```

The system then compiles the engine and eventually loads successfully with all TensorRT acceleration enabled.
