#!/usr/bin/env python3
"""Test post_process_color processor initialization"""

import sys
import torch

# Test import
try:
    from src.streamdiffusion.preprocessing.processors import PostProcessColorPreprocessor
    print("[OK] Successfully imported PostProcessColorPreprocessor")
except Exception as e:
    print(f"[FAIL] Failed to import: {e}")
    sys.exit(1)

# Test metadata
try:
    metadata = PostProcessColorPreprocessor.get_preprocessor_metadata()
    print("\n[OK] Metadata retrieved successfully:")
    print(f"  Display name: {metadata['display_name']}")
    print(f"  Parameters: {', '.join(metadata['parameters'].keys())}")
except Exception as e:
    print(f"[FAIL] Failed to get metadata: {e}")
    sys.exit(1)

# Test initialization
try:
    processor = PostProcessColorPreprocessor(
        image_resolution=512,
        brightness=0.1,
        saturation=1.2,
        contrast=1.1,
        black_level=0.05,
        gamma=1.0,
        temperature=0.1
    )
    print("\n[OK] Processor initialized successfully")
    print(f"  Device: {processor.device}")
    print(f"  Brightness: {processor.brightness}")
    print(f"  Saturation: {processor.saturation}")
    print(f"  Contrast: {processor.contrast}")
except Exception as e:
    print(f"[FAIL] Failed to initialize: {e}")
    sys.exit(1)

# Test processing
try:
    # Create a simple test tensor [B, C, H, W]
    test_tensor = torch.rand(1, 3, 512, 512, device=processor.device, dtype=processor.dtype)
    print(f"\n[OK] Created test tensor: {test_tensor.shape}")

    # Process
    output = processor._process_tensor_core(test_tensor)
    print(f"[OK] Processing successful: {output.shape}")
    print(f"  Input range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

except Exception as e:
    print(f"[FAIL] Failed to process: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")
