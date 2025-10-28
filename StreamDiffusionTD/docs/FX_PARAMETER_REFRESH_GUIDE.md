# Fx Parameter Refresh Guide

**Date:** 2025-10-27
**Purpose:** How to refresh Fx metadata and parameters when adding new processors

---

## Problem

When you add a new processor to the codebase, TouchDesigner doesn't automatically know about it. The `table_preprocessors` DAT table caches processor metadata, and needs to be refreshed for new processors to appear.

**Symptoms:**
- Enable new processor toggle (e.g., `Usepostprocessxformcc`)
- No Fx* parameters appear on the Fx page
- Processor is registered in `__init__.py` but TouchDesigner doesn't see it

---

## Solution

### Step 1: Add Pulse Parameter in TouchDesigner

You need to add a pulse parameter called `Refreshfxmetadata` on the **Fx page**:

1. Open TouchDesigner
2. Select the StreamDiffusion component
3. Navigate to the **Fx** page in the parameter editor
4. Add a new **Pulse** parameter:
   - **Name:** `Refreshfxmetadata`
   - **Label:** "Refresh Fx Metadata"
   - **Type:** Pulse
   - **Page:** Fx

### Step 2: Pulse to Refresh

After adding a new processor to the codebase:

1. **Enable the processor toggle** (e.g., `Usepostprocessxformcc`)
2. **Pulse `Refreshfxmetadata`** button on the Fx page
3. **Fx* parameters should appear** automatically

---

## What the Refresh Does

When you pulse `Refreshfxmetadata`, it triggers the `Refreshfxmetadata()` method in `StreamDiffusionExt.py` (line 6156), which:

1. **Refreshes metadata table** - Calls `Getpreprocessors()` to scan `__init__.py` and extract metadata from all processors
2. **Regenerates Fx parameters** - Calls `update_fx_dynamic_parameters()` to create Fx* parameters based on updated metadata
3. **Logs success** - Outputs "Fx metadata refreshed and parameters updated" to console

---

## Code Reference

**File:** `StreamDiffusionTD/StreamDiffusionExt.py`
**Lines:** 6156-6173

```python
def Refreshfxmetadata(self):
    """
    Refresh Fx metadata table and regenerate dynamic parameters.
    Called when Refreshfxmetadata parameter is pulsed.
    This should be triggered after adding new processors to refresh the UI.
    """
    try:
        # Refresh the metadata table
        self.Getpreprocessors()

        # Regenerate Fx parameters with updated metadata
        self.update_fx_dynamic_parameters()

        self.logger.log('Fx metadata refreshed and parameters updated', level='INFO')
    except Exception as e:
        self.logger.log(f'ERROR refreshing Fx metadata: {e}', level='ERROR')
        import traceback
        traceback.print_exc()
```

---

## Workflow for Adding New Processor

### 1. Create Processor File
```python
# src/streamdiffusion/preprocessing/processors/my_processor.py
class MyProcessor(PipelineAwareProcessor):
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "My Processor",
            "parameters": {
                "strength": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 1.0],
                    ...
                }
            }
        }
```

### 2. Register Processor
```python
# src/streamdiffusion/preprocessing/processors/__init__.py
from .my_processor import MyProcessor

_preprocessor_registry = {
    # ... existing processors ...
    "my_processor": MyProcessor,
}
```

### 3. Add TouchDesigner Toggle
```python
# StreamDiffusionTD/StreamDiffusionExt.py (line ~6505)
if hasattr(self.ownerComp.par, 'Usemyprocessor') and self.ownerComp.par.Usemyprocessor.eval():
    active_fx.append('my_processor')
```

### 4. Refresh Metadata in TouchDesigner
1. Open TouchDesigner
2. Enable `Usemyprocessor` toggle on Fx page
3. **Pulse `Refreshfxmetadata` button**
4. Verify Fx* parameters appear (e.g., `Fxmyprocessorstrength`)

---

## Troubleshooting

### Parameters Still Don't Appear

**Check 1: Is processor registered?**
```python
# In __init__.py, verify your processor is in the registry
_preprocessor_registry = {
    "my_processor": MyProcessor,  # ← Should be here
}
```

**Check 2: Is metadata correct?**
```python
# Your processor must implement get_preprocessor_metadata()
@classmethod
def get_preprocessor_metadata(cls):
    return { ... }  # Must return dict with "parameters" key
```

**Check 3: Is toggle added?**
```python
# In update_fx_dynamic_parameters() around line 6505
if hasattr(self.ownerComp.par, 'Usemyprocessor') and self.ownerComp.par.Usemyprocessor.eval():
    active_fx.append('my_processor')
```

**Check 4: Check console logs**
After pulsing `Refreshfxmetadata`, check console for:
- "Loaded N preprocessors with metadata into table_preprocessors"
- "Fx metadata refreshed and parameters updated"
- Any ERROR messages

### Metadata Table Empty

If `table_preprocessors` is empty after refresh:

1. Check `Basefolder` parameter is set correctly
2. Verify `__init__.py` path: `{Basefolder}/src/streamdiffusion/preprocessing/processors/__init__.py`
3. Check for Python syntax errors in `__init__.py`

### Parameters Appear but Values Wrong

1. Pulse `Refreshfxmetadata` again to reload metadata
2. Check `get_preprocessor_metadata()` return dict format
3. Verify parameter types match TouchDesigner expectations:
   - `float` → slider with range
   - `int` → integer slider
   - `string` with `options` → dropdown menu

---

## Alternative: Manual Refresh via Textport

If you prefer not to add the pulse parameter, you can manually refresh from textport:

```python
# In TouchDesigner textport
op('StreamDiffusion').Refreshfxmetadata()
```

But the pulse button approach is cleaner and more user-friendly.

---

## Related Documentation

- **CLAUDE.md** - Complete guide to creating processors
- **multistage_processing_system_architecture.md** - Pipeline architecture details
- **CUDA_12_API_FIX.md** - CUDA compatibility fix

---

## Summary

**Problem:** New processor doesn't show parameters in TouchDesigner
**Solution:** Add `Refreshfxmetadata` pulse parameter and pulse after enabling new processor toggle
**What it does:** Refreshes metadata table and regenerates Fx* parameters
**Where to add:** Fx page in TouchDesigner parameter editor
**Method called:** `Refreshfxmetadata()` in StreamDiffusionExt.py:6156
