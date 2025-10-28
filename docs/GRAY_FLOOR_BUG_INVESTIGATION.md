# Gray Floor Bug Investigation & Fix

**Date:** 2025-10-28
**Issue:** Black backgrounds becoming progressively gray/white with feedback enabled
**Status:** ✅ FIXED

---

## Executive Summary

A critical bug in `feedback_transform` processor was causing black backgrounds to appear as 50% gray. After multiple false starts, we discovered the issue was a **missing output range conversion** - the processor was returning blended results in `[0, 1]` range when the pipeline expected `[-1, 1]` range for VAE encoding.

**Root Cause:** Output range mismatch
**Primary Fix:** Convert blended output from `[0, 1]` to `[-1, 1]` before returning
**Secondary Fix:** Remap VAE output range to prevent gray floor accumulation
**Bonus:** Changed black_level from lift to crush for better creative control

---

## Timeline of Investigation

### Initial Problem Report
**User:** "we now have that grey issue again with the vae decode where it centers to grey"

With feedback enabled and color correction added, black backgrounds were becoming progressively gray. With high feedback strength (0.957), blacks would eventually turn white.

---

## Attempted Fixes (All Wrong!)

### ❌ Attempt 1: Brightness Compensation
**Theory:** VAE has inherent brightness bias, compensate with negative brightness
**Action:** Set brightness default to -0.1
**Result:** FAILED - Wrong approach entirely
**User Feedback:** "nooo no no. in the post process color we remapped it in a totally different way that fixes this completely"

---

### ❌ Attempt 2: Add VAE Output Range Detection
**Theory:** Need to properly detect and convert VAE output range
**Action:** Added range detection in feedback_transform for prev_output conversion
**Code:**
```python
if tensor.max() > 1.0:
    tensor = tensor / 255.0
elif tensor.min() < 0.0:
    tensor = (tensor / 2.0 + 0.5).clamp(0, 1)
```
**Result:** FAILED - Still gray
**User Feedback:** "no its still not fixed"

---

### ❌ Attempt 3: Simplified prev_output Conversion
**Theory:** We know prev_output is always from VAE, just convert it directly
**Action:** Always convert prev_output from [-1, 1] to [0, 1]
**Code:**
```python
prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)
```
**Result:** FAILED - Still gray
**User Feedback:** Still seeing gray floor issue

---

### ❌ Attempt 4: Add Input Tensor Normalization
**Theory:** Input from VaeImageProcessor needs conversion too
**Action:** Added input tensor range detection and conversion
**Code:**
```python
if input_tensor.min() < 0.0:
    input_tensor = (input_tensor / 2.0 + 0.5).clamp(0, 1)
```
**Result:** FAILED - Still gray
**User Feedback:** "still the same behavior unfortunately"

---

### ❌ Attempt 5: Remap VAE Output Range
**Theory:** VAE doesn't actually output full [-1, 1] range, has gray floor
**Action:** Remap actual VAE range to [0, 1] to force blacks to be black
**Code:**
```python
actual_min = prev_output.min()
actual_max = prev_output.max()
prev_output = (prev_output - actual_min) / (actual_max - actual_min)
```
**Result:** Partially helped, but still wrong
**User Feedback:** "the colors are different now but still when i crank up the steps and lower the feedback to get close to the original image its just double the brightness it should be"

---

## The Real Problem Discovered

**User Insight:** "when i lower the feedback to 0 and send in pure black, 50% grey fucking comes out!"

This was the key observation that led to the actual bug.

### Debug Output Analysis
```
[FEEDBACK DEBUG] input_tensor AFTER normalization: min=0.0000, max=1.0000
[FEEDBACK DEBUG] OUTPUT range after conversion to [-1,1]: min=-1.0000, max=1.0000
```

Wait... we were converting input from `[-1, 1]` to `[0, 1]`, blending in `[0, 1]` space, but then... **returning in `[0, 1]` range!**

The pipeline expects preprocessor hooks to return tensors in `[-1, 1]` range for VAE encoding, but we were returning `[0, 1]`.

---

## ✅ The Actual Fix

### Root Cause
```
Input: image_processor.preprocess() outputs [-1, 1]
↓
feedback_transform converts to [0, 1] for blending
↓
feedback_transform blends and clamps in [0, 1]
↓
feedback_transform returns [0, 1]  ← BUG!
↓
Pipeline passes [0, 1] to VAE encoder expecting [-1, 1]
↓
Black (0.0 in [0,1]) treated as mid-gray (0 in [-1,1])
↓
Result: 50% gray instead of black (100% brightness error!)
```

### The Fix
**File:** `feedback_transform.py:590-594`

**Before:**
```python
# Clamp to [0, 1] to prevent color space drift/accumulation
blended_tensor = blended_tensor.clamp(0, 1)

# Return directly (BUG - wrong range!)
return blended_tensor
```

**After:**
```python
# Clamp to [0, 1] to prevent color space drift/accumulation
blended_tensor = blended_tensor.clamp(0, 1)

# CRITICAL FIX: Convert back to [-1, 1] range for VAE encoder
# Pipeline expects input in [-1, 1] range (black=-1, gray=0, white=1)
# We processed in [0, 1] for easier blending, now convert back
blended_tensor = (blended_tensor * 2.0) - 1.0

return blended_tensor
```

### Range Conversion Math
```
[0, 1] → [-1, 1] conversion:
- Black: 0.0 × 2.0 - 1.0 = -1.0 ✅
- Gray:  0.5 × 2.0 - 1.0 =  0.0 ✅
- White: 1.0 × 2.0 - 1.0 =  1.0 ✅
```

---

## Secondary Fix: VAE Range Remapping

While the main bug was the output range, we also improved the handling of VAE's limited reconstruction range.

### Problem
VAE decoder doesn't actually output the full `[-1, 1]` range. It typically outputs something like `[-0.8765, 1.0547]`, which has an inherent gray floor.

### Solution
Remap the actual VAE output range to `[0, 1]` to force blacks to be black:

```python
# Get actual min/max from VAE output
actual_min = prev_output.min()
actual_max = prev_output.max()

# Remap actual range to [0, 1] to force blacks to be black
if actual_max > actual_min:
    prev_output = (prev_output - actual_min) / (actual_max - actual_min)
```

**Before remapping:**
```
VAE output: min=-0.8765, max=1.0547
After (x/2+0.5): min=0.0618, max=1.0000  ← gray floor!
```

**After remapping:**
```
VAE output: min=-0.8765, max=1.0547
After remapping: min=0.0000, max=1.0000  ← true black!
```

This prevents the VAE's gray floor from accumulating through the feedback loop.

---

## Bonus: Black Level Crush

While fixing this, we also improved the `black_level` parameter from a "lift" to a "crush".

### Old Behavior (Black Lift)
```python
# Compress range: [0, 1] → [black_level, 1]
result = result * (1.0 - self.black_level) + self.black_level
```
- black_level=0.167 made black (0.0) become gray (0.167)
- Lifted shadows, never crushed to pure black
- Not very useful creatively

### New Behavior (Black Crush)
```python
# Remap: [black_level, 1] → [0, 1], values below threshold → 0
result = torch.clamp((result - self.black_level) / (1.0 - self.black_level + 1e-8), 0, 1)
```
- black_level=0.0 → no crush (normal)
- black_level=0.3 → values below 0.3 become black
- black_level=0.5 → values below 0.5 become black
- black_level=1.0 → everything becomes black

Much more useful for creating dramatic high-contrast looks!

---

## Key Lessons Learned

### 1. Always Check Output Range Expectations
Preprocessor hooks can return PIL Images (which get converted by the pipeline) or tensors (which must be in the correct range). We were returning tensors in the wrong range.

### 2. The "With Feedback=0, Black Should Be Black" Test
This simple test immediately revealed the core issue. If you can't get black-in → black-out with feedback disabled, something fundamental is wrong.

### 3. VAE Range is Not [-1, 1]
Despite the convention, VAE decoders don't actually output the full `[-1, 1]` range. They have an inherent gray floor that needs to be accounted for in feedback loops.

### 4. Debug Logging is Essential
Without the debug output showing actual min/max values at each stage, we would never have found the issue.

### 5. Trust the User's Intuition
When the user said "it works fine in TouchDesigner when processing externally before feeding back in", that was a huge clue that our internal processing was doing something wrong with the ranges.

---

## Files Modified

### feedback_transform.py
**Lines changed:**
1. **476-493:** Added VAE range remapping for prev_output
2. **379-393:** Added VAE range remapping for prev_output (PIL path)
3. **499-523:** Added input tensor normalization from [-1, 1] to [0, 1]
4. **590-594:** **CRITICAL FIX** - Convert blended output back to [-1, 1]
5. **299-306:** Changed black_level from lift to crush
6. **80-85:** Updated metadata for black_level parameter

### Commits
1. `9c57090` - Added input tensor normalization (wrong approach)
2. `8944a29` - Added VAE range remapping (partially helpful)
3. `62ca4e3` - **THE FIX** - Convert output back to [-1, 1] range
4. `9500a5f` - Changed black_level to crush behavior

---

## Testing Recommendations

### Test 1: Black Passthrough (feedback_strength=0)
```yaml
feedback_strength: 0.0
brightness: 0.0
saturation: 1.0
contrast: 1.0
black_level: 0.0
```
**Input:** Pure black (0, 0, 0)
**Expected Output:** Pure black (0, 0, 0)
**Before Fix:** 50% gray (127, 127, 127) ❌
**After Fix:** Pure black (0, 0, 0) ✅

### Test 2: High Feedback Stability
```yaml
feedback_strength: 0.95
brightness: 0.0
saturation: 1.0
contrast: 1.0
black_level: 0.0
```
**Input:** Black background with subject
**Expected:** Black stays black, doesn't drift to gray/white
**Before Fix:** Progressive brightness increase, eventually white ❌
**After Fix:** Black stays black across frames ✅

### Test 3: Black Crush
```yaml
feedback_strength: 0.5
black_level: 0.3
```
**Input:** Image with values [0, 0.2, 0.4, 0.6, 0.8, 1.0]
**Expected Output:** Values [0, 0, ~0.14, ~0.43, ~0.71, 1.0]
(Values below 0.3 crushed to black, rest remapped)

---

## Conclusion

What seemed like a VAE reconstruction issue or color space problem was actually a simple but critical bug: **returning tensors in the wrong range**. The fix was a single line of code, but finding it required understanding:

1. The pipeline's range expectations (`[-1, 1]` for VAE input)
2. The preprocessor's internal processing (`[0, 1]` for blending)
3. The need to convert between these ranges
4. The VAE's actual output characteristics (limited range with gray floor)

**Final Status:** Black backgrounds now stay black, feedback loops are stable, and we have bonus black crush control for creative looks. 🎉

---

**Related Documentation:**
- `CLAUDE.md` - Processor development guide
- `multistage_processing_system_architecture.md` - Pipeline architecture
- Commit history: `9c57090`, `8944a29`, `62ca4e3`, `9500a5f`
