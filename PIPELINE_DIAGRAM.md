# StreamDiffusion Pipeline Architecture

Complete visual guide to the StreamDiffusion processing pipeline and feedback loops.

---

## Complete Pipeline Flow

```
┌──────────────────────┐
│   INPUT IMAGE        │
│   from TouchDesigner │
│   [B, 3, 512, 512]   │
│   RGB [0, 1]         │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: IMAGE PREPROCESSING (Before VAE Encode)                                │
│ Location: stream.image_preprocessing_hooks                                      │
│                                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐                │
│ │  Available Processors:                                      │                │
│ │  - FeedbackPreprocessor (image-space feedback)              │                │
│ │  - FeedbackTransformPreprocessor (zoom/pan/rotate + FB)     │                │
│ │  - TemporalNetTensorRT (optical flow)                       │                │
│ │  - ColorCorrectionPreprocessor (color grading + feedback)   │                │
│ └─────────────────────────────────────────────────────────────┘                │
│                                                                                  │
│ Access to: prev_image_result (output from Frame N-1)                            │
│ requires_sync_processing: true (to avoid 1-frame delay)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   VAE ENCODE         │
│   x_t_latent         │
│   [B, 4, 64, 64]     │
│   (8x8 downscale)    │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: LATENT PREPROCESSING (After VAE Encode, Before Diffusion)              │
│ Location: stream.latent_preprocessing_hooks                                     │
│                                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐                │
│ │  Available Processors:                                      │                │
│ │  - LatentFeedbackPreprocessor (latent-space feedback)       │                │
│ │  - LatentTransformPreprocessor (zoom/pan/rotate) ⭐          │                │
│ └─────────────────────────────────────────────────────────────┘                │
│                                                                                  │
│ Access to: prev_latent_result (diffusion output from Frame N-1)                 │
│                                                                                  │
│ ⚠️  CRITICAL: prev_latent_result comes from STAGE 3 (diffusion output),         │
│     NOT from VAE encode! So it doesn't see image preprocessing changes!         │
└─────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   UNET DIFFUSION     │
│   + ControlNet       │
│   + IPAdapter        │
│   x_0_pred           │
│   [B, 4, 64, 64]     │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: LATENT POSTPROCESSING (After Diffusion, Before VAE Decode)             │
│ Location: stream.latent_postprocessing_hooks                                    │
│                                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐                │
│ │  Available Processors:                                      │                │
│ │  - ColorCorrectionFeedbackPreprocessor (latent color)       │                │
│ │  - (Relatively unexplored stage!)                           │                │
│ └─────────────────────────────────────────────────────────────┘                │
│                                                                                  │
│ Access to: prev_latent_result (previous frame's latent postprocessing output)   │
│                                                                                  │
│ ⚠️  STORED HERE: prev_latent_result = x_0_pred_out                              │
│     This is what LatentTransformPreprocessor sees next frame!                   │
└─────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   VAE DECODE         │
│   x_output           │
│   [B, 3, 512, 512]   │
│   RGB [-1, 1]        │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: IMAGE POSTPROCESSING (After VAE Decode)                                │
│ Location: stream.image_postprocessing_hooks                                     │
│                                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐                │
│ │  Available Processors:                                      │                │
│ │  - SharpenPreprocessor                                      │                │
│ │  - RealESRGANProcessor (upscale)                            │                │
│ │  - PostProcessColorPreprocessor (stateless color grading)   │                │
│ └─────────────────────────────────────────────────────────────┘                │
│                                                                                  │
│ Access to: prev_image_result (previous frame's final output)                    │
│                                                                                  │
│ ⚠️  STORED HERE: prev_image_result = final_output                               │
│     This is what image preprocessing sees next frame!                           │
└─────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   OUTPUT IMAGE       │
│   to TouchDesigner   │
│   [B, 3, 512, 512]   │
│   RGB [-1, 1]        │
└──────────────────────┘
```

---

## Feedback Loop Visualization

```
Frame N-1:
  Output ────────────────────────┐
  [After Stage 4]                │
                                 │ Stored as prev_image_result
                                 │
  Diffusion Output ──────────────┼─────────┐
  [After Stage 3]                │         │ Stored as prev_latent_result
                                 │         │
                                 │         │
Frame N:                         │         │
  Input                          │         │
    │                            │         │
    ▼                            │         │
  Stage 1 ◄──────────────────────┘         │
    │ (Can use prev_image_result)          │
    ▼                                      │
  VAE Encode                               │
    │                                      │
    ▼                                      │
  Stage 2 ◄──────────────────────────────┘
    │ (Uses prev_latent_result)
    ▼
  Diffusion
    │
    ▼
  Stage 3
    │
    ▼
  VAE Decode
    │
    ▼
  Stage 4
    │
    ▼
  Output
```

---

## Key Insights

### Two Separate Feedback Paths

**Image Feedback Path (prev_image_result):**
- Stored after Stage 4 (final output)
- Available to Stage 1 processors
- Use for: Image-space feedback, color correction, optical flow
- Example: `FeedbackTransformPreprocessor`

**Latent Feedback Path (prev_latent_result):**
- Stored after Stage 3 (diffusion output, before VAE decode)
- Available to Stage 2 processors
- Use for: Latent-space transforms, efficient motion effects
- Example: `LatentTransformPreprocessor`

### Critical Limitation

**Image preprocessing CANNOT influence latent_transform feedback!**

Why? Because:
1. Image preprocessing modifies the current input
2. That gets VAE encoded to create `x_t_latent` (current)
3. `latent_transform` transforms `prev_latent_result` (from previous frame's diffusion)
4. These are separate data paths!

```
Frame N:
  Input → [Color Correction] → VAE Encode → x_t_latent (current, color corrected ✓)

  latent_transform transforms:
    prev_latent_result (from Frame N-1 diffusion, NOT color corrected ❌)
```

---

## Common Patterns

### Pattern 1: Stateless Post-Processing

**Use Case:** Apply effects to final output without feedback

**Stage:** Stage 4 (Image Postprocessing)

**Example:** `PostProcessColorPreprocessor`

```yaml
image_postprocessing:
  enabled: true
  processors:
    - type: "post_process_color"
      params:
        brightness: 0.1
        saturation: 1.2
```

**Flow:**
```
Diffusion → VAE Decode → [Color Correction] → Output
```

**Pros:**
- ✅ Simple, predictable
- ✅ No feedback artifacts
- ✅ Fast (no pipeline dependencies)

**Cons:**
- ❌ Doesn't influence motion/feedback
- ❌ Applied after everything else

---

### Pattern 2: Image-Space Feedback with Color Correction

**Use Case:** Color correction that participates in feedback loop

**Stage:** Stage 1 (Image Preprocessing)

**Example:** `ColorCorrectionPreprocessor` (NEW!)

```yaml
image_preprocessing:
  enabled: true
  processors:
    - type: "color_correction"
      params:
        brightness: 0.1
        saturation: 1.2
        feedback_strength: 0.3
        requires_sync_processing: true
```

**Flow:**
```
Input → [Color Correction with Feedback] → VAE Encode → Diffusion
        ↑
        └───────── prev_image_result ───────────────────┘
```

**Pros:**
- ✅ Color correction participates in feedback
- ✅ Temporal smoothing of colors
- ✅ Can combine with other Stage 1 processors

**Cons:**
- ❌ Doesn't influence `latent_transform` (separate feedback path)
- ❌ Requires `requires_sync_processing: true`

---

### Pattern 3: Image-Space Motion + Color

**Use Case:** Zoom/pan/rotate with color-aware feedback

**Stage:** Stage 1 (Image Preprocessing)

**Example:** `FeedbackTransformPreprocessor` + `ColorCorrectionPreprocessor`

```yaml
image_preprocessing:
  enabled: true
  processors:
    - type: "color_correction"
      order: 1
      params:
        brightness: 0.1
        saturation: 1.2
        feedback_strength: 0.3
        requires_sync_processing: true
    - type: "feedback_transform"
      order: 2
      params:
        zoom: 1.02
        feedback_strength: 0.8
        requires_sync_processing: true
```

**Flow:**
```
Input → [Color Correction] → [Transform + Feedback] → VAE Encode → Diffusion
        ↑                     ↑
        └─────────────────────┴──── prev_image_result ──────────────┘
```

**Pros:**
- ✅ True color-aware motion
- ✅ Both effects participate in feedback
- ✅ Image-space operations (intuitive)

**Cons:**
- ❌ Slower than latent operations (512x512 vs 64x64)
- ❌ Requires `requires_sync_processing: true` for both

---

### Pattern 4: Latent-Space Motion (Fast)

**Use Case:** Efficient zoom/pan/rotate without color correction

**Stage:** Stage 2 (Latent Preprocessing)

**Example:** `LatentTransformPreprocessor`

```yaml
latent_preprocessing:
  enabled: true
  processors:
    - type: "latent_transform"
      params:
        zoom: 1.02
        feedback_blend: 0.8
```

**Flow:**
```
VAE Encode → [Transform prev_latent_result] → Diffusion
             ↑                                 ↓
             └──── prev_latent_result ─────────┘
```

**Pros:**
- ✅ 8x faster (64x64 vs 512x512)
- ✅ Efficient feedback in latent space
- ✅ No sync requirement

**Cons:**
- ❌ Cannot see color corrections from Stage 1 or Stage 4
- ❌ Works with diffusion output, not input

---

## Performance Comparison

| Stage | Resolution | Speed | Feedback Type | Use Case |
|-------|-----------|-------|---------------|----------|
| Stage 1 | 512x512 | Slow | Image (prev_image_result) | Color + Motion |
| Stage 2 | 64x64 | **Fast** | Latent (prev_latent_result) | Motion only |
| Stage 3 | 64x64 | **Fast** | Latent (prev_latent_result) | Latent effects |
| Stage 4 | 512x512 | Slow | None (stateless) | Final polish |

**Rule of Thumb:**
- Use **Stage 2 (latent)** for fast motion effects
- Use **Stage 1 (image)** for color-aware motion (accept slower speed)
- Use **Stage 4 (post)** for final color grading without feedback

---

## Best Practices

### 1. Choose the Right Stage

**Need speed?** → Stage 2 (latent operations)

**Need color awareness?** → Stage 1 (image operations)

**Just final polish?** → Stage 4 (stateless post-processing)

### 2. Understand Feedback Paths

- Image feedback (`prev_image_result`) and latent feedback (`prev_latent_result`) are **separate**
- Stage 1 processors see final output from previous frame
- Stage 2 processors see diffusion output from previous frame (before VAE decode)

### 3. Use `requires_sync_processing` Correctly

- **Required** for Stage 1 processors that use `prev_image_result`
- Prevents 1-frame delay from pipelined orchestrator
- Not needed for Stage 2 (latent) processors

### 4. Combine Stages Strategically

**Good combinations:**
```yaml
# Fast motion + Final color grading
latent_preprocessing:
  - latent_transform  # Fast motion in latent space

image_postprocessing:
  - post_process_color  # Stateless color grading
```

```yaml
# Color-aware motion (slower but integrated)
image_preprocessing:
  - color_correction  # With feedback
  - feedback_transform  # Sees color-corrected feedback
```

**Avoid conflicts:**
- Don't use both `latent_transform` and `feedback_transform` together (competing motion)
- Don't do heavy processing in both Stage 1 and Stage 4 (double slowdown)

---

## Troubleshooting

### "My color correction doesn't affect motion"

**Problem:** Using `post_process_color` (Stage 4) with `latent_transform` (Stage 2)

**Solution:** Move color correction to Stage 1, or use `feedback_transform` instead of `latent_transform`

### "Feedback creates 1-frame delay"

**Problem:** Missing `requires_sync_processing: true` for Stage 1 processors

**Solution:** Add `requires_sync_processing: true` to YAML config

### "Performance is slow"

**Problem:** Using Stage 1 (image space) for motion effects

**Solution:** Use Stage 2 (latent space) for motion, reserve Stage 1 for color/optical flow

### "Colors are unstable/flickering"

**Problem:** Stateless color correction in Stage 4 with feedback motion

**Solution:** Add `feedback_strength` to color correction in Stage 1 for temporal smoothing

---

## References

- **Pipeline Code:** `src/streamdiffusion/pipeline.py`
- **Processors:** `src/streamdiffusion/preprocessing/processors/`
- **TouchDesigner Integration:** `StreamDiffusionTD/StreamDiffusionExt.py`
- **Processor Development Guide:** `CLAUDE.md`
