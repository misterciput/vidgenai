# LoFi Video Generator - Google Colab Setup Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/lofi-video-generator/blob/main/notebooks/colab_quickstart.ipynb)

## Quick Start Instructions

### Step 1: Setup Google Colab Environment

```python
# Check GPU availability
!nvidia-smi

# Install requirements
!pip install -r requirements.txt

# Optional: Install additional optimizations
!pip install --upgrade diffusers[torch]
```

### Step 2: Basic Usage Example

```python
from lofi_video_generator import LoFiVideoGenerator
from PIL import Image
import torch

# Initialize generator
generator = LoFiVideoGenerator(
    enable_memory_optimization=True,
    use_int8=True  # Reduces memory usage for Colab
)

# Generate your first LoFi video
output_path = generator.generate_lofi_video(
    image_path="path/to/your/image.jpg",  # Upload your image first
    prompt="Gentle wind moving through tall grass, soft morning light, peaceful nature scene",
    negative_prompt="blurry, low quality, distorted, deformed, fast motion",
    num_frames=49,  # ~5 seconds at 10fps
    fps=10,
    guidance_scale=6.0,
    num_inference_steps=50,
    generator_seed=42,
    output_path="lofi_video.mp4"
)

print(f"Video generated: {output_path}")
```

### Step 3: Upload Images to Colab

```python
from google.colab import files
import os

# Upload your LoFi images
print("Upload your LoFi images:")
uploaded = files.upload()

# List uploaded files
for filename in uploaded.keys():
    print(f"Uploaded: {filename}")
```

### Step 4: Generate Multiple Videos

```python
# Define your prompts for different scenes
lofi_prompts = [
    "Gentle wind moving through tall grass, soft morning light",
    "Calm lake with small ripples, reflecting clouds slowly drifting",
    "Tree branches swaying gently in the breeze, dappled sunlight",
    "Soft rain drops creating ripples in a puddle, cozy rainy day",
    "Steam rising from a hot cup, warm lighting, comfortable scene"
]

# Get list of uploaded images
image_files = list(uploaded.keys())

# Generate videos for each image
output_paths = generator.generate_multiple_videos(
    image_paths=image_files[:len(lofi_prompts)],
    prompts=lofi_prompts,
    output_dir="lofi_videos",
    num_frames=49,
    fps=10,
    guidance_scale=6.0,
    generator_seed=42
)

# Download generated videos
from google.colab import files
for path in output_paths:
    files.download(path)
```

## Advanced Configuration

### Memory Optimization Settings

```python
# For maximum memory efficiency (slower but fits in Colab)
generator = LoFiVideoGenerator(
    enable_memory_optimization=True,
    use_int8=True,
)

# For faster generation (requires more memory)
generator = LoFiVideoGenerator(
    enable_memory_optimization=False,
    use_int8=False,
)
```

### Video Quality Settings

```python
# High quality (longer generation time)
generator.generate_lofi_video(
    image_path="your_image.jpg",
    prompt="your prompt here",
    num_frames=73,  # ~7 seconds at 10fps
    fps=12,
    guidance_scale=7.0,
    num_inference_steps=75,
    output_path="high_quality.mp4"
)

# Fast generation (lower quality)
generator.generate_lofi_video(
    image_path="your_image.jpg",
    prompt="your prompt here",
    num_frames=25,  # ~3 seconds at 8fps
    fps=8,
    guidance_scale=5.0,
    num_inference_steps=25,
    output_path="fast_generation.mp4"
)
```

## Best Practices for LoFi Natural Scenery

### Recommended Prompts

- **Grass/Fields**: "Gentle wind moving through tall grass, soft morning light, peaceful nature scene"
- **Water Scenes**: "Calm lake with small ripples, reflecting clouds slowly drifting by, serene atmosphere"
- **Forest**: "Tree branches swaying gently in the breeze, dappled sunlight, tranquil forest scene"
- **Rain**: "Soft rain drops creating ripples, cozy rainy day atmosphere, gentle movement"
- **Sky**: "Clouds slowly drifting across the sky, peaceful day, subtle movement"

### Negative Prompts to Use

- "blurry, low quality, distorted, deformed, fast motion, jerky movement, unrealistic"

### Optimal Settings for LoFi Aesthetic

- **FPS**: 8-12 (LoFi aesthetic works well with lower frame rates)
- **Guidance Scale**: 5.0-7.0 (higher values for more prompt adherence)
- **Inference Steps**: 30-50 (balance between quality and speed)
- **Frames**: 25-49 (3-5 seconds is perfect for loops)

## Troubleshooting

### Memory Issues

```python
# Clear CUDA memory if you get OOM errors
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

# Reduce parameters
num_frames = 25  # Instead of 49
num_inference_steps = 25  # Instead of 50
```

### Generation Taking Too Long

```python
# Speed up generation
guidance_scale = 5.0  # Lower guidance
num_inference_steps = 25  # Fewer steps
num_frames = 25  # Shorter video
```

### Video Quality Issues

```python
# Improve quality
guidance_scale = 7.0  # Higher guidance
num_inference_steps = 50  # More steps
# Use better negative prompts
negative_prompt = "blurry, low quality, distorted, deformed, fast motion, jerky movement"
```

## Expected Performance

- **Generation Time**: 10-15 minutes per 5-second video on Colab T4 GPU
- **Memory Usage**: ~12-14GB with optimizations enabled
- **Video Quality**: 720p, smooth loops, LoFi aesthetic
- **Batch Processing**: 4-6 videos per hour

## File Management

```python
# Create organized output structure
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"lofi_videos_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Generate with organized naming
generator.generate_lofi_video(
    image_path="scene1.jpg",
    prompt="your prompt",
    output_path=f"{output_dir}/lofi_scene1.mp4"
)
```

This setup is optimized for Google Colab's constraints and should work reliably for generating high-quality LoFi video loops!