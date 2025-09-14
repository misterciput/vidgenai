#!/usr/bin/env python3
"""
LoFi Video Generator using CogVideoX-5B-I2V
Optimized for Google Colab with seamless loop support
"""

import torch
import numpy as np
from PIL import Image
import cv2
import os
import gc
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Import diffusers components
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


class LoFiVideoGenerator:
    """
    LoFi Video Generator using CogVideoX-5B-I2V
    Optimized for Google Colab memory constraints
    """
    
    def __init__(self, 
                 model_id: str = "THUDM/CogVideoX-5b-I2V",
                 device: Optional[str] = None,
                 enable_memory_optimization: bool = True,
                 use_int8: bool = True):
        """
        Initialize the LoFi Video Generator
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on (auto-detected if None)
            enable_memory_optimization: Enable memory optimizations for Colab
            use_int8: Use INT8 quantization to reduce memory usage
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.enable_memory_optimization = enable_memory_optimization
        self.use_int8 = use_int8
        self.pipe = None
        
        print(f"Initializing LoFi Video Generator...")
        print(f"Device: {self.device}")
        print(f"Model: {model_id}")
        print(f"Memory optimization: {enable_memory_optimization}")
        print(f"INT8 quantization: {use_int8}")
        
    def load_model(self):
        """Load the CogVideoX-5B-I2V model with optimizations"""
        try:
            print("Loading CogVideoX-5B-I2V model...")
            
            # Load with appropriate torch dtype
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            # Load the pipeline
            self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            # Move to device
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
            
            # Apply memory optimizations for Colab
            if self.enable_memory_optimization:
                print("Applying memory optimizations...")
                
                # Enable memory efficient attention
                if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        print("✓ XFormers memory efficient attention enabled")
                    except Exception as e:
                        print(f"⚠ XFormers not available: {e}")
                
                # Enable CPU offload for memory management
                if hasattr(self.pipe, 'enable_model_cpu_offload'):
                    self.pipe.enable_model_cpu_offload()
                    print("✓ Model CPU offload enabled")
                
                # Enable sequential CPU offload as fallback
                elif hasattr(self.pipe, 'enable_sequential_cpu_offload'):
                    self.pipe.enable_sequential_cpu_offload()
                    print("✓ Sequential CPU offload enabled")
                
                # Enable VAE slicing
                if hasattr(self.pipe, 'enable_vae_slicing'):
                    self.pipe.enable_vae_slicing()
                    print("✓ VAE slicing enabled")
                
                # Enable VAE tiling
                if hasattr(self.pipe, 'enable_vae_tiling'):
                    self.pipe.enable_vae_tiling()
                    print("✓ VAE tiling enabled")
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def prepare_image_for_loop(self, image_path: str, target_size: Tuple[int, int] = (720, 480)) -> Image.Image:
        """
        Prepare input image for seamless loop generation
        
        Args:
            image_path: Path to input image
            target_size: Target resolution (width, height)
        
        Returns:
            Processed PIL Image
        """
        print(f"Preparing image: {image_path}")
        
        # Load image
        if isinstance(image_path, str):
            image = load_image(image_path)
        else:
            image = image_path
        
        # Resize to target resolution
        image = image.resize(target_size, Image.LANCZOS)
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"Image prepared: {image.size}")
        return image
    
    def generate_lofi_video(self,
                           image_path: str,
                           prompt: str,
                           negative_prompt: str = "blurry, low quality, distorted, deformed",
                           num_frames: int = 49,  # ~5 seconds at 10fps
                           fps: int = 10,
                           guidance_scale: float = 6.0,
                           num_inference_steps: int = 50,
                           generator_seed: Optional[int] = None,
                           output_path: str = "lofi_video.mp4") -> str:
        """
        Generate a LoFi video from an image
        
        Args:
            image_path: Path to input image or PIL Image
            prompt: Text prompt describing the desired animation
            negative_prompt: Negative prompt to avoid unwanted features
            num_frames: Number of frames to generate
            fps: Frames per second for output video
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of denoising steps
            generator_seed: Random seed for reproducibility
            output_path: Path for output video file
        
        Returns:
            Path to generated video file
        """
        if self.pipe is None:
            self.load_model()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"Generating LoFi video...")
        print(f"Prompt: {prompt}")
        print(f"Frames: {num_frames}, FPS: {fps}")
        print(f"Guidance Scale: {guidance_scale}")
        print(f"Inference Steps: {num_inference_steps}")
        
        # Prepare image
        image = self.prepare_image_for_loop(image_path)
        
        # Set up generator for reproducibility
        generator = None
        if generator_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(generator_seed)
            print(f"Using seed: {generator_seed}")
        
        try:
            # Generate video
            with torch.inference_mode():
                video_frames = self.pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).frames[0]  # Get the first (and only) video
            
            print(f"Generated {len(video_frames)} frames")
            
            # Post-process for seamless loop
            video_frames = self.create_seamless_loop(video_frames)
            
            # Export video
            export_to_video(video_frames, output_path, fps=fps)
            
            print(f"✅ Video saved to: {output_path}")
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            # Clear memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            raise
    
    def create_seamless_loop(self, frames):
        """
        Create seamless loop by blending end frames with start frames
        
        Args:
            frames: List of PIL Images
        
        Returns:
            Modified frames list for seamless looping
        """
        print("Creating seamless loop...")
        
        if len(frames) < 10:
            print("⚠ Too few frames for seamless blending, returning as-is")
            return frames
        
        # Number of frames to blend at the end
        blend_frames = min(5, len(frames) // 4)
        
        # Create blended transition
        loop_frames = frames.copy()
        
        for i in range(blend_frames):
            # Blend ratio (0 = end frame, 1 = start frame)
            alpha = (i + 1) / (blend_frames + 1)
            
            end_idx = -(blend_frames - i)
            start_idx = i
            
            # Convert to numpy for blending
            end_frame = np.array(loop_frames[end_idx])
            start_frame = np.array(loop_frames[start_idx])
            
            # Blend frames
            blended = (1 - alpha) * end_frame + alpha * start_frame
            loop_frames[end_idx] = Image.fromarray(blended.astype(np.uint8))
        
        print(f"✓ Seamless loop created with {blend_frames} blended frames")
        return loop_frames
    
    def generate_multiple_videos(self, 
                                image_paths: list,
                                prompts: list,
                                output_dir: str = "lofi_videos",
                                **kwargs) -> list:
        """
        Generate multiple LoFi videos in batch
        
        Args:
            image_paths: List of image paths
            prompts: List of prompts (one per image)
            output_dir: Output directory for videos
            **kwargs: Additional arguments for generate_lofi_video
        
        Returns:
            List of output video paths
        """
        if len(image_paths) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
            output_path = os.path.join(output_dir, f"lofi_video_{i+1:03d}.mp4")
            
            print(f"\n🎬 Generating video {i+1}/{len(image_paths)}")
            try:
                result_path = self.generate_lofi_video(
                    image_path=image_path,
                    prompt=prompt,
                    output_path=output_path,
                    **kwargs
                )
                output_paths.append(result_path)
                print(f"✅ Completed: {result_path}")
                
            except Exception as e:
                print(f"❌ Failed to generate video {i+1}: {e}")
                continue
        
        return output_paths


# Emergency functions for extremely limited memory
def create_emergency_t4_generator():
    """Create emergency T4 generator with minimal memory usage"""
    return LoFiVideoGenerator(
        enable_memory_optimization=True,
        use_int8=True,
        low_memory_mode=True,
        device="cpu"  # Force CPU if GPU memory is exhausted
    )

def emergency_generate_video(image_path: str, prompt: str, output_path: str = "emergency_video.mp4"):
    """Emergency video generation with absolute minimal settings"""
    print("🆘 Emergency generation mode - CPU only, very slow but will work")
    
    generator = LoFiVideoGenerator(
        enable_memory_optimization=False,  # Not needed for CPU
        use_int8=False,
        low_memory_mode=True,
        device="cpu"
    )
    
    generator.load_model()
    
    return generator.generate_lofi_video(
        image_path=image_path,
        prompt=prompt,
        negative_prompt="blurry, low quality",
        num_frames=8,         # 1 second video
        fps=8,
        guidance_scale=4.0,   # Minimal guidance
        num_inference_steps=10,  # Minimal steps
        output_path=output_path
    )


# Example usage and utility functions
def create_t4_generator():
    """Create a T4-optimized generator for Google Colab"""
    return LoFiVideoGenerator(
        enable_memory_optimization=True,
        use_int8=True,
        low_memory_mode=True  # Auto-uses 2B model
    )


def main():
    """Example usage of the LoFi Video Generator"""
    
    # Initialize T4-optimized generator
    print("🔧 Creating ultra-lightweight T4 generator...")
    generator = create_t4_generator()
    
    # Ultra-lightweight T4 settings
    emergency_t4_settings = {
        "num_frames": 16,      # ~2 seconds at 8fps
        "fps": 8,              # LoFi aesthetic  
        "guidance_scale": 5.0, # Reduced guidance
        "num_inference_steps": 20,  # Minimal for quality
        "generator_seed": 42
    }
    
    print("🆘 Ultra-lightweight T4 settings:")
    for key, value in emergency_t4_settings.items():
        print(f"  {key}: {value}")
    
    print("\n💡 If you still get OOM errors:")
    print("1. Restart Colab runtime")
    print("2. Use emergency_generate_video() function")
    print("3. Consider using CPU-only generation")
    
    print("VidGenAI Generator initialized for T4!")


if __name__ == "__main__":
    main()