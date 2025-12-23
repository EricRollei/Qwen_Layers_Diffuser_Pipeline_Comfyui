"""
Quick test script for Qwen-Image-Layered
Run with: A:\Comfy25\ComfyUI_windows_portable\python_embeded\python.exe test_qwen_layered.py
"""

import torch
from PIL import Image
from diffusers import QwenImageLayeredPipeline
import os

# Configuration
INPUT_IMAGE = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\input\test.png"  # Change this!
OUTPUT_DIR = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\output\qwen_layers_test"
NUM_LAYERS = 4
SEED = 777

def main():
    print("Loading Qwen-Image-Layered pipeline...")
    print("(First run will download ~30GB of model weights)")
    
    # Load pipeline
    pipeline = QwenImageLayeredPipeline.from_pretrained(
        "Qwen/Qwen-Image-Layered",
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)
    
    print(f"Loading input image: {INPUT_IMAGE}")
    if not os.path.exists(INPUT_IMAGE):
        print(f"ERROR: Image not found at {INPUT_IMAGE}")
        print("Please update INPUT_IMAGE path in this script")
        return
    
    image = Image.open(INPUT_IMAGE).convert("RGBA")
    print(f"Image size: {image.size}")
    
    # Prepare inputs
    inputs = {
        "image": image,
        "generator": torch.Generator(device='cuda').manual_seed(SEED),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
        "layers": NUM_LAYERS,
        "resolution": 640,  # 640 or 1024
        "cfg_normalize": False,
        "use_en_prompt": True,
    }
    
    print(f"Running decomposition with {NUM_LAYERS} layers...")
    
    with torch.inference_mode():
        output = pipeline(**inputs)
        layer_images = output.images[0]  # List of PIL RGBA images
    
    print(f"Generated {len(layer_images)} layers")
    
    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i, layer in enumerate(layer_images):
        output_path = os.path.join(OUTPUT_DIR, f"layer_{i:02d}.png")
        layer.save(output_path)
        print(f"Saved: {output_path}")
    
    # Create composite
    composite = Image.new("RGBA", layer_images[0].size, (255, 255, 255, 0))
    for layer in layer_images:
        composite = Image.alpha_composite(composite, layer)
    
    composite_path = os.path.join(OUTPUT_DIR, "_composite.png")
    composite.save(composite_path)
    print(f"Saved composite: {composite_path}")
    
    print("\nDone! Check output folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
