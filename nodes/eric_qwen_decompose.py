"""Eric Qwen Decompose Node
Wraps the Qwen-Image-Layered pipeline for proper layer decomposition.

Author: Eric Hiss (GitHub: EricRollei)
Contact: [eric@historic.camera, eric@rollei.us]
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: This software is licensed under the terms of the
   Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
   
2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at [eric@historic.camera, eric@rollei.us] for licensing options.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any

import folder_paths


def get_diffusion_model_files():
    """Get list of safetensors files AND diffusers repos from diffusion_models folder."""
    diffusion_path = folder_paths.get_folder_paths("diffusion_models")
    files = ["HuggingFace (Download)"]  # Default option
    found_repos = set()
    
    for base_path in diffusion_path:
        if os.path.exists(base_path):
            # First, check for full diffusers repos (folders with model_index.json)
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    model_index = os.path.join(item_path, "model_index.json")
                    if os.path.exists(model_index):
                        # This is a full diffusers repo
                        repo_name = f"[Local Repo] {item}"
                        if repo_name not in found_repos:
                            files.append(repo_name)
                            found_repos.add(repo_name)
            
            # Also check for individual safetensors files
            for root, dirs, filenames in os.walk(base_path):
                for filename in filenames:
                    if filename.endswith(('.safetensors', '.ckpt', '.pt')):
                        rel_path = os.path.relpath(os.path.join(root, filename), base_path)
                        # Skip files inside repos we already found
                        if not any(rel_path.startswith(repo.replace("[Local Repo] ", "")) for repo in found_repos):
                            files.append(rel_path)
    
    return files


# Global pipeline cache for cross-instance access
_QWEN_PIPELINE_CACHE = {
    "pipeline": None,
    "source": None,
}


def clear_qwen_pipeline():
    """Clear the cached Qwen pipeline and free VRAM."""
    import gc
    import torch
    
    global _QWEN_PIPELINE_CACHE
    
    if _QWEN_PIPELINE_CACHE["pipeline"] is not None:
        print("[EricQwenLayer] Unloading Qwen-Image-Layered pipeline...")
        del _QWEN_PIPELINE_CACHE["pipeline"]
        _QWEN_PIPELINE_CACHE["pipeline"] = None
        _QWEN_PIPELINE_CACHE["source"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[EricQwenLayer] Pipeline unloaded, VRAM freed.")
        return True
    else:
        print("[EricQwenLayer] No pipeline loaded.")
        return False


class EricQwenDecompose:
    """
    Decompose an image into multiple RGBA layers using Qwen-Image-Layered.
    
    This node provides two operation modes:
    1. Direct pipeline execution (if diffusers is installed)
    2. Loading pre-generated layers from files
    
    The output is a proper IMAGE batch where each item in the batch
    represents a layer (not a video frame).
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "decompose"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING")
    RETURN_NAMES = ("layers", "alpha_masks", "layer_count", "layer_info", "auto_caption")
    OUTPUT_IS_LIST = (False, False, False, False, False)
    
    # Check if diffusers pipeline is available
    DIFFUSERS_AVAILABLE = False
    try:
        from diffusers import QwenImageLayeredPipeline
        DIFFUSERS_AVAILABLE = True
    except ImportError:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        model_files = get_diffusion_model_files()
        
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "num_layers": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of layers to decompose into (2-10)"
                }),
                "resolution": (["1024", "640"], {
                    "default": "1024",
                    "tooltip": "Processing resolution. 1024 gives ~1MP output, 640 gives ~0.4MP."
                }),
                "model_source": (model_files, {
                    "default": "HuggingFace (Download)",
                    "tooltip": "Choose HuggingFace download or local safetensors file"
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional prompt to guide decomposition. Leave empty for auto-caption."
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional negative prompt"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG scale (true_cfg_scale in pipeline)"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Number of inference steps"
                }),
                "cfg_normalize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable CFG normalization"
                }),
                "use_en_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use English for auto-captioning"
                }),
                "upscale_to_original": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Upscale output layers back to original input image size"
                }),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "nearest"], {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for upscaling"
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM after processing. Disable to free memory after each run."
                }),
                "caption_layers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use vision model to generate descriptive names for each layer (slower but more accurate)"
                }),
            }
        }
        return inputs
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when seed changes"""
        return kwargs.get("seed", 0)
    
    def decompose(
        self,
        image: torch.Tensor,
        num_layers: int = 4,
        resolution: str = "640",
        model_source: str = "HuggingFace (Download)",
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 0,
        cfg_scale: float = 4.0,
        steps: int = 50,
        cfg_normalize: bool = False,
        use_en_prompt: bool = True,
        upscale_to_original: bool = False,
        upscale_method: str = "lanczos",
        keep_model_loaded: bool = True,
        caption_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """
        Decompose an image into layers.
        
        Args:
            image: Input image tensor [B, H, W, C] (RGB or RGBA)
            num_layers: Number of layers to generate
            resolution: Processing resolution (640 or 1024)
            model_source: HuggingFace or local safetensors path
            prompt: Optional text prompt
            negative_prompt: Optional negative prompt
            seed: Random seed
            cfg_scale: CFG scale
            steps: Inference steps
            cfg_normalize: Enable CFG normalization
            use_en_prompt: Use English auto-caption
            upscale_to_original: Upscale output to match input size
            upscale_method: Interpolation method for upscaling
            
        Returns:
            layers: IMAGE batch [num_layers, H, W, 4] - RGBA layers
            alpha_masks: MASK batch [num_layers, H, W] - Alpha channels
            layer_count: Number of layers
            layer_info: JSON string with layer metadata
            auto_caption: The caption generated by the vision model (or user prompt)
        """
        import json
        
        # Validate input image
        if image is None or len(image.shape) == 0 or image.shape[0] == 0:
            raise ValueError("No input image provided. Please connect an image to the 'image' input.")
        
        print(f"[EricQwenDecompose] Input image shape: {image.shape}")
        
        # Convert input image to PIL
        input_pil = self._tensor_to_pil(image[0])  # Take first image from batch
        original_size = (input_pil.width, input_pil.height)
        
        # Ensure RGBA
        if input_pil.mode != "RGBA":
            input_pil = input_pil.convert("RGBA")
        
        # Get layers from pipeline or simulation
        auto_caption = ""
        if self.DIFFUSERS_AVAILABLE:
            layer_images, auto_caption = self._run_pipeline(
                input_pil,
                num_layers=num_layers,
                resolution=int(resolution),
                model_source=model_source,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                cfg_scale=cfg_scale,
                steps=steps,
                cfg_normalize=cfg_normalize,
                use_en_prompt=use_en_prompt,
            )
        else:
            # Fallback: simulate layer decomposition for testing
            # In production, users should install diffusers
            layer_images = self._simulate_layers(input_pil, num_layers)
            auto_caption = "(simulated - diffusers not installed)"
            print("[EricQwenDecompose] Warning: diffusers not installed, using simulated layers")
        
        # Upscale layers to original size if requested
        if upscale_to_original and layer_images:
            output_size = (layer_images[0].width, layer_images[0].height)
            if output_size != original_size:
                print(f"[EricQwenDecompose] Upscaling from {output_size} to {original_size}")
                
                # Map method names to PIL resampling filters
                resample_map = {
                    "lanczos": Image.LANCZOS,
                    "bicubic": Image.BICUBIC,
                    "bilinear": Image.BILINEAR,
                    "nearest": Image.NEAREST,
                }
                resample = resample_map.get(upscale_method, Image.LANCZOS)
                
                layer_images = [
                    img.resize(original_size, resample=resample)
                    for img in layer_images
                ]
        
        # Convert PIL layers to tensors
        layers_tensor, alpha_tensor = self._layers_to_tensors(layer_images)
        
        # Build layer info
        layer_info = {
            "count": len(layer_images),
            "original_width": original_size[0],
            "original_height": original_size[1],
            "output_width": layer_images[0].width if layer_images else 0,
            "output_height": layer_images[0].height if layer_images else 0,
            "processing_resolution": int(resolution),
            "upscaled": upscale_to_original,
            "source": "qwen-image-layered" if self.DIFFUSERS_AVAILABLE else "simulated",
            "layers": []
        }
        
        # First pass: analyze coverage for all layers
        layer_coverages = []
        for i, img in enumerate(layer_images):
            alpha_np = np.array(img.split()[-1]) if img.mode == "RGBA" else np.ones((img.height, img.width)) * 255
            coverage = (alpha_np > 10).sum() / alpha_np.size * 100
            has_transparency = (alpha_np < 255).any().item() if hasattr((alpha_np < 255).any(), 'item') else bool((alpha_np < 255).any())
            layer_coverages.append({
                "index": i,
                "coverage": coverage,
                "has_transparency": has_transparency,
                "alpha_np": alpha_np,
            })
        
        # Generate meaningful names based on coverage
        layer_names = self._generate_layer_names(layer_coverages)
        
        # If caption_layers is enabled, use vision model to get descriptive names
        if caption_layers and self.DIFFUSERS_AVAILABLE:
            try:
                caption_names = self._caption_layers(layer_images, int(resolution), model_source, use_en_prompt)
                if caption_names:
                    layer_names = caption_names
            except Exception as e:
                print(f"[EricQwenDecompose] Layer captioning failed, using coverage-based names: {e}")
        
        for i, info in enumerate(layer_coverages):
            layer_info["layers"].append({
                "index": i,
                "name": layer_names[i],
                "coverage_percent": round(info["coverage"], 2),
                "has_transparency": info["has_transparency"],
            })
        
        # Include auto_caption in layer_info for convenience
        layer_info["auto_caption"] = auto_caption
        
        # Unload model if keep_model_loaded is False
        if not keep_model_loaded:
            clear_qwen_pipeline()
        
        return (
            layers_tensor,
            alpha_tensor,
            len(layer_images),
            json.dumps(layer_info, indent=2),
            auto_caption
        )
    
    def _generate_layer_names(self, layer_coverages: List[dict]) -> List[str]:
        """
        Generate meaningful layer names based on coverage analysis.
        
        Naming logic:
        - Highest coverage (>90%) with transparency: Background
        - Highest coverage (>90%) without transparency: Base
        - Medium coverage (20-90%): Main_Subject, Subject_2, etc.
        - Low coverage (<20%): Element_1, Element_2, etc.
        - Very low coverage (<5%): Detail_1, Detail_2, etc.
        """
        names = []
        
        # Sort by coverage to identify roles
        sorted_layers = sorted(layer_coverages, key=lambda x: x["coverage"], reverse=True)
        
        background_assigned = False
        subject_count = 0
        element_count = 0
        detail_count = 0
        
        # Create a mapping from original index to name
        name_map = {}
        
        for layer in sorted_layers:
            idx = layer["index"]
            coverage = layer["coverage"]
            has_transparency = layer["has_transparency"]
            
            if coverage >= 90 and not background_assigned:
                # Highest coverage layer is likely background
                if has_transparency:
                    name_map[idx] = "Background"
                else:
                    name_map[idx] = "Base"
                background_assigned = True
            elif coverage >= 20:
                # Medium coverage - main subjects
                subject_count += 1
                if subject_count == 1:
                    name_map[idx] = "Main_Subject"
                else:
                    name_map[idx] = f"Subject_{subject_count}"
            elif coverage >= 5:
                # Lower coverage - elements
                element_count += 1
                name_map[idx] = f"Element_{element_count}"
            else:
                # Very low coverage - details
                detail_count += 1
                name_map[idx] = f"Detail_{detail_count}"
        
        # Build names list in original order
        for i in range(len(layer_coverages)):
            names.append(name_map.get(i, f"Layer_{i}"))
        
        return names
    
    def _caption_layers(
        self,
        layer_images: List[Image.Image],
        resolution: int,
        model_source: str,
        use_en_prompt: bool,
    ) -> List[str]:
        """
        Use the vision model to generate descriptive captions for each layer.
        
        Args:
            layer_images: List of RGBA PIL images (layers)
            resolution: Processing resolution for resizing
            model_source: Model source for pipeline
            use_en_prompt: Use English for captions
            
        Returns:
            List of descriptive layer names
        """
        import math
        
        pipeline = self._get_pipeline(model_source)
        device = next(pipeline.transformer.parameters()).device
        
        layer_names = []
        print(f"[EricQwenDecompose] Captioning {len(layer_images)} layers with vision model...")
        
        for i, layer_img in enumerate(layer_images):
            try:
                # Composite transparent layer onto white background for better captioning
                if layer_img.mode == "RGBA":
                    # Create white background
                    bg = Image.new("RGBA", layer_img.size, (255, 255, 255, 255))
                    composite = Image.alpha_composite(bg, layer_img)
                    caption_img = composite.convert("RGB")
                else:
                    caption_img = layer_img.convert("RGB") if layer_img.mode != "RGB" else layer_img
                
                # Resize to target resolution (same logic as main captioning)
                target_area = resolution * resolution
                ratio = caption_img.width / caption_img.height
                new_w = math.sqrt(target_area * ratio)
                new_h = new_w / ratio
                new_w = round(new_w / 32) * 32
                new_h = round(new_h / 32) * 32
                new_w = int(new_w)
                new_h = int(new_h)
                
                if (new_w, new_h) != (caption_img.width, caption_img.height):
                    caption_img = caption_img.resize((new_w, new_h), Image.LANCZOS)
                
                # Convert back to RGBA for the pipeline (it expects RGBA)
                caption_img = caption_img.convert("RGBA")
                
                # Get caption
                caption = pipeline.get_image_caption(
                    prompt_image=caption_img,
                    use_en_prompt=use_en_prompt,
                    device=device
                )
                
                # Extract a short name from the caption (first meaningful phrase)
                short_name = self._extract_short_name(caption, i)
                layer_names.append(short_name)
                print(f"[EricQwenDecompose]   Layer {i}: {short_name}")
                
            except Exception as e:
                print(f"[EricQwenDecompose]   Layer {i} captioning failed: {e}")
                layer_names.append(f"Layer_{i}")
        
        return layer_names
    
    def _extract_short_name(self, caption: str, layer_index: int) -> str:
        """
        Extract a short, filesystem-safe name from a long caption.
        
        Args:
            caption: Full caption text
            layer_index: Layer index for fallback naming
            
        Returns:
            Short name suitable for layer naming
        """
        if not caption or len(caption.strip()) == 0:
            return f"Layer_{layer_index}"
        
        # Take first sentence or phrase
        caption = caption.strip()
        
        # Split on common delimiters
        for delimiter in ['. ', ', ', ' - ', ': ']:
            if delimiter in caption:
                caption = caption.split(delimiter)[0]
                break
        
        # Limit length
        max_len = 40
        if len(caption) > max_len:
            # Try to break at word boundary
            caption = caption[:max_len].rsplit(' ', 1)[0]
        
        # Make filesystem-safe (keep alphanumeric, spaces become underscores)
        safe_name = ''.join(c if c.isalnum() or c == ' ' else '_' for c in caption)
        safe_name = '_'.join(safe_name.split())  # Replace spaces with underscores
        
        # Ensure not empty
        if not safe_name:
            return f"Layer_{layer_index}"
        
        return safe_name
    
    def _run_pipeline(
        self,
        image: Image.Image,
        num_layers: int,
        resolution: int,
        model_source: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        cfg_scale: float,
        steps: int,
        cfg_normalize: bool,
        use_en_prompt: bool,
    ) -> Tuple[List[Image.Image], str]:
        """Run the actual Qwen-Image-Layered pipeline.
        
        Returns:
            layer_images: List of RGBA PIL Images
            auto_caption: The caption generated by the pipeline's vision model
        """
        from diffusers import QwenImageLayeredPipeline
        import torch
        
        # Get or create pipeline
        pipeline = self._get_pipeline(model_source)
        
        # Get the auto-generated caption if no prompt provided
        auto_caption = ""
        if not prompt:
            # The pipeline has a get_image_caption method we can call directly
            try:
                device = next(pipeline.transformer.parameters()).device
                
                # Resize image for captioning using the SAME logic the pipeline uses internally
                # This avoids double-scaling and ensures caption matches what gets decomposed
                import math
                target_area = resolution * resolution  # 640*640 = 409600 or 1024*1024 = 1048576
                ratio = image.width / image.height
                
                # Calculate dimensions same as pipeline's calculate_dimensions()
                new_w = math.sqrt(target_area * ratio)
                new_h = new_w / ratio
                new_w = round(new_w / 32) * 32
                new_h = round(new_h / 32) * 32
                new_w = int(new_w)
                new_h = int(new_h)
                
                if (new_w, new_h) != (image.width, image.height):
                    caption_image = image.resize((new_w, new_h), Image.LANCZOS)
                    print(f"[EricQwenDecompose] Resized for captioning: {image.size} -> ({new_w}, {new_h})")
                else:
                    caption_image = image
                
                auto_caption = pipeline.get_image_caption(
                    prompt_image=caption_image,
                    use_en_prompt=use_en_prompt,
                    device=device
                )
                print(f"[EricQwenDecompose] Auto-caption: {auto_caption}")
            except Exception as e:
                print(f"[EricQwenDecompose] Could not get auto-caption: {e}")
                auto_caption = ""
        else:
            # User provided a prompt, use that
            auto_caption = prompt
        
        # Prepare inputs
        inputs = {
            "image": image,
            "generator": torch.Generator(device='cuda').manual_seed(seed),
            "true_cfg_scale": cfg_scale,
            "negative_prompt": negative_prompt if negative_prompt else " ",
            "num_inference_steps": steps,
            "num_images_per_prompt": 1,
            "layers": num_layers,
            "resolution": resolution,
            "cfg_normalize": cfg_normalize,
            "use_en_prompt": use_en_prompt,
        }
        
        if prompt:
            inputs["prompt"] = prompt
        
        # Run pipeline
        with torch.inference_mode():
            output = pipeline(**inputs)
            layer_images = output.images[0]  # List of PIL RGBA images
        
        return layer_images, auto_caption
    
    def _get_pipeline(self, model_source: str = "HuggingFace (Download)"):
        """Get or load the Qwen-Image-Layered pipeline."""
        from diffusers import QwenImageLayeredPipeline
        import torch
        
        global _QWEN_PIPELINE_CACHE
        
        # Create cache key based on model source
        cache_key = model_source
        
        # Check if pipeline is cached with same model
        if _QWEN_PIPELINE_CACHE["pipeline"] is not None:
            if _QWEN_PIPELINE_CACHE["source"] == cache_key:
                print(f"[EricQwenDecompose] Using cached pipeline")
                return _QWEN_PIPELINE_CACHE["pipeline"]
            else:
                # Clear old pipeline - different model requested
                print(f"[EricQwenDecompose] Clearing cached pipeline (switching models)")
                clear_qwen_pipeline()
        
        print(f"[EricQwenDecompose] Loading pipeline from: {model_source}")
        
        if model_source == "HuggingFace (Download)":
            # Load from HuggingFace
            pipeline = QwenImageLayeredPipeline.from_pretrained(
                "Qwen/Qwen-Image-Layered",
                torch_dtype=torch.bfloat16,
            )
        elif model_source.startswith("[Local Repo] "):
            # Load from local diffusers repo folder
            repo_name = model_source.replace("[Local Repo] ", "")
            diffusion_paths = folder_paths.get_folder_paths("diffusion_models")
            local_repo_path = None
            
            for base_path in diffusion_paths:
                full_path = os.path.join(base_path, repo_name)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "model_index.json")):
                    local_repo_path = full_path
                    break
            
            if local_repo_path is None:
                raise ValueError(f"Local repo not found: {repo_name}")
            
            print(f"[EricQwenDecompose] Loading from local repo: {local_repo_path}")
            
            # Load the full pipeline from local path
            pipeline = QwenImageLayeredPipeline.from_pretrained(
                local_repo_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        else:
            # Load from local safetensors file (just transformer weights)
            diffusion_paths = folder_paths.get_folder_paths("diffusion_models")
            local_model_path = None
            
            for base_path in diffusion_paths:
                full_path = os.path.join(base_path, model_source)
                if os.path.exists(full_path):
                    local_model_path = full_path
                    break
            
            if local_model_path is None:
                raise ValueError(f"Model file not found: {model_source}")
            
            print(f"[EricQwenDecompose] Using local transformer: {local_model_path}")
            
            # For local model, we need to load the pipeline with custom transformer
            # First load the base pipeline to get VAE/text encoder
            # Then replace the transformer with local weights
            from diffusers.models import QwenImageTransformer2DModel
            from safetensors.torch import load_file
            
            # Load base pipeline for other components (VAE, text encoder)
            pipeline = QwenImageLayeredPipeline.from_pretrained(
                "Qwen/Qwen-Image-Layered",
                torch_dtype=torch.bfloat16,
                transformer=None,  # Don't load transformer yet
            )
            
            # Load local transformer weights
            print(f"[EricQwenDecompose] Loading transformer weights from: {local_model_path}")
            state_dict = load_file(local_model_path)
            
            # Create transformer and load weights
            transformer = QwenImageTransformer2DModel.from_config(
                pipeline.transformer.config if hasattr(pipeline, 'transformer') and pipeline.transformer else 
                QwenImageTransformer2DModel._get_default_config(),
                torch_dtype=torch.bfloat16
            )
            transformer.load_state_dict(state_dict, strict=False)
            pipeline.transformer = transformer
        
        pipeline = pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=None)
        
        # Store in global cache
        _QWEN_PIPELINE_CACHE["pipeline"] = pipeline
        _QWEN_PIPELINE_CACHE["source"] = cache_key
        
        return pipeline
    
    def _simulate_layers(self, image: Image.Image, num_layers: int) -> List[Image.Image]:
        """
        Simulate layer decomposition for testing when diffusers is not available.
        Creates fake layers by extracting regions based on luminance/color.
        """
        import numpy as np
        
        img_array = np.array(image.convert("RGBA"))
        height, width = img_array.shape[:2]
        
        layers = []
        
        for i in range(num_layers):
            # Create a layer with partial transparency
            layer = np.zeros((height, width, 4), dtype=np.uint8)
            
            if i == 0:
                # Background layer - lower portion
                layer[:, :, :3] = img_array[:, :, :3]
                # Create gradient alpha from bottom
                alpha_gradient = np.linspace(255, 100, height).reshape(-1, 1)
                layer[:, :, 3] = np.broadcast_to(alpha_gradient, (height, width)).astype(np.uint8)
            else:
                # Other layers - horizontal bands with varying alpha
                band_height = height // num_layers
                start_y = i * band_height
                end_y = min((i + 1) * band_height, height)
                
                layer[start_y:end_y, :, :3] = img_array[start_y:end_y, :, :3]
                layer[start_y:end_y, :, 3] = 200  # Semi-transparent
            
            layers.append(Image.fromarray(layer, mode="RGBA"))
        
        return layers
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a ComfyUI image tensor to PIL Image."""
        # tensor is [H, W, C] with values 0-1
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if img_np.shape[-1] == 4:
            return Image.fromarray(img_np, mode="RGBA")
        elif img_np.shape[-1] == 3:
            return Image.fromarray(img_np, mode="RGB")
        else:
            return Image.fromarray(img_np.squeeze(), mode="L")
    
    def _layers_to_tensors(
        self,
        layers: List[Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of PIL RGBA images to ComfyUI tensors.
        
        Returns:
            layers_tensor: [N, H, W, 4] RGBA image batch
            alpha_tensor: [N, H, W] mask batch
        """
        if not layers:
            # Return empty tensors
            return torch.zeros((1, 64, 64, 4)), torch.zeros((1, 64, 64))
        
        layer_arrays = []
        alpha_arrays = []
        
        for layer in layers:
            # Ensure RGBA
            if layer.mode != "RGBA":
                layer = layer.convert("RGBA")
            
            # Convert to numpy
            layer_np = np.array(layer).astype(np.float32) / 255.0
            layer_arrays.append(layer_np)
            
            # Extract alpha
            alpha_np = layer_np[:, :, 3]
            alpha_arrays.append(alpha_np)
        
        # Stack into batches
        layers_tensor = torch.from_numpy(np.stack(layer_arrays, axis=0))
        alpha_tensor = torch.from_numpy(np.stack(alpha_arrays, axis=0))
        
        return layers_tensor, alpha_tensor


class EricQwenUnloadModel:
    """
    Unload the Qwen-Image-Layered pipeline from VRAM.
    
    Use this node to free up GPU memory when you're done with layer decomposition.
    Can be connected anywhere in your workflow as an output node.
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {"tooltip": "Connect any output here to trigger unload after that node completes"}),
            }
        }
    
    def unload(self, trigger=None):
        """Unload the pipeline and free VRAM."""
        if clear_qwen_pipeline():
            return ("Pipeline unloaded successfully",)
        else:
            return ("No pipeline was loaded",)


# For testing
if __name__ == "__main__":
    node = EricQwenDecompose()
    print(f"Diffusers available: {node.DIFFUSERS_AVAILABLE}")
    print(f"Input types: {node.INPUT_TYPES()}")
