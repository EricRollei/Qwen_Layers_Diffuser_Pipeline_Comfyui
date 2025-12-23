"""Eric Qwen Native Nodes
Native ComfyUI workflow nodes for Qwen-Image-Edit and layer handling.

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
import json
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any

import folder_paths
import comfy.utils
import comfy.model_management


class EricQwenAddAlpha:
    """
    Add alpha channel to RGB images for RGBA VAE compatibility.
    
    The Qwen-Image-Layered VAE requires 4-channel (RGBA) input,
    but LoadImage outputs 3-channel RGB. This node adds an alpha
    channel to make images compatible with the Qwen Layered VAE.
    """
    
    CATEGORY = "Eric Qwen Layer/Native"
    FUNCTION = "add_alpha"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba_image",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "alpha_value": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Alpha value if no mask provided (1.0 = fully opaque)"
                }),
            }
        }
    
    def add_alpha(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        alpha_value: float = 1.0,
    ) -> Tuple[torch.Tensor]:
        """
        Add alpha channel to RGB image.
        
        Args:
            image: RGB image tensor [B, H, W, 3]
            mask: Optional mask to use as alpha [B, H, W] or [H, W]
            alpha_value: Default alpha if no mask (1.0 = opaque)
            
        Returns:
            RGBA image tensor [B, H, W, 4]
        """
        batch_size, height, width, channels = image.shape
        
        if channels == 4:
            # Already RGBA
            return (image,)
        
        if channels != 3:
            raise ValueError(f"Expected RGB (3 channels) or RGBA (4 channels), got {channels}")
        
        # Create alpha channel
        if mask is not None:
            # Use provided mask
            if mask.dim() == 2:
                # Single mask [H, W] -> expand to batch
                alpha = mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Batch mask [B, H, W]
                alpha = mask
            
            # Ensure mask matches image dimensions
            if alpha.shape[1:] != (height, width):
                alpha = torch.nn.functional.interpolate(
                    alpha.unsqueeze(1),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
        else:
            # Create solid alpha
            alpha = torch.full(
                (batch_size, height, width),
                alpha_value,
                device=image.device,
                dtype=image.dtype
            )
        
        # Concatenate alpha channel
        rgba = torch.cat([image, alpha.unsqueeze(-1)], dim=-1)
        
        return (rgba,)


class EricQwenEncode:
    """
    Qwen-Image encoding node for native ComfyUI workflows.
    
    Creates conditioning for Qwen-Image-Edit models by combining
    text prompt with reference image latents.
    
    This is similar to ComfyUI's TextEncodeQwenImageEdit but with
    additional options and layer-focused defaults.
    """
    
    CATEGORY = "Eric Qwen Layer/Native"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive_prompt": ("STRING", {
                    "default": "decompose this image into separate layers with transparent backgrounds",
                    "multiline": True,
                    "tooltip": "Prompt for layer decomposition"
                }),
                "negative_prompt": ("STRING", {
                    "default": "blurry, low quality",
                    "multiline": True,
                    "tooltip": "Negative prompt"
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
            }
        }
    
    def encode(
        self,
        clip,
        positive_prompt: str,
        negative_prompt: str,
        vae=None,
        image=None,
    ):
        """Encode prompts and optional reference image for Qwen-Image."""
        import node_helpers
        import math
        
        # Process reference image if provided
        ref_latent = None
        images = []
        
        if image is not None:
            # Scale image for vision processing
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)
            
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            processed_image = s.movedim(1, -1)
            images = [processed_image[:, :, :, :3]]
            
            # Encode to latent if VAE provided
            if vae is not None:
                ref_latent = vae.encode(processed_image[:, :, :, :3])
        
        # Encode positive prompt
        tokens = clip.tokenize(positive_prompt, images=images)
        positive_cond = clip.encode_from_tokens_scheduled(tokens)
        
        if ref_latent is not None:
            positive_cond = node_helpers.conditioning_set_values(
                positive_cond, 
                {"reference_latents": [ref_latent]}, 
                append=True
            )
        
        # Encode negative prompt (no images)
        neg_tokens = clip.tokenize(negative_prompt, images=[])
        negative_cond = clip.encode_from_tokens_scheduled(neg_tokens)
        
        return (positive_cond, negative_cond)


class EricQwenLayerExtract:
    """
    Extract layers from a video/batch tensor output.
    
    Qwen-Image models (when sampled with multiple frames) output
    layers as video frames. This node converts that format to
    proper layer format with RGBA support.
    
    Use this after VAE decoding to convert video frames to layers.
    """
    
    CATEGORY = "Eric Qwen Layer/Native"
    FUNCTION = "extract"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("layers", "alpha_masks", "layer_count", "layer_info")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of images [B, H, W, C]
                "treat_as": (["layers", "video_frames", "auto_detect"], {
                    "default": "layers",
                    "tooltip": "How to interpret the batch dimension"
                }),
            },
            "optional": {
                "alpha_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Threshold for generating alpha masks from RGB"
                }),
                "generate_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate alpha from dark regions if no alpha channel"
                }),
            }
        }
    
    def extract(
        self,
        images: torch.Tensor,
        treat_as: str = "layers",
        alpha_threshold: float = 0.1,
        generate_alpha: bool = True,
    ):
        """Extract and format layers from batch tensor."""
        import json
        
        batch_size = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        channels = images.shape[3]
        
        # Determine if this is layers or video
        is_layers = treat_as == "layers" or (
            treat_as == "auto_detect" and batch_size <= 10
        )
        
        # Process each image in batch
        layers = []
        alpha_masks = []
        layer_info = {
            "count": batch_size,
            "width": width,
            "height": height,
            "interpretation": "layers" if is_layers else "video_frames",
            "layers": []
        }
        
        for i in range(batch_size):
            img = images[i]  # [H, W, C]
            
            # Handle alpha channel
            if channels == 4:
                # Already has alpha
                rgb = img[:, :, :3]
                alpha = img[:, :, 3]
            else:
                # RGB only - generate alpha if requested
                rgb = img
                if generate_alpha:
                    # Generate alpha based on luminance (dark = transparent)
                    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
                    alpha = (luminance > alpha_threshold).float()
                else:
                    alpha = torch.ones((height, width), device=img.device)
            
            # Create RGBA layer
            rgba = torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1)
            layers.append(rgba)
            alpha_masks.append(alpha)
            
            # Analyze layer
            alpha_np = alpha.cpu().numpy()
            coverage = (alpha_np > 0.1).sum() / alpha_np.size * 100
            
            layer_info["layers"].append({
                "index": i,
                "name": f"Layer_{i}" if is_layers else f"Frame_{i}",
                "coverage_percent": round(coverage, 2),
                "has_transparency": bool((alpha_np < 0.99).any()),
            })
        
        # Stack into tensors
        layers_tensor = torch.stack(layers, dim=0)
        alpha_tensor = torch.stack(alpha_masks, dim=0)
        
        return (
            layers_tensor,
            alpha_tensor,
            batch_size,
            json.dumps(layer_info, indent=2)
        )


class EricQwenMultiLatent:
    """
    Create multi-frame latent for Qwen layer generation.
    
    Qwen-Image-Layered works by generating multiple frames where
    each frame represents a layer. This node prepares the latent
    tensor with the correct shape for layer generation.
    """
    
    CATEGORY = "Eric Qwen Layer/Native"
    FUNCTION = "create_latent"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                }),
                "num_layers": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of layers to generate"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
            }
        }
    
    def create_latent(
        self,
        width: int,
        height: int,
        num_layers: int,
        batch_size: int = 1,
    ):
        """Create empty latent tensor for layer generation."""
        # Qwen uses 16 latent channels and 8x compression
        latent_height = height // 8
        latent_width = width // 8
        latent_channels = 16
        
        # Shape: [batch, frames, channels, height, width]
        # For layers: frames = num_layers + 1 (composite + individual layers)
        latent = torch.zeros(
            (batch_size, num_layers + 1, latent_channels, latent_height, latent_width),
            device=comfy.model_management.intermediate_device()
        )
        
        return ({"samples": latent},)


class EricQwenLatentCutToBatch:
    """
    Convert video/multi-frame latent to batch format.
    
    This node is essential for Qwen-Image-Layered workflow.
    After KSampler outputs a video latent [B, F, C, H, W],
    this converts it to batch format [B*F, C, H, W] for VAE decoding.
    
    Equivalent to the LatentCutToBatch node used in reference workflows.
    """
    
    CATEGORY = "Eric Qwen Layer/Native"
    FUNCTION = "cut_to_batch"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "skip_first": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip the first frame (composite) and only output layer frames"
                }),
            }
        }
    
    def cut_to_batch(self, latent: dict, skip_first: bool = False):
        """
        Convert video latent [B, F, C, H, W] to batch [B*F, C, H, W].
        
        The Qwen-Image-Layered model outputs layers as video frames.
        This reshapes them into a batch for VAEDecode.
        """
        samples = latent["samples"]
        
        # Check if this is a 5D video latent [B, F, C, H, W]
        if samples.dim() == 5:
            batch, frames, channels, height, width = samples.shape
            
            if skip_first and frames > 1:
                # Skip composite frame, only keep layer frames
                samples = samples[:, 1:, :, :, :]
                frames = frames - 1
            
            # Reshape: [B, F, C, H, W] -> [B*F, C, H, W]
            samples = samples.reshape(batch * frames, channels, height, width)
        
        elif samples.dim() == 4:
            # Already in batch format [B, C, H, W], nothing to do
            if skip_first and samples.shape[0] > 1:
                samples = samples[1:]
        
        else:
            raise ValueError(f"Unexpected latent dimensions: {samples.dim()}")
        
        return ({"samples": samples},)


class EricQwenBatchToVideo:
    """
    Convert batch latent back to video format.
    
    Reverse of LatentCutToBatch - converts [B, C, H, W] to [1, B, C, H, W].
    Useful if you need to process layers back through video-aware nodes.
    """
    
    CATEGORY = "Eric Qwen Layer/Native"
    FUNCTION = "batch_to_video"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }
    
    def batch_to_video(self, latent: dict):
        """Convert batch [B, C, H, W] to video [1, B, C, H, W]."""
        samples = latent["samples"]
        
        if samples.dim() == 4:
            # Add frame dimension: [B, C, H, W] -> [1, B, C, H, W]
            samples = samples.unsqueeze(0)
        
        return ({"samples": samples},)


class EricQwenLayerPrompts:
    """
    Generate layer-specific prompts for decomposition.
    
    Provides pre-built prompts optimized for layer decomposition
    with Qwen-Image-Edit models.
    """
    
    CATEGORY = "Eric Qwen Layer/Native"
    FUNCTION = "get_prompts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([
                    "layer_decomposition",
                    "foreground_background",
                    "subject_extraction",
                    "text_extraction",
                    "color_separation",
                    "custom"
                ], {
                    "default": "layer_decomposition",
                }),
            },
            "optional": {
                "custom_positive": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "custom_negative": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "num_layers": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 10,
                }),
            }
        }
    
    PROMPTS = {
        "layer_decomposition": (
            "Decompose this image into {n} separate transparent layers. "
            "Each layer should contain distinct elements with clear boundaries. "
            "Background layer first, then foreground elements in order of depth.",
            "blurry, overlapping, merged layers, opaque backgrounds"
        ),
        "foreground_background": (
            "Separate the main subject from the background. "
            "Create a clean background layer and a foreground subject layer with transparency.",
            "blurry edges, halo artifacts, incomplete separation"
        ),
        "subject_extraction": (
            "Extract each distinct subject or object as a separate layer with transparent background. "
            "Maintain original colors and details.",
            "merged subjects, missing objects, color bleeding"
        ),
        "text_extraction": (
            "Extract any text or typography as a separate layer with transparent background. "
            "Keep text sharp and legible.",
            "blurry text, broken letters, background noise"
        ),
        "color_separation": (
            "Separate elements by their dominant colors into distinct layers. "
            "Each layer contains elements of similar color with transparency.",
            "mixed colors, gradient splitting"
        ),
    }
    
    def get_prompts(
        self,
        preset: str,
        custom_positive: str = "",
        custom_negative: str = "",
        num_layers: int = 4,
    ):
        """Get prompts for the selected preset."""
        if preset == "custom":
            return (custom_positive, custom_negative)
        
        positive, negative = self.PROMPTS.get(preset, self.PROMPTS["layer_decomposition"])
        positive = positive.format(n=num_layers)
        
        return (positive, negative)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "EricQwenAddAlpha": EricQwenAddAlpha,
    "EricQwenEncode": EricQwenEncode,
    "EricQwenLayerExtract": EricQwenLayerExtract,
    "EricQwenMultiLatent": EricQwenMultiLatent,
    "EricQwenLayerPrompts": EricQwenLayerPrompts,
    "EricQwenLatentCutToBatch": EricQwenLatentCutToBatch,
    "EricQwenBatchToVideo": EricQwenBatchToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EricQwenAddAlpha": "Eric Qwen Add Alpha (RGBA)",
    "EricQwenEncode": "Eric Qwen Encode (Native)",
    "EricQwenLayerExtract": "Eric Qwen Layer Extract",
    "EricQwenMultiLatent": "Eric Qwen Multi-Layer Latent",
    "EricQwenLayerPrompts": "Eric Qwen Layer Prompts",
    "EricQwenLatentCutToBatch": "Eric Qwen Latent Cut to Batch",
    "EricQwenBatchToVideo": "Eric Qwen Batch to Video",
}
