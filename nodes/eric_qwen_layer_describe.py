"""
Eric Qwen Layer Describe Node
Uses a vision model to generate descriptions for each layer.

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

import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple


class EricQwenLayerDescribe:
    """
    Generate layer names/descriptions using a vision model.
    
    Takes an IMAGE batch (layers) and processes each through a vision model
    to generate descriptive names. Outputs a multiline string for layer_names.
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "describe_layers"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("layer_names", "layer_descriptions")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE",),
                "prompt_template": ("STRING", {
                    "default": "Describe this image layer in 2-4 words for use as a layer name. Focus on the main subject or content visible. Be concise.",
                    "multiline": True,
                    "tooltip": "Prompt template for the vision model"
                }),
                "name_style": (["short", "descriptive", "numbered"], {
                    "default": "short",
                    "tooltip": "short=2-3 words, descriptive=full sentence, numbered=Name_01 style"
                }),
            },
            "optional": {
                "vision_model": ("VISION_MODEL",),
                "prefix": ("STRING", {
                    "default": "",
                    "tooltip": "Prefix for each layer name"
                }),
            }
        }
    
    def describe_layers(
        self,
        layers: torch.Tensor,
        prompt_template: str,
        name_style: str = "short",
        vision_model=None,
        prefix: str = "",
    ) -> Tuple[str, str]:
        """
        Generate descriptions for each layer.
        
        If no vision_model is provided, generates names based on layer analysis.
        """
        num_layers = layers.shape[0]
        names = []
        descriptions = []
        
        for i in range(num_layers):
            layer = layers[i]
            
            if vision_model is not None:
                # Use vision model to describe the layer
                desc = self._describe_with_model(layer, vision_model, prompt_template)
            else:
                # Fallback: analyze layer content
                desc = self._analyze_layer(layer, i)
            
            descriptions.append(desc)
            
            # Format name based on style
            if name_style == "short":
                # Take first 3-4 words, clean up
                words = desc.split()[:4]
                name = "_".join(w.capitalize() for w in words if w.isalnum() or w.replace("'", "").isalnum())
                name = name[:30]  # Limit length
            elif name_style == "numbered":
                # Simple numbered format
                name = f"Layer_{i:02d}"
            else:  # descriptive
                name = desc[:50]  # Limit length
            
            if prefix:
                name = f"{prefix}_{name}"
            
            names.append(name)
        
        # Join with newlines for layer_names input
        layer_names_str = "\n".join(names)
        layer_descriptions_str = "\n---\n".join(descriptions)
        
        return (layer_names_str, layer_descriptions_str)
    
    def _describe_with_model(self, layer: torch.Tensor, model, prompt: str) -> str:
        """Use a vision model to describe the layer."""
        # Convert tensor to PIL
        if layer.shape[-1] == 4:
            # RGBA
            img_np = (layer.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np, mode="RGBA")
        else:
            img_np = (layer.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np, mode="RGB")
        
        # This is a placeholder - actual implementation depends on the vision model type
        # Common ComfyUI vision models: LLaVA, Florence, BLIP, etc.
        try:
            # Try common interfaces
            if hasattr(model, 'generate'):
                # LLaVA-style
                result = model.generate(image=img, prompt=prompt)
                return result if isinstance(result, str) else str(result)
            elif hasattr(model, 'caption'):
                # BLIP-style
                result = model.caption(img)
                return result if isinstance(result, str) else str(result)
            elif hasattr(model, '__call__'):
                # Callable model
                result = model(img, prompt)
                return result if isinstance(result, str) else str(result)
            else:
                return self._analyze_layer(layer, 0)
        except Exception as e:
            print(f"[EricQwenLayerDescribe] Vision model error: {e}")
            return self._analyze_layer(layer, 0)
    
    def _analyze_layer(self, layer: torch.Tensor, index: int) -> str:
        """Analyze layer content when no vision model is available."""
        layer_np = layer.cpu().numpy()
        
        # Check alpha coverage
        if layer_np.shape[-1] == 4:
            alpha = layer_np[:, :, 3]
            coverage = (alpha > 0.04).sum() / alpha.size
            avg_alpha = alpha.mean()
        else:
            coverage = 1.0
            avg_alpha = 1.0
        
        # Analyze colors
        if layer_np.shape[-1] >= 3:
            rgb = layer_np[:, :, :3]
            # Get average color where visible
            if layer_np.shape[-1] == 4:
                mask = alpha > 0.04
                if mask.sum() > 0:
                    avg_r = rgb[:, :, 0][mask].mean()
                    avg_g = rgb[:, :, 1][mask].mean()
                    avg_b = rgb[:, :, 2][mask].mean()
                else:
                    avg_r = avg_g = avg_b = 0
            else:
                avg_r, avg_g, avg_b = rgb.mean(axis=(0, 1))
        else:
            avg_r = avg_g = avg_b = 0.5
        
        # Generate description based on analysis
        if coverage < 0.02:
            return f"Empty transparent layer {index}"
        elif coverage > 0.95 and avg_alpha > 0.9:
            # Describe dominant color
            if avg_r > 0.6 and avg_g > 0.6 and avg_b > 0.6:
                return f"Light background layer"
            elif avg_r < 0.3 and avg_g < 0.3 and avg_b < 0.3:
                return f"Dark background layer"
            elif avg_r > avg_g and avg_r > avg_b:
                return f"Warm background layer"
            elif avg_b > avg_r and avg_b > avg_g:
                return f"Cool background layer"
            else:
                return f"Full background layer"
        elif coverage > 0.5:
            return f"Main subject layer {index}"
        elif coverage > 0.2:
            return f"Foreground element layer {index}"
        else:
            return f"Detail overlay layer {index}"


class EricQwenStringJoin:
    """
    Join multiple strings into a multiline string.
    Useful for collecting layer names from multiple sources.
    """
    
    CATEGORY = "Eric Qwen Layer/Utils"
    FUNCTION = "join_strings"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {
                    "default": "\n",
                    "tooltip": "Separator between strings (\\n for newline)"
                }),
            },
            "optional": {
                "string_1": ("STRING", {"forceInput": True}),
                "string_2": ("STRING", {"forceInput": True}),
                "string_3": ("STRING", {"forceInput": True}),
                "string_4": ("STRING", {"forceInput": True}),
                "string_5": ("STRING", {"forceInput": True}),
                "string_6": ("STRING", {"forceInput": True}),
                "string_7": ("STRING", {"forceInput": True}),
                "string_8": ("STRING", {"forceInput": True}),
            }
        }
    
    def join_strings(
        self,
        separator: str = "\n",
        string_1: str = "",
        string_2: str = "",
        string_3: str = "",
        string_4: str = "",
        string_5: str = "",
        string_6: str = "",
        string_7: str = "",
        string_8: str = "",
    ) -> Tuple[str]:
        # Handle escape sequences
        if separator == "\\n":
            separator = "\n"
        elif separator == "\\t":
            separator = "\t"
        
        # Collect non-empty strings
        strings = [s for s in [string_1, string_2, string_3, string_4, 
                               string_5, string_6, string_7, string_8] if s]
        
        return (separator.join(strings),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "EricQwenLayerDescribe": EricQwenLayerDescribe,
    "EricQwenStringJoin": EricQwenStringJoin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EricQwenLayerDescribe": "Eric Qwen Layer Describe (Vision)",
    "EricQwenStringJoin": "Eric Qwen String Join",
}
