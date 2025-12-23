"""Eric Qwen Layer Utilities
Nodes for manipulating layers from Qwen-Image-Layered output.

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
from typing import Optional, List, Tuple, Dict, Any
import json


class EricQwenLayerSelector:
    """
    Select specific layers from a layer batch.
    Can select by index, range, or pick multiple specific layers.
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "select_layers"
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("selected_layers", "selected_masks", "count")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE",),
                "selection_mode": (["single", "range", "pick_multiple", "all_except"], {
                    "default": "single",
                    "tooltip": "How to select layers"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99,
                    "tooltip": "Layer index for single selection (0 = first layer)"
                }),
            },
            "optional": {
                "alpha_masks": ("MASK",),
                "end_index": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 99,
                    "tooltip": "End index for range selection"
                }),
                "indices_csv": ("STRING", {
                    "default": "0,1,2",
                    "tooltip": "Comma-separated indices for pick_multiple mode"
                }),
            }
        }
    
    def select_layers(
        self,
        layers: torch.Tensor,
        selection_mode: str = "single",
        index: int = 0,
        alpha_masks: Optional[torch.Tensor] = None,
        end_index: int = 1,
        indices_csv: str = "0,1,2",
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Select layers based on mode."""
        
        num_layers = layers.shape[0]
        
        if selection_mode == "single":
            # Single layer
            idx = min(index, num_layers - 1)
            selected = layers[idx:idx+1]
            
        elif selection_mode == "range":
            # Range of layers
            start = min(index, num_layers - 1)
            end = min(end_index + 1, num_layers)  # +1 because end is inclusive
            selected = layers[start:end]
            
        elif selection_mode == "pick_multiple":
            # Pick specific indices
            try:
                indices = [int(i.strip()) for i in indices_csv.split(",") if i.strip()]
                indices = [i for i in indices if 0 <= i < num_layers]
                if not indices:
                    indices = [0]
                selected = layers[indices]
            except ValueError:
                selected = layers[0:1]
                
        elif selection_mode == "all_except":
            # All layers except the specified index
            idx = min(index, num_layers - 1)
            indices = [i for i in range(num_layers) if i != idx]
            if not indices:
                indices = [0]
            selected = layers[indices]
        else:
            selected = layers
        
        # Handle alpha masks
        if alpha_masks is not None:
            if selection_mode == "single":
                idx = min(index, alpha_masks.shape[0] - 1)
                selected_masks = alpha_masks[idx:idx+1]
            elif selection_mode == "range":
                start = min(index, alpha_masks.shape[0] - 1)
                end = min(end_index + 1, alpha_masks.shape[0])
                selected_masks = alpha_masks[start:end]
            elif selection_mode == "pick_multiple":
                try:
                    indices = [int(i.strip()) for i in indices_csv.split(",") if i.strip()]
                    indices = [i for i in indices if 0 <= i < alpha_masks.shape[0]]
                    if not indices:
                        indices = [0]
                    selected_masks = alpha_masks[indices]
                except ValueError:
                    selected_masks = alpha_masks[0:1]
            elif selection_mode == "all_except":
                idx = min(index, alpha_masks.shape[0] - 1)
                indices = [i for i in range(alpha_masks.shape[0]) if i != idx]
                if not indices:
                    indices = [0]
                selected_masks = alpha_masks[indices]
            else:
                selected_masks = alpha_masks
        else:
            # Extract alpha from RGBA layers
            if selected.shape[-1] == 4:
                selected_masks = selected[:, :, :, 3]
            else:
                selected_masks = torch.ones(selected.shape[0], selected.shape[1], selected.shape[2])
        
        return (selected, selected_masks, selected.shape[0])


class EricQwenLayerReorder:
    """
    Reorder layers in a batch.
    Can reverse, rotate, or specify custom order.
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "reorder_layers"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("reordered_layers", "reordered_masks")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE",),
                "mode": (["reverse", "rotate_up", "rotate_down", "custom", "shuffle"], {
                    "default": "reverse",
                    "tooltip": "Reordering mode"
                }),
            },
            "optional": {
                "alpha_masks": ("MASK",),
                "custom_order": ("STRING", {
                    "default": "0,1,2,3",
                    "tooltip": "Custom order as comma-separated indices"
                }),
                "rotate_amount": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "tooltip": "How many positions to rotate"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Seed for shuffle mode"
                }),
            }
        }
    
    def reorder_layers(
        self,
        layers: torch.Tensor,
        mode: str = "reverse",
        alpha_masks: Optional[torch.Tensor] = None,
        custom_order: str = "0,1,2,3",
        rotate_amount: int = 1,
        seed: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reorder layers."""
        
        num_layers = layers.shape[0]
        
        if mode == "reverse":
            indices = list(range(num_layers - 1, -1, -1))
            
        elif mode == "rotate_up":
            # Move layers up (first becomes last)
            amount = rotate_amount % num_layers
            indices = list(range(amount, num_layers)) + list(range(amount))
            
        elif mode == "rotate_down":
            # Move layers down (last becomes first)
            amount = rotate_amount % num_layers
            indices = list(range(num_layers - amount, num_layers)) + list(range(num_layers - amount))
            
        elif mode == "custom":
            try:
                indices = [int(i.strip()) for i in custom_order.split(",") if i.strip()]
                # Validate and clamp indices
                indices = [max(0, min(i, num_layers - 1)) for i in indices]
                if not indices:
                    indices = list(range(num_layers))
            except ValueError:
                indices = list(range(num_layers))
                
        elif mode == "shuffle":
            # Seeded random shuffle
            rng = np.random.RandomState(seed)
            indices = list(range(num_layers))
            rng.shuffle(indices)
        else:
            indices = list(range(num_layers))
        
        # Reorder layers
        reordered = layers[indices]
        
        # Handle masks
        if alpha_masks is not None:
            # Ensure indices are valid for mask tensor
            mask_indices = [min(i, alpha_masks.shape[0] - 1) for i in indices]
            reordered_masks = alpha_masks[mask_indices]
        else:
            if reordered.shape[-1] == 4:
                reordered_masks = reordered[:, :, :, 3]
            else:
                reordered_masks = torch.ones(reordered.shape[0], reordered.shape[1], reordered.shape[2])
        
        return (reordered, reordered_masks)


class EricQwenLayerComposite:
    """
    Composite layers together using various blend modes.
    Stacks layers from bottom to top with proper alpha blending.
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "composite_layers"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composite", "composite_alpha")
    
    BLEND_MODES = [
        "normal",
        "multiply", 
        "screen",
        "overlay",
        "darken",
        "lighten",
        "color_dodge",
        "color_burn",
        "hard_light",
        "soft_light",
        "difference",
        "exclusion",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE",),
                "blend_mode": (cls.BLEND_MODES, {
                    "default": "normal",
                    "tooltip": "Blend mode for compositing layers"
                }),
            },
            "optional": {
                "alpha_masks": ("MASK",),
                "background_color": (["transparent", "white", "black", "gray"], {
                    "default": "transparent",
                    "tooltip": "Background for the composite"
                }),
                "global_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Global opacity applied to all layers"
                }),
            }
        }
    
    def composite_layers(
        self,
        layers: torch.Tensor,
        blend_mode: str = "normal",
        alpha_masks: Optional[torch.Tensor] = None,
        background_color: str = "transparent",
        global_opacity: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Composite all layers into a single image."""
        
        if layers.shape[0] == 0:
            return (torch.zeros((1, 64, 64, 4)), torch.zeros((1, 64, 64)))
        
        height, width = layers.shape[1], layers.shape[2]
        num_channels = layers.shape[3] if len(layers.shape) > 3 else 3
        
        # Initialize background
        if background_color == "white":
            composite = torch.ones((height, width, 4))
        elif background_color == "black":
            composite = torch.zeros((height, width, 4))
            composite[:, :, 3] = 1.0  # Full alpha
        elif background_color == "gray":
            composite = torch.ones((height, width, 4)) * 0.5
            composite[:, :, 3] = 1.0
        else:  # transparent
            composite = torch.zeros((height, width, 4))
        
        # Composite each layer from bottom to top
        for i in range(layers.shape[0]):
            layer = layers[i]
            
            # Get alpha
            if num_channels == 4:
                layer_alpha = layer[:, :, 3] * global_opacity
                layer_rgb = layer[:, :, :3]
            else:
                if alpha_masks is not None and i < alpha_masks.shape[0]:
                    layer_alpha = alpha_masks[i] * global_opacity
                else:
                    layer_alpha = torch.ones((height, width)) * global_opacity
                layer_rgb = layer[:, :, :3] if num_channels >= 3 else layer.unsqueeze(-1).repeat(1, 1, 3)
            
            # Apply blend mode
            blended_rgb = self._blend(composite[:, :, :3], layer_rgb, blend_mode)
            
            # Alpha compositing: result = src * src_alpha + dst * dst_alpha * (1 - src_alpha)
            src_alpha = layer_alpha.unsqueeze(-1)
            dst_alpha = composite[:, :, 3:4]
            
            out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
            out_alpha_safe = torch.clamp(out_alpha, min=1e-8)
            
            out_rgb = (blended_rgb * src_alpha + composite[:, :, :3] * dst_alpha * (1 - src_alpha)) / out_alpha_safe
            
            composite[:, :, :3] = out_rgb
            composite[:, :, 3:4] = out_alpha
        
        # Add batch dimension
        composite = composite.unsqueeze(0)
        composite_alpha = composite[:, :, :, 3]
        
        return (composite, composite_alpha)
    
    def _blend(self, dst: torch.Tensor, src: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply blend mode between destination and source."""
        
        if mode == "normal":
            return src
            
        elif mode == "multiply":
            return dst * src
            
        elif mode == "screen":
            return 1 - (1 - dst) * (1 - src)
            
        elif mode == "overlay":
            mask = dst < 0.5
            result = torch.zeros_like(dst)
            result[mask] = 2 * dst[mask] * src[mask]
            result[~mask] = 1 - 2 * (1 - dst[~mask]) * (1 - src[~mask])
            return result
            
        elif mode == "darken":
            return torch.min(dst, src)
            
        elif mode == "lighten":
            return torch.max(dst, src)
            
        elif mode == "color_dodge":
            return torch.clamp(dst / (1 - src + 1e-8), 0, 1)
            
        elif mode == "color_burn":
            return 1 - torch.clamp((1 - dst) / (src + 1e-8), 0, 1)
            
        elif mode == "hard_light":
            mask = src < 0.5
            result = torch.zeros_like(dst)
            result[mask] = 2 * dst[mask] * src[mask]
            result[~mask] = 1 - 2 * (1 - dst[~mask]) * (1 - src[~mask])
            return result
            
        elif mode == "soft_light":
            return (1 - 2 * src) * dst * dst + 2 * src * dst
            
        elif mode == "difference":
            return torch.abs(dst - src)
            
        elif mode == "exclusion":
            return dst + src - 2 * dst * src
            
        else:
            return src


class EricQwenLayerInfo:
    """
    Display information about layers in a batch.
    Outputs detailed statistics and generates a preview.
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "analyze_layers"
    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("layer_info_json", "layer_preview", "layer_count")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE",),
            },
            "optional": {
                "alpha_masks": ("MASK",),
                "preview_mode": (["grid", "stack_vertical", "stack_horizontal", "thumbnails"], {
                    "default": "grid",
                    "tooltip": "How to display layer preview"
                }),
                "thumbnail_size": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 512,
                    "tooltip": "Size for thumbnail previews"
                }),
            }
        }
    
    def analyze_layers(
        self,
        layers: torch.Tensor,
        alpha_masks: Optional[torch.Tensor] = None,
        preview_mode: str = "grid",
        thumbnail_size: int = 128,
    ) -> Tuple[str, torch.Tensor, int]:
        """Analyze and visualize layers."""
        
        num_layers = layers.shape[0]
        height = layers.shape[1]
        width = layers.shape[2]
        channels = layers.shape[3] if len(layers.shape) > 3 else 1
        
        # Build info dict
        info = {
            "total_layers": num_layers,
            "dimensions": {
                "width": width,
                "height": height,
                "channels": channels,
            },
            "has_alpha": channels == 4,
            "layers": []
        }
        
        for i in range(num_layers):
            layer = layers[i]
            
            # Get alpha channel
            if channels == 4:
                alpha = layer[:, :, 3].numpy()
            elif alpha_masks is not None and i < alpha_masks.shape[0]:
                alpha = alpha_masks[i].numpy()
            else:
                alpha = np.ones((height, width))
            
            # Calculate statistics
            rgb = layer[:, :, :3].numpy() if channels >= 3 else layer.numpy()
            
            layer_info = {
                "index": i,
                "name": f"Layer_{i}",
                "alpha_stats": {
                    "min": float(alpha.min()),
                    "max": float(alpha.max()),
                    "mean": float(alpha.mean()),
                    "coverage_percent": float((alpha > 0.1).sum() / alpha.size * 100),
                    "fully_opaque_percent": float((alpha > 0.99).sum() / alpha.size * 100),
                },
                "rgb_stats": {
                    "mean": [float(rgb[:, :, c].mean()) for c in range(min(3, rgb.shape[-1]))],
                    "std": [float(rgb[:, :, c].std()) for c in range(min(3, rgb.shape[-1]))],
                },
            }
            info["layers"].append(layer_info)
        
        # Generate preview
        preview = self._generate_preview(layers, alpha_masks, preview_mode, thumbnail_size)
        
        return (json.dumps(info, indent=2), preview, num_layers)
    
    def _generate_preview(
        self,
        layers: torch.Tensor,
        alpha_masks: Optional[torch.Tensor],
        mode: str,
        thumb_size: int,
    ) -> torch.Tensor:
        """Generate a visual preview of all layers."""
        
        num_layers = layers.shape[0]
        if num_layers == 0:
            return torch.zeros((1, thumb_size, thumb_size, 4))
        
        height = layers.shape[1]
        width = layers.shape[2]
        
        # Resize layers to thumbnails
        thumbs = []
        for i in range(num_layers):
            layer = layers[i]
            
            # Convert to PIL for resizing
            if layer.shape[-1] == 4:
                img = Image.fromarray((layer.numpy() * 255).astype(np.uint8), mode="RGBA")
            else:
                img = Image.fromarray((layer.numpy() * 255).astype(np.uint8)[:, :, :3], mode="RGB")
            
            # Resize maintaining aspect ratio
            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            
            # Pad to square
            thumb = Image.new("RGBA", (thumb_size, thumb_size), (40, 40, 40, 255))
            paste_x = (thumb_size - img.width) // 2
            paste_y = (thumb_size - img.height) // 2
            if img.mode == "RGBA":
                thumb.paste(img, (paste_x, paste_y), img)
            else:
                thumb.paste(img, (paste_x, paste_y))
            
            thumbs.append(np.array(thumb).astype(np.float32) / 255.0)
        
        if mode == "grid":
            # Arrange in a grid
            cols = int(np.ceil(np.sqrt(num_layers)))
            rows = int(np.ceil(num_layers / cols))
            
            grid = np.ones((rows * thumb_size, cols * thumb_size, 4), dtype=np.float32) * 0.2
            grid[:, :, 3] = 1.0
            
            for i, thumb in enumerate(thumbs):
                row = i // cols
                col = i % cols
                y = row * thumb_size
                x = col * thumb_size
                grid[y:y+thumb_size, x:x+thumb_size] = thumb
            
            preview = grid
            
        elif mode == "stack_vertical":
            preview = np.vstack(thumbs)
            
        elif mode == "stack_horizontal":
            preview = np.hstack(thumbs)
            
        else:  # thumbnails - same as grid
            cols = int(np.ceil(np.sqrt(num_layers)))
            rows = int(np.ceil(num_layers / cols))
            
            grid = np.ones((rows * thumb_size, cols * thumb_size, 4), dtype=np.float32) * 0.2
            grid[:, :, 3] = 1.0
            
            for i, thumb in enumerate(thumbs):
                row = i // cols
                col = i % cols
                y = row * thumb_size
                x = col * thumb_size
                grid[y:y+thumb_size, x:x+thumb_size] = thumb
            
            preview = grid
        
        return torch.from_numpy(preview).unsqueeze(0)
