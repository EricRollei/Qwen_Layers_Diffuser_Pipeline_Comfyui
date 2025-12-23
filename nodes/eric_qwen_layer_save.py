"""Eric Qwen Layer Save Node
Save layers as PSD, TIFF, or PNG sequence with proper layer metadata.

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
import re
import unicodedata
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

import folder_paths


def sanitize_layer_name(name: str, max_length: int = 63) -> str:
    """
    Sanitize a layer name for PSD compatibility.
    
    PSD files use macroman encoding which can't handle Unicode/CJK characters.
    This function transliterates or removes incompatible characters.
    
    Args:
        name: The original layer name
        max_length: Maximum length for layer name (PSD limit is ~255, but shorter is safer)
        
    Returns:
        A sanitized ASCII-compatible layer name
    """
    if not name:
        return "Layer"
    
    # Try to normalize Unicode characters to ASCII equivalents
    # NFKD decomposes characters (e.g., é -> e + combining accent)
    normalized = unicodedata.normalize('NFKD', name)
    
    # Encode to ASCII, ignoring non-ASCII characters
    try:
        ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
    except:
        ascii_name = ""
    
    # If we lost everything (e.g., all Chinese), create a placeholder
    if not ascii_name.strip():
        # Try to detect language and give a meaningful placeholder
        if any('\u4e00' <= c <= '\u9fff' for c in name):
            ascii_name = "Chinese_Text"
        elif any('\u3040' <= c <= '\u30ff' for c in name):
            ascii_name = "Japanese_Text"
        elif any('\uac00' <= c <= '\ud7af' for c in name):
            ascii_name = "Korean_Text"
        else:
            ascii_name = "Unicode_Text"
    
    # Clean up: replace multiple spaces/underscores, trim
    ascii_name = re.sub(r'[\s_]+', '_', ascii_name).strip('_')
    
    # Ensure it's not empty
    if not ascii_name:
        ascii_name = "Layer"
    
    # Truncate if too long
    if len(ascii_name) > max_length:
        ascii_name = ascii_name[:max_length].rstrip('_')
    
    return ascii_name

# Check for psd-tools
try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    from psd_tools.constants import BlendMode as PSDBlendMode
    PSD_TOOLS_AVAILABLE = True
except ImportError:
    PSD_TOOLS_AVAILABLE = False

# Check for tifffile
try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False


class EricQwenLayerSave:
    """
    Save layers to various formats with full layer metadata preservation.
    
    Supported formats:
    - PSD: Photoshop format with layers, blend modes, opacity
    - TIFF: Multi-page TIFF with layer tags
    - PNG Sequence: Individual PNG files with manifest
    - ZIP: All PNGs in a ZIP archive
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "save_layers"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("saved_path", "layer_preview")
    OUTPUT_NODE = True
    
    BLEND_MODE_MAP = {
        "normal": "normal",
        "multiply": "multiply",
        "screen": "screen",
        "overlay": "overlay",
        "darken": "darken",
        "lighten": "lighten",
        "color_dodge": "color dodge",
        "color_burn": "color burn",
        "hard_light": "hard light",
        "soft_light": "soft light",
        "difference": "difference",
        "exclusion": "exclusion",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        
        formats = ["png_sequence"]
        if PSD_TOOLS_AVAILABLE:
            formats.insert(0, "psd")
        if TIFF_AVAILABLE:
            formats.append("tiff")
        formats.append("zip")
        
        return {
            "required": {
                "layers": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "qwen_layers",
                    "tooltip": "Prefix for output filename"
                }),
                "output_format": (formats, {
                    "default": formats[0],
                    "tooltip": "Output file format"
                }),
                "also_save_pngs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also save individual layer PNGs alongside PSD/TIFF"
                }),
            },
            "optional": {
                "alpha_masks": ("MASK",),
                "auto_name_layers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-generate layer names based on content (Background, Subject, etc.)"
                }),
                "layer_names": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom layer names, one per line. Overrides auto-naming."
                }),
                "blend_modes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Blend modes, one per line (normal, multiply, screen, etc.)"
                }),
                "opacities": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated opacities (0-100) for each layer"
                }),
                "include_composite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include a flattened composite image"
                }),
                "include_manifest": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include JSON manifest with layer metadata"
                }),
                "subfolder": ("STRING", {
                    "default": "",
                    "tooltip": "Optional subfolder within output directory"
                }),
                "layer_info": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON layer info from decompose node (names extracted automatically)"
                }),
                "original_image": ("IMAGE", {
                    "tooltip": "Original input image - layers will be scaled to match this size"
                }),
                "include_original_as_base": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include original image as the bottom layer in PSD"
                }),
            }
        }
    
    def save_layers(
        self,
        layers: torch.Tensor,
        filename_prefix: str = "qwen_layers",
        output_format: str = "psd",
        also_save_pngs: bool = True,
        alpha_masks: Optional[torch.Tensor] = None,
        auto_name_layers: bool = True,
        layer_names: str = "",
        blend_modes: str = "",
        opacities: str = "",
        include_composite: bool = True,
        include_manifest: bool = True,
        subfolder: str = "",
        layer_info: str = "",
        original_image: Optional[torch.Tensor] = None,
        include_original_as_base: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        """Save layers to file."""
        
        # Get target size from original image if provided
        target_size = None
        original_pil = None
        if original_image is not None and len(original_image.shape) >= 3:
            # Extract first image from batch
            orig = original_image[0] if len(original_image.shape) == 4 else original_image
            orig_np = (orig.numpy() * 255).astype(np.uint8)
            if orig_np.shape[-1] == 3:
                original_pil = Image.fromarray(orig_np, mode="RGB").convert("RGBA")
            elif orig_np.shape[-1] == 4:
                original_pil = Image.fromarray(orig_np, mode="RGBA")
            else:
                original_pil = Image.fromarray(orig_np).convert("RGBA")
            target_size = (original_pil.width, original_pil.height)
            print(f"[EricQwenLayerSave] Will scale layers to original size: {target_size}")
        
        # Get output directory
        output_dir = folder_paths.get_output_directory()
        if subfolder:
            output_dir = os.path.join(output_dir, subfolder)
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        # Parse layer metadata
        num_layers = layers.shape[0]
        
        # Parse or auto-generate layer names
        # Priority: 1. User-provided names, 2. layer_info JSON, 3. auto-naming, 4. generic names
        if layer_names.strip():
            # User provided custom names (highest priority)
            names = [n.strip() for n in layer_names.strip().split("\n")]
        elif layer_info.strip():
            # Extract names from layer_info JSON (from decompose node)
            try:
                info = json.loads(layer_info)
                if "layers" in info and isinstance(info["layers"], list):
                    names = [layer.get("name", f"Layer_{i}") for i, layer in enumerate(info["layers"])]
                else:
                    names = [f"Layer_{i}" for i in range(num_layers)]
            except (json.JSONDecodeError, KeyError):
                names = [f"Layer_{i}" for i in range(num_layers)]
        elif auto_name_layers:
            # Auto-generate names based on coverage analysis
            names = self._auto_name_layers(layers)
        else:
            names = [f"Layer_{i}" for i in range(num_layers)]
        # Pad or truncate to match layer count
        while len(names) < num_layers:
            names.append(f"Layer_{len(names)}")
        names = names[:num_layers]
        
        # Parse blend modes
        if blend_modes.strip():
            modes = [m.strip().lower() for m in blend_modes.strip().split("\n")]
        else:
            modes = ["normal"] * num_layers
        while len(modes) < num_layers:
            modes.append("normal")
        modes = modes[:num_layers]
        
        # Parse opacities
        if opacities.strip():
            try:
                ops = [float(o.strip()) / 100.0 for o in opacities.split(",")]
            except ValueError:
                ops = [1.0] * num_layers
        else:
            ops = [1.0] * num_layers
        while len(ops) < num_layers:
            ops.append(1.0)
        ops = ops[:num_layers]
        
        # Convert layers to PIL images
        layer_images = []
        for i in range(num_layers):
            layer = layers[i]
            
            # Handle alpha
            if layer.shape[-1] == 4:
                img_np = (layer.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_np, mode="RGBA")
            elif alpha_masks is not None and i < alpha_masks.shape[0]:
                rgb_np = (layer.numpy() * 255).astype(np.uint8)
                alpha_np = (alpha_masks[i].numpy() * 255).astype(np.uint8)
                img = Image.fromarray(rgb_np[:, :, :3], mode="RGB")
                img.putalpha(Image.fromarray(alpha_np, mode="L"))
            else:
                img_np = (layer.numpy() * 255).astype(np.uint8)
                if img_np.shape[-1] == 3:
                    img = Image.fromarray(img_np, mode="RGB")
                else:
                    img = Image.fromarray(img_np.squeeze(), mode="L")
            
            layer_images.append(img)
        
        # Scale layers to original image size if provided
        if target_size is not None:
            current_size = (layer_images[0].width, layer_images[0].height) if layer_images else (0, 0)
            if current_size != target_size:
                print(f"[EricQwenLayerSave] Scaling {len(layer_images)} layers from {current_size} to {target_size}")
                layer_images = [
                    img.resize(target_size, Image.LANCZOS)
                    for img in layer_images
                ]
        
        # Add original image as base layer if requested
        if original_pil is not None and include_original_as_base:
            # Insert original at the beginning (will be bottom layer)
            layer_images.insert(0, original_pil)
            names.insert(0, "Original")
            modes.insert(0, "normal")
            ops.insert(0, 1.0)
            num_layers += 1
            print(f"[EricQwenLayerSave] Added original image as base layer")
        
        # Build manifest
        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "source": "eric_qwen_layer",
            "layer_count": num_layers,
            "dimensions": {
                "width": layer_images[0].width if layer_images else layers.shape[2],
                "height": layer_images[0].height if layer_images else layers.shape[1],
            },
            "original_dimensions": {
                "width": target_size[0] if target_size else layers.shape[2],
                "height": target_size[1] if target_size else layers.shape[1],
            } if target_size else None,
            "layers": []
        }
        
        for i in range(num_layers):
            manifest["layers"].append({
                "index": i,
                "name": names[i],
                "blend_mode": modes[i],
                "opacity": ops[i],
                "visible": True,
            })
        
        # Save based on format
        if output_format == "psd" and PSD_TOOLS_AVAILABLE:
            output_path = self._save_psd(
                output_dir, base_filename,
                layer_images, names, modes, ops,
                include_composite, also_save_pngs
            )
        elif output_format == "tiff" and TIFF_AVAILABLE:
            output_path = self._save_tiff(
                output_dir, base_filename,
                layer_images, names
            )
        elif output_format == "zip":
            output_path = self._save_zip(
                output_dir, base_filename,
                layer_images, names, manifest
            )
        else:  # png_sequence
            output_path = self._save_png_sequence(
                output_dir, base_filename,
                layer_images, names, include_composite
            )
        
        # Save manifest
        if include_manifest:
            manifest_path = os.path.splitext(output_path)[0] + "_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        
        # Generate preview
        preview = self._generate_layer_preview(layer_images, names, modes, ops)
        
        return (output_path, preview)
    
    def _auto_name_layers(self, layers: torch.Tensor) -> List[str]:
        """
        Auto-generate meaningful layer names based on content analysis.
        
        Analyzes alpha coverage to determine layer type:
        - ~100% coverage with high alpha = Background
        - ~0% coverage = Empty/Transparent
        - Partial coverage = Subject/Element layers
        """
        names = []
        num_layers = layers.shape[0]
        
        # Analyze each layer
        coverages = []
        for i in range(num_layers):
            layer = layers[i]
            if layer.shape[-1] == 4:
                alpha = layer[:, :, 3].numpy()
            else:
                alpha = np.ones((layer.shape[0], layer.shape[1]))
            
            coverage = (alpha > 0.04).sum() / alpha.size  # >10/255 threshold
            avg_alpha = alpha.mean()
            coverages.append((coverage, avg_alpha))
        
        # Assign names based on analysis
        background_assigned = False
        element_count = 0
        
        for i, (coverage, avg_alpha) in enumerate(coverages):
            if coverage > 0.95 and avg_alpha > 0.9:
                # High coverage, high alpha = Background
                if not background_assigned:
                    names.append("Background")
                    background_assigned = True
                else:
                    names.append(f"Fill_{i}")
            elif coverage < 0.02:
                # Very low coverage = Empty/Transparent
                names.append(f"Empty_{i}")
            elif coverage > 0.7:
                # Large element
                names.append(f"Main_Subject_{i}")
            elif coverage > 0.3:
                # Medium element
                element_count += 1
                names.append(f"Element_{element_count}")
            else:
                # Small element - likely foreground detail
                element_count += 1
                names.append(f"Detail_{element_count}")
        
        return names
    
    def _save_psd(
        self,
        output_dir: str,
        base_filename: str,
        layers: List[Image.Image],
        names: List[str],
        blend_modes: List[str],
        opacities: List[float],
        include_composite: bool,
        also_save_pngs: bool = True,
    ) -> str:
        """Save as PSD with proper layer support."""
        from psd_tools import PSDImage
        from psd_tools.api.layers import PixelLayer
        from psd_tools.constants import BlendMode as PSDBlendMode
        
        output_path = os.path.join(output_dir, f"{base_filename}.psd")
        
        # Get dimensions from first layer
        width = layers[0].width
        height = layers[0].height
        
        # Create a new PSD document
        psd = PSDImage.new(mode="RGBA", size=(width, height))
        
        # Map blend mode strings to PSD constants
        blend_mode_map = {
            "normal": PSDBlendMode.NORMAL,
            "multiply": PSDBlendMode.MULTIPLY,
            "screen": PSDBlendMode.SCREEN,
            "overlay": PSDBlendMode.OVERLAY,
            "darken": PSDBlendMode.DARKEN,
            "lighten": PSDBlendMode.LIGHTEN,
            "color_dodge": PSDBlendMode.COLOR_DODGE,
            "color_burn": PSDBlendMode.COLOR_BURN,
            "hard_light": PSDBlendMode.HARD_LIGHT,
            "soft_light": PSDBlendMode.SOFT_LIGHT,
            "difference": PSDBlendMode.DIFFERENCE,
            "exclusion": PSDBlendMode.EXCLUSION,
        }
        
        # Add each layer to the PSD (layer 0 = background at bottom, added first)
        # append() adds to top, so we add in normal order: 0, 1, 2...
        for i in range(len(layers)):
            layer_img = layers[i]
            if layer_img.mode != "RGBA":
                layer_img = layer_img.convert("RGBA")
            
            # Apply opacity to alpha channel
            if opacities[i] < 1.0:
                r, g, b, a = layer_img.split()
                a = a.point(lambda x: int(x * opacities[i]))
                layer_img = Image.merge("RGBA", (r, g, b, a))
            
            # Sanitize layer name for PSD compatibility (macroman encoding)
            safe_layer_name = sanitize_layer_name(names[i])
            
            # Create pixel layer from PIL image
            pixel_layer = PixelLayer.frompil(
                layer_img,
                psd_file=psd,
                layer_name=safe_layer_name,
                top=0,
                left=0,
            )
            
            # Set blend mode if supported
            mode_key = blend_modes[i].lower().replace(" ", "_")
            if mode_key in blend_mode_map:
                pixel_layer._record.blend_mode = blend_mode_map[mode_key]
            
            # Append layer to PSD
            psd.append(pixel_layer)
        
        # Save the PSD
        psd.save(output_path)
        print(f"[EricQwenLayerSave] Saved PSD with {len(layers)} layers: {output_path}")
        
        # Also save individual PNGs if requested
        if also_save_pngs:
            layer_dir = os.path.join(output_dir, base_filename + "_layers")
            os.makedirs(layer_dir, exist_ok=True)
            
            for i, layer in enumerate(layers):
                safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in names[i])
                layer_path = os.path.join(layer_dir, f"{i:02d}_{safe_name}.png")
                layer.save(layer_path, "PNG")
            
            # Save composite
            if include_composite:
                composite = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                for layer in layers:
                    if layer.mode != "RGBA":
                        layer = layer.convert("RGBA")
                    composite = Image.alpha_composite(composite, layer)
                composite.save(os.path.join(layer_dir, "_composite.png"), "PNG")
            
            print(f"[EricQwenLayerSave] Also saved individual PNGs to: {layer_dir}")
        
        return output_path
    
    def _save_tiff(
        self,
        output_dir: str,
        base_filename: str,
        layers: List[Image.Image],
        names: List[str],
    ) -> str:
        """Save as multi-page TIFF."""
        import tifffile
        
        output_path = os.path.join(output_dir, f"{base_filename}.tiff")
        
        # Convert to numpy arrays
        layer_arrays = []
        for layer in layers:
            if layer.mode != "RGBA":
                layer = layer.convert("RGBA")
            layer_arrays.append(np.array(layer))
        
        # Stack and save
        stacked = np.stack(layer_arrays, axis=0)
        
        # Write multi-page TIFF
        tifffile.imwrite(
            output_path,
            stacked,
            photometric='rgb',
            metadata={'layer_names': names},
        )
        
        return output_path
    
    def _save_png_sequence(
        self,
        output_dir: str,
        base_filename: str,
        layers: List[Image.Image],
        names: List[str],
        include_composite: bool,
    ) -> str:
        """Save as PNG sequence in a folder."""
        
        # Create subfolder for layers
        layer_dir = os.path.join(output_dir, base_filename)
        os.makedirs(layer_dir, exist_ok=True)
        
        # Save each layer
        for i, layer in enumerate(layers):
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in names[i])
            layer_path = os.path.join(layer_dir, f"{i:02d}_{safe_name}.png")
            layer.save(layer_path, "PNG")
        
        # Save composite if requested
        if include_composite and layers:
            composite = Image.new("RGBA", layers[0].size, (255, 255, 255, 0))
            for layer in layers:
                if layer.mode != "RGBA":
                    layer = layer.convert("RGBA")
                composite = Image.alpha_composite(composite, layer)
            
            composite_path = os.path.join(layer_dir, "_composite.png")
            composite.save(composite_path, "PNG")
        
        return layer_dir
    
    def _save_zip(
        self,
        output_dir: str,
        base_filename: str,
        layers: List[Image.Image],
        names: List[str],
        manifest: Dict,
    ) -> str:
        """Save as ZIP archive with PNGs and manifest."""
        import zipfile
        from io import BytesIO
        
        output_path = os.path.join(output_dir, f"{base_filename}.zip")
        
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add manifest
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            
            # Add layers
            for i, layer in enumerate(layers):
                safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in names[i])
                
                buffer = BytesIO()
                layer.save(buffer, format="PNG")
                buffer.seek(0)
                
                zf.writestr(f"layers/{i:02d}_{safe_name}.png", buffer.getvalue())
            
            # Add composite
            if layers:
                composite = Image.new("RGBA", layers[0].size, (255, 255, 255, 0))
                for layer in layers:
                    if layer.mode != "RGBA":
                        layer = layer.convert("RGBA")
                    composite = Image.alpha_composite(composite, layer)
                
                buffer = BytesIO()
                composite.save(buffer, format="PNG")
                buffer.seek(0)
                
                zf.writestr("composite.png", buffer.getvalue())
        
        return output_path
    
    def _generate_layer_preview(
        self,
        layers: List[Image.Image],
        names: List[str],
        blend_modes: List[str],
        opacities: List[float],
    ) -> torch.Tensor:
        """Generate a visual layer stack preview."""
        
        if not layers:
            return torch.zeros((1, 128, 128, 4))
        
        # Preview settings
        thumb_size = 64
        padding = 10
        text_height = 20
        row_height = thumb_size + text_height + padding
        
        num_layers = len(layers)
        width = 300
        height = max(128, num_layers * row_height + padding * 2)
        
        # Create preview canvas
        preview = Image.new("RGBA", (width, height), (40, 40, 40, 255))
        
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(preview)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw each layer row (top to bottom = last layer to first)
            for i in range(num_layers):
                layer_idx = num_layers - 1 - i  # Reverse order
                layer = layers[layer_idx]
                
                y_pos = padding + i * row_height
                
                # Create thumbnail
                thumb = layer.copy()
                thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                
                # Create checkerboard background for transparency
                checker = Image.new("RGBA", (thumb_size, thumb_size))
                for cy in range(0, thumb_size, 8):
                    for cx in range(0, thumb_size, 8):
                        color = (80, 80, 80, 255) if (cx // 8 + cy // 8) % 2 == 0 else (60, 60, 60, 255)
                        for py in range(min(8, thumb_size - cy)):
                            for px in range(min(8, thumb_size - cx)):
                                checker.putpixel((cx + px, cy + py), color)
                
                # Center thumbnail
                paste_x = padding + (thumb_size - thumb.width) // 2
                paste_y = y_pos + (thumb_size - thumb.height) // 2
                
                # Paste with alpha
                if thumb.mode == "RGBA":
                    checker.paste(thumb, ((thumb_size - thumb.width) // 2, (thumb_size - thumb.height) // 2), thumb)
                else:
                    checker.paste(thumb, ((thumb_size - thumb.width) // 2, (thumb_size - thumb.height) // 2))
                
                preview.paste(checker, (padding, y_pos))
                
                # Draw layer info
                text_x = padding + thumb_size + 10
                text_y = y_pos + 5
                
                # Layer name
                draw.text((text_x, text_y), names[layer_idx], fill=(255, 255, 255), font=font)
                
                # Blend mode and opacity
                mode_text = f"{blend_modes[layer_idx]} @ {int(opacities[layer_idx] * 100)}%"
                draw.text((text_x, text_y + 16), mode_text, fill=(180, 180, 180), font=font)
                
                # Index
                draw.text((text_x, text_y + 32), f"[{layer_idx}]", fill=(120, 120, 120), font=font)
                
        except Exception as e:
            print(f"[EricQwenLayerSave] Preview generation note: {e}")
        
        # Convert to tensor
        preview_np = np.array(preview).astype(np.float32) / 255.0
        return torch.from_numpy(preview_np).unsqueeze(0)
