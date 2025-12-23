"""Eric Qwen Layer Load Node
Load layered images from PSD, TIFF, or PNG sequence.

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
import glob
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any

import folder_paths

# Check for psd-tools
try:
    from psd_tools import PSDImage
    PSD_TOOLS_AVAILABLE = True
except ImportError:
    PSD_TOOLS_AVAILABLE = False

# Check for tifffile
try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False


class EricQwenLayerLoad:
    """
    Load layers from various file formats.
    
    Supported formats:
    - PSD: Photoshop format with layers
    - TIFF: Multi-page TIFF
    - PNG Sequence: Folder of PNG files
    - ZIP: Archive with PNGs and manifest
    """
    
    CATEGORY = "Eric Qwen Layer"
    FUNCTION = "load_layers"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("layers", "alpha_masks", "layer_count", "layer_info")
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        
        # Collect available files
        files = []
        
        # PSD files
        if PSD_TOOLS_AVAILABLE:
            files.extend(glob.glob(os.path.join(input_dir, "**/*.psd"), recursive=True))
            files.extend(glob.glob(os.path.join(input_dir, "**/*.psb"), recursive=True))
        
        # TIFF files
        if TIFF_AVAILABLE:
            files.extend(glob.glob(os.path.join(input_dir, "**/*.tiff"), recursive=True))
            files.extend(glob.glob(os.path.join(input_dir, "**/*.tif"), recursive=True))
        
        # ZIP files
        files.extend(glob.glob(os.path.join(input_dir, "**/*.zip"), recursive=True))
        
        # PNG folders (look for manifest files)
        files.extend(glob.glob(os.path.join(input_dir, "**/*_manifest.json"), recursive=True))
        
        # Make paths relative
        files = [os.path.relpath(f, input_dir) for f in files]
        
        if not files:
            files = ["(no layered files found)"]
        
        return {
            "required": {
                "source": (sorted(files), {
                    "tooltip": "Select a layered file to load"
                }),
            },
            "optional": {
                "source_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Or specify a folder path with PNG sequence"
                }),
                "include_hidden": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include hidden/invisible layers"
                }),
                "flatten_groups": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Flatten layer groups into single layers"
                }),
                "max_layers": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Maximum number of layers to load"
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, source, **kwargs):
        """Check if source file has changed."""
        input_dir = folder_paths.get_input_directory()
        full_path = os.path.join(input_dir, source)
        if os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return source
    
    def load_layers(
        self,
        source: str,
        source_folder: str = "",
        include_hidden: bool = False,
        flatten_groups: bool = True,
        max_layers: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """Load layers from file."""
        
        input_dir = folder_paths.get_input_directory()
        
        # Determine source path
        if source_folder:
            source_path = source_folder
        else:
            source_path = os.path.join(input_dir, source)
        
        # Determine format and load
        ext = os.path.splitext(source_path)[1].lower()
        
        if ext in [".psd", ".psb"] and PSD_TOOLS_AVAILABLE:
            layers, names, metadata = self._load_psd(source_path, include_hidden, flatten_groups, max_layers)
        elif ext in [".tiff", ".tif"] and TIFF_AVAILABLE:
            layers, names, metadata = self._load_tiff(source_path, max_layers)
        elif ext == ".zip":
            layers, names, metadata = self._load_zip(source_path, max_layers)
        elif ext == ".json" or os.path.isdir(source_path):
            layers, names, metadata = self._load_png_sequence(source_path, max_layers)
        else:
            # Try to load as single image
            layers, names, metadata = self._load_single_image(source_path)
        
        # Convert to tensors
        if not layers:
            empty_tensor = torch.zeros((1, 64, 64, 4))
            return (empty_tensor, empty_tensor[:, :, :, 0], 0, json.dumps({"error": "No layers found"}))
        
        layer_tensors = []
        alpha_tensors = []
        
        for layer in layers:
            # Ensure RGBA
            if layer.mode != "RGBA":
                layer = layer.convert("RGBA")
            
            # Convert to numpy then tensor
            layer_np = np.array(layer).astype(np.float32) / 255.0
            layer_tensors.append(layer_np)
            alpha_tensors.append(layer_np[:, :, 3])
        
        # Stack into batches
        layers_batch = torch.from_numpy(np.stack(layer_tensors, axis=0))
        alpha_batch = torch.from_numpy(np.stack(alpha_tensors, axis=0))
        
        # Build info
        info = {
            "source": source_path,
            "layer_count": len(layers),
            "dimensions": {
                "width": layers[0].width,
                "height": layers[0].height,
            },
            "layers": []
        }
        
        for i, name in enumerate(names):
            layer_info = {"index": i, "name": name}
            if metadata and i < len(metadata):
                layer_info.update(metadata[i])
            info["layers"].append(layer_info)
        
        return (layers_batch, alpha_batch, len(layers), json.dumps(info, indent=2))
    
    def _load_psd(
        self,
        path: str,
        include_hidden: bool,
        flatten_groups: bool,
        max_layers: int,
    ) -> Tuple[List[Image.Image], List[str], List[Dict]]:
        """Load layers from PSD file."""
        from psd_tools import PSDImage
        
        psd = PSDImage.open(path)
        
        layers = []
        names = []
        metadata = []
        
        def process_layer(layer, depth=0):
            if len(layers) >= max_layers:
                return
            
            # Skip hidden layers if not requested
            if not include_hidden and not layer.is_visible():
                return
            
            if layer.is_group():
                if flatten_groups:
                    # Composite the group
                    try:
                        composite = layer.composite()
                        if composite is not None:
                            layers.append(composite)
                            names.append(layer.name)
                            metadata.append({
                                "blend_mode": str(layer.blend_mode).split(".")[-1].lower(),
                                "opacity": layer.opacity / 255.0,
                                "visible": layer.is_visible(),
                                "is_group": True,
                            })
                    except:
                        pass
                else:
                    # Process children
                    for child in layer:
                        process_layer(child, depth + 1)
            else:
                # Regular layer
                try:
                    img = layer.composite()
                    if img is not None:
                        layers.append(img)
                        names.append(layer.name)
                        metadata.append({
                            "blend_mode": str(layer.blend_mode).split(".")[-1].lower(),
                            "opacity": layer.opacity / 255.0,
                            "visible": layer.is_visible(),
                        })
                except:
                    pass
        
        # Process all layers
        for layer in psd:
            process_layer(layer)
        
        return layers, names, metadata
    
    def _load_tiff(
        self,
        path: str,
        max_layers: int,
    ) -> Tuple[List[Image.Image], List[str], List[Dict]]:
        """Load layers from multi-page TIFF."""
        import tifffile
        
        with tifffile.TiffFile(path) as tif:
            # Read all pages
            pages = []
            for i, page in enumerate(tif.pages):
                if i >= max_layers:
                    break
                pages.append(page.asarray())
            
            # Get metadata if available
            try:
                if 'layer_names' in tif.imagej_metadata:
                    layer_names = tif.imagej_metadata['layer_names']
                else:
                    layer_names = [f"Layer_{i}" for i in range(len(pages))]
            except:
                layer_names = [f"Layer_{i}" for i in range(len(pages))]
        
        layers = []
        metadata = []
        
        for i, page_data in enumerate(pages):
            # Convert to PIL
            if page_data.ndim == 2:
                img = Image.fromarray(page_data, mode="L")
            elif page_data.shape[-1] == 3:
                img = Image.fromarray(page_data, mode="RGB")
            elif page_data.shape[-1] == 4:
                img = Image.fromarray(page_data, mode="RGBA")
            else:
                img = Image.fromarray(page_data)
            
            layers.append(img)
            metadata.append({
                "blend_mode": "normal",
                "opacity": 1.0,
                "visible": True,
            })
        
        return layers, layer_names[:len(layers)], metadata
    
    def _load_zip(
        self,
        path: str,
        max_layers: int,
    ) -> Tuple[List[Image.Image], List[str], List[Dict]]:
        """Load layers from ZIP archive."""
        import zipfile
        from io import BytesIO
        
        layers = []
        names = []
        metadata = []
        
        with zipfile.ZipFile(path, "r") as zf:
            # Try to load manifest
            manifest = None
            if "manifest.json" in zf.namelist():
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            
            # Find layer files
            layer_files = sorted([
                f for f in zf.namelist()
                if f.startswith("layers/") and f.endswith(".png")
            ])
            
            for i, layer_file in enumerate(layer_files):
                if i >= max_layers:
                    break
                
                # Load image
                img_data = zf.read(layer_file)
                img = Image.open(BytesIO(img_data))
                layers.append(img.convert("RGBA"))
                
                # Get name from filename
                filename = os.path.basename(layer_file)
                name = os.path.splitext(filename)[0]
                if name[0:3].isdigit() and name[2] == "_":
                    name = name[3:]  # Remove index prefix
                names.append(name)
                
                # Get metadata from manifest
                if manifest and i < len(manifest.get("layers", [])):
                    metadata.append(manifest["layers"][i])
                else:
                    metadata.append({
                        "blend_mode": "normal",
                        "opacity": 1.0,
                        "visible": True,
                    })
        
        return layers, names, metadata
    
    def _load_png_sequence(
        self,
        path: str,
        max_layers: int,
    ) -> Tuple[List[Image.Image], List[str], List[Dict]]:
        """Load layers from PNG sequence folder."""
        
        # Handle manifest file path
        if path.endswith("_manifest.json"):
            folder = os.path.dirname(path)
            manifest_path = path
        else:
            folder = path
            manifest_path = None
            
            # Look for manifest in folder
            for mf in ["manifest.json", "_manifest.json"]:
                mp = os.path.join(folder, mf)
                if os.path.exists(mp):
                    manifest_path = mp
                    break
        
        # Load manifest if available
        manifest = None
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        
        # Find PNG files
        png_files = sorted(glob.glob(os.path.join(folder, "*.png")))
        
        # Filter out composite
        png_files = [f for f in png_files if not os.path.basename(f).startswith("_")]
        
        layers = []
        names = []
        metadata = []
        
        for i, png_file in enumerate(png_files):
            if i >= max_layers:
                break
            
            img = Image.open(png_file).convert("RGBA")
            layers.append(img)
            
            # Get name from filename
            filename = os.path.basename(png_file)
            name = os.path.splitext(filename)[0]
            if len(name) > 3 and name[2] == "_" and name[:2].isdigit():
                name = name[3:]  # Remove index prefix
            names.append(name)
            
            # Get metadata
            if manifest and i < len(manifest.get("layers", [])):
                metadata.append(manifest["layers"][i])
            else:
                metadata.append({
                    "blend_mode": "normal",
                    "opacity": 1.0,
                    "visible": True,
                })
        
        return layers, names, metadata
    
    def _load_single_image(
        self,
        path: str,
    ) -> Tuple[List[Image.Image], List[str], List[Dict]]:
        """Load a single image as one layer."""
        
        try:
            img = Image.open(path).convert("RGBA")
            return (
                [img],
                [os.path.splitext(os.path.basename(path))[0]],
                [{"blend_mode": "normal", "opacity": 1.0, "visible": True}]
            )
        except Exception as e:
            print(f"[EricQwenLayerLoad] Error loading {path}: {e}")
            return [], [], []
