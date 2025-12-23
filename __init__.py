"""
Eric Qwen Layer Nodes
A ComfyUI node package for working with Qwen-Image-Layered model.

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

from .nodes.eric_qwen_decompose import EricQwenDecompose, EricQwenUnloadModel
from .nodes.eric_qwen_layer_utils import (
    EricQwenLayerSelector,
    EricQwenLayerReorder,
    EricQwenLayerComposite,
    EricQwenLayerInfo,
)
from .nodes.eric_qwen_layer_save import EricQwenLayerSave
from .nodes.eric_qwen_layer_load import EricQwenLayerLoad
from .nodes.eric_qwen_layer_describe import EricQwenLayerDescribe, EricQwenStringJoin
from .nodes.eric_qwen_native import (
    EricQwenAddAlpha,
    EricQwenEncode,
    EricQwenLayerExtract,
    EricQwenMultiLatent,
    EricQwenLayerPrompts,
    EricQwenLatentCutToBatch,
    EricQwenBatchToVideo,
)
from .nodes.eric_qwen_rgba_vae import EricQwenRGBAVAELoader

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Diffusers-based (full pipeline)
    "EricQwenDecompose": EricQwenDecompose,
    "EricQwenUnloadModel": EricQwenUnloadModel,
    
    # Native ComfyUI workflow nodes
    "EricQwenAddAlpha": EricQwenAddAlpha,
    "EricQwenEncode": EricQwenEncode,
    "EricQwenLayerExtract": EricQwenLayerExtract,
    "EricQwenMultiLatent": EricQwenMultiLatent,
    "EricQwenLayerPrompts": EricQwenLayerPrompts,
    "EricQwenLatentCutToBatch": EricQwenLatentCutToBatch,
    "EricQwenBatchToVideo": EricQwenBatchToVideo,
    
    # Loaders
    "EricQwenRGBAVAELoader": EricQwenRGBAVAELoader,
    
    # Layer manipulation nodes
    "EricQwenLayerSelector": EricQwenLayerSelector,
    "EricQwenLayerReorder": EricQwenLayerReorder,
    "EricQwenLayerComposite": EricQwenLayerComposite,
    "EricQwenLayerInfo": EricQwenLayerInfo,
    
    # File I/O
    "EricQwenLayerSave": EricQwenLayerSave,
    "EricQwenLayerLoad": EricQwenLayerLoad,
    
    # Layer naming/description
    "EricQwenLayerDescribe": EricQwenLayerDescribe,
    "EricQwenStringJoin": EricQwenStringJoin,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    # Diffusers-based
    "EricQwenDecompose": "Eric Qwen Layer Decompose (Diffusers)",
    "EricQwenUnloadModel": "Eric Qwen Unload Model",
    
    # Native ComfyUI
    "EricQwenAddAlpha": "Eric Qwen Add Alpha (RGBA)",
    "EricQwenEncode": "Eric Qwen Encode (Native)",
    "EricQwenLayerExtract": "Eric Qwen Layer Extract",
    "EricQwenMultiLatent": "Eric Qwen Multi-Layer Latent",
    "EricQwenLayerPrompts": "Eric Qwen Layer Prompts",
    "EricQwenLatentCutToBatch": "Eric Qwen Latent Cut to Batch",
    "EricQwenBatchToVideo": "Eric Qwen Batch to Video",
    
    # Loaders
    "EricQwenRGBAVAELoader": "Eric Qwen RGBA VAE Loader",
    
    # Layer manipulation
    "EricQwenLayerSelector": "Eric Qwen Layer Selector",
    "EricQwenLayerReorder": "Eric Qwen Layer Reorder",
    "EricQwenLayerComposite": "Eric Qwen Layer Composite",
    "EricQwenLayerInfo": "Eric Qwen Layer Info",
    
    # File I/O
    "EricQwenLayerSave": "Eric Qwen Layer Save (PSD/TIFF)",
    "EricQwenLayerLoad": "Eric Qwen Layer Load",
    
    # Layer naming/description
    "EricQwenLayerDescribe": "Eric Qwen Layer Describe (Vision)",
    "EricQwenStringJoin": "Eric Qwen String Join",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

WEB_DIRECTORY = "./web"
