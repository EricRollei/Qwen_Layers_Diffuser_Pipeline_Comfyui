"""
Eric Qwen RGBA VAE Loader
Custom VAE loader for Qwen-Image-Layered RGBA VAE.

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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.ops
from comfy.ldm.modules.diffusionmodules.model import vae_attention

# Import the original WanVAE components
from comfy.ldm.wan.vae import (
    CausalConv3d, RMS_norm, Resample, ResidualBlock, AttentionBlock,
    CACHE_T
)

ops = comfy.ops.disable_weight_init


class Encoder3dRGBA(nn.Module):
    """
    Modified Encoder3d that supports configurable input channels.
    Based on comfy.ldm.wan.vae.Encoder3d but with in_channels parameter.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        in_channels=4,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.in_channels = in_channels

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block - configurable input channels
        self.conv1 = CausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block - skip last iteration
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3dRGBA(nn.Module):
    """
    Modified Decoder3d that supports configurable output channels.
    Based on comfy.ldm.wan.vae.Decoder3d but with out_channels parameter.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        out_channels=4,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.out_channels = out_channels

        # dimensions - match original: [dim_mult[-1]] + reversed dim_mult
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block - dims[0] is the largest (dim * dim_mult[-1])
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks - match original structure exactly
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            # Special handling for certain iterations
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):  # Note: +1 vs encoder
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block - skip last iteration
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks - configurable output channels (original uses 3)
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, out_channels, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class WanVAE_RGBA(nn.Module):
    """
    WanVAE variant that supports 4-channel (RGBA) input/output.
    """

    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        in_channels=4,
        out_channels=4,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = Encoder3dRGBA(
            dim, z_dim, dim_mult, num_res_blocks,
            attn_scales, self.temperal_downsample, dropout,
            in_channels=in_channels
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3dRGBA(
            dim, z_dim, dim_mult, num_res_blocks,
            attn_scales, self.temperal_upsample, dropout,
            out_channels=out_channels
        )

    def encode(self, x):
        # Add temporal dimension if needed
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
        
        out = self.encoder(x)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def decode(self, z):
        out = self.conv2(z)
        out = self.decoder(out)
        
        # Remove temporal dimension if single frame
        if out.shape[2] == 1:
            out = out.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        
        return out


class EricQwenRGBAVAELoader:
    """
    Load Qwen-Image-Layered RGBA VAE.
    
    The standard VAELoader doesn't support 4-channel VAEs.
    This node properly loads the Qwen Layered VAE with RGBA support.
    """
    
    CATEGORY = "Eric Qwen Layer/Loaders"
    FUNCTION = "load_vae"
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    
    @classmethod
    def INPUT_TYPES(cls):
        vae_folder = folder_paths.get_folder_paths("vae")
        vae_files = []
        for folder in vae_folder:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith(('.safetensors', '.pt', '.pth', '.ckpt')):
                        vae_files.append(f)
        
        return {
            "required": {
                "vae_name": (sorted(set(vae_files)) if vae_files else ["none"],),
            },
        }
    
    def load_vae(self, vae_name: str):
        """Load and configure the RGBA VAE."""
        # Find VAE file
        vae_path = None
        for folder in folder_paths.get_folder_paths("vae"):
            potential_path = os.path.join(folder, vae_name)
            if os.path.exists(potential_path):
                vae_path = potential_path
                break
        
        if vae_path is None:
            raise FileNotFoundError(f"VAE not found: {vae_name}")
        
        print(f"[EricQwenRGBAVAELoader] Loading: {vae_path}")
        
        # Load state dict
        sd = comfy.utils.load_torch_file(vae_path)
        
        # Detect dimensions from weights
        in_channels = 4
        out_channels = 4
        
        if "encoder.conv1.weight" in sd:
            weight_shape = sd["encoder.conv1.weight"].shape
            in_channels = weight_shape[1]
            print(f"[EricQwenRGBAVAELoader] Detected input channels: {in_channels}")
        
        if "decoder.head.2.weight" in sd:
            weight_shape = sd["decoder.head.2.weight"].shape
            out_channels = weight_shape[0]
            print(f"[EricQwenRGBAVAELoader] Detected output channels: {out_channels}")
        
        # Detect dim from decoder.head.0.gamma
        dim = 96
        if "decoder.head.0.gamma" in sd:
            dim = sd["decoder.head.0.gamma"].shape[0]
            print(f"[EricQwenRGBAVAELoader] Detected dim: {dim}")
        
        # Create model
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16
        
        model = WanVAE_RGBA(
            dim=dim,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        
        # Load weights with detailed error reporting
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[EricQwenRGBAVAELoader] Missing keys: {len(missing)}")
                for k in missing[:10]:
                    print(f"  - {k}")
            if unexpected:
                print(f"[EricQwenRGBAVAELoader] Unexpected keys: {len(unexpected)}")
                for k in unexpected[:10]:
                    print(f"  - {k}")
        except Exception as e:
            print(f"[EricQwenRGBAVAELoader] Error loading state dict: {e}")
            raise
        
        model.to(device=device, dtype=dtype)
        model.eval()
        
        # Create a VAE wrapper compatible with ComfyUI
        class VAEWrapper:
            def __init__(self, model, device, dtype):
                self.first_stage_model = model
                self.device = device
                self.dtype = dtype
                self.latent_channels = 16
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                self.output_channels = out_channels
                
            def decode(self, z):
                """Decode latent to image."""
                device = comfy.model_management.get_torch_device()
                z = z.to(device=device, dtype=self.dtype)
                
                with torch.no_grad():
                    x = self.first_stage_model.decode(z)
                
                # Handle 5D output [B, C, T, H, W]
                if x.dim() == 5:
                    # Combine batch and time: [B, C, T, H, W] -> [B*T, H, W, C]
                    b, c, t, h, w = x.shape
                    x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
                    x = x.reshape(b * t, h, w, c)
                elif x.dim() == 4:
                    # [B, C, H, W] -> [B, H, W, C]
                    x = x.permute(0, 2, 3, 1)
                
                return x.float().clamp(0, 1)
            
            def encode(self, x):
                """Encode image to latent."""
                device = comfy.model_management.get_torch_device()
                
                # Handle input shape
                if x.dim() == 4:
                    if x.shape[-1] in [3, 4]:
                        # [B, H, W, C] -> [B, C, H, W]
                        x = x.permute(0, 3, 1, 2)
                
                x = x.to(device=device, dtype=self.dtype)
                
                with torch.no_grad():
                    z = self.first_stage_model.encode(x)
                
                return z
        
        vae = VAEWrapper(model, device, dtype)
        
        print(f"[EricQwenRGBAVAELoader] Successfully loaded {vae_name}")
        print(f"[EricQwenRGBAVAELoader] Channels: {in_channels} in / {out_channels} out")
        
        return (vae,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "EricQwenRGBAVAELoader": EricQwenRGBAVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EricQwenRGBAVAELoader": "Eric Qwen RGBA VAE Loader",
}
