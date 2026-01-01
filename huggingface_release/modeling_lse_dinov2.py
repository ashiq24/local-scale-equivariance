"""
LSE-DINOv2: Local Scale Equivariant DINOv2 Model

This module implements a DINOv2 Vision Transformer enhanced with Deep Equilibrium
Model (DEM) based local scale adaptation. The model learns to apply content-aware
local scaling transformations at multiple layers for improved scale equivariance.

Key Components:
    1. Base DINOv2 ViT with register tokens (from timm)
    2. DEMAdapter: Deep Equilibrium Model for learning local scaling parameters
    3. Local scaling operations applied at specified transformer layers

Example:
    >>> from huggingface_release import LSEDinoV2ForImageClassification
    >>> model = LSEDinoV2ForImageClassification.from_pretrained("path/to/model")
    >>> outputs = model(pixel_values)
    >>> logits = outputs.logits

License: Apache-2.0
"""

import os
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.functional import grid_sample as GridSample
from torchvision.transforms import Compose, Resize, Normalize
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

# Handle both direct run and package import
try:
    from .configuration_lse_dinov2 import LSEDinoV2Config
except ImportError:
    from configuration_lse_dinov2 import LSEDinoV2Config

# Optional torchdeq dependency
try:
    from torchdeq import get_deq
    HAS_TORCHDEQ = True
except ImportError:
    HAS_TORCHDEQ = False


# ============================================================================
# Utility Functions for Local Scaling
# ============================================================================

def apply_smoothing(grid, kernel_size=5):
    """Apply Gaussian smoothing to parameter grids."""
    smoothing_kernel = T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.2, 0.2))
    return smoothing_kernel(grid.unsqueeze(1)).squeeze(1)


def normalize_cumsum(grid):
    """Create monotonic cumulative sum along the last dimension."""
    monotonic_grid = torch.cumsum(grid, dim=-1)
    return monotonic_grid


def get_coarse_adaptive_grid(params_x, params_y):
    """Generate adaptive sampling grid from learnable parameters."""
    params_x_smooth = F.softmax(3 * apply_smoothing(params_x), dim=-1)
    params_y_smooth = F.softmax(3 * apply_smoothing(params_y), dim=-1)
    params_x_smooth = torch.cat([torch.zeros_like(params_x_smooth[:, :, 0]).unsqueeze(2), params_x_smooth], dim=2)
    params_y_smooth = torch.cat([torch.zeros_like(params_y_smooth[:, :, 0]).unsqueeze(2), params_y_smooth], dim=2)
    monotonic_params_x = normalize_cumsum(params_x_smooth)
    monotonic_params_y = normalize_cumsum(params_y_smooth)
    return torch.stack([monotonic_params_x, torch.transpose(monotonic_params_y, -1, -2)], dim=1)


def grid_sample_custom(image, grid, **kwargs):
    """Custom grid sampling with bilinear interpolation."""
    N, C, IH, IW = image.shape
    _, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)
        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)
        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)
        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.reshape(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def deform(params_x, params_y, images, *, resolution=None, mode='bilinear'):
    """Apply local scaling deformation to images."""
    if resolution is None:
        resolution = images.shape[-2:]
        original_resolution = None
    else:
        original_resolution = images.shape[-2:]
        images = F.interpolate(images, size=(resolution[0], resolution[1]), mode='bilinear', align_corners=True, antialias=True)
    
    if params_x.shape[-2] == 1:
        expand_size = (params_x.shape[0], params_x.shape[-1]+1, params_x.shape[-1])
        params_x = params_x.expand(expand_size)
    if params_y.shape[-2] == 1:
        expand_size = (params_y.shape[0], params_y.shape[-1]+1, params_y.shape[-1])
        params_y = params_y.expand(expand_size)

    coarse_grid = get_coarse_adaptive_grid(params_x, params_y).to(images.device)
    denser_grid = F.interpolate(coarse_grid, size=(resolution[0], resolution[1]), mode='bilinear', align_corners=True).to(images.device)
    denser_grid = denser_grid * 2 - 1
    
    sampled_images = GridSample(images, denser_grid.permute(0, 2, 3, 1), mode=mode) if mode == 'nearest' else \
                     grid_sample_custom(images, denser_grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=True)

    if original_resolution is not None:
        sampled_images = F.interpolate(sampled_images, size=(original_resolution[0], original_resolution[1]), mode='bilinear', align_corners=True, antialias=True)

    return sampled_images


def rerrange_and_scale_tokens(phi_x, phi_y, tokens, inv_transform=False, cls_token=None,
                              num_prefix_tokens=None, mode='bilinear', defom_resolution=None):
    """Apply local scaling to Vision Transformer tokens."""
    if cls_token is not None:
        if num_prefix_tokens is None:
            cls_token = tokens[:, 0, :]
            hidden_states = tokens[:, 1:, :]
        else:
            cls_token = tokens[:, :num_prefix_tokens, :]
            hidden_states = tokens[:, num_prefix_tokens:, :]
    else:
        hidden_states = tokens

    resolution = hidden_states.shape[-2]
    h = int(resolution**0.5)
    hidden_states = hidden_states.transpose(1, 2).reshape(-1, tokens.shape[-1], h, h)
    hidden_states = deform(phi_x, phi_y, hidden_states, mode=mode, resolution=defom_resolution)
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[1], h * h).transpose(1, 2)

    if cls_token is not None:
        if num_prefix_tokens is None:
            cls_token = cls_token.unsqueeze(1)
        return torch.cat([cls_token, hidden_states], dim=1)
    
    return hidden_states


# ============================================================================
# DEM (Deep Equilibrium Model) Components
# ============================================================================

class DEMResidualBlock(nn.Module):
    """Residual block for Deep Equilibrium Model."""
    
    def __init__(self, channels_in, channels_out, filter_size, pool_rate):
        super(DEMResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels_in, channels_out, filter_size, 
                               padding=filter_size // 2, 
                               padding_mode='reflect', 
                               dilation=1)
        self.norm1 = nn.InstanceNorm2d(channels_out, affine=True)
        self.activation = nn.Softplus()
        self.projection = nn.Conv2d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()
        
        self.pool_rate = pool_rate
        if pool_rate > 1:
            self.pool = nn.MaxPool2d(pool_rate)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        residual = self.projection(x)
        out_1 = self.conv1(x)
        out = out_1 + residual
        out = self.norm1(out)
        out = self.activation(out)
        
        if self.pool_rate != 1:
            out = self.pool(out)
        
        return out


class DEMAdapter(nn.Module):
    """
    Deep Equilibrium Model Adapter for learning local scaling parameters.
    
    Uses fixed-point iteration to find equilibrium deformation parameters
    that are content-aware and spatially adaptive.
    """
    
    def __init__(self, config: LSEDinoV2Config):
        super().__init__()
        
        self.config = config
        self.num_layers = config.adapter_num_layers
        self.channels = config.adapter_channels
        self.filter_size = config.adapter_filter_size
        self.pool_rate = config.adapter_pool_rate
        self.num_phi_layers = config.num_phi_layers
        self.deep_equilibrium_steps = config.inner_epochs
        self.scale_image = config.dem_image_scale
        
        self.transform = Compose([
            Resize((224, 224)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # DEQ solvers for each phi layer
        if HAS_TORCHDEQ:
            self.deq = [get_deq(ift=False,
                               f_solver='fixed_point_iter', f_max_iter=config.inner_epochs, f_tol=1e-6,
                               b_solver='fixed_point_iter', b_max_iter=config.inner_epochs, b_tol=1e-6) 
                       for _ in range(self.num_phi_layers)]
        else:
            self.deq = None
        
        # Build CNN modules for each phi layer
        self.module_list = nn.ModuleList()
        for j in range(config.num_phi_layers):
            layers = []
            for i in range(self.num_layers):
                layers.append(DEMResidualBlock(self.channels[i], self.channels[i+1], 
                                              self.filter_size[i], self.pool_rate[i]))
            layers.append(nn.Conv2d(self.channels[-1], 2, 1))
            layers.append(nn.InstanceNorm2d(2, affine=True))
            layers.append(nn.AdaptiveAvgPool2d((config.adapter_coarse_resolution[j], 
                                                config.adapter_coarse_resolution[j])))
            layers.append(nn.Sigmoid())
            self.module_list.append(nn.Sequential(*layers))
    
    def _forward_single_phi(self, x, phi_x_batch_i, phi_y_batch_i, i):
        """Forward pass for a single phi layer."""
        x_aug = deform(phi_x_batch_i, phi_y_batch_i, x)
        score = self.module_list[i](x_aug)
        return score[:, 0, :, :-1], score[:, 1, 1:, :].transpose(-1, -2)
    
    def _simple_fixed_point_iter(self, x, phi_x, phi_y, layer_idx, num_iters=5):
        """Simple fixed-point iteration when torchdeq is not available."""
        for _ in range(num_iters):
            phi_x, phi_y = self._forward_single_phi(x, phi_x, phi_y, layer_idx)
        return phi_x, phi_y
    
    def _DEQ(self, x, phi_x_batch, phi_y_batch):
        """Run Deep Equilibrium to find optimal phi parameters."""
        updated_phi_x_batch = []
        updated_phi_y_batch = []

        if self.scale_image:
            x = self.transform(x.clone())
        else:
            x = x.clone()
            
        for i in range(self.num_phi_layers):
            if HAS_TORCHDEQ and self.deq is not None:
                f_lambda = lambda phi_x, phi_y: self._forward_single_phi(x, phi_x, phi_y, i)
                phi_xy, info = self.deq[i](f_lambda, (phi_x_batch[i], phi_y_batch[i]))
                phi_x = phi_xy[-1][0]
                phi_y = phi_xy[-1][1]
            else:
                phi_x, phi_y = self._simple_fixed_point_iter(
                    x, phi_x_batch[i], phi_y_batch[i], i, self.deep_equilibrium_steps
                )
            
            updated_phi_x_batch.append(phi_x)
            updated_phi_y_batch.append(phi_y)

        return updated_phi_x_batch, updated_phi_y_batch


class PerlayerAdapterParams(nn.Module):
    """Learnable phi parameters for each adaptive layer."""
    
    def __init__(self, num_layers: int, adapter_coarse_resolution: List[int]):
        super().__init__()
        self.num_layers = num_layers
        self.adapter_coarse_resolution = adapter_coarse_resolution
        
        # Initialize phi parameters
        # Note: DEM outputs (B, 2, res, res), then slices to (B, res, res-1) for phi_x/phi_y
        # So initial phi should match that shape
        self.param_x_list = nn.ParameterList()
        self.param_y_list = nn.ParameterList()
        
        for i in range(num_layers):
            res = adapter_coarse_resolution[i]
            # Match the shape that DEM._forward_single_phi returns:
            # phi_x = score[:, 0, :, :-1] → (B, res, res-1)
            # phi_y = score[:, 1, 1:, :].transpose(-1, -2) → (B, res-1, res) → transposed to (B, res, res-1)
            self.param_x_list.append(nn.Parameter(torch.ones(1, res, res - 1)))
            self.param_y_list.append(nn.Parameter(torch.ones(1, res, res - 1)))


# ============================================================================
# Main LSE-DINOv2 Model
# ============================================================================

class LSEDinoV2PreTrainedModel(PreTrainedModel):
    """Base class for LSE-DINOv2 models."""
    
    config_class = LSEDinoV2Config
    base_model_prefix = ""  # No prefix - keys are stored without model prefix
    supports_gradient_checkpointing = True
    _no_split_modules = ["backbone"]  # Don't split the backbone during loading
    
    def _init_weights(self, module):
        """Initialize weights - only called for newly initialized modules, not loaded ones."""
        # Skip initialization for timm backbone modules (they're loaded from checkpoint)
        if hasattr(module, '_from_timm'):
            return
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class LSEDinoV2ForImageClassification(LSEDinoV2PreTrainedModel):
    """
    LSE-DINOv2 for Image Classification.
    
    This model combines DINOv2 ViT backbone with Deep Equilibrium Model (DEM)
    based local scale adaptation for improved scale equivariance.
    """
    
    def __init__(self, config: LSEDinoV2Config):
        super().__init__(config)
        self.config = config
        
        # Import timm for base model
        try:
            from timm import create_model
        except ImportError:
            raise ImportError("timm is required for LSE-DINOv2. Install with: pip install timm")
        
        # Create base DINOv2 model
        self.backbone = create_model(
            'vit_base_patch14_reg4_dinov2',
            pretrained=False,
            num_classes=config.num_classes,
            img_size=config.image_size,
        )
        
        # DEM adapter
        self.dem = DEMAdapter(config)
        
        # Learnable phi parameters
        self.local_scale_params = PerlayerAdapterParams(
            num_layers=config.num_phi_layers,
            adapter_coarse_resolution=config.adapter_coarse_resolution
        )
        
        # Store config for forward pass
        self.augment_layer_id = config.augment_layer_id
        self.unaugment_layer_id = config.unaugment_layer_id
        self.interpolation_mode = config.interpolation_mode
        self.deform_resolution = config.deform_resolution
        self.num_prefix_tokens = config.num_prefix_tokens
        
        # Don't call post_init here - it will be called by from_pretrained
        # and we don't want to reinitialize weights after loading
        # self.post_init()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load model with proper weight handling for timm backbone.
        
        This override ensures the timm backbone weights are loaded correctly,
        bypassing HuggingFace's automatic loading which may not handle
        nested timm models properly.
        
        Args:
            pretrained_model_name_or_path: Path to model directory or HuggingFace Hub model ID
            **kwargs: Additional arguments passed to model initialization
            
        Returns:
            LSEDinoV2ForImageClassification: Loaded model with weights
        """
        from safetensors import safe_open
        
        # Load config
        config = kwargs.pop('config', None)
        if config is None:
            config = LSEDinoV2Config.from_pretrained(pretrained_model_name_or_path)
        
        # Create model
        model = cls(config)
        
        # Determine if this is a HuggingFace Hub model ID or local path
        is_hub_model = '/' in pretrained_model_name_or_path and not os.path.exists(pretrained_model_name_or_path)
        
        if is_hub_model:
            # Download from HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                print(f"Downloading model weights from HuggingFace Hub...")
                safetensor_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="model.safetensors",
                    cache_dir=kwargs.get('cache_dir', None)
                )
                print(f"✓ Model weights downloaded successfully")
            except Exception as e:
                print(f"Warning: Could not download model.safetensors from Hub: {e}")
                safetensor_path = None
        else:
            # Local path
            safetensor_path = os.path.join(pretrained_model_name_or_path, 'model.safetensors')
            if not os.path.exists(safetensor_path):
                safetensor_path = None
        
        # Load weights from safetensors
        if safetensor_path and os.path.exists(safetensor_path):
            print(f"Loading model weights from {safetensor_path}...")
            state_dict = {}
            with safe_open(safetensor_path, framework='pt') as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                # Filter out local_scale_params which are initialized separately
                missing_keys = [k for k in missing_keys if 'local_scale_params' not in k]
                if missing_keys:
                    print(f"Warning: Missing keys: {missing_keys[:5]}...")
            
            num_loaded = len(state_dict)
            print(f"✓ Loaded {num_loaded} parameters into model")
        else:
            print(f"Warning: No safetensors file found. Model weights not loaded.")
        
        return model
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_phi: bool = False,
    ) -> Union[Tuple, ImageClassifierOutput]:
        """
        Forward pass with DEM-based local scaling adaptation.
        
        Args:
            pixel_values: Input images of shape (batch, channels, height, width)
            labels: Optional classification labels
            return_dict: Whether to return a dict or tuple
            output_phi: Whether to return the learned phi parameters
        
        Returns:
            ImageClassifierOutput or tuple with logits and optional loss
        """
        batch_size = pixel_values.size(0)
        
        # Get initial phi parameters
        phi_x_batch = [
            phi_x.detach().clone().repeat(batch_size, 1, 1).to(pixel_values.device) 
            for phi_x in self.local_scale_params.param_x_list
        ]
        phi_y_batch = [
            phi_y.detach().clone().repeat(batch_size, 1, 1).to(pixel_values.device) 
            for phi_y in self.local_scale_params.param_y_list
        ]
        
        # Run DEM to find equilibrium phi parameters
        phi_x_batch, phi_y_batch = self.dem._DEQ(pixel_values, phi_x_batch, phi_y_batch)
        
        # Forward through backbone with adaptive deformations
        x = self._forward_features(pixel_values, phi_x_batch, phi_y_batch)
        logits = self.backbone.forward_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))
        
        if not return_dict:
            output = (logits,)
            if output_phi:
                output = output + (phi_x_batch, phi_y_batch)
            return ((loss,) + output) if loss is not None else output
        
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
    def _forward_features(self, x, phi_x_batch, phi_y_batch):
        """
        Forward through transformer blocks with local scaling.
        
        Matches the original timm_ada_dinov2.py implementation:
        - Applies deformation at layers specified in augment_layer_id
        - Optionally undoes deformation at layers in unaugment_layer_id
        - Preserves prefix tokens (CLS + registers) during deformation
        """
        aug_index = 0
        unaug_index = 0
        
        # Optional: Deform raw input image before patch embedding (layer_id = -1)
        if -1 in self.augment_layer_id:
            x = deform(phi_x_batch[aug_index], phi_y_batch[aug_index], x, 
                      mode=self.interpolation_mode, resolution=self.deform_resolution)
            aug_index += 1
        
        # Patch embedding
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        
        # Optional: Undo deformation after embedding (layer_id = -1)
        if -1 in self.unaugment_layer_id:
            x = rerrange_and_scale_tokens(
                phi_x_batch[unaug_index], 
                phi_y_batch[unaug_index], 
                x, 
                cls_token=True, 
                num_prefix_tokens=self.num_prefix_tokens, 
                inv_transform=True,
                mode=self.interpolation_mode, 
                defom_resolution=self.deform_resolution
            )
            unaug_index += 1
        
        # Transformer blocks with optional per-layer deformation
        for i, blk in enumerate(self.backbone.blocks):
            # Apply deformation before block if layer is in augment list
            if aug_index < len(phi_x_batch) and i in self.augment_layer_id:
                x = rerrange_and_scale_tokens(
                    phi_x_batch[aug_index], 
                    phi_y_batch[aug_index], 
                    x, 
                    cls_token=True, 
                    num_prefix_tokens=self.num_prefix_tokens, 
                    mode=self.interpolation_mode, 
                    defom_resolution=self.deform_resolution
                )
                aug_index += 1
            
            # Forward through transformer block
            x = blk(x)
            
            # Optional: Undo deformation after block
            if unaug_index < len(phi_x_batch) and i in self.unaugment_layer_id:
                x = rerrange_and_scale_tokens(
                    phi_x_batch[unaug_index], 
                    phi_y_batch[unaug_index], 
                    x, 
                    cls_token=True, 
                    num_prefix_tokens=self.num_prefix_tokens, 
                    inv_transform=True,
                    mode=self.interpolation_mode, 
                    defom_resolution=self.deform_resolution
                )
                unaug_index += 1
        
        x = self.backbone.norm(x)
        return x
    
    def get_phi_parameters(self, pixel_values: torch.Tensor):
        """
        Get the learned phi parameters for visualization.
        
        Args:
            pixel_values: Input images of shape (batch, channels, height, width)
        
        Returns:
            phi_x_batch, phi_y_batch: Lists of phi parameters for each layer
        """
        batch_size = pixel_values.size(0)
        
        phi_x_batch = [
            phi_x.detach().clone().repeat(batch_size, 1, 1).to(pixel_values.device) 
            for phi_x in self.local_scale_params.param_x_list
        ]
        phi_y_batch = [
            phi_y.detach().clone().repeat(batch_size, 1, 1).to(pixel_values.device) 
            for phi_y in self.local_scale_params.param_y_list
        ]
        
        with torch.no_grad():
            phi_x_batch, phi_y_batch = self.dem._DEQ(pixel_values, phi_x_batch, phi_y_batch)
        
        return phi_x_batch, phi_y_batch


# Register the model for auto classes
LSEDinoV2Config.register_for_auto_class()
LSEDinoV2ForImageClassification.register_for_auto_class("AutoModelForImageClassification")

