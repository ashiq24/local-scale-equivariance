import torch
from timm.models._manipulate import checkpoint
from torch import nn
from types import MethodType
from utils.sampling_utils import *
from models.canonicalizer_wrapper import CanonicalizeWrapper

########################################################################################################
# This file does monkey patching to timm's Eva-based DINOv3 models to make them adaptive.
#
# IMPORTANT: DINOv3 uses the Eva architecture (NOT VisionTransformer like DINOv2!)
# Key differences from DINOv2:
#   1. Uses RoPE (Rotary Position Embeddings) - _pos_embed returns (x, rot_pos_embed)
#   2. No patch_drop module
#   3. Blocks take rope=rot_pos_embed as an argument
#   4. Based on timm's Eva class, not VisionTransformer
#
# The adapter:
#   1. Adds DEM, phi parameters, layer IDs to the model
#   2. Interleaves local scaling at different stages of the forward pass
#   3. Handles num_prefix_tokens (CLS + register tokens) correctly
########################################################################################################


def custom_forward(self, x):
    """
    Custom forward pass with DEM-based local scaling adaptation for Eva/DINOv3.

    Flow:
    1. Get phi parameters from DEM (Deep Equilibrium)
    2. Apply deformations at specified layers
    3. Store phi for potential unique optima loss
    """
    phi_x_list = self.local_scale_params.param_x_list
    phi_y_list = self.local_scale_params.param_y_list
    batch_size = x.size(0)

    # Clone and expand phi parameters to batch size
    phi_x_batch = [phi_x.clone().repeat(batch_size, 1, 1).requires_grad_(True).to(x.device) for phi_x in phi_x_list]
    phi_y_batch = [phi_y.clone().repeat(batch_size, 1, 1).requires_grad_(True).to(x.device) for phi_y in phi_y_list]

    # DEM: Find equilibrium phi parameters
    phi_x_batch, phi_y_batch = self.DEM._DEQ(x, phi_x_batch, phi_y_batch)

    # Store local scaling parameters in the model temporarily for unique optima loss
    self.tem_x_batch = phi_x_batch
    self.tem_y_batch = phi_y_batch

    # Forward pass with adaptive deformations
    x = self.forward_features(x, phi_x_batch, phi_y_batch)
    x = self.forward_head(x)
    return x


def custom_forward_features(self, x, phi_x_batch, phi_y_batch):
    """
    Forward through Eva/DINOv3 blocks with local scaling at specified layers.

    Eva/DINOv3 specifics:
    - Uses RoPE (Rotary Position Embeddings)
    - _pos_embed returns (x, rot_pos_embed) tuple
    - No patch_drop module
    - Blocks take rope=rot_pos_embed argument
    - num_prefix_tokens includes CLS + register tokens (typically 5 for reg4 models)
    """
    aug_index = 0
    unaug_index = 0

    # Optional: Deform raw input image before patch embedding
    if -1 in self.augment_layer_id:
        x = deform(phi_x_batch[aug_index], phi_y_batch[aug_index], x, mode=self.interpolation_mode, resolution=self.deform_resolution)
        aug_index += 1

    # Patch embedding
    x = self.patch_embed(x)
    
    # Position embedding - Eva returns (x, rot_pos_embed) tuple for RoPE
    pos_result = self._pos_embed(x)
    if isinstance(pos_result, tuple):
        x, rot_pos_embed = pos_result
    else:
        # Fallback for models without RoPE
        x = pos_result
        rot_pos_embed = None
    
    # norm_pre (Eva has this, apply if exists)
    if hasattr(self, 'norm_pre') and self.norm_pre is not None:
        x = self.norm_pre(x)

    # Optional: Undo deformation after embedding
    if -1 in self.unaugment_layer_id:
        x = rerrange_and_scale_tokens(
            phi_x_batch[unaug_index],
            phi_y_batch[unaug_index],
            x,
            cls_token=True,
            num_prefix_tokens=self.num_prefix_tokens,
            inv_transform=True,
            mode=self.interpolation_mode,
            defom_resolution=self.deform_resolution,
        )
        unaug_index += 1

    # Check if using mixed mode RoPE (depth-dependent)
    rope_mixed = getattr(self, 'rope_mixed', False)

    # Transformer blocks with optional per-layer deformation
    for i, blk in enumerate(self.blocks):
        # Apply deformation before block if layer is in augment list
        if aug_index < len(phi_x_batch) and i in self.augment_layer_id:
            x = rerrange_and_scale_tokens(
                phi_x_batch[aug_index],
                phi_y_batch[aug_index],
                x,
                cls_token=True,
                num_prefix_tokens=self.num_prefix_tokens,
                mode=self.interpolation_mode,
                defom_resolution=self.deform_resolution,
            )
            aug_index += 1

        # Forward through transformer block
        # Eva blocks take rope argument for RoPE
        if rope_mixed and rot_pos_embed is not None:
            # Depth-dependent RoPE
            rope_for_block = rot_pos_embed[i]
        else:
            rope_for_block = rot_pos_embed
            
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(blk, x, rope=rope_for_block)
        else:
            x = blk(x, rope=rope_for_block)

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
                defom_resolution=self.deform_resolution,
            )
            unaug_index += 1

    x = self.norm(x)
    return x


def custom_forward_cannon(self, x):
    """
    Forward pass with traditional canonicalization (discrete set of deformations).
    Used for baseline comparison.
    """
    cannoned_x, deform_params_x_repeated, deform_params_y_repeated = self.cannonicalizer(x)
    cannoned_x = self.forward_features(cannoned_x)
    cannoned_x = self.forward_head(cannoned_x)
    return cannoned_x


def convert_dinov3_to_canonicalizer(model, adaptation_config):
    """Add discrete canonicalization wrapper to an Eva/DINOv3 model."""
    cannon_wrapper = CanonicalizeWrapper(
        None,
        num_layers=adaptation_config.canon_num_layers,
        num_channels=adaptation_config.canon_num_channels,
        kernel_sizes=adaptation_config.canon_kernel_sizes,
        task=adaptation_config.task,
        unique_params_limit=adaptation_config.unique_params_limit,
        discrete_values=adaptation_config.can_discrete_vals,
    )

    model.cannonicalizer = cannon_wrapper
    model.forward = MethodType(custom_forward_cannon, model)


def convert_dinov3_to_dem(model, local_scale_params, DEM_model, aug_ids, unaug_ids, deform_res, interpolation_mode="bilinear"):
    """
    Add DEM-based adaptive local scaling to an Eva/DINOv3 model.

    Args:
        model: Eva backbone with DINOv3 weights (e.g., vit_base_patch16_dinov3.lvd1689m)
        local_scale_params: PerlayerAdapterParams with phi_x/phi_y lists
        DEM_model: DEMAdapter that finds equilibrium phi parameters
        aug_ids: Layer IDs to apply deformation (e.g., [-1, 4, 8])
        unaug_ids: Layer IDs to undo deformation (e.g., [-100] for never)
        deform_res: Target resolution for deformation processing
        interpolation_mode: 'bilinear' or 'nearest' for grid sampling
    """
    # Add additional components to the model
    model.local_scale_params = local_scale_params
    model.DEM = DEM_model
    model.tem_x_batch = None  # Temporary storage for phi (used in unique optima loss)
    model.tem_y_batch = None
    model.augment_layer_id = aug_ids
    model.unaugment_layer_id = unaug_ids
    model.interpolation_mode = interpolation_mode
    model.deform_resolution = deform_res

    # Replace forward methods with adaptive versions
    model.forward_features = MethodType(custom_forward_features, model)
    model.forward = MethodType(custom_forward, model)
    return model


def convert_dinov3_model(model, adaptation_config, local_scale_params, DEM_model):
    """
    Main entry point for converting an Eva/DINOv3 backbone to adaptive version.

    Supports two modes:
    1. Canonicalization: Discrete set of deformations
    2. DEM adaptation: Learned continuous deformations via Deep Equilibrium
    """
    if adaptation_config.do_cannonicalization:
        convert_dinov3_to_canonicalizer(model, adaptation_config)
    else:
        model = convert_dinov3_to_dem(
            model,
            local_scale_params,
            DEM_model,
            adaptation_config.augment_layer_id,
            adaptation_config.unaugment_layer_id,
            adaptation_config.deform_resolution,
            adaptation_config.interpolation_mode,
        )
    return model
