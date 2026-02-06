from timm.models.vision_transformer import *
from timm.models._manipulate import checkpoint
from torch import nn
from types import MethodType
from utils.sampling_utils import *
from models.canonicalizer_wrapper import CanonicalizeWrapper

########################################################################################################
# this file does monkey patching to the timm DINOv2 model to make it adaptive
# The steps are:
# 1. add additional components to the model (DEM, phi parameters, layer IDs)
# 2. add interleaving local scaling canonicalization by DEM at different stages of the forward pass
# 3. **DINOv2 with registers:** Model has num_prefix_tokens=5 (1 CLS + 4 register tokens)
#    These prefix tokens are preserved and not deformed during local scaling
########################################################################################################

def custom_forward(self, x):
    """
    Custom forward pass with DEM-based local scaling adaptation.
    
    Flow:
    1. Get phi parameters from DEM (Deep Equilibrium)
    2. Apply deformations at specified layers
    3. Store phi for potential unique optima loss
    """
    phi_x_list = self.local_scale_params.param_x_list
    phi_y_list = self.local_scale_params.param_y_list 
    batch_size = x.size(0)
    
    # Clone and expand phi parameters to batch size
    phi_x_batch = [phi_x.detach().to(x.device).repeat(batch_size, 1, 1).requires_grad_(True) for phi_x in phi_x_list]
    phi_y_batch = [phi_y.detach().to(x.device).repeat(batch_size, 1, 1).requires_grad_(True) for phi_y in phi_y_list]
    
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
    Forward through transformer blocks with local scaling at specified layers.
    
    DINOv2 specifics:
    - num_prefix_tokens = 5 (1 CLS + 4 registers for reg4 model)
    - These tokens are automatically handled by rerrange_and_scale_tokens
    - Only spatial patch tokens (14x14 = 196 for 224x224 input) are deformed
    """
    aug_index = 0
    unaug_index = 0
    
    # Optional: Deform raw input image before patch embedding
    if -1 in self.augment_layer_id:
        x = deform(phi_x_batch[aug_index], phi_y_batch[aug_index], x, mode=self.interpolation_mode, resolution=self.deform_resolution)
        aug_index += 1
        
    # Patch embedding
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    
    # Optional: Undo deformation after embedding
    if -1 in self.unaugment_layer_id:
        # Note: For DINOv2 with registers, num_prefix_tokens=5
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
                defom_resolution=self.deform_resolution
            )
            aug_index += 1
            
        # Forward through transformer block
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(blk, x)
        else:
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
    

def convert_dinov2_to_canonicalizer(model, adaptation_config):
    """Add discrete canonicalization wrapper to DINOv2 model."""
    cannon_wrapper = CanonicalizeWrapper(
        None, 
        num_layers=adaptation_config.canon_num_layers,
        num_channels=adaptation_config.canon_num_channels,
        kernel_sizes=adaptation_config.canon_kernel_sizes,
        task=adaptation_config.task,
        unique_params_limit=adaptation_config.unique_params_limit,
        discrete_values=adaptation_config.can_discrete_vals
    )
    
    model.cannonicalizer = cannon_wrapper
    model.forward = MethodType(custom_forward_cannon, model)
   
def convert_dinov2_to_dem(model, local_scale_params, DEM_model, aug_ids, unaug_ids, deform_res, interpolation_mode='bilinear'):
    """
    Add DEM-based adaptive local scaling to DINOv2 model.
    
    Args:
        model: timm DINOv2 model (e.g., vit_base_patch14_reg4_dinov2)
        local_scale_params: PerlayerAdapterParams with phi_x/phi_y lists
        DEM_model: DEMAdapter that finds equilibrium phi parameters
        aug_ids: Layer IDs to apply deformation (e.g., [-1, 0, 1])
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

def convert_dinov2_model(model, adaptation_config, local_scale_params, DEM_model):
    """
    Main entry point for converting timm DINOv2 to adaptive version.
    
    Supports two modes:
    1. Canonicalization: Discrete set of deformations
    2. DEM adaptation: Learned continuous deformations via Deep Equilibrium
    """
    if adaptation_config.do_cannonicalization:
        convert_dinov2_to_canonicalizer(model, adaptation_config)
    else:
        model = convert_dinov2_to_dem(
            model, 
            local_scale_params, 
            DEM_model, 
            adaptation_config.augment_layer_id, 
            adaptation_config.unaugment_layer_id, 
            adaptation_config.deform_resolution,
            adaptation_config.interpolation_mode
        )
    return model
