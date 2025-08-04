from timm.models.beit import *
from torch import nn
from timm.models._manipulate import checkpoint
from types import MethodType
from utils.sampling_utils import *
from models.canonicalizer_wrapper import CanonicalizeWrapper

########################################################################################################
# this file do monkey patching to the model to make it adaptive
# The steps are:
# 1. add additional components to the model
# 2. add interleaving local scaling canonicalization by DEM at different stages of the forward pass
########################################################################################################

def custom_forward(self, x):
    phi_x_list = self.local_scale_params.param_x_list
    phi_y_list = self.local_scale_params.param_y_list
    batch_size = x.size(0)
    phi_x_batch = [phi_x.clone().repeat(batch_size, 1, 1).requires_grad_(True).to(x.device) for phi_x in phi_x_list]
    phi_y_batch = [phi_y.clone().repeat(batch_size, 1, 1).requires_grad_(True).to(x.device) for phi_y in phi_y_list]
    phi_x_batch, phi_y_batch = self.DEM._DEQ(x, phi_x_batch, phi_y_batch)
    
    # store local scaling parameters in the model temporalily
    self.tem_x_batch = phi_x_batch
    self.tem_y_batch = phi_y_batch
    
    x = self.forward_features(x, phi_x_batch, phi_y_batch)
    x = self.forward_head(x)
    return x

def custom_forward_features(self, x, phi_x_batch, phi_y_batch):
    
    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    aug_index = 0
    unaug_index = 0
    for i, blk in enumerate(self.blocks):
         
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
            
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
        else:
            x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        
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
    
    x

def custom_forward_cannon(self, x):
    cannoned_x, _, _ = self.cannonicalizer(x)
    cannoned_x = self.forward_features(cannoned_x)
    cannoned_x = self.forward_head(cannoned_x)
    return cannoned_x
    

def convert_beit_to_canonicalizer(model, adaptation_config):
    cannon_wrapper = CanonicalizeWrapper(None, 
                                  num_layers=adaptation_config.canon_num_layers,
                                  num_channels=adaptation_config.canon_num_channels,
                                  kernel_sizes=adaptation_config.canon_kernel_sizes,
                                  task=adaptation_config.task,
                                  unique_params_limit=adaptation_config.unique_params_limit,
                                  discrete_values=adaptation_config.can_discrete_vals)
    
    model.cannonicalizer = cannon_wrapper
    model.forward = MethodType(custom_forward_cannon, model)
   
def convert_beit_to_dem(model, local_scale_params, DEM_model, aug_ids, unaug_ids, deform_res, interpolation_mode):
    
    # add additional components to the model
    model.local_scale_params = local_scale_params
    model.DEM = DEM_model
    model.tem_x_batch = None
    model.tem_y_batch = None
    model.augment_layer_id = aug_ids
    model.unaugment_layer_id = unaug_ids
    model.interpolation_mode = interpolation_mode
    model.deform_resolution = deform_res
    
    print("Converting beit model")
    model.forward_features = MethodType(custom_forward_features, model)
    model.forward = MethodType(custom_forward, model)
    return model

def convert_beit_model(model, adaptation_config, local_scale_params, DEM_model):
    if adaptation_config.do_cannonicalization:
        convert_beit_to_canonicalizer(model, adaptation_config)
    else:
        model = convert_beit_to_dem(
            model, 
            local_scale_params, 
            DEM_model, 
            adaptation_config.augment_layer_id, 
            adaptation_config.unaugment_layer_id, 
            adaptation_config.deform_resolution,
            adaptation_config.interpolation_mode
        )
    return model