from .timm_ada_swin import *
from .timm_ada_vit import *
from .timm_ada_beit import *

dict = { 'swin': convert_swin_model,
        'vit': convert_vit_model,
        'deit': convert_vit_model,
        'beit': convert_beit_model }


def convert_model(model, adaptation_config, local_scale_params, DEM_model):
    return dict[adaptation_config.model_name](model, adaptation_config, local_scale_params, DEM_model)