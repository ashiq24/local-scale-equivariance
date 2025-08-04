import torch
from torch import nn
from train.training_handler import *
from utils.sampling_utils import *
from layers.adapter import *
import higher
import gc


class AdapterWrapper(nn.Module):
    '''
    combines the backbone model and the adapter model (DEM) to create a single model to ease evaluation.
    '''
    def __init__(self,
                 *, 
                 model, 
                 surrogate_model,
                 phi_x_list,
                 phi_y_list,
                 params):
        super(AdapterWrapper, self).__init__()
        self.model = model
        self.surrogate_model = surrogate_model
        self.phi_x_list = phi_x_list
        self.phi_y_list = phi_y_list
        self.params = params


    def forward(self,
                images,
                return_params=False,
                return_features=False):
        if self.phi_x_list is None:
            return self.model(images, return_features=return_features)
        batch_size = images.size(0)
        # expand the param_x and param_y to the batch size
        self.phi_x_list[0] = self.phi_x_list[0].requires_grad_(True)

        phi_x_batch, phi_y_batch = inner_optimization(model=self.model,
                                                      canonicalizer_model=self.surrogate_model,
                                                      images=images,
                                                      params=self.params,
                                                      phi_x_list=self.phi_x_list,
                                                      phi_y_list=self.phi_y_list,
                                                      create_grad_graph=False, # wrapper is only used in 
                                                      wandb_log=self.params.wandb_log)

        y = self.model(
            images,
            phi_x_list=phi_x_batch,
            phi_y_list=phi_y_batch,
            return_features=return_features,
            skip_input_augmentation=getattr(
                self.params,
                "skip_input_augmentation",
                False))

        if return_params:
            return y, phi_x_batch, phi_y_batch

        return y
