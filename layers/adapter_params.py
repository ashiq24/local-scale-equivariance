import torch
from torch import nn
import torch.nn.functional as F
from layers.adapter import *

class PerlayerAdapterParams:
    def __init__(self,
                 *,
                 num_layers: int,
                 adapter_coarse_resolution: list[int]):
        """
        num_layers: number of blocks in the backbone model to do the adaptation. For example,
                    Swin transfor have 4 stages.
                    So for Swin we can set num_layers=4, to do the adaptation (cannonicalization) at the begining and end of each stage.
                    And if num_layers=2, it will do the adaptation at the begining and end of the first two stages.

        adapter_coarse_resolution: The grid size for local scaling operation by monotine function.
        """
        self.num_layers = num_layers
        self.adapter_coarse_resolution = adapter_coarse_resolution

        shape_fix = 1
        
        normalization_factor = [ 1 for i in range(num_layers)]

        '''
        Identity initialization
        '''
        self.param_x_list = [
                normalization_factor[i] *
                torch.ones(
                    1,
                    adapter_coarse_resolution[i],
                    adapter_coarse_resolution[i] - shape_fix,
                    requires_grad=True) for i in range(
                self.num_layers)]
        self.param_y_list = [
                normalization_factor[i] *
                torch.ones(
                    1,
                    adapter_coarse_resolution[i],
                    adapter_coarse_resolution[i] - shape_fix,
                    requires_grad=True) for i in range(
                self.num_layers)]
        

    def to(self, device, dtype=None):
        self.param_x_list = [param.to(device, dtype=dtype) for param in self.param_x_list]
        self.param_y_list = [param.to(device, dtype=dtype) for param in self.param_y_list]
