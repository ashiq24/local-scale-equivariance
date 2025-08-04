import torch
import torch.nn as nn
from utils.sampling_utils import *
import numpy as np
import torch.nn.functional as F
from torchdeq import get_deq
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


######
# Deep Equilibrium Model (DEM)
######

class DEMResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out, filter_size, pool_rate):
        '''
        Residual block for Deep Equilibrium Model (DEM)
        '''
        super(DEMResidualBlock, self).__init__()
        
        # First convolution (dilation=1)
        self.conv1 = nn.Conv2d(channels_in, channels_out, filter_size, 
                               padding=filter_size // 2, 
                               padding_mode='reflect', 
                               dilation=1)
        
        self.norm1 = nn.InstanceNorm2d(channels_out, affine=True)
        
        # Activation
        self.activation = nn.Softplus()
        
        # Projection for residual connection when in/out channels differ
        self.projection = nn.Conv2d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()
        
        self.pool_rate = pool_rate
        if pool_rate > 1:
            self.pool = nn.MaxPool2d(pool_rate)
        else:
            self.pool = nn.Identity()

    def forward(self, x):

        # Residual (skip) connection
        residual = self.projection(x)
        
        # First conv + norm + activation
        out_1 = self.conv1(x)
        
        # Skip connection
        out = out_1 + residual
        out = self.norm1(out)
        out = self.activation(out)
        
        # Optional pooling
        if self.pool_rate != 1:
            out = self.pool(out)
        
        return out


class DEMAdapter(nn.Module):
    def __init__(self,
                 params):
        super().__init__()

        self.params = params

        self.module_list = nn.ModuleList()
        self.num_layers = params.adapter_num_layers # number of layers in DEM 
        self.channels = params.adapter_channels # number of channels in each layer
        self.filter_size = params.adapter_filter_size # filter size for each layer
        self.pool_rate = params.adapter_pool_rate # pooling rate for each layer
        self.num_phi_layers = params.num_phi_layers # number of adaptive layers used in backbone, we have one DEM for each adaptive layer
        self.deep_equilibrium_steps = params.inner_epochs
        self.scale_image = getattr(params, 'dem_image_scale', True)
        self.transform = Compose([
                                Resize((224, 224)),  # Resize image
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
                                ])
        # this module finds the equilibrium state of the DEM
        self.deq = [get_deq(ift=False,
                            f_solver='anderson', f_max_iter=params.inner_epochs, f_tol=1e-6,
                            b_solver='anderson', b_max_iter=params.inner_epochs + 3 , b_tol=1e-6) for _ in range(self.num_phi_layers)]
        
        self.module_list = nn.ModuleList()
        for j in range(params.num_phi_layers):
            k = []
            for i in range(self.num_layers):
                k.append(DEMResidualBlock(self.channels[i], self.channels[i+1], self.filter_size[i], self.pool_rate[i]))

            k.append(nn.Conv2d(self.channels[-1], 2, 1))
            k.append(nn.InstanceNorm2d(2, affine=True))
            k.append(torch.nn.AdaptiveAvgPool2d((params.adapter_coarse_resolution[j], params.adapter_coarse_resolution[j])))
            k.append(nn.Sigmoid())
            self.module_list.append(nn.Sequential(*k))
    
    def _DEQ(self, x, phi_x_batch, phi_y_batch):
        updated_phi_x_batch = []
        updated_phi_y_batch = []

        if self.scale_image:
            x = self.transform(x.clone())
        else:
            x = x.clone()
            
        for i in range(self.num_phi_layers):
            f_lambda = lambda phi_x, phi_y : self._forward_single_phi(x, phi_x, phi_y , i)
            phi_xy, info = self.deq[i]( f_lambda, (phi_x_batch[i], phi_y_batch[i]))
            phi_x = phi_xy[-1][0]
            phi_y = phi_xy[-1][1]
            updated_phi_x_batch.append(phi_x)
            updated_phi_y_batch.append(phi_y)

        return updated_phi_x_batch, updated_phi_y_batch


    def _forward_single_phi(self, x, phi_x_batch_i, phi_y_batch_i, i):
        x_aug = deform(phi_x_batch_i, phi_y_batch_i, x)
        score = self.module_list[i](x_aug)
        return score[:, 0, :, :-1], score[:, 1, 1:, :].transpose(-1, -2)


######
# Model for inner optimization i.e., direct optimization based canonicalization
######


class ResidualBlock(nn.Module):
    '''
    generic residual block
    '''
    def __init__(self, channels_in, channels_out, filter_size, pool_rate):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, filter_size, padding=filter_size // 2)
        self.norm = nn.InstanceNorm2d(channels_out, affine=False)
        self.activation = nn.GELU()
        self.pool_rate = pool_rate
        if pool_rate > 1:
            self.pool = nn.AvgPool2d(pool_rate)
        else:
            self.pool = nn.Identity()

        self.projection = nn.Conv2d(channels_in, channels_out, 1)
    def forward(self, x):
        residual = self.projection(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)  + residual
        return out


class Adapter(nn.Module):
    def __init__(self,
                 params):
        super().__init__()

        self.params = params

        self.module_list = nn.ModuleList()
        self.num_layers = params.adapter_num_layers
        self.channels = params.adapter_channels
        self.filter_size = params.adapter_filter_size
        self.pool_rate = params.adapter_pool_rate
        self.num_phi_layers = params.num_phi_layers
        self.transform = Compose([
                                Resize((224, 224)),  # Resize image
                                ToTensor(),          # Convert to tensor
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
                                ])
        
        self.module_list = nn.ModuleList()
        for j in range(params.num_phi_layers):
            k = []
            for i in range(self.num_layers):
                k.append(ResidualBlock(self.channels[i], self.channels[i+1], self.filter_size[i], self.pool_rate[i]))
            k.append(nn.Conv2d(self.channels[-1], 1, 1))
            self.module_list.append(nn.Sequential(*k))
    def forward(self, x, phi_x_batch, phi_y_batch):
        loss = 0
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        for i in range(self.num_phi_layers):
            x_aug = deform(phi_x_batch[i], phi_y_batch[i], x)
            score = self.module_list[i](x_aug)
            loss += -1*torch.mean(score, dim=(1,2,3))
        return loss
