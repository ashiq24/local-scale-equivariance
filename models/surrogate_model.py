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
        Actual residual block for Deep Equilibrium Model (DEM)
        Standard residual structure: Conv -> Norm -> Activation -> Conv -> Norm -> Add + Activation
        '''
        super(DEMResidualBlock, self).__init__()
        
        padding = (filter_size - 1) // 2
        
        # First convolution branch - bias=False since norm follows
        self.conv1 = nn.Conv2d(channels_in, channels_out, filter_size, 
                               padding=padding, 
                               bias=True)
        self.norm1 = nn.InstanceNorm2d(channels_out, affine=False)
        
        # Second convolution branch - bias=False since norm follows
        self.conv2 = nn.Conv2d(channels_out, channels_out, filter_size, 
                               padding=padding, 
                               bias=True)
        self.norm2 = nn.InstanceNorm2d(channels_out, affine=False)
        
        # Projection for residual connection when in/out channels differ
        self.projection = nn.Conv2d(channels_in, channels_out, 1, bias=True) if channels_in != channels_out else None
        
        self.activation = nn.ReLU(inplace=True)
        
        self.pool_rate = pool_rate
        self.pool = nn.MaxPool2d(pool_rate) if pool_rate > 1 else None

    def forward(self, x):
        # Save input for skip connection
        residual = self.projection(x) if self.projection is not None else x
        
        # Main branch: Conv -> Norm -> Activation -> Conv -> Norm
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Add skip connection and activate
        out = out + residual
        out = self.activation(out)
        
        # Optional pooling
        if self.pool is not None:
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
        
        # Pre-compute normalization constants for efficiency
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # this module finds the equilibrium state of the DEM
        # Reduced tolerance for faster convergence in practice
        self.deq = [get_deq(ift=False,
                            f_solver='fixed_point_iter', f_max_iter=params.inner_epochs, f_tol=1e-4,  # Relaxed from 1e-6
                            b_solver='fixed_point_iter', b_max_iter=params.inner_epochs, b_tol=1e-4) for _ in range(self.num_phi_layers)]
        
        self.module_list = nn.ModuleList()
        for j in range(params.num_phi_layers):
            k = []
            for i in range(self.num_layers):
                k.append(DEMResidualBlock(self.channels[i], self.channels[i+1], self.filter_size[i], self.pool_rate[i]))

            # Combine final layers efficiently - bias=False where norm follows
            k.append(nn.Conv2d(self.channels[-1], 2, 1, bias=True))
            k.append(nn.InstanceNorm2d(2, affine=False))
            k.append(torch.nn.AdaptiveAvgPool2d((params.adapter_coarse_resolution[j], params.adapter_coarse_resolution[j])))
            # Sigmoid will be applied in functional form for potential inplace optimization
            self.module_list.append(nn.Sequential(*k))
    
    def _DEQ(self, x, phi_x_batch, phi_y_batch):
        updated_phi_x_batch = []
        updated_phi_y_batch = []

        if self.scale_image:
            # Efficient resize + normalize without unnecessary clones or multiple transforms
            # Use functional API for potentially better memory usage
            x_scaled = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x_scaled = (x_scaled - self.mean) / self.std
        else:
            x_scaled = x  # Avoid clone, use directly
            
        for i in range(self.num_phi_layers):
            f_lambda = lambda phi_x, phi_y : self._forward_single_phi(x_scaled, phi_x, phi_y , i)
            if self.training:
                # add small noise to phi_x_batch and phi_y_batch for stability during training
                noise_level = 0.1
                phi_x_batch_i = phi_x_batch[i] + noise_level/phi_x_batch[i].shape[-1] * torch.randn_like(phi_x_batch[i])
                phi_y_batch_i = phi_y_batch[i] + noise_level/phi_y_batch[i].shape[-1] * torch.randn_like(phi_y_batch[i])
                # sigmoid to ensure positivity
                phi_x_batch_i = torch.sigmoid(phi_x_batch_i)
                phi_y_batch_i = torch.sigmoid(phi_y_batch_i)
            else:
                phi_x_batch_i = phi_x_batch[i]
                phi_y_batch_i = phi_y_batch[i]
            phi_xy, info = self.deq[i]( f_lambda, (phi_x_batch_i, phi_y_batch_i))
            phi_x = phi_xy[-1][0]
            phi_y = phi_xy[-1][1]
            updated_phi_x_batch.append(phi_x)
            updated_phi_y_batch.append(phi_y)

        return updated_phi_x_batch, updated_phi_y_batch


    def _forward_single_phi(self, x, phi_x_batch_i, phi_y_batch_i, i):
        x_aug = deform(phi_x_batch_i, phi_y_batch_i, x)
        score = self.module_list[i](x_aug)
        # Apply sigmoid efficiently - functional version is often faster
        score = torch.sigmoid(score)
        new_phi_x = score[:, 0, :, :-1]
        new_phi_y = score[:, 1, 1:, :].transpose(-1, -2)
        
        # alpha blending for stability
        alpha = 0.95
        new_phi_x = alpha * new_phi_x + (1 - alpha) * phi_x_batch_i
        new_phi_y = alpha * new_phi_y + (1 - alpha) * phi_y_batch_i
        
        return new_phi_x, new_phi_y


######
# Model for inner optimization i.e., direct optimization based canonicalization
######


class ResidualBlock(nn.Module):
    '''
    generic residual block - optimized version
    '''
    def __init__(self, channels_in, channels_out, filter_size, pool_rate):
        super(ResidualBlock, self).__init__()
        # bias=False since norm follows
        self.conv = nn.Conv2d(channels_in, channels_out, filter_size, padding=filter_size // 2, bias=False)
        self.norm = nn.InstanceNorm2d(channels_out, affine=False)
        # GELU doesn't have native inplace, but we can use functional for efficiency
        self.pool_rate = pool_rate
        self.pool = nn.AvgPool2d(pool_rate) if pool_rate > 1 else None

        self.projection = nn.Conv2d(channels_in, channels_out, 1, bias=False) if channels_in != channels_out else None
        
    def forward(self, x):
        residual = self.projection(x) if self.projection is not None else x
        out = self.conv(x)
        out = self.norm(out)
        # Fused activation + residual
        out = F.gelu(out) + residual
        return out if self.pool is None else self.pool(out)


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
