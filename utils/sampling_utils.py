import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.nn.functional import grid_sample as GridSample


def apply_smoothing(grid, kernel_size=5):
    """
    Apply Gaussian smoothing to parameter grids to ensure smooth deformations.
    
    This function smooths the adaptive sampling parameters to prevent abrupt changes
    in the sampling grid, which could cause artifacts or instability during training.
    
    Args:
        grid (torch.Tensor): Input parameter grid of shape (batch, h, w)
                           Contains the raw adaptive sampling parameters
        kernel_size (int): Size of the Gaussian blur kernel (default: 5)
                         Larger values create smoother deformations
    
    Returns:
        torch.Tensor: Smoothed grid of same shape as input (batch, h, w)
    """
    smoothing_kernel = T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.2, 0.2))
    return smoothing_kernel(grid.unsqueeze(1)).squeeze(1)


def normalize_cumsum(grid):
    """
    Create monotonic cumulative sum along the last dimension for grid generation.
    
    This function converts probability-like parameters into monotonic coordinates
    by computing cumulative sums. This ensures that the resulting grid coordinates
    are monotonically increasing, which is required for valid spatial transformations.
    
    Args:
        grid (torch.Tensor): Input parameter grid of shape (batch, h, w)
                           The w-axis represents the grid direction where cumsum is applied
                           Values should typically be positive (post-softmax)
    
    Returns:
        torch.Tensor: Monotonic grid of same shape (batch, h, w)
                     Values increase monotonically along the last dimension
    """
    monotonic_grid = torch.cumsum(grid, dim=-1)
    monotonic_grid = monotonic_grid 
    return monotonic_grid 


def get_coarse_adaptive_grid(params_x, params_y):
    """
    Generate a coarse adaptive (locally scaled) sampling grid from learnable parameters to perfrom local scaling of an image.
    
    This is the core function that converts raw learnable parameters into a valid
    sampling grid. The grid defines how to spatially transform input images/features
    by specifying new coordinate locations for each pixel.
    
    Args:
        params_x (torch.Tensor): X-axis sampling parameters of shape (batch, h, w)
                                Raw learnable parameters for horizontal deformation
        params_y (torch.Tensor): Y-axis sampling parameters of shape (batch, h, w)  
                                Raw learnable parameters for vertical deformation
    
    Returns:
        torch.Tensor: Adaptive sampling grid of shape (batch, 2, h+1, w+1)
                     dim=1 contains [x_coordinates, y_coordinates]
                     Grid values are in range [0, 1] representing normalized coordinates
    
    """
    params_x_smooth = F.softmax(3 * apply_smoothing(params_x), dim=-1)
    params_y_smooth = F.softmax(3 * apply_smoothing(params_y), dim=-1)
    params_x_smooth = torch.cat([torch.zeros_like(params_x_smooth[:, :, 0]).unsqueeze(2), params_x_smooth], dim=2)
    params_y_smooth = torch.cat([torch.zeros_like(params_y_smooth[:, :, 0]).unsqueeze(2), params_y_smooth], dim=2)
    monotonic_params_x = normalize_cumsum(params_x_smooth)
    monotonic_params_y = normalize_cumsum(params_y_smooth)
    return torch.stack([monotonic_params_x, torch.transpose(monotonic_params_y, -1, -2)], dim=1)


def deform(params_x, params_y, images, *,
           resolution=None,
           mode='bilinear'):
    """
    Apply local scaling to input images using learned parameters.
    
    This is the main forward transformation function that applies the learned
    local scaling to the input images or feature maps. It first deforms the regular grid of the image accoring to the local scaling parameters, and then use the deformed grid to sample the image.
    
    Args:
        params_x (torch.Tensor): X-axis deformation parameters of shape (batch, h, w)
                                Raw learnable parameters for horizontal deformation
        params_y (torch.Tensor): Y-axis deformation parameters of shape (batch, h, w)
                                Raw learnable parameters for vertical deformation  
        images (torch.Tensor): Input images/features of shape (batch, channels, height, width)
                              The spatial dimensions will be transformed according to params
        resolution (tuple, optional): Target resolution (height, width) for processing
                                    If None, uses original image resolution
        mode (str): Interpolation mode for grid sampling ('bilinear' or 'nearest')
                   Default: 'bilinear' for smooth transformations
    
    Returns:
        torch.Tensor: Deformed images of same shape as input (batch, channels, height, width)
                     Spatial content is rearranged according to learned parameters
    """
    if resolution is None:
        resolution = images.shape[-2:]
        original_resolution = None
    else:
        original_resolution = images.shape[-2:]
        images = F.interpolate(images, size=(resolution[0], resolution[1]), mode='bilinear', align_corners=True, antialias=True)
    
    if params_x.shape[-2] == 1:
        expand_size = (params_x.shape[0], params_x.shape[-1]+1,  params_x.shape[-1])
        params_x = params_x.expand(expand_size)
    if params_y.shape[-2] == 1:
        expand_size = (params_y.shape[0], params_y.shape[-1]+1, params_y.shape[-1])
        params_y = params_y.expand(expand_size)

    coarse_grid = get_coarse_adaptive_grid(params_x, params_y).to(images.device)
    denser_grid = F.interpolate(coarse_grid, size=(resolution[0], resolution[1]), mode='bilinear', align_corners=True).to(images.device)
    denser_grid = denser_grid * 2 - 1
    sampled_images = GridSample(images, denser_grid.permute(0, 2, 3, 1), mode=mode) if mode == 'nearest' else \
                     grid_sample(images, denser_grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=True)

    if original_resolution is not None:
        sampled_images = F.interpolate(sampled_images, size=(original_resolution[0], original_resolution[1]), mode='bilinear', align_corners=True, antialias=True)

    return sampled_images


def inv_deform(params_x, params_y, images, 
               resolution=None,
               mode='bilinear'):
    """
    Apply inverse local scaling to reverse the effect of deform().
    
    This function computes the inverse transformation of the local scaling.
    This is required for dense prediction tasks or inverse transforming the hidden features of the deep nets.
    
    Args:
        params_x (torch.Tensor): X-axis deformation parameters of shape (batch, h, w)
                                Same parameters used in forward deform() function
        params_y (torch.Tensor): Y-axis deformation parameters of shape (batch, h, w)
                                Same parameters used in forward deform() function
        images (torch.Tensor): Input images/features of shape (batch, channels, height, width)
                              These should be the OUTPUT of a previous deform() operation
        resolution (tuple, optional): Target resolution (height, width) for processing
                                    If None, uses original image resolution
        mode (str): Interpolation mode for grid sampling ('bilinear' or 'nearest')
                   Default: 'bilinear' for smooth transformations
    
    Returns:
        torch.Tensor: Inverse deformed images of same shape as input (batch, channels, height, width)
    Should approximately recover the original input to deform() function
    """
    if resolution is None:
        resolution = images.shape[-2:]
        original_resolution = None
    else:
        original_resolution = images.shape[-2:]
        images = F.interpolate(images, size=(resolution[0], resolution[1]), mode='bilinear', align_corners=True, antialias=True)
    if params_x.shape[-2] == 1:
        expand_size = (params_x.shape[0], params_x.shape[-1]+1,  params_x.shape[-1])
        params_x = params_x.expand(expand_size)
    if params_y.shape[-2] == 1:
        expand_size = (params_y.shape[0], params_y.shape[-1]+1, params_y.shape[-1])
        params_y = params_y.expand(expand_size)
    denser_grid = F.interpolate(get_coarse_adaptive_grid(params_x, params_y).to(images.device),
                                size=(resolution[0],resolution[1]),
                                mode='bilinear',
                                align_corners=True).to(images.device)

    x_uniform = torch.linspace(0, 1, resolution[1], device=params_x.device).to(images.device)
    y_uniform = torch.linspace(0, 1, resolution[0], device=params_x.device).to(images.device)
    inverse_mesh_x = interpolate_nonuniform_to_uniform(denser_grid[:, 0, :, :], x_uniform)
    inverse_mesh_y = interpolate_nonuniform_to_uniform(denser_grid[:, 1, :, :].transpose(-1, -2), y_uniform)
    inverse_grid = torch.concat([inverse_mesh_x, inverse_mesh_y.transpose(-1, -2)], dim=1) * 2 - 1
    sampled_images = GridSample(images, inverse_grid.permute(0, 2, 3, 1), mode=mode) if mode == 'nearest' else \
        grid_sample(images, inverse_grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=True)
    
    if original_resolution is not None:
        sampled_images = F.interpolate(sampled_images, size=(original_resolution[0], original_resolution[1]), mode='bilinear', align_corners=True, antialias=True)

    return sampled_images


def rerrange_and_scale_tokens(phi_x,
                              phi_y,
                              tokens,
                              inv_transform=False,
                              cls_token=None,
                              num_prefix_tokens=None,
                              mode='bilinear',
                              defom_resolution=None):
    """
    Apply local scaling to Vision Transformer tokens. It handles the conversion between token
    sequences and spatial feature maps, applies deformation, and converts back.
    
    Args:
        phi_x (torch.Tensor): X-axis deformation parameters of shape (batch, h, w)
        phi_y (torch.Tensor): Y-axis deformation parameters of shape (batch, h, w)
        tokens (torch.Tensor): ViT tokens of shape (batch, num_tokens, embed_dim)
                              where num_tokens = num_prefix_tokens + h*w
        inv_transform (bool): If True, applies inverse deformation. Default: False
        cls_token (torch.Tensor, optional): Classification token(s) to preserve
                                          Will be extracted and reattached
        num_prefix_tokens (int, optional): Number of special tokens (CLS, etc.) at the beginning
                                         If None and cls_token is not None, assumes 1 token
        mode (str): Interpolation mode for deformation ('bilinear' or 'nearest')
        defom_resolution (tuple, optional): Target resolution for deformation processing
    
    Returns:
        torch.Tensor: Transformed tokens of same shape as input (batch, num_tokens, embed_dim)
                     Spatial tokens are rearranged according to learned parameters
    
    """

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
    if inv_transform:
        hidden_states = inv_deform(phi_x, phi_y, hidden_states, mode=mode, resolution=defom_resolution)
    else:
        hidden_states = deform(phi_x, phi_y, hidden_states, mode=mode, resolution=defom_resolution)
        
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[1], h * h).transpose(1, 2)

    
    if cls_token is not None:
        if num_prefix_tokens is None:
            cls_token = cls_token.unsqueeze(1)
        return torch.cat([cls_token, hidden_states], dim=1)
    
    return hidden_states



###########
### General utility Functions for Grid Interpolation and Sampling
###########


def interpolate_nonuniform_to_uniform(x_nonuniform, x_uniform):
    
    B, N = x_nonuniform.shape[:2]
    M = x_uniform.shape[0]

    x_nonuniform = x_nonuniform.unsqueeze(1).contiguous()
    x_uniform = x_uniform.unsqueeze(0).unsqueeze(
        0).unsqueeze(0).expand(B, 1, N, M).contiguous()
    y_nonuniform = x_uniform.clone()

    # Find indices of intervals
    idxs = torch.searchsorted(x_nonuniform, x_uniform)

    # Clip indices to be within bounds
    idxs = torch.clamp(idxs, 0, M - 1)

    # Gather left and right values for interpolation
    idxs_l = torch.clamp(idxs - 1, 0, M - 1)

    x_left = torch.gather(x_nonuniform, -1, idxs_l)
    x_right = torch.gather(x_nonuniform, -1, idxs)
    y_left = torch.gather(y_nonuniform, -1, idxs_l)
    y_right = torch.gather(y_nonuniform, -1, idxs)

    # Linear interpolation
    denom = (x_right - x_left).clamp(min=1e-8)
    weight_right = (x_uniform - x_left) / denom
    weight_left = 1 - weight_right

    return weight_left * y_left + weight_right * y_right


def grid_sample(image, grid, **kwargs):
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

    nw_val = torch.gather(
        image,
        2,
        (iy_nw *
         IW +
         ix_nw).long().view(
            N,
            1,
            H *
            W).repeat(
            1,
            C,
            1))
    ne_val = torch.gather(
        image,
        2,
        (iy_ne *
         IW +
         ix_ne).long().view(
            N,
            1,
            H *
            W).repeat(
            1,
            C,
            1))
    sw_val = torch.gather(
        image,
        2,
        (iy_sw *
         IW +
         ix_sw).long().view(
            N,
            1,
            H *
            W).repeat(
            1,
            C,
            1))
    se_val = torch.gather(
        image,
        2,
        (iy_se *
         IW +
         ix_se).long().view(
            N,
            1,
            H *
            W).repeat(
            1,
            C,
            1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val
