"""
Configuration class for LSE-DINOv2 (Local Scale Equivariant DINOv2).

This module defines the configuration for the DINOv2 model enhanced with
Deep Equilibrium Model (DEM) based local scale adaptation.

Example:
    >>> from huggingface_release import LSEDinoV2Config
    >>> config = LSEDinoV2Config()
    >>> print(config.num_phi_layers)  # 3
    
License: Apache-2.0
"""

from transformers import PretrainedConfig


class LSEDinoV2Config(PretrainedConfig):
    """
    Configuration class for LSE-DINOv2 model.
    
    This configuration stores all hyperparameters needed to instantiate a
    Local Scale Equivariant DINOv2 model with DEM adaptation.
    
    Args:
        image_size (int): Input image size. Default: 224
        patch_size (int): Patch size for ViT. Default: 14
        num_channels (int): Number of input channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 768
        depth (int): Number of transformer blocks. Default: 12
        num_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0
        num_classes (int): Number of classification classes. Default: 1000
        num_register_tokens (int): Number of register tokens. Default: 4
        
        # DEM (Deep Equilibrium Model) parameters
        num_phi_layers (int): Number of adaptive phi layers. Default: 3
        adapter_coarse_resolution (list): Resolution for each phi layer. Default: [16, 8, 8]
        adapter_num_layers (int): Number of layers in DEM CNN. Default: 2
        adapter_channels (list): Channel sizes in DEM. Default: [3, 64, 128]
        adapter_filter_size (list): Filter sizes in DEM. Default: [5, 3]
        adapter_pool_rate (list): Pooling rates in DEM. Default: [4, 2]
        inner_epochs (int): DEQ fixed-point iterations. Default: 5
        
        # Local scaling parameters
        augment_layer_id (list): Layer IDs to apply deformation. Default: [-1, 4, 8]
        unaugment_layer_id (list): Layer IDs to undo deformation. Default: [-100]
        interpolation_mode (str): Interpolation mode. Default: 'bilinear'
        deform_resolution (int or None): Target resolution for deformation. Default: None
    """
    
    model_type = "lse-dinov2"
    
    def __init__(
        self,
        # Base ViT parameters
        image_size=224,
        patch_size=14,
        num_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        num_register_tokens=4,
        
        # DEM parameters
        num_phi_layers=3,
        adapter_coarse_resolution=[16, 8, 8],
        adapter_num_layers=2,
        adapter_channels=[3, 64, 128],
        adapter_filter_size=[5, 3],
        adapter_pool_rate=[4, 2],
        inner_epochs=5,
        dem_image_scale=True,
        
        # Local scaling parameters
        augment_layer_id=[-1, 4, 8],
        unaugment_layer_id=[-100],
        interpolation_mode='bilinear',
        deform_resolution=None,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Base ViT parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.num_register_tokens = num_register_tokens
        self.num_prefix_tokens = 1 + num_register_tokens  # CLS + registers
        
        # DEM parameters
        self.num_phi_layers = num_phi_layers
        self.adapter_coarse_resolution = adapter_coarse_resolution
        self.adapter_num_layers = adapter_num_layers
        self.adapter_channels = adapter_channels
        self.adapter_filter_size = adapter_filter_size
        self.adapter_pool_rate = adapter_pool_rate
        self.inner_epochs = inner_epochs
        self.dem_image_scale = dem_image_scale
        
        # Local scaling parameters
        self.augment_layer_id = augment_layer_id
        self.unaugment_layer_id = unaugment_layer_id
        self.interpolation_mode = interpolation_mode
        self.deform_resolution = deform_resolution

