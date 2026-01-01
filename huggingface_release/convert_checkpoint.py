#!/usr/bin/env python3
"""
Convert trained LSE-DINOv2 checkpoint to HuggingFace format.

This script:
1. Loads the original model using the codebase
2. Creates the HuggingFace model
3. Copies weights directly from original model to HF model
4. Saves in SafeTensors format
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from safetensors.torch import save_file

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import original codebase
from timm import create_model
from YParams import YParams
from models.model_handler import get_dem_model
from layers.adapter_params import PerlayerAdapterParams
from imagenet.timm_model_handler import convert_model

# Handle both direct run and package import
try:
    from .configuration_lse_dinov2 import LSEDinoV2Config
    from .modeling_lse_dinov2 import LSEDinoV2ForImageClassification
except ImportError:
    from configuration_lse_dinov2 import LSEDinoV2Config
    from modeling_lse_dinov2 import LSEDinoV2ForImageClassification


def load_original_model(checkpoint_path: str, config_file: str, config_name: str, device: str = 'cpu'):
    """Load the original model using the codebase."""
    print(f"Loading original model from: {checkpoint_path}")
    print(f"Using config: {config_file} / {config_name}")
    
    # Load adapter config
    config_path = os.path.join(Path(__file__).parent.parent, 'imagenet', 'config', config_file)
    adapter_config = YParams(config_path, config_name, print_params=False)
    
    # Create base model
    model = create_model(
        'vit_base_patch14_reg4_dinov2',
        pretrained=False,
        num_classes=1000,
        img_size=224,
    )
    
    # Create DEM and local scale params
    DEM = get_dem_model(adapter_config)
    local_scale_params = PerlayerAdapterParams(
        num_layers=adapter_config.num_phi_layers,
        adapter_coarse_resolution=adapter_config.adapter_coarse_resolution
    )
    
    # Convert model to adaptive version
    model = convert_model(model, adapter_config, local_scale_params, DEM)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print(f"  ✓ Loaded {len(state_dict)} parameters")
    
    return model, adapter_config, checkpoint.get('epoch', 'unknown')


def create_hf_config(adapter_config):
    """Create HuggingFace config from adapter config."""
    return LSEDinoV2Config(
        # Base DINOv2 parameters
        img_size=224,
        patch_size=14,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        num_register_tokens=4,
        
        # DEM parameters from adapter config
        do_adaptation=adapter_config.do_adaptation,
        num_phi_layers=adapter_config.num_phi_layers,
        adapter_coarse_resolution=adapter_config.adapter_coarse_resolution,
        adapter_num_layers=adapter_config.adapter_num_layers,
        adapter_channels=adapter_config.adapter_channels,
        adapter_filter_size=adapter_config.adapter_filter_size,
        adapter_pool_rate=adapter_config.adapter_pool_rate,
        augment_layer_id=adapter_config.augment_layer_id,
        unaugment_layer_id=adapter_config.unaugment_layer_id,
        interpolation_mode=adapter_config.interpolation_mode,
        dem_image_scale=adapter_config.dem_image_scale,
        inner_epochs=adapter_config.inner_epochs,
    )


def copy_weights(orig_model, hf_model):
    """Copy weights from original model to HuggingFace model."""
    print("\nCopying weights from original model to HuggingFace model...")
    
    orig_sd = orig_model.state_dict()
    hf_sd = hf_model.state_dict()
    
    # Build mapping
    copied = 0
    missing = []
    
    for hf_key in hf_sd.keys():
        # Try to find corresponding original key
        orig_key = None
        
        # backbone.X -> X
        if hf_key.startswith('backbone.'):
            orig_key = hf_key.replace('backbone.', '')
            
            # Handle LayerScale naming: HF model uses .gamma, original may use .weight
            # e.g., backbone.blocks.0.ls1.gamma -> blocks.0.ls1.gamma (try first)
            #       then try blocks.0.ls1.weight if .gamma not found
            if orig_key not in orig_sd and '.gamma' in orig_key:
                alt_key = orig_key.replace('.gamma', '.weight')
                if alt_key in orig_sd:
                    orig_key = alt_key
                    
        # dem.X -> DEM.X
        elif hf_key.startswith('dem.'):
            orig_key = hf_key.replace('dem.', 'DEM.')
        # local_scale_params stays the same
        elif hf_key.startswith('local_scale_params.'):
            orig_key = hf_key
        
        if orig_key and orig_key in orig_sd:
            if hf_sd[hf_key].shape == orig_sd[orig_key].shape:
                hf_sd[hf_key] = orig_sd[orig_key].clone()
                copied += 1
            else:
                print(f"  Shape mismatch: {hf_key} ({hf_sd[hf_key].shape}) vs {orig_key} ({orig_sd[orig_key].shape})")
                missing.append(hf_key)
        else:
            missing.append(hf_key)
    
    print(f"  ✓ Copied {copied}/{len(hf_sd)} parameters")
    
    if missing:
        print(f"  Missing {len(missing)} parameters:")
        for key in missing[:10]:
            print(f"    - {key}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    
    # Load the copied weights
    hf_model.load_state_dict(hf_sd)
    
    return hf_model


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    config_file: str = 'adapter_config.yaml',
    config_name: str = 'dem_dinov2_v3',
    device: str = 'cpu'
):
    """
    Convert checkpoint to HuggingFace format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load original model using codebase
    orig_model, adapter_config, epoch = load_original_model(
        checkpoint_path, config_file, config_name, device
    )
    
    # Step 2: Create HuggingFace config
    print("\nCreating HuggingFace config...")
    hf_config = create_hf_config(adapter_config)
    
    # Step 3: Create HuggingFace model
    print("Creating HuggingFace model...")
    hf_model = LSEDinoV2ForImageClassification(hf_config)
    hf_model.to(device)
    
    # Step 4: Copy weights from original to HF model
    hf_model = copy_weights(orig_model, hf_model)
    hf_model.eval()
    
    # Step 5: Verify outputs match
    print("\nVerifying outputs match...")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)
        
        orig_out = orig_model(test_input)
        hf_out = hf_model(test_input)
        
        max_diff = torch.abs(orig_out - hf_out.logits).max().item()
        mean_diff = torch.abs(orig_out - hf_out.logits).mean().item()
        
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print("  ✓ Outputs match!")
        else:
            print("  ✗ Outputs differ - investigating...")
    
    # Step 6: Save model
    print(f"\nSaving model to {output_dir}...")
    
    # Get final state dict
    final_state_dict = hf_model.state_dict()
    
    # Save weights with metadata
    metadata = {"format": "pt"}
    save_file(final_state_dict, os.path.join(output_dir, "model.safetensors"), metadata=metadata)
    
    # Save config
    hf_config.save_pretrained(output_dir)
    
    print("\n✓ Conversion complete!")
    print(f"  Config saved to: {output_dir}/config.json")
    print(f"  Weights saved to: {output_dir}/model.safetensors")
    
    return hf_model, hf_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert LSE-DINOv2 checkpoint to HuggingFace format')
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/rahman79/Desktop/ray_ashiq/Projects/local-scale-equivariance/logs/output/dinov2_dem_dinov2_v3_1gpu_5344460/vit_base_p_dem_dinov2_v3_20251217-040755/model_best.pth.tar',
                        help='Path to original checkpoint')
    parser.add_argument('--output', type=str,
                        default='./lse-dinov2-base',
                        help='Output directory for converted model')
    parser.add_argument('--config-file', type=str, default='adapter_config.yaml',
                        help='Adapter config file name')
    parser.add_argument('--config-name', type=str, default='dem_dinov2_v3',
                        help='Config name within the file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for loading checkpoint')
    
    args = parser.parse_args()
    
    convert_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        config_file=args.config_file,
        config_name=args.config_name,
        device=args.device
    )
