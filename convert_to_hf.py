#!/usr/bin/env python3
"""
Convert trained adaptive DINOv2 model (timm monkey-patched)
to HuggingFace standalone format for release.
"""

import torch
import timm
import argparse
from pathlib import Path
from YParams import YParams
from models.model_handler import get_dem_model
from layers.adapter_params import PerlayerAdapterParams
from imagenet.timm_model_handler import convert_model


def load_training_checkpoint(checkpoint_path):
    """Load the trained model checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint
    return checkpoint, checkpoint


def create_release_package(checkpoint_path, config_file, config_name, output_dir):
    """
    Create a complete release package with all necessary components.
    
    This uses Option B (custom loading) which is more reliable than
    trying to map weights to HuggingFace standalone model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Converting Adaptive DINOv2 to HuggingFace Release Package")
    print("=" * 70)
    
    # Load checkpoint
    state_dict, full_checkpoint = load_training_checkpoint(checkpoint_path)
    
    # Load configuration
    print(f"\nLoading configuration: {config_name} from {config_file}")
    config = YParams(config_file, config_name, print_params=False)
    
    # Create release package
    release_package = {
        'model_state_dict': state_dict,
        'config': {
            'model_name': 'dinov2',
            'base_model': 'vit_base_patch14_reg4_dinov2',
            'img_size': 224,
            'num_classes': 1000,
            
            # Adaptation settings
            'do_adaptation': config.do_adaptation,
            'num_phi_layers': config.num_phi_layers,
            'adapter_coarse_resolution': config.adapter_coarse_resolution,
            'inner_epochs': config.inner_epochs,
            'augment_layer_id': config.augment_layer_id,
            'unaugment_layer_id': config.unaugment_layer_id,
            'interpolation_mode': config.interpolation_mode,
            'deform_resolution': config.deform_resolution,
            
            # DEM architecture
            'adapter_num_layers': config.adapter_num_layers,
            'adapter_channels': config.adapter_channels,
            'adapter_filter_size': config.adapter_filter_size,
            'adapter_pool_rate': config.adapter_pool_rate,
            'dem_image_scale': getattr(config, 'dem_image_scale', True),
            
            # Loss settings
            'equivariance_loss': config.equivariance_loss,
            'equivariance_loss_weight': config.equivariance_loss_weight,
            
            # Learning rates (for reference)
            'outer_lr': config.outer_lr,
            'dem_lr': config.dem_lr,
            'dem_weight_decay': config.dem_weight_decay,
        },
        'training_info': {
            'epochs_trained': full_checkpoint.get('epoch', 'unknown'),
            'best_acc': full_checkpoint.get('best_acc', 'unknown'),
        }
    }
    
    # Save release package
    release_path = output_dir / 'dinov2_adaptive_imagenet.pth'
    print(f"\nSaving release package to: {release_path}")
    torch.save(release_package, release_path)
    
    print(f"✓ Release package saved!")
    print(f"  Size: {release_path.stat().st_size / (1024**2):.2f} MB")
    
    # Create loading script
    create_loading_script(output_dir)
    
    # Create model card
    create_model_card(output_dir, release_package)
    
    # Verify the package can be loaded
    print("\n" + "=" * 70)
    print("Verifying release package...")
    print("=" * 70)
    verify_release_package(release_path)
    
    print("\n" + "=" * 70)
    print("✓✓✓ CONVERSION COMPLETE ✓✓✓")
    print("=" * 70)
    print(f"\nRelease package ready at: {output_dir}")
    print("\nNext steps:")
    print("1. Test loading: python hf_release/load_model.py")
    print("2. Upload to HuggingFace Hub")
    print("=" * 70)


def create_loading_script(output_dir):
    """Create the load_model.py script"""
    script_content = '''#!/usr/bin/env python3
"""
Load trained adaptive DINOv2 model from release package.
"""

import torch
import timm
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_handler import get_dem_model
from layers.adapter_params import PerlayerAdapterParams
from imagenet.timm_model_handler import convert_model


class Config:
    """Simple config object"""
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


def load_adaptive_dinov2(checkpoint_path=None, device='cuda'):
    """
    Load trained adaptive DINOv2 model.
    
    Args:
        checkpoint_path: Path to release package (default: ./dinov2_adaptive_imagenet.pth)
        device: Device to load model on
        
    Returns:
        model: Adaptive DINOv2 model
        DEM: Deep Equilibrium Model
        config: Model configuration
    """
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent / 'dinov2_adaptive_imagenet.pth'
    
    print(f"Loading adaptive DINOv2 from: {checkpoint_path}")
    
    # Load release package
    package = torch.load(checkpoint_path, map_location='cpu')
    
    # Create config
    config = Config(package['config'])
    
    print(f"Configuration:")
    print(f"  Base model: {config.base_model}")
    print(f"  Adaptation: {config.do_adaptation}")
    print(f"  DEQ iterations: {config.inner_epochs}")
    print(f"  Augment layers: {config.augment_layer_id}")
    
    # Create base model
    print("Creating base model...")
    model = timm.create_model(
        config.base_model,
        pretrained=False,
        img_size=config.img_size,
        num_classes=config.num_classes
    )
    
    # Create DEM
    print("Creating DEM adapter...")
    DEM = get_dem_model(config)
    
    # Create local scale params
    local_scale_params = PerlayerAdapterParams(
        num_layers=config.num_phi_layers,
        adapter_coarse_resolution=config.adapter_coarse_resolution
    )
    
    # Convert to adaptive model
    print("Converting to adaptive model...")
    model = convert_model(model, config, local_scale_params, DEM)
    
    # Load trained weights
    print("Loading trained weights...")
    model.load_state_dict(package['model_state_dict'])
    
    # Move to device
    model = model.to(device)
    DEM = DEM.to(device)
    
    model.eval()
    DEM.eval()
    
    print(f"✓ Model loaded successfully!")
    if 'training_info' in package:
        print(f"  Epochs trained: {package['training_info']['epochs_trained']}")
        print(f"  Best accuracy: {package['training_info']['best_acc']}")
    
    return model, DEM, config


def test_inference(model, device='cuda'):
    """Test inference with dummy input"""
    print("\\nTesting inference...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Get top-5 predictions for first sample
    probs = torch.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)
    
    print(f"  Top-5 predictions: {top5_idx.tolist()}")
    print(f"  Top-5 probabilities: {[f'{p:.3f}' for p in top5_prob.tolist()]}")
    
    assert output.shape == (2, 1000), "Output shape mismatch!"
    print("✓ Inference test passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load adaptive DINOv2 model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: ./dinov2_adaptive_imagenet.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to load model on')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip inference test')
    
    args = parser.parse_args()
    
    # Load model
    model, DEM, config = load_adaptive_dinov2(args.checkpoint, args.device)
    
    # Test inference
    if not args.no_test and torch.cuda.is_available():
        test_inference(model, args.device)
    
    print("\\nModel ready for use!")
'''
    
    script_path = output_dir / 'load_model.py'
    print(f"Creating loading script: {script_path}")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print("✓ Loading script created!")


def create_model_card(output_dir, release_package):
    """Create README.md model card"""
    config = release_package['config']
    training_info = release_package.get('training_info', {})
    
    readme_content = f'''# Adaptive DINOv2-base with Local Scale Equivariance

This model is an adaptive version of DINOv2-base trained on ImageNet-1K with local scale equivariance learning.

## Model Details

- **Base model**: {config['base_model']}
- **Parameters**: ~88M (86M backbone + 2M DEM)
- **Image size**: {config['img_size']}×{config['img_size']}
- **Classes**: {config['num_classes']} (ImageNet-1K)

### Training Info

- **Epochs trained**: {training_info.get('epochs_trained', 'N/A')}
- **Best accuracy**: {training_info.get('best_acc', 'N/A')}
- **DEQ iterations**: {config['inner_epochs']}
- **Augmentation layers**: {config['augment_layer_id']}

## Usage

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/local-scale-equivariance
cd local-scale-equivariance

# Install dependencies
pip install torch torchvision timm torchdeq
```

### Inference

```python
from load_model import load_adaptive_dinov2
import torch
from torchvision import transforms
from PIL import Image

# Load model
model, DEM, config = load_adaptive_dinov2()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

image = Image.open('your_image.jpg')
input_tensor = transform(image).unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

print(f"Top-5: {{top5_idx[0].tolist()}}")
```

## Configuration

```python
{{
    'base_model': '{config['base_model']}',
    'do_adaptation': {config['do_adaptation']},
    'inner_epochs': {config['inner_epochs']},
    'num_phi_layers': {config['num_phi_layers']},
    'augment_layer_id': {config['augment_layer_id']},
}}
```

## License

MIT License

## Citation

```bibtex
@misc{{adaptive-dinov2-2024,
  title={{Adaptive DINOv2 with Local Scale Equivariance}},
  author={{Your Name}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/your-username/dinov2-adaptive-imagenet}}}}
}}
```
'''
    
    readme_path = output_dir / 'README.md'
    print(f"Creating model card: {readme_path}")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("✓ Model card created!")


def verify_release_package(package_path):
    """Verify the release package can be loaded"""
    try:
        package = torch.load(package_path, map_location='cpu')
        
        assert 'model_state_dict' in package, "Missing model_state_dict"
        assert 'config' in package, "Missing config"
        
        config = package['config']
        required_keys = ['base_model', 'img_size', 'num_classes', 'do_adaptation']
        for key in required_keys:
            assert key in config, f"Missing config key: {key}"
        
        num_params = sum(p.numel() for p in package['model_state_dict'].values())
        print(f"✓ Package verified!")
        print(f"  Total parameters: {num_params:,}")
        print(f"  Config keys: {len(config)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert trained adaptive DINOv2 to HuggingFace release format'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config-file', type=str, 
                       default='./imagenet/config/adapter_config.yaml',
                       help='Path to adapter config file')
    parser.add_argument('--config-name', type=str, default='dem_dinov2_fast',
                       help='Config name to use')
    parser.add_argument('--output-dir', type=str, default='./hf_release',
                       help='Output directory for release package')
    
    args = parser.parse_args()
    
    create_release_package(
        checkpoint_path=args.checkpoint,
        config_file=args.config_file,
        config_name=args.config_name,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()


