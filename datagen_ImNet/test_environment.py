#!/usr/bin/env python3
"""
Test script to verify the inpaint environment is properly set up.
Run this script after setting up the environment to check if all dependencies are working.
"""

import sys
import importlib
from packaging import version

def test_import(module_name, min_version=None):
    """Test if a module can be imported and optionally check version."""
    try:
        module = importlib.import_module(module_name)
        if min_version and hasattr(module, '__version__'):
            if version.parse(module.__version__) < version.parse(min_version):
                print(f"❌ {module_name}: Version {module.__version__} < required {min_version}")
                return False
        print(f"✅ {module_name}: OK" + (f" (v{module.__version__})" if hasattr(module, '__version__') else ""))
        return True
    except (ImportError, ValueError, AttributeError) as e:
        error_msg = str(e).split('\n')[0]  # Only show first line of error
        print(f"❌ {module_name}: Import failed - {error_msg}")
        return False
    except Exception as e:
        error_msg = str(e).split('\n')[0]  # Only show first line of error
        print(f"⚠️  {module_name}: Unexpected error - {error_msg}")
        return False

def test_cuda():
    """Test CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available (v{torch.version.cuda})")
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA: Not available (CPU only)")
            return False
    except Exception as e:
        print(f"❌ CUDA: Error - {e}")
        return False

def main():
    """Run environment tests."""
    print("🔍 Testing inpaint environment setup...")
    print("=" * 50)
    
    # Core dependencies (using actual import names)
    core_deps = [
        ("torch", "2.0.0"),
        ("torchvision", "0.15.0"),
        ("tensorflow", "2.10.0"),
        ("numpy", "1.20.0"),
        ("cv2", None),  # opencv-python imports as cv2
        ("matplotlib", "3.0.0"),
        ("PIL", None),  # pillow imports as PIL
        ("transformers", "4.20.0"),
        ("timm", "0.9.0"),
    ]
    
    # Optional but important (using actual import names)
    optional_deps = [
        "kornia",
        "albumentations", 
        "hydra",  # hydra-core imports as hydra
        "omegaconf",
        "pytorch_lightning",  # pytorch-lightning imports as pytorch_lightning
        "skimage",  # scikit-image imports as skimage
        "supervision",
        "groundingdino",  # groundingdino-py imports as groundingdino
    ]
    
    failed_core = 0
    failed_optional = 0
    
    print("Core Dependencies:")
    for dep, min_ver in core_deps:
        if not test_import(dep, min_ver):
            failed_core += 1
    
    print("\nOptional Dependencies:")
    for dep in optional_deps:
        if not test_import(dep):
            failed_optional += 1
    
    print("\nSystem Tests:")
    cuda_available = test_cuda()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"Core dependencies: {len(core_deps) - failed_core}/{len(core_deps)} passed")
    print(f"Optional dependencies: {len(optional_deps) - failed_optional}/{len(optional_deps)} passed")
    print(f"CUDA support: {'✅ Available' if cuda_available else '⚠️ Not available'}")
    
    if failed_core > 0:
        print(f"\n❌ {failed_core} core dependencies failed. Please fix these before proceeding.")
        return 1
    elif failed_optional > 0:
        print(f"\n⚠️  {failed_optional} optional dependencies missing. Some features may not work.")
        return 0
    else:
        print("\n🎉 All tests passed! Environment is ready to use.")
        return 0

if __name__ == "__main__":
    sys.exit(main())