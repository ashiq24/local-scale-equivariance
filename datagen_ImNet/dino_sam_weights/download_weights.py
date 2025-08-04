#!/usr/bin/env python3
"""
Download script for large model weights required by Inpaint-Anything.

This script downloads the pre-trained model weights that are too large 
to be stored in the git repository.
"""

import os
import urllib.request
import hashlib
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, filename, expected_hash=None):
    """Download a file with progress bar and optional hash verification."""
    if os.path.exists(filename):
        print(f"✅ {filename} already exists")
        return
    
    print(f"📥 Downloading {filename}...")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
    
    if expected_hash:
        print(f"🔍 Verifying hash for {filename}...")
        with open(filename, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash == expected_hash:
            print(f"✅ Hash verification successful")
        else:
            print(f"❌ Hash verification failed. Expected: {expected_hash}, Got: {file_hash}")
            os.remove(filename)
            return False
    
    print(f"✅ Downloaded {filename}")
    return True


def main():
    # Create weights directory if it doesn't exist
    weights_dir = "dino_sam_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Model download URLs and hashes
    models = [
        {
            "name": "SAM ViT-H model",
            "filename": os.path.join(weights_dir, "sam_vit_h_4b8939.pth"),
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "hash": None  # Add hash if known
        },
        {
            "name": "Grounding DINO model",
            "filename": os.path.join(weights_dir, "groundingdino_swint_ogc.pth"),
            "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            "hash": None  # Add hash if known
        }
    ]
    
    print("🚀 Starting model weights download...")
    print("=" * 50)
    
    success_count = 0
    for model in models:
        print(f"\n📦 {model['name']}")
        if download_file(model["url"], model["filename"], model.get("hash")):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Download complete! {success_count}/{len(models)} models downloaded successfully.")
    
    if success_count == len(models):
        print("\n🎉 All model weights are ready! You can now run the inpainting scripts.")
    else:
        print("\n⚠️  Some downloads failed. Please check your internet connection and try again.")


if __name__ == "__main__":
    main()