#!/usr/bin/env python3
"""
Upload LSE-DINOv2 model to HuggingFace Hub.

Usage:
    python upload_to_hub.py --repo-name "ashiq24/lse-dinov2-base"
    
Make sure to login first:
    huggingface-cli login
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def upload_model(
    local_dir: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload LSE-DINOv2 model"
):
    """
    Upload model to HuggingFace Hub.
    
    Args:
        local_dir: Local directory containing model files
        repo_name: HuggingFace repo name (e.g., "username/lse-dinov2-base")
        private: Whether to make the repo private
        commit_message: Commit message for the upload
    """
    api = HfApi()
    
    # Create repo if it doesn't exist
    print(f"Creating/checking repository: {repo_name}")
    try:
        create_repo(repo_name, private=private, exist_ok=True)
        print(f"✓ Repository ready: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Copy model files to upload directory
    upload_dir = Path(local_dir)
    
    # Ensure required files exist
    required_files = ["config.json", "model.safetensors", "README.md"]
    for f in required_files:
        if not (upload_dir / f).exists():
            print(f"Warning: {f} not found in {local_dir}")
    
    # Copy model code files
    code_files = [
        "configuration_lse_dinov2.py",
        "modeling_lse_dinov2.py",
    ]
    
    parent_dir = Path(__file__).parent
    for code_file in code_files:
        src = parent_dir / code_file
        dst = upload_dir / code_file
        if src.exists():
            shutil.copy(src, dst)
            print(f"✓ Copied {code_file}")
    
    # Upload
    print(f"\nUploading to {repo_name}...")
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_name,
        commit_message=commit_message,
    )
    
    print(f"\n✓ Upload complete!")
    print(f"  Model URL: https://huggingface.co/{repo_name}")
    print(f"\nTo use the model:")
    print(f'  from transformers import AutoModelForImageClassification')
    print(f'  model = AutoModelForImageClassification.from_pretrained("{repo_name}", trust_remote_code=True)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload LSE-DINOv2 to HuggingFace Hub")
    parser.add_argument("--local-dir", type=str, 
                        default="/home/rahman79/Desktop/ray_ashiq/Projects/local-scale-equivariance/huggingface_release/lse-dinov2-base",
                        help="Local directory with model files")
    parser.add_argument("--repo-name", type=str, required=True,
                        help="HuggingFace repo name (e.g., 'username/lse-dinov2-base')")
    parser.add_argument("--private", action="store_true",
                        help="Make the repository private")
    parser.add_argument("--message", type=str, default="Upload LSE-DINOv2 model",
                        help="Commit message")
    
    args = parser.parse_args()
    
    upload_model(
        local_dir=args.local_dir,
        repo_name=args.repo_name,
        private=args.private,
        commit_message=args.message
    )

