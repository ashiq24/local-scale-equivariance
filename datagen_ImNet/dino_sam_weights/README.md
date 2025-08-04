# Model Weights Directory

This directory contains the pre-trained model weights required for the Inpaint-Anything functionality. Due to GitHub's file size limitations, these large model files are not stored in the repository.

## Required Model Files

You need to download the following model weights and place them in this directory:

### 1. SAM (Segment Anything Model) - ViT-H
- **File:** `sam_vit_h_4b8939.pth`
- **Size:** ~2.4GB
- **Download URL:** https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- **Description:** Pre-trained Segment Anything Model with Vision Transformer Huge backbone

### 2. Grounding DINO
- **File:** `groundingdino_swint_ogc.pth`
- **Size:** ~662MB
- **Download URL:** https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
- **Description:** Grounding DINO model with Swin Transformer backbone for open-set object detection

## Download Methods

### Option 1: Automatic Download (Recommended)
Run the provided download script from the `datagen_ImNet` directory:

```bash
# Python script (with progress bar)
python download_weights.py

# Or shell script
./download_weights.sh
```

### Option 2: Manual Download
```bash
# Create this directory if it doesn't exist
mkdir -p dino_sam_weights
cd dino_sam_weights

# Download SAM model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download Grounding DINO model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Option 3: Manual Download (Browser)
1. Download `sam_vit_h_4b8939.pth` from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
2. Download `groundingdino_swint_ogc.pth` from: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
3. Place both files in this directory (`datagen_ImNet/dino_sam_weights/`)

## Verification

After downloading, this directory should contain:
```
dino_sam_weights/
├── README.md                    # This file
├── sam_vit_h_4b8939.pth        # ~2.4GB
└── groundingdino_swint_ogc.pth # ~662MB
```

## Notes

- These model weights are required for the inpainting functionality to work properly
- The files are excluded from git tracking due to their large size (see `.gitignore`)
- Total download size: ~3.1GB
- Ensure you have sufficient disk space before downloading