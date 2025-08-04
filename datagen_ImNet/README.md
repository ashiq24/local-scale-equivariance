# Multi-Scale ImageNet Dataset Generation

This directory contains tools for generating multi-scale ImageNet datasets using automated object detection, segmentation, and inpainting techniques. The generated datasets are used to train and evaluate local scale equivariance methods.

## 🚨 Important Attribution

**This data generation pipeline is heavily based on the ["Inpaint Anything: Segment Anything Meets Image Inpainting"](https://github.com/geekyutao/Inpaint-Anything) project by Yu et al.** We extend their framework to create multi-scale datasets for our research. Please refer to their repository for the original implementation and cite their work appropriately.

## 🛠️ Installation

### Prerequisites

- **CUDA 12.4+** with compatible NVIDIA drivers
- **GPU Memory**: 8GB+ recommended for large models
- **Python**: 3.12.9

### Using Conda (Recommended)

```bash
# Create environment from provided file
conda env create -f environment.yml
conda activate inpaint

# Manual installation for Segment Anything (required)
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

### Using pip (Alternative)

```bash
pip install -r requirements.txt

# Manual installation for Segment Anything (required)  
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

### Key Dependencies

- **PyTorch**: 2.6.0 (with CUDA 12.4 support)
- **TensorFlow**: 2.18.0
- **OpenCV**: 4.11.0.86
- **Transformers**: 4.48.3
- **GroundingDINO**: 0.4.0
- **Segment Anything**: Latest from GitHub

## 📊 Multi-Scale Dataset Generation Pipeline

The `inpaint_imagenet.py` script creates multi-scale ImageNet variations through:

1. **Object Detection**: GroundingDINO detects objects by class name
2. **Segmentation**: Segment Anything Model (SAM) precisely segments detected objects
3. **Background Inpainting**: LaMa inpaints backgrounds where objects were removed
4. **Multi-Scale Generation**: Objects are scaled by different factors and pasted back
5. **Dataset Creation**: Generates new ImageNet-structured datasets with local scale variations

<!-- ## Updates
| Date | News |
| ------ | --------
| 2023-04-12 | Release the Fill Anything feature | 
| 2023-04-10 | Release the Remove Anything feature |
| 2023-04-10 | Release the first version of Inpaint Anything | -->

## 🚀 Usage

### Basic Command

```bash
# Activate environment
conda activate inpaint

# Generate multi-scale dataset
python inpaint_imagenet.py \
    --dataset_path /path/to/original/imagenet \
    --output_path /path/to/output/multiscale_imagenet \
    --subset train \
    --max_images 10000 \
    --scale_factors 1.3 1.2 1.1 1.0 0.9 0.8 0.7
```

### Advanced Configuration

```bash
python inpaint_imagenet.py \
    --dataset_path /datasets/imagenet \
    --output_path ../scale_imagenet_custom \
    --subset train \
    --scale_factors 1.5 1.2 1.0 0.8 0.5 \
    --max_images 5000 \
    --mask_inflation 5 \
    --mask_blur 2 \
    --debug
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_path` | `/datasets/imagenet` | Path to original ImageNet directory |
| `--output_path` | `../demo_train_1` | Output directory for multi-scale dataset |
| `--subset` | `train` | Dataset subset (`train` or `val`) |
| `--scale_factors` | `[1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7]` | Object scaling factors |
| `--max_images` | `10000` | Maximum number of images to process |
| `--mask_inflation` | `5` | Mask expansion parameter for better inpainting |
| `--mask_blur` | `2` | Mask edge blurring for seamless blending |
| `--debug` | `False` | Save intermediate results for debugging |

### Output Structure

The generated dataset maintains ImageNet's directory structure:

```
output_path/
├── n01440764/  # Class directories (same as original ImageNet)
│   ├── image1_scale_1.3.JPEG
│   ├── image1_scale_1.2.JPEG
│   ├── image1_scale_1.1.JPEG
│   ├── image1_scale_1.0.JPEG
│   ├── image1_scale_0.9.JPEG
│   └── ...
└── n01443537/
    └── ...
```

## 🔧 Model Requirements

### Required Model Checkpoints

Download and place these models in `./pretrained_models/`:

1. **Segment Anything Model**: [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
2. **LaMa Inpainting Model**: [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)
3. **GroundingDINO Model**: Automatically downloaded or place in `./dino_sam_weights/`

## 🚀 Quick Start Example

```bash
# 1. Activate environment
conda activate inpaint

# 2. Generate small test dataset
python inpaint_imagenet.py \
    --dataset_path /path/to/imagenet \
    --output_path ./test_multiscale \
    --subset val \
    --max_images 100 \

# 3. Check output
ls ./test_multiscale/  # Should show ImageNet class directories
```

## ⚠️ Common Issues

1. **CUDA Memory Issues**: Reduce `max_images` or use smaller models
2. **Segmentation Failures**: Some objects may not be detected/segmented properly (logged as warnings)
3. **Path Issues**: Ensure ImageNet directory structure follows standard format
4. **Model Downloads**: All required models must be downloaded before running



## 📚 Citation

If you use this data generation pipeline, please cite both our work and the original "Inpaint Anything" project:

**Our Work:**
```bibtex
@inproceedings{rahman2025local,
  title={Local Scale Equivariance with Latent Deep Equilibrium Canonicalizer},
  author={Rahman, Md Ashiqur and Yang, Chiao-An and Cheng, Michael N. and Lim, Jun Hao and Jiang, Jeremiah and Lim, Teck-Yian and Yeh, Raymond A.},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

```bibtex
@article{yu2023inpaint,
  title={Inpaint Anything: Segment Anything Meets Image Inpainting},
  author={Yu, Tao and Feng, Runseng and Feng, Ruoyu and Liu, Jinming and Jin, Xin and Zeng, Wenjun and Chen, Zhibo},
  journal={arXiv preprint arXiv:2304.06790},
  year={2023}
}
```

## 🔗 Related Projects

- **Original Implementation**: [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything)
- **Segment Anything**: [Meta's SAM](https://github.com/facebookresearch/segment-anything)
- **LaMa Inpainting**: [Samsung's LaMa](https://github.com/advimman/lama)
- **GroundingDINO**: [IDEA Research](https://github.com/IDEA-Research/GroundingDINO)
