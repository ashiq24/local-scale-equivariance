# Local Scale Equivariance with Latent Deep Equilibrium Canonicalizer

📢 **Accepted at ICCV 2025** • 🌐 [![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://ashiq24.github.io/local-scale-equivariance/)

[Md Ashiqur Rahman¹](mailto:rahman79@purdue.edu), [Chiao-An Yang¹](mailto:yang2892@purdue.edu), [Michael N. Cheng¹](mailto:cheng610@purdue.edu), [Lim Jun Hao²](mailto:ljunhao@dso.org.sg), [Jeremiah Jiang²](mailto:jjiang@dso.org.sg), [Teck-Yian Lim²](mailto:tylim@dso.org.sg), [Raymond A. Yeh¹](mailto:raymond.yeh@purdue.edu)

¹Purdue University, ²DSO National Laboratories
| Global Scaling | Local Scaling |
|----------------|---------------|
| <img src="https://github.com/ashiq24/local-scale-equivariance/blob/gh-pages/assets/bird_1_gscale.gif" width="450"/> | <img src="https://github.com/ashiq24/local-scale-equivariance/blob/gh-pages/assets/bird_1.gif" width="450"/> |


## Abstract

In computer vision, scale variation is a persistent challenge where objects of the same class can appear at different sizes due to intrinsic differences and extrinsic factors like camera distance. While prior work addresses global scale invariance, real-world images often experience **local scale changes** where only parts of the image are resized. Existing neural networks, including state-of-the-art vision transformers, are not inherently robust to such transformations.

We propose **Deep Equilibrium Canonicalizer (DEC)**, a novel method that achieves local scale equivariance through:
- **Monotone Scaling Group**: A mathematically tractable approximation to local scaling
- **Deep Equilibrium Models (DEQs)**: Efficient canonicalization via fixed-point computation
- **Latent Canonicalization**: Applied to feature space as well as the input images  

Our method consistently improves both accuracy and scale consistency metrics across various architectures (ViT, Swin, BEiT, DINOv2, ResNet, DeiT) on synthetic and real-world datasets.

## 🔥 Key Contributions

1. **First group-theoretic approach** to local scale equivariance with monotone scaling groups
2. **Novel use of Deep Equilibrium Models** for efficient, differentiable canonicalization  
3. **Latent canonicalization** that integrates with any pretrained architecture
4. **Extensive validation** across 6 architectures on MNIST, and ImageNet


## Workflow of DEC Module
<video src="https://github.com/user-attachments/assets/d57409d6-ba93-47c8-b69e-da45a1ac371b" 
       width="200" height="200" controls>
</video>
## 📋 Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
  - [MNIST Experiments](#mnist-experiments)
  - [ImageNet Experiments](#imagenet-experiments)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/local-scale-equivariance.git
cd local-scale-equivariance

# Create and activate the main environment
conda env create -f environment.yml
conda activate local_scale

# For data generation pipeline (optional)
cd datagen_ImNet
conda env create -f environment.yml
conda activate inpaint
```


### Dependencies

**Core Requirements:**
- Python 3.11+
- PyTorch 2.1.2+ with CUDA 12.1+
- timm 1.0.14
- torchdeq 0.1.0
- transformers 4.44.2+

**Full dependency list:** See `environment.yml`

## 📊 Datasets

### MNIST Multi-Scale
- **Auto-downloaded**: Automatically handled by the training scripts
- **Description**: Multi-digit MNIST with local scaling transformations

### ImageNet
- **Required**: Standard ImageNet-1K dataset
- **Multi-Scale ImageNet**: Generated using our data generation pipeline. Available on Huggingface (https://huggingface.co/datasets/ashiq24/Multi_Scale_ImageNet). 

### Custom Multi-Scale Dataset Generation
```bash
# Activate data generation environment
conda activate inpaint
cd datagen_ImNet

# Generate multi-scale datasets (see datagen_ImNet/README.md for details)
python inpaint_imagenet.py --help
```

##  🧪 Experiments

### MNIST Example 

```bash
# Single experiment with Swin Transformer + DEC
bash run_mnist.sh config_1.yaml swin_data_augmentation 42
bash run_mnist.sh config_1.yaml ada_swin_dem 42
```

### ImageNet Example

```bash
# baselines
bash imagenet_baselines.sh swin base_swin 20

# DEC method experiment  
bash imagenet_dem.sh swin dem_swin 20
```

### Batch ImageNet Experiments

**Run All Baselines:**
```bash
# Runs all baseline variants for all models (swin, vit, deit, beit)
bash imnet_all_baselines.sh
```

**Run All DEC Experiments:**
```bash
# Runs DEC experiments for all models
bash imnet_all_dem.sh
```

These scripts automatically run experiments with multiple model-configuration pairs.

### MNIST Experiments

#### Single Experiment
```bash
bash run_mnist.sh <config_file> <config_name> <random_seed>

# Examples:
bash run_mnist.sh config_1.yaml swin_data_augmentation 70
bash run_mnist.sh config_1.yaml ada_vit_dem_1 71
```

#### Batch Experiments (All Models)
```bash
# Run all data augmentation baselines
CONFIGS_TO_RUN=$data_aug_configs bash run_all_mnist.sh

# Run all DEC experiments  
CONFIGS_TO_RUN=$ada_dem_configs bash run_all_mnist.sh

# Run all canonicalization baselines
CONFIGS_TO_RUN=$canon_configs bash run_all_mnist.sh
```

## 📁 Repository Structure

```
local-scale-equivariance/
├── 📁 config/                    # MNIST experiment configurations
│   └── config_1.yaml
├── 📁 imagenet/                  # ImageNet-specific code
│   ├── config/adapter_config.yaml
│   └── timm_*.py                 # Model implementations
├── 📁 models/                    # Core model implementations
│   ├── canonicalizer_wrapper.py # Canonicalization baseline
│   ├── surrogate_model.py        # DEC implementation
│   └── ada_*.py                  # Adaptive model wrappers
├── 📁 layers/                    # Core DEC layers
│   └── adapter.py                # Deep Equilibrium adaptation
├── 📁 train/                     # Training infrastructure  
│   └── trainer.py                # Bi-level optimization
├── 📁 evaluation/                # Metrics and evaluation
├── 📁 utils/                     # Utility functions
├── 📁 dataloader/                # Data loading utilities
├── 📁 datagen_ImNet/             # Data generation pipeline
│   ├── environment.yml           # Separate environment
│   └── README.md                 # Data generation docs
├── 📄 train_mnist.py             # MNIST training script
├── 📄 train_imagenet.py          # ImageNet training script
├── 📄 imagenet_baselines.sh      # Baseline experiments
├── 📄 imagenet_dem.sh            # DEC experiments  
├── 📄 imnet_all_baselines.sh     # Batch baseline experiments
├── 📄 imnet_all_dem.sh           # Batch DEC experiments
├── 📄 run_mnist.sh               # Single MNIST experiment
├── 📄 run_all_mnist.sh           # Batch MNIST experiments
├── 📄 YParams.py                 # YAML parameter utilities
└── 📄 TUTORIAL.ipynb             # Tutorial notebook
```


### Logging and Monitoring

- **WandB Integration**: Automatic logging enabled
- **Checkpoints**: Saved in `./weights/` directory  
- **Logs**: Console output shows training progress


## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{rahman2025local,
  title={Local Scale Equivariance with Latent Deep Equilibrium Canonicalizer},
  author={Md Ashiqur Rahman and Chiao-An Yang and Michael N. Cheng and Jun Hao Lim and Jeremiah Jiang and Teck-Yian Lim and Raymond A. Yeh},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```


## 🙏 Acknowledgments

- PyTorch Image Models ([timm](https://github.com/rwightman/pytorch-image-models)) for model implementations
- Deep Equilibrium Models ([torchdeq](https://github.com/locuslab/torchdeq)) for DEQ implementations  
- HuggingFace Transformers for transformer utilities
- The computer vision community for inspiring this work

---
For questions, please contact [rahman79@purdue.edu](mailto:rahman79@purdue.edu)
