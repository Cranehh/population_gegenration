# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This is a PyTorch implementation of Diffusion Transformers (DiT) for image generation. Set up the environment using:

```bash
conda env create -f environment.yml
conda activate DiT
```

## Core Commands

### Model Training
```bash
# Single-node multi-GPU training (replace N with number of GPUs)
torchrun --nnodes=1 --nproc_per_node=N train.py --model DiT-XL/2 --data-path /path/to/imagenet/train

# Key training arguments:
# --model: DiT-XL/2, DiT-L/4, DiT-B/4, etc.
# --image-size: 256 or 512
# --global-batch-size: Total batch size across all GPUs
# --epochs: Number of training epochs
# --data-path: Path to ImageNet training data (required)
```

### Sampling/Inference
```bash
# Single image generation
python sample.py --image-size 512 --seed 1

# Custom checkpoint sampling
python sample.py --model DiT-L/4 --image-size 256 --ckpt /path/to/model.pt

# Large-scale evaluation sampling (distributed)
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000
```

### Model Download
```bash
# Download pre-trained checkpoints
python download.py
```

## Architecture Overview

### Core Components

**DiT Model (models.py)**
- Transformer-based diffusion model operating on latent patches
- Key classes: `DiTBlock`, `FinalLayer`, `DiT`  
- Available model sizes: DiT-S/2, DiT-B/2, DiT-B/4, DiT-L/2, DiT-L/4, DiT-XL/2
- Uses patch embedding similar to Vision Transformers
- Incorporates timestep and class label conditioning

**Diffusion Pipeline (diffusion/)**
- `gaussian_diffusion.py`: Core diffusion process implementation
- `__init__.py`: Main `create_diffusion()` factory function
- `respace.py`: Timestep scheduling and respacing
- Supports both training and sampling modes
- Implements DDPM/DDIM sampling strategies

**VAE Integration**
- Uses Stability AI's VAE for latent space encoding/decoding
- Images (256x256) → latent patches (32x32x4)
- Latent space reduces computational requirements significantly

### Data Flow

**Training Pipeline:**
1. ImageNet images → VAE encoder → latent representations
2. Add noise at random timesteps → noisy latents
3. DiT predicts noise given (noisy_latents, timestep, class_label)
4. MSE loss between predicted and actual noise

**Sampling Pipeline:**
1. Random noise → iterative denoising via DiT
2. Class conditioning with Classifier-Free Guidance
3. Final latents → VAE decoder → generated images

### Key Design Patterns

**Distributed Training**
- Uses PyTorch DDP with NCCL backend
- EMA (Exponential Moving Average) model tracking
- Automatic gradient accumulation and synchronization

**Conditioning**
- Class-conditional generation (ImageNet 1000 classes)
- Timestep embedding using sinusoidal encoding
- Classifier-Free Guidance for improved sample quality

**Model Scaling**
- Patch size and model depth/width determine computational cost
- Higher Gflops models consistently achieve lower FID scores
- Models scale from 33M (DiT-S/2) to 675M (DiT-XL/2) parameters

## Important Notes

- TF32 is enabled by default for faster A100 training
- Pre-trained models available for 256x256 and 512x512 resolution
- Evaluation uses ADM's TensorFlow evaluation suite for FID computation
- The codebase includes detailed Chinese documentation in `train_py_详细流程说明.md`