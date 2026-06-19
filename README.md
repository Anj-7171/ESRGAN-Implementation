# ESRGAN Implementation (PyTorch)

This repository contains a PyTorch implementation of **ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)** for single image super-resolution. The goal is to reconstruct high-resolution images from low-resolution inputs while preserving sharp textures and perceptual quality.

Built as part of a deep learning super-resolution project.
---

## Overview

ESRGAN improves upon SRGAN by introducing:

- Residual-in-Residual Dense Blocks (RRDB)
- Removal of Batch Normalization layers for better stability
- Relativistic average GAN discriminator
- Improved perceptual loss using VGG features before activation

This project is trained and evaluated on the DIV2K dataset.

---

## Key Features

- PyTorch implementation of ESRGAN
- RRDB-based Generator architecture
- Relativistic GAN training
- Perceptual + Pixel + Adversarial loss combination
- PSNR and SSIM evaluation metrics
- Training + Evaluation pipelines

---

## Project Structure
ESRGAN-Implementation/
│
├── models/ # Generator and Discriminator (RRDBNet, etc.)
├── datasets/ # Dataset loading scripts
├── utils/ # Loss functions, metrics, helper functions
├── train.py # Training script
├── evaluate.py # Evaluation script (PSNR/SSIM)
├── config.py # Hyperparameters and config
├── degradation.py # LR image generation
├── checkpoints/ # Saved model weights
├── results/ # Super-resolved outputs
└── README.md

---

## Installation

Clone the repository:
git clone https://github.com/Anj-7171/ESRGAN-Implementation.git
cd ESRGAN-Implementation

## Install Dependencies
pip install -r requirements.txt

## Dataset
This project uses the DIV2K dataset:

High-Resolution images (HR)
Low-Resolution images generated via degradation pipeline

Expected Structure: 
data/
├── train/
├── valid/
└── test/

## Training
Stage 1: Pretraining (Pixel Loss Only)
python train.py --mode pretrain

Stage 2: GAN Training (Full ESRGAN)
python train.py --mode gan

## Evaluation

Run evaluation:

python evaluate.py --checkpoint checkpoints/esrgan_best.pth

This will output:

PSNR score
SSIM score
Generated super-resolved images in results/

## References:
ESRGAN Paper: https://arxiv.org/abs/1809.00219
