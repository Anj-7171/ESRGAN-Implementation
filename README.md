# ESRGAN Implementation (PyTorch)

This repository contains a PyTorch implementation of **ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)** for single image super-resolution. The goal is to reconstruct high-resolution images from low-resolution inputs while preserving sharp textures and perceptual quality.

<img width="1424" height="569" alt="image" src="https://github.com/user-attachments/assets/ac0e2c63-013a-481c-8190-3a45b8e0dfdc" />


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
```
ESRGAN-Implementation/
├── models/
├── datasets/
├── utils/
├── train.py
├── evaluate.py
├── config.py
├── degradation.py
├── checkpoints/
├── results/
└── README.md
```

---

## Installation

Clone the repository:
```
git clone https://github.com/Anj-7171/ESRGAN-Implementation.git
cd ESRGAN-Implementation
```

## Install Dependencies
```
pip install -r requirements.txt
```
## Dataset
This project uses the DIV2K dataset:

High-Resolution images (HR)
Low-Resolution images generated via degradation pipeline

Expected Structure: 
```
data/
├── train/
├── valid/
└── test/
```

## Training
Stage 1: Pretraining (Pixel Loss Only)
```
python train.py --mode pretrain
```
Stage 2: GAN Training (Full ESRGAN)
```
python train.py --mode gan
```
## Evaluation

Run evaluation:
```
python evaluate.py --checkpoint checkpoints/esrgan_best.pth
```
This will output:

PSNR score
SSIM score
Generated super-resolved images in results/

Results:

<img width="757" height="297" alt="image" src="https://github.com/user-attachments/assets/6ed4526e-fc2c-46ae-bfeb-3640b0b547bb" />

<img width="767" height="456" alt="image" src="https://github.com/user-attachments/assets/42c5527a-dafa-4b1a-9a0f-44077b0e987f" />

After GAN Training:

<img width="756" height="370" alt="image" src="https://github.com/user-attachments/assets/c2f113f3-ede9-45e0-991f-efa02aa085ae" />

<img width="792" height="513" alt="image" src="https://github.com/user-attachments/assets/598e9af3-d9b4-42db-a0a9-0e735d1d7f9a" />

Final ESRGAN Outputs:

<img width="668" height="762" alt="image" src="https://github.com/user-attachments/assets/8a3fb857-c2fa-4059-8b98-d354bb6e4b0a" />

<img width="801" height="96" alt="image" src="https://github.com/user-attachments/assets/0aa1d155-9c3c-4394-9a91-48608221b753" />

## References:
ESRGAN Paper: https://arxiv.org/abs/1809.00219
