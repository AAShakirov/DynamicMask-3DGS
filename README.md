# Overview

This repository implements multiple strategies for handling dynamic objects in 3D Gaussian Splatting (3DGS). The project addresses a key limitation of the original 3DGS method — its inability to handle moving objects — through three distinct approaches that filter, adapt, or inpaint dynamic regions.
# Key Features

Three comprehensive strategies for dynamic object handling

Pre-training filtering of SfM points belonging to moving objects

Training-time adaptation preventing Gaussian creation in dynamic regions

Automatic cleanup of Gaussians persistently marked as dynamic

Post-hoc inpainting of removed dynamic regions

Modular architecture allowing strategy combination

# Implemented Strategies
## Strategy A: Static Scene Extraction

Post-training approach that filters trained Gaussians using segmentation masks and retrains the model on static background only.
## Strategy B: Three-Stage Training Pipeline

B1: SfM Point Filtering - Preprocessing removal of 3D points projected onto dynamic masks

B2: Mask-Aware Gaussian Adaptation - Training-time prevention of Gaussian cloning/splitting in masked regions

B3: Dynamic Gaussian Removal - Lifecycle management removing persistently dynamic Gaussians

## Strategy C: Background Inpainting

Post-processing approach using a specialized inpainting network to fill regions where dynamic objects were removed.

## Colab

Original notebook (with dependency bug fix) https://colab.research.google.com/drive/1jHoQgsLjBGV77cueaWQrTHd8jeD8ARtX?usp=sharing

Notebook for strategy A https://colab.research.google.com/drive/1l8ugB-nEF44WNiiXSUe2vgcoSYLZkc7g?usp=sharing
