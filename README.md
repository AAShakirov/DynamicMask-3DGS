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

Notebook for strategy B https://colab.research.google.com/drive/1w3D69U4VF8a6QabxXQ6i6tQTwzZrtyL6?usp=sharing

## GIF
<table>
  <tr>
    <td><img src="assets/3DGS-original-defects-optimize-1.gif" width="280" /></td>
    <td><img src="assets/3DGS-StrategyA-defects-optimize-1.gif" width="280" /></td>
    <td><img src="assets/3DGS-StrategyB-defects-optimize-1.gif" width="280" /></td>
  </tr>
  <tr>
    <td align="center">Original Gaussian</td>
    <td align="center">Gaussin with method A</td>
    <td align="center">Gaussin with method B</td>
  </tr>
</table>

## Video
[Link for Animation of original Gaussian Splatting](https://drive.google.com/file/d/1kqp4lyJWO0o0oVZn_5GsjTuTIpB-rPiW/view?usp=drive_link)

[Link for Animation of Gaussian Splatting modified for strategy A](https://drive.google.com/file/d/1L2_8ohsEKnm82Ca_QIbLiklidQHH2FOi/view?usp=drive_link)

[Link for Animation of Gaussian Splatting modified for strategy B](https://drive.google.com/file/d/1_skuuS7fpmcJr9liHdSUGlkYZsFtqs_i/view?usp=drive_link)

## Metric

<table>
  <tr>
    <td align="center">Methods</td>
    <td align="center">SSIM</td>
    <td align="center">PSNR</td>
    <td align="center">LPIPS</td>
  </tr>
  <tr>
    <td align="center">Original</td>
    <td align="center">0.7399</td>
    <td align="center">18.3676</td>
    <td align="center">0.3325</td>
  </tr>
    <tr>
    <td align="center">Method A</td>
    <td align="center">***0.7989***</td>
    <td align="center">18.7507</td>
    <td align="center">**0.2675**</td>
  </tr>
    <tr>
    <td align="center">Method B</td>
    <td align="center">0.7753</td>
    <td align="center">**20.0513**</td>
    <td align="center">0.3215</td>
  </tr>
</table>
