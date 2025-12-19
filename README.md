# Overview
This repository implements multiple strategies for handling dynamic objects in 3D Gaussian Splatting (3DGS). The project addresses a key limitation of the original 3DGS method — its inability to handle moving objects — through three distinct approaches that filter, adapt, or inpaint dynamic regions.\
[Project Presentation](https://docs.google.com/presentation/d/1HwW4lar9bygGFCRijcWwV9zkDp8vtOkT/edit?usp=sharing)
# Key Features

Three comprehensive strategies for dynamic object handling

Pre-training filtering of SfM points belonging to moving objects

Training-time adaptation preventing Gaussian creation in dynamic regions

Automatic cleanup of Gaussians persistently marked as dynamic

Post-hoc inpainting of removed dynamic regions

Modular architecture allowing strategy combination
# 3D Gaussian Splatting baseline
The pipeline starts from sparse 3D points reconstructed using Structure-from-Motion (SfM). These points are initialized as 3D Gaussians, each parameterized by position, covariance, color, and opacity.  

Given known camera parameters, the Gaussians are projected into the image plane and rendered using a differentiable tile-based rasterizer.  

The rendering process is fully differentiable, allowing image reconstruction loss to be backpropagated to the Gaussian parameters. During optimization, an adaptive density control mechanism dynamically splits or removes Gaussians to better capture scene geometry and appearance.  

This results in a compact, continuous, and efficiently renderable 3D scene representation optimized directly from multi-view images.


# Implemented Strategies
## Strategy A: Static Scene Extraction

Post-training approach that filters trained Gaussians using segmentation masks and retrains the model on static background only.
## Strategy B: Three-Stage Training Pipeline

B1: SfM Point Filtering - Preprocessing removal of 3D points projected onto dynamic masks

B2: Mask-Aware Gaussian Adaptation - Training-time prevention of Gaussian cloning/splitting in masked regions

B3: Dynamic Gaussian Removal - Lifecycle management removing persistently dynamic Gaussians

## Git workflow

### Core Branches  
*Long-lived, protected branches representing stable and integration states.*

| Branch    | Purpose                                                                 |
|-----------|-------------------------------------------------------------------------|
| `main`    | Stable, production-ready code. Only updated via merges from `develop` after thorough testing and version tagging. |
| `develop` | Primary integration branch. Contains the latest tested features, ready for release preparation. All feature work is merged here first. |

---

### Feature Branches  
*Short-lived branches for new functionality or experiments. Always branched from and merged back into `develop`.*

| Branch                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `feature/preprocessing`    | Implements dynamic object mask generation. Outputs masks in `.npy` (NumPy array) and `.png` (raster image) formats. |
| `feature/strategy_A`       | Implements the first gradient-zeroing strategy for dynamic regions. |
| `feature/strategy_B`       | Implements an alternative strategy for handling dynamic regions. |

---

### Maintenance Branches  
*Branches for non-functional work: tooling, configuration, documentation, and infrastructure.*

| Branch                | Description                                                |
|-----------------------|------------------------------------------------------------|
| `chore/setup`         | Initial repository setup: dependency management, build tooling, CI configuration, and project scaffolding. |
| `chore/edit-readme`   |  Formatting and structuring of documentation in `README.md` — layout, sections, tables, code blocks, and Markdown styling. |

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

## Metrics

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
    <td align="center">0.7989 [+ 0.0591 (~8%)]</td>
    <td align="center">18.7507 [+ 0.383(~2%)]</td>
    <td align="center">0.2675 [- 0.0651 (~19.6%)]</td>
  </tr>
    <tr>
    <td align="center">Method B</td>
    <td align="center">0.7753b [+ 0.0354 (~5%)]</td>
    <td align="center">20.0513 [+ 1.684 (~9%)]</td>
    <td align="center">0.3215 [- 0.0111 (~3%)]</td>
  </tr>
</table>

## Conclusion

If the goal is to minimize mathematical error (for example, in scientific measurements, medicine), then method B is better. 

If the goal is to obtain an image that looks as natural as possible and similar to the original to the human eye, then method A is better.
