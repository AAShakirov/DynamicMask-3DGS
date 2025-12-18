"""
Strategy B: Dynamic Filtering
Стратегия интеграции 2D-масок в пайплайн 3DGS для подавления динамических артефактов.

Компоненты:
(а) SfM Filtering - фильтрация 3D-точек перед инициализацией
(б) Controlled Splitting - запрет создания гауссианов в динамических областях
(в) Gaussian Pruning - удаление "старых" динамических гауссианов
"""

from .sfm_filtering import filter_point_cloud_with_masks, filter_sfm_points
from .controlled_splitting import (
    MaskAwareDensifier,
    densify_and_clone_masked,
    densify_and_split_masked,
    densify_and_prune_masked
)
from .gaussian_pruning import (
    DynamicGaussianTracker,
    prune_dynamic_gaussians,
    update_tracker_after_standard_pruning,
    AdaptivePruningScheduler
)

__all__ = [
    # SfM Filtering
    'filter_point_cloud_with_masks',
    'filter_sfm_points',
    
    # Controlled Splitting
    'MaskAwareDensifier',
    'densify_and_clone_masked',
    'densify_and_split_masked',
    'densify_and_prune_masked',
    
    # Gaussian Pruning
    'DynamicGaussianTracker',
    'prune_dynamic_gaussians',
    'update_tracker_after_standard_pruning',
    'AdaptivePruningScheduler',
]
