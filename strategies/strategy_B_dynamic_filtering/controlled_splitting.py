"""
Strategy B Component (b): Controlled Splitting
Модификация логики адаптивной денсификации для запрета создания новых гауссианов
в 2D-областях, покрытых маской динамических объектов.

Идея: при клонировании и разделении гауссианов проверять, проецируются ли они
на динамические области в обучающих изображениях, и исключать такие гауссианы
из процесса денсификации.
"""

import torch
import numpy as np
from typing import Dict, Optional


class MaskAwareDensifier:
    """
    Класс для управления денсификацией с учетом масок динамических объектов.
    """
    
    def __init__(self, gaussians, cameras, masks: Dict[str, torch.Tensor]):
        """
        Args:
            gaussians: модель гауссианов
            cameras: список камер для проекции
            masks: словарь {image_name: mask_tensor} с масками динамических объектов
        """
        self.gaussians = gaussians
        self.cameras = cameras
        self.masks = masks
        
        # Кэш для хранения информации о том, какие гауссианы попадают на маски
        self.dynamic_scores = None
        self.last_check_iteration = -1
        
    def project_gaussians_to_cameras(self, gaussian_indices: torch.Tensor) -> torch.Tensor:
        """
        Проецирует гауссианы на камеры и проверяет, попадают ли они на маски.
        
        Args:
            gaussian_indices: индексы гауссианов для проверки
        
        Returns:
            Тензор с долей наблюдений на динамических областях для каждого гауссиана [0, 1]
        """
        xyz = self.gaussians.get_xyz[gaussian_indices]  # [N, 3]
        n_gaussians = xyz.shape[0]
        
        dynamic_hits = torch.zeros(n_gaussians, device="cuda")
        total_hits = torch.zeros(n_gaussians, device="cuda")
        
        # Проверяем проекцию на каждую камеру
        for camera in self.cameras:
            # Проверяем, есть ли маска для этой камеры
            if camera.image_name not in self.masks:
                continue
            
            mask = self.masks[camera.image_name]
            if mask is None:
                continue
            
            # Проецируем точки в систему координат камеры
            # world_view_transform: [4, 4] преобразование из мира в камеру
            xyz_homo = torch.cat([xyz, torch.ones(n_gaussians, 1, device="cuda")], dim=1)  # [N, 4]
            xyz_cam = (camera.world_view_transform @ xyz_homo.T).T  # [N, 4]
            
            # Проверяем, что точки перед камерой
            valid_depth = xyz_cam[:, 2] > 0.01
            
            # Проецируем на плоскость изображения
            # Используем full_proj_transform для полной проекции
            xyz_proj = (camera.full_proj_transform @ xyz_homo.T).T  # [N, 4]
            xyz_proj = xyz_proj / (xyz_proj[:, 3:4] + 1e-7)  # нормализация
            
            # Переводим из NDC в пиксельные координаты
            # NDC: [-1, 1] -> pixel: [0, width/height]
            u = ((xyz_proj[:, 0] + 1.0) * 0.5 * camera.image_width).long()
            v = ((xyz_proj[:, 1] + 1.0) * 0.5 * camera.image_height).long()
            
            # Проверяем, что точки внутри изображения
            valid_u = (u >= 0) & (u < camera.image_width)
            valid_v = (v >= 0) & (v < camera.image_height)
            valid = valid_depth & valid_u & valid_v
            
            # Для валидных точек проверяем маску
            valid_indices = torch.where(valid)[0]
            if len(valid_indices) > 0:
                u_valid = u[valid_indices]
                v_valid = v[valid_indices]
                
                # Проверяем значения маски (mask > 0 означает динамический объект)
                mask_values = mask[v_valid, u_valid]
                
                # Обновляем счетчики
                total_hits[valid_indices] += 1
                dynamic_hits[valid_indices] += (mask_values > 0).float()
        
        # Вычисляем долю попаданий на динамические области
        dynamic_ratio = torch.zeros(n_gaussians, device="cuda")
        valid_mask = total_hits > 0
        dynamic_ratio[valid_mask] = dynamic_hits[valid_mask] / total_hits[valid_mask]
        
        return dynamic_ratio
    
    def filter_candidates_for_densification(self, candidate_mask: torch.Tensor,
                                          dynamic_threshold: float = 0.3) -> torch.Tensor:
        """
        Фильтрует кандидатов для денсификации, исключая гауссианы на динамических объектах.
        
        Args:
            candidate_mask: булева маска кандидатов для денсификации [N_gaussians]
            dynamic_threshold: порог для отсечения динамических гауссианов
        
        Returns:
            Отфильтрованная маска кандидатов
        """
        # Если нет масок, возвращаем исходную маску
        if not self.masks or all(m is None for m in self.masks.values()):
            return candidate_mask
        
        # Получаем индексы кандидатов
        candidate_indices = torch.where(candidate_mask)[0]
        
        if len(candidate_indices) == 0:
            return candidate_mask
        
        # Проверяем, какие кандидаты попадают на динамические области
        dynamic_scores = self.project_gaussians_to_cameras(candidate_indices)
        
        # Фильтруем: оставляем только те, у которых dynamic_score < threshold
        keep_mask = dynamic_scores < dynamic_threshold
        
        # Создаем новую маску кандидатов
        filtered_mask = torch.zeros_like(candidate_mask)
        filtered_indices = candidate_indices[keep_mask]
        filtered_mask[filtered_indices] = True
        
        n_removed = candidate_mask.sum().item() - filtered_mask.sum().item()
        if n_removed > 0:
            print(f"[Strategy B-b] Filtered {n_removed} gaussians from densification (on dynamic areas)")
        
        return filtered_mask


def densify_and_clone_masked(gaussians, grads, grad_threshold, scene_extent, 
                            mask_densifier: Optional[MaskAwareDensifier] = None,
                            dynamic_threshold: float = 0.3):
    """
    Модифицированная версия densify_and_clone с учетом масок динамических объектов.
    
    Args:
        gaussians: модель гауссианов
        grads: градиенты
        grad_threshold: порог градиента для клонирования
        scene_extent: размер сцены
        mask_densifier: объект для фильтрации по маскам
        dynamic_threshold: порог для отсечения динамических гауссианов
    """
    # Исходная логика выбора кандидатов для клонирования
    selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
    selected_pts_mask = torch.logical_and(selected_pts_mask,
                                         torch.max(gaussians.get_scaling, dim=1).values <= gaussians.percent_dense * scene_extent)
    
    # Применяем фильтрацию по маскам, если доступна
    if mask_densifier is not None:
        selected_pts_mask = mask_densifier.filter_candidates_for_densification(
            selected_pts_mask, dynamic_threshold
        )
    
    # Клонируем отфильтрованные гауссианы
    new_xyz = gaussians._xyz[selected_pts_mask]
    new_features_dc = gaussians._features_dc[selected_pts_mask]
    new_features_rest = gaussians._features_rest[selected_pts_mask]
    new_opacities = gaussians._opacity[selected_pts_mask]
    new_scaling = gaussians._scaling[selected_pts_mask]
    new_rotation = gaussians._rotation[selected_pts_mask]
    new_tmp_radii = gaussians.tmp_radii[selected_pts_mask]

    gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                   new_opacities, new_scaling, new_rotation, new_tmp_radii)


def densify_and_split_masked(gaussians, grads, grad_threshold, scene_extent, N=2,
                            mask_densifier: Optional[MaskAwareDensifier] = None,
                            dynamic_threshold: float = 0.3):
    """
    Модифицированная версия densify_and_split с учетом масок динамических объектов.
    
    Args:
        gaussians: модель гауссианов
        grads: градиенты
        grad_threshold: порог градиента для разделения
        scene_extent: размер сцены
        N: количество новых гауссианов при разделении
        mask_densifier: объект для фильтрации по маскам
        dynamic_threshold: порог для отсечения динамических гауссианов
    """
    from utils.general_utils import build_rotation
    
    n_init_points = gaussians.get_xyz.shape[0]
    
    # Исходная логика выбора кандидатов для разделения
    padded_grad = torch.zeros((n_init_points), device="cuda")
    padded_grad[:grads.shape[0]] = grads.squeeze()
    selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    selected_pts_mask = torch.logical_and(selected_pts_mask,
                                         torch.max(gaussians.get_scaling, dim=1).values > gaussians.percent_dense * scene_extent)
    
    # Применяем фильтрацию по маскам, если доступна
    if mask_densifier is not None:
        selected_pts_mask = mask_densifier.filter_candidates_for_densification(
            selected_pts_mask, dynamic_threshold
        )
    
    # Разделяем отфильтрованные гауссианы
    stds = gaussians.get_scaling[selected_pts_mask].repeat(N, 1)
    means = torch.zeros((stds.size(0), 3), device="cuda")
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(gaussians._rotation[selected_pts_mask]).repeat(N, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyz[selected_pts_mask].repeat(N, 1)
    new_scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
    new_rotation = gaussians._rotation[selected_pts_mask].repeat(N, 1)
    new_features_dc = gaussians._features_dc[selected_pts_mask].repeat(N, 1, 1)
    new_features_rest = gaussians._features_rest[selected_pts_mask].repeat(N, 1, 1)
    new_opacity = gaussians._opacity[selected_pts_mask].repeat(N, 1)
    new_tmp_radii = gaussians.tmp_radii[selected_pts_mask].repeat(N)

    gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                   new_opacity, new_scaling, new_rotation, new_tmp_radii)

    prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
    gaussians.prune_points(prune_filter)


def densify_and_prune_masked(gaussians, max_grad, min_opacity, extent, max_screen_size, radii,
                            mask_densifier: Optional[MaskAwareDensifier] = None,
                            dynamic_threshold: float = 0.3):
    """
    Полная замена для densify_and_prune с учетом масок.
    
    Args:
        gaussians: модель гауссианов
        max_grad: максимальный градиент для денсификации
        min_opacity: минимальная прозрачность для удаления
        extent: размер сцены
        max_screen_size: максимальный размер на экране
        radii: радиусы гауссианов на экране
        mask_densifier: объект для фильтрации по маскам
        dynamic_threshold: порог для отсечения динамических гауссианов
    """
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads[grads.isnan()] = 0.0

    gaussians.tmp_radii = radii
    
    # Используем модифицированные версии с фильтрацией по маскам
    densify_and_clone_masked(gaussians, grads, max_grad, extent, 
                            mask_densifier, dynamic_threshold)
    densify_and_split_masked(gaussians, grads, max_grad, extent, 
                            mask_densifier=mask_densifier, 
                            dynamic_threshold=dynamic_threshold)

    # Обычная обрезка (без изменений)
    prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = gaussians.max_radii2D > max_screen_size
        big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    gaussians.prune_points(prune_mask)
    
    gaussians.tmp_radii = None
    torch.cuda.empty_cache()
