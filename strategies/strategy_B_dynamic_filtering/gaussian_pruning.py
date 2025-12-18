"""
Strategy B Component (c): Dynamic Gaussian Pruning
Удаление "старых" гауссианов, которые постоянно проецируются на динамические области.

Идея: отслеживать, как часто гауссианы попадают на маски динамических объектов,
и удалять те, которые стабильно находятся в динамических областях.
"""

import torch
import numpy as np
from typing import Dict, Optional


class DynamicGaussianTracker:
    """
    Класс для отслеживания и удаления гауссианов, находящихся в динамических областях.
    """
    
    def __init__(self, n_gaussians: int, tracking_window: int = 100):
        """
        Args:
            n_gaussians: начальное количество гауссианов
            tracking_window: размер окна для отслеживания (в итерациях)
        """
        self.tracking_window = tracking_window
        
        # Счетчики для каждого гауссиана
        self.dynamic_hit_count = torch.zeros(n_gaussians, device="cuda")
        self.total_check_count = torch.zeros(n_gaussians, device="cuda")
        
        # История для скользящего окна
        self.iteration_counter = 0
    
    def resize(self, new_size: int):
        """
        Изменяет размер трекеров при добавлении/удалении гауссианов.
        
        Args:
            new_size: новое количество гауссианов
        """
        old_size = self.dynamic_hit_count.shape[0]
        
        if new_size > old_size:
            # Добавляем новые гауссианы
            additional = new_size - old_size
            self.dynamic_hit_count = torch.cat([
                self.dynamic_hit_count,
                torch.zeros(additional, device="cuda")
            ])
            self.total_check_count = torch.cat([
                self.total_check_count,
                torch.zeros(additional, device="cuda")
            ])
        elif new_size < old_size:
            # Уменьшаем (обычно не нужно, так как pruning делается через маску)
            self.dynamic_hit_count = self.dynamic_hit_count[:new_size]
            self.total_check_count = self.total_check_count[:new_size]
    
    def update_after_densification(self, old_count: int, new_count: int):
        """
        Обновляет трекеры после денсификации.
        
        Args:
            old_count: количество гауссианов до денсификации
            new_count: количество гауссианов после денсификации
        """
        if new_count != old_count:
            self.resize(new_count)
    
    def update_after_pruning(self, keep_mask: torch.Tensor):
        """
        Обновляет трекеры после удаления гауссианов.
        
        Args:
            keep_mask: булева маска гауссианов, которые остаются
        """
        self.dynamic_hit_count = self.dynamic_hit_count[keep_mask]
        self.total_check_count = self.total_check_count[keep_mask]
    
    def update_dynamic_scores(self, gaussians, camera, mask: torch.Tensor):
        """
        Обновляет счетчики попаданий на динамические области для текущей камеры.
        
        Args:
            gaussians: модель гауссианов
            camera: текущая камера
            mask: маска динамических объектов для этой камеры [H, W]
        """
        if mask is None:
            return
        
        xyz = gaussians.get_xyz  # [N, 3]
        n_gaussians = xyz.shape[0]
        
        # Убеждаемся, что размеры совпадают
        if self.dynamic_hit_count.shape[0] != n_gaussians:
            self.resize(n_gaussians)
        
        # Проецируем все гауссианы на камеру
        xyz_homo = torch.cat([xyz, torch.ones(n_gaussians, 1, device="cuda")], dim=1)  # [N, 4]
        xyz_cam = (camera.world_view_transform @ xyz_homo.T).T  # [N, 4]
        
        # Проверяем, что точки перед камерой
        valid_depth = xyz_cam[:, 2] > 0.01
        
        # Полная проекция
        xyz_proj = (camera.full_proj_transform @ xyz_homo.T).T  # [N, 4]
        xyz_proj = xyz_proj / (xyz_proj[:, 3:4] + 1e-7)
        
        # NDC -> пиксельные координаты
        u = ((xyz_proj[:, 0] + 1.0) * 0.5 * camera.image_width).long()
        v = ((xyz_proj[:, 1] + 1.0) * 0.5 * camera.image_height).long()
        
        # Валидация границ
        valid_u = (u >= 0) & (u < camera.image_width)
        valid_v = (v >= 0) & (v < camera.image_height)
        valid = valid_depth & valid_u & valid_v
        
        # Обновляем счетчики для валидных проекций
        valid_indices = torch.where(valid)[0]
        if len(valid_indices) > 0:
            u_valid = u[valid_indices]
            v_valid = v[valid_indices]
            
            # Проверяем маску
            mask_values = mask[v_valid, u_valid]
            is_dynamic = mask_values > 0
            
            # Обновляем счетчики
            self.total_check_count[valid_indices] += 1
            self.dynamic_hit_count[valid_indices] += is_dynamic.float()
        
        self.iteration_counter += 1
    
    def get_dynamic_gaussians(self, prune_threshold: float = 0.7, 
                             min_observations: int = 10) -> torch.Tensor:
        """
        Возвращает маску гауссианов, которые следует удалить как динамические.
        
        Args:
            prune_threshold: порог доли попаданий на динамические области (0-1)
            min_observations: минимальное количество наблюдений для принятия решения
        
        Returns:
            Булева маска гауссианов для удаления
        """
        # Вычисляем долю попаданий на динамические области
        dynamic_ratio = torch.zeros_like(self.dynamic_hit_count)
        observed = self.total_check_count >= min_observations
        
        dynamic_ratio[observed] = (
            self.dynamic_hit_count[observed] / self.total_check_count[observed]
        )
        
        # Помечаем для удаления те, у которых доля >= порога
        prune_mask = (dynamic_ratio >= prune_threshold) & observed
        
        return prune_mask
    
    def reset_statistics(self):
        """
        Сбрасывает статистику (например, после сброса прозрачности).
        """
        self.dynamic_hit_count.zero_()
        self.total_check_count.zero_()
        self.iteration_counter = 0


def prune_dynamic_gaussians(gaussians, tracker: DynamicGaussianTracker,
                           prune_threshold: float = 0.7,
                           min_observations: int = 10) -> int:
    """
    Удаляет гауссианы, которые постоянно находятся в динамических областях.
    
    Args:
        gaussians: модель гауссианов
        tracker: трекер динамических гауссианов
        prune_threshold: порог для удаления
        min_observations: минимальное количество наблюдений
    
    Returns:
        Количество удаленных гауссианов
    """
    # Получаем маску гауссианов для удаления
    prune_mask = tracker.get_dynamic_gaussians(prune_threshold, min_observations)
    
    n_to_prune = prune_mask.sum().item()
    
    if n_to_prune > 0:
        print(f"[Strategy B-c] Pruning {n_to_prune} dynamic gaussians "
              f"(threshold={prune_threshold}, min_obs={min_observations})")
        
        # Удаляем гауссианы
        gaussians.prune_points(prune_mask)
        
        # Обновляем трекер
        keep_mask = ~prune_mask
        tracker.update_after_pruning(keep_mask)
    
    return n_to_prune


def update_tracker_after_standard_pruning(tracker: DynamicGaussianTracker, 
                                         prune_mask: torch.Tensor):
    """
    Обновляет трекер после стандартного pruning (по opacity, размеру и т.д.).
    
    Args:
        tracker: трекер динамических гауссианов
        prune_mask: маска удаленных гауссианов
    """
    keep_mask = ~prune_mask
    tracker.update_after_pruning(keep_mask)


class AdaptivePruningScheduler:
    """
    Планировщик для адаптивного удаления динамических гауссианов.
    """
    
    def __init__(self, 
                 start_iter: int = 3000,
                 end_iter: int = 15000,
                 prune_interval: int = 500,
                 initial_threshold: float = 0.8,
                 final_threshold: float = 0.6):
        """
        Args:
            start_iter: итерация начала удаления динамических гауссианов
            end_iter: итерация окончания удаления
            prune_interval: интервал между удалениями
            initial_threshold: начальный порог (более строгий)
            final_threshold: конечный порог (менее строгий)
        """
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.prune_interval = prune_interval
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
    
    def should_prune(self, iteration: int) -> bool:
        """
        Проверяет, нужно ли удалять на текущей итерации.
        
        Args:
            iteration: текущая итерация
        
        Returns:
            True, если нужно выполнить удаление
        """
        if iteration < self.start_iter or iteration > self.end_iter:
            return False
        
        return (iteration - self.start_iter) % self.prune_interval == 0
    
    def get_threshold(self, iteration: int) -> float:
        """
        Вычисляет текущий порог удаления (линейная интерполяция).
        
        Args:
            iteration: текущая итерация
        
        Returns:
            Текущий порог
        """
        if iteration < self.start_iter:
            return self.initial_threshold
        if iteration > self.end_iter:
            return self.final_threshold
        
        progress = (iteration - self.start_iter) / (self.end_iter - self.start_iter)
        threshold = self.initial_threshold + progress * (self.final_threshold - self.initial_threshold)
        
        return threshold
