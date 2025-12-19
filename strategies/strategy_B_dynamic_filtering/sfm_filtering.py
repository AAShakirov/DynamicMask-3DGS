"""
Strategy B Component (a): SfM Point Filtering
Фильтрация 3D-точек из SfM перед инициализацией гауссианов.

Идея: удалить 3D-точки, которые наблюдаются преимущественно на динамических объектах,
чтобы предотвратить создание гауссианов в этих областях.
"""

import os
import torch
import numpy as np
from typing import Dict, List
from utils.graphics_utils import BasicPointCloud


def load_masks_for_images(image_folder: str, image_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Загружает маски для всех изображений.
    
    Args:
        image_folder: путь к папке с изображениями
        image_names: список имен изображений
    
    Returns:
        Словарь {image_name: mask_array}
    """
    
    masks = {}
    for image_name in image_names:
        # Убираем расширение и добавляем _mask.npy
        base_name = os.path.splitext(image_name)[0]
        mask_path = os.path.join(image_folder, f"{base_name}_mask.npy")
        
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            masks[image_name] = mask
        else:
            print(f"[Warning] Mask not found for {image_name}, assuming no dynamic objects")
            # Создаем пустую маску (все статично)
            masks[image_name] = None
    
    return masks


def project_point_to_camera(point_3d: np.ndarray, R: np.ndarray, T: np.ndarray, 
                           K: np.ndarray, width: int, height: int) -> tuple:
    """
    Проецирует 3D-точку на плоскость изображения камеры.
    
    Args:
        point_3d: 3D координаты точки [x, y, z]
        R: матрица поворота камеры (3x3)
        T: вектор трансляции камеры (3,)
        K: матрица внутренних параметров камеры (3x3)
        width: ширина изображения
        height: высота изображения
    
    Returns:
        (u, v, is_valid): 2D координаты и флаг валидности проекции
    """
    # Преобразование в систему координат камеры
    point_cam = R @ point_3d + T
    
    # Проверка, что точка перед камерой
    if point_cam[2] <= 0:
        return None, None, False
    
    # Проекция на плоскость изображения
    point_2d = K @ point_cam
    u = int(point_2d[0] / point_2d[2])
    v = int(point_2d[1] / point_2d[2])
    
    # Проверка, что точка внутри изображения
    if 0 <= u < width and 0 <= v < height:
        return u, v, True
    
    return None, None, False


def filter_sfm_points(point_cloud: BasicPointCloud, cam_infos: List, 
                     masks: Dict[str, np.ndarray],
                     dynamic_threshold: float = 0.5) -> BasicPointCloud:
    """
    Фильтрует 3D-точки из SfM на основе 2D-масок динамических объектов.
    
    Args:
        point_cloud: исходное облако точек из SfM
        cam_infos: список с информацией о камерах
        masks: словарь с масками для каждого изображения
        dynamic_threshold: порог доли наблюдений на динамических объектах (0-1)
                          Точка удаляется, если >= threshold наблюдений попадают на маску
    
    Returns:
        Отфильтрованное облако точек
    """
    print("\n[Strategy B-a] Filtering SfM points based on dynamic masks...")
    
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    normals = np.asarray(point_cloud.normals)
    
    n_points = points.shape[0]
    point_dynamic_scores = np.zeros(n_points)  # Счетчик "динамичности" для каждой точки
    point_observations = np.zeros(n_points)    # Счетчик наблюдений для каждой точки
    
    # Для каждой камеры проверяем, попадают ли точки на маски
    for cam_info in cam_infos:
        image_name = cam_info.image_name
        
        # Если маски нет, пропускаем
        if image_name not in masks or masks[image_name] is None:
            continue
        
        mask = masks[image_name]
        R = cam_info.R
        T = cam_info.T
        width = cam_info.width
        height = cam_info.height
        
        # Построение матрицы внутренних параметров
        # Используем FoV для восстановления фокусного расстояния
        fx = width / (2 * np.tan(cam_info.FovX / 2))
        fy = height / (2 * np.tan(cam_info.FovY / 2))
        cx = width / 2
        cy = height / 2
        K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
        
        # Проецируем каждую точку на изображение
        for i, point_3d in enumerate(points):
            u, v, is_valid = project_point_to_camera(point_3d, R, T, K, width, height)
            
            if is_valid:
                point_observations[i] += 1
                
                # Проверяем, попадает ли точка на маску (mask > 0 означает динамический объект)
                if mask[v, u] > 0:
                    point_dynamic_scores[i] += 1
    
    # Вычисляем долю "динамичных" наблюдений для каждой точки
    valid_points = point_observations > 0
    dynamic_ratio = np.zeros(n_points)
    dynamic_ratio[valid_points] = point_dynamic_scores[valid_points] / point_observations[valid_points]
    
    # Фильтруем точки: оставляем только те, у которых dynamic_ratio < threshold
    keep_mask = dynamic_ratio < dynamic_threshold
    
    n_removed = n_points - keep_mask.sum()
    n_kept = keep_mask.sum()
    
    print(f"[Strategy B-a] Points before filtering: {n_points}")
    print(f"[Strategy B-a] Points removed (dynamic): {n_removed} ({100*n_removed/n_points:.1f}%)")
    print(f"[Strategy B-a] Points kept (static): {n_kept} ({100*n_kept/n_points:.1f}%)")
    
    # Создаем новое облако точек только из статичных точек
    filtered_points = points[keep_mask]
    filtered_colors = colors[keep_mask]
    filtered_normals = normals[keep_mask]
    
    return BasicPointCloud(
        points=filtered_points,
        colors=filtered_colors,
        normals=filtered_normals
    )


def filter_point_cloud_with_masks(point_cloud: BasicPointCloud, 
                                 cam_infos: List,
                                 source_path: str,
                                 dynamic_threshold: float = 0.5) -> BasicPointCloud:
    """
    Wrapper-функция для фильтрации облака точек с автоматической загрузкой масок.
    
    Args:
        point_cloud: исходное облако точек
        cam_infos: информация о камерах
        source_path: путь к датасету
        dynamic_threshold: порог фильтрации
    
    Returns:
        Отфильтрованное облако точек
    """    
    # Загружаем маски
    image_folder = os.path.join(source_path, "images")
    if not os.path.exists(image_folder):
        # Пробуем альтернативные пути
        image_folder = os.path.join(source_path, "colmap", "images")
        if not os.path.exists(image_folder):
            print("[Warning] Cannot find images folder, skipping SfM filtering")
            return point_cloud
    
    image_names = [cam_info.image_name for cam_info in cam_infos]
    masks = load_masks_for_images(image_folder, image_names)
    
    # Проверяем, есть ли хоть одна маска
    has_masks = any(mask is not None for mask in masks.values())
    if not has_masks:
        print("[Strategy B-a] No masks found, skipping SfM filtering")
        return point_cloud
    
    # Фильтруем точки
    return filter_sfm_points(point_cloud, cam_infos, masks, dynamic_threshold)
