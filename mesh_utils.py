"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".
Описание: Вспомогательные инструменты для работы с мешами: функции визуализации, сохранения в формат .obj и обработки полигональных сеток.
Дата: 2026
================================================================================
"""
import numpy as np
import trimesh
from typing import Tuple, Optional
import torch
from skimage import measure


def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Загрузка и нормализация меша.
    
    Args:
        path: путь к файлу (.obj, .off, .ply)
        
    Returns:
        Нормализованный trimesh объект
    """
    mesh = trimesh.load(path, force='mesh')
    
    # Обработка Scene
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Пустая сцена: {path}")
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    # Нормализация в единичный куб [-0.5, 0.5]
    mesh = normalize_mesh(mesh)
    
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Нормализация меша в единичный куб [-0.5, 0.5].
    """
    vertices = mesh.vertices.copy()
    
    # Центрирование
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    
    # Масштабирование
    max_dist = np.abs(vertices).max()
    if max_dist > 1e-6:
        vertices = vertices / max_dist * 0.5
    
    mesh.vertices = vertices
    return mesh


def sample_points_from_mesh(
    mesh: trimesh.Trimesh,
    num_surface: int = 1024,
    num_uniform: int = 1024,
    surface_noise: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Сэмплирование точек для обучения Occupancy Network.
    
    Args:
        mesh: входной меш
        num_surface: количество точек около поверхности
        num_uniform: количество равномерно распределённых точек
        surface_noise: std шума для поверхностных точек
        
    Returns:
        points: [N, 3] координаты точек
        occupancies: [N] метки (1 = внутри, 0 = снаружи)
    """
    points_list = []
    occupancies_list = []
    
    # 1. Точки на поверхности + шум
    if num_surface > 0:
        surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface)
        
        # Добавляем шум
        noise = np.random.randn(num_surface, 3) * surface_noise
        surface_points = surface_points + noise
        
        # Проверяем occupancy
        surface_occ = mesh.contains(surface_points).astype(np.float32)
        
        points_list.append(surface_points)
        occupancies_list.append(surface_occ)
    
    # 2. Равномерно распределённые точки в кубе [-0.55, 0.55]
    if num_uniform > 0:
        uniform_points = np.random.uniform(-0.55, 0.55, (num_uniform, 3))
        uniform_occ = mesh.contains(uniform_points).astype(np.float32)
        
        points_list.append(uniform_points)
        occupancies_list.append(uniform_occ)
    
    points = np.concatenate(points_list, axis=0).astype(np.float32)
    occupancies = np.concatenate(occupancies_list, axis=0).astype(np.float32)
    
    return points, occupancies


def extract_mesh_marching_cubes(
    occupancy_fn,
    resolution: int = 64,
    threshold: float = 0.5,
    bounds: Tuple[float, float] = (-0.55, 0.55),
    batch_size: int = 32768,
    device: str = 'cuda'
) -> Optional[trimesh.Trimesh]:
    """
    Извлечение меша из occupancy function через Marching Cubes.
    
    Args:
        occupancy_fn: функция (points: Tensor) -> occupancy: Tensor
        resolution: разрешение 3D сетки
        threshold: порог для определения поверхности
        bounds: границы пространства
        batch_size: размер батча для запросов
        device: устройство для вычислений
        
    Returns:
        trimesh.Trimesh или None если меш не извлечён
    """
    # Создаём 3D сетку точек
    min_bound, max_bound = bounds
    
    x = np.linspace(min_bound, max_bound, resolution)
    y = np.linspace(min_bound, max_bound, resolution)
    z = np.linspace(min_bound, max_bound, resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Вычисляем occupancy батчами
    occupancy_values = []
    
    with torch.no_grad():
        for i in range(0, len(grid_points), batch_size):
            batch_points = torch.from_numpy(
                grid_points[i:i+batch_size]
            ).float().to(device)
            
            batch_occ = occupancy_fn(batch_points)
            occupancy_values.append(batch_occ.cpu().numpy())
    
    occupancy_grid = np.concatenate(occupancy_values).reshape(resolution, resolution, resolution)
    
    # Marching Cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(
            occupancy_grid, 
            level=threshold,
            spacing=((max_bound - min_bound) / resolution,) * 3
        )
        
        # Смещаем вершины в правильную систему координат
        vertices = vertices + min_bound
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        
        return mesh
        
    except Exception as e:
        print(f"[mesh_utils] Ошибка Marching Cubes: {e}")
        return None


def simplify_mesh(
    mesh: trimesh.Trimesh, 
    target_faces: int = 10000
) -> trimesh.Trimesh:
    """
    Упрощение меша до заданного количества граней.
    """
    if len(mesh.faces) <= target_faces:
        return mesh
    
    # Используем quadric decimation
    simplified = mesh.simplify_quadric_decimation(target_faces)
    
    return simplified


def save_mesh(mesh: trimesh.Trimesh, path: str) -> None:
    """Сохранение меша в файл."""
    mesh.export(path)
    print(f"[mesh_utils] Сохранено: {path} ({len(mesh.vertices)} вершин, {len(mesh.faces)} граней)")