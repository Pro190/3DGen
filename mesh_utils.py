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
from typing import Tuple, Optional, Callable, List
import torch
from skimage import measure
from tqdm import tqdm


def load_mesh(path: str, normalize: bool = True) -> trimesh.Trimesh:
    """
    Загрузка меша из файла.
    
    Args:
        path: путь к файлу (.obj, .off, .ply)
        normalize: нормализовать в [-0.5, 0.5]
        
    Returns:
        Trimesh объект
    """
    mesh = trimesh.load(path, force='mesh', process=False)
    
    # Обработка Scene
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Пустая сцена: {path}")
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    if normalize:
        mesh = normalize_mesh(mesh)
    
    return mesh


def normalize_mesh(
    mesh: trimesh.Trimesh, 
    scale: float = 0.45
) -> trimesh.Trimesh:
    """
    Нормализация меша в единичный куб.
    
    Args:
        mesh: входной меш
        scale: масштаб (0.5 = полный куб [-0.5, 0.5])
    """
    vertices = mesh.vertices.copy()
    
    # Центрирование
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    
    # Масштабирование
    max_dist = np.abs(vertices).max()
    if max_dist > 1e-6:
        vertices = vertices / max_dist * scale
    
    mesh.vertices = vertices
    return mesh


def sample_points_from_mesh(
    mesh: trimesh.Trimesh,
    num_surface: int = 10000,
    num_uniform: int = 10000,
    surface_noise: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Сэмплирование точек для обучения.
    
    Returns:
        points: [N, 3]
        occupancies: [N] (1 = внутри, 0 = снаружи)
    """
    points_list = []
    occupancies_list = []
    
    # Точки на поверхности + шум
    if num_surface > 0:
        try:
            surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface)
            noise = np.random.randn(num_surface, 3) * surface_noise
            surface_points = surface_points + noise
        except:
            indices = np.random.choice(len(mesh.vertices), num_surface, replace=True)
            surface_points = mesh.vertices[indices]
        
        # Проверяем occupancy
        try:
            surface_occ = mesh.contains(surface_points).astype(np.float32)
        except:
            surface_occ = np.ones(len(surface_points), dtype=np.float32) * 0.5
        
        points_list.append(surface_points.astype(np.float32))
        occupancies_list.append(surface_occ)
    
    # Равномерные точки
    if num_uniform > 0:
        uniform_points = np.random.uniform(-0.55, 0.55, (num_uniform, 3))
        
        try:
            uniform_occ = mesh.contains(uniform_points).astype(np.float32)
        except:
            uniform_occ = np.zeros(num_uniform, dtype=np.float32)
        
        points_list.append(uniform_points.astype(np.float32))
        occupancies_list.append(uniform_occ)
    
    points = np.concatenate(points_list, axis=0)
    occupancies = np.concatenate(occupancies_list, axis=0)
    
    return points, occupancies


def create_grid_points(
    resolution: int,
    bounds: Tuple[float, float] = (-0.55, 0.55)
) -> np.ndarray:
    """
    Создание 3D сетки точек.
    
    Returns:
        points: [resolution^3, 3]
    """
    min_bound, max_bound = bounds
    
    x = np.linspace(min_bound, max_bound, resolution)
    y = np.linspace(min_bound, max_bound, resolution)
    z = np.linspace(min_bound, max_bound, resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    return grid_points.astype(np.float32)


def extract_mesh_marching_cubes(
    occupancy_fn: Callable[[torch.Tensor], torch.Tensor],
    resolution: int = 64,
    threshold: float = 0.5,
    bounds: Tuple[float, float] = (-0.55, 0.55),
    batch_size: int = 65536,
    device: str = 'cuda',
    verbose: bool = True
) -> Optional[trimesh.Trimesh]:
    """
    Извлечение меша из occupancy function через Marching Cubes.
    
    Args:
        occupancy_fn: функция (points: Tensor[N, 3]) -> occupancy: Tensor[N]
        resolution: разрешение 3D сетки
        threshold: порог для поверхности
        bounds: границы пространства
        batch_size: размер батча
        device: устройство
        verbose: показывать прогресс
        
    Returns:
        trimesh.Trimesh или None
    """
    min_bound, max_bound = bounds
    
    # Создаём сетку
    grid_points = create_grid_points(resolution, bounds)
    total_points = len(grid_points)
    
    # Вычисляем occupancy батчами
    occupancy_values = []
    
    iterator = range(0, total_points, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Computing occupancy", leave=False)
    
    with torch.no_grad():
        for i in iterator:
            batch_points = torch.from_numpy(
                grid_points[i:i+batch_size]
            ).float().to(device)
            
            batch_occ = occupancy_fn(batch_points)
            occupancy_values.append(batch_occ.cpu().numpy())
    
    occupancy_grid = np.concatenate(occupancy_values).reshape(
        resolution, resolution, resolution
    )
    
    # Проверка на пустой/полный grid
    occ_ratio = (occupancy_grid > threshold).mean()
    if occ_ratio < 0.001 or occ_ratio > 0.999:
        if verbose:
            print(f"[mesh_utils] ⚠️ Occupancy ratio: {occ_ratio:.4f} (слишком {'низкий' if occ_ratio < 0.5 else 'высокий'})")
        
        # Пробуем адаптивный порог
        if occ_ratio < 0.001:
            adaptive_threshold = np.percentile(occupancy_grid, 95)
        else:
            adaptive_threshold = np.percentile(occupancy_grid, 5)
        
        if verbose:
            print(f"[mesh_utils] Пробую адаптивный порог: {adaptive_threshold:.4f}")
        
        threshold = adaptive_threshold
    
    # Marching Cubes
    try:
        spacing = (max_bound - min_bound) / resolution
        
        vertices, faces, normals, _ = measure.marching_cubes(
            occupancy_grid, 
            level=threshold,
            spacing=(spacing, spacing, spacing)
        )
        
        # Смещаем в правильную систему координат
        vertices = vertices + min_bound
        
        mesh = trimesh.Trimesh(
            vertices=vertices, 
            faces=faces, 
            vertex_normals=normals
        )
        
        if verbose:
            print(f"[mesh_utils] Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
        
    except Exception as e:
        if verbose:
            print(f"[mesh_utils] Ошибка Marching Cubes: {e}")
        return None


def extract_mesh_multiresolution(
    occupancy_fn: Callable[[torch.Tensor], torch.Tensor],
    resolutions: List[int] = [32, 64, 128],
    threshold: float = 0.5,
    device: str = 'cuda',
    verbose: bool = True
) -> Optional[trimesh.Trimesh]:
    """
    Извлечение меша с постепенным увеличением разрешения.
    Более эффективно для высоких разрешений.
    """
    mesh = None
    
    for res in resolutions:
        if verbose:
            print(f"[mesh_utils] Resolution: {res}")
        
        mesh = extract_mesh_marching_cubes(
            occupancy_fn,
            resolution=res,
            threshold=threshold,
            device=device,
            verbose=verbose
        )
        
        if mesh is not None and len(mesh.vertices) > 100:
            # Уточняем bounds на основе текущего меша
            pass
    
    return mesh


def simplify_mesh(
    mesh: trimesh.Trimesh, 
    target_faces: int = 10000,
    method: str = 'quadric'
) -> trimesh.Trimesh:
    """
    Упрощение меша.
    
    Args:
        mesh: входной меш
        target_faces: целевое количество граней
        method: метод упрощения ('quadric', 'vertex_clustering')
    """
    if len(mesh.faces) <= target_faces:
        return mesh
    
    try:
        if method == 'quadric':
            simplified = mesh.simplify_quadric_decimation(target_faces)
        else:
            # Vertex clustering (быстрее, но менее точно)
            voxel_size = mesh.extents.max() / (target_faces ** (1/3))
            simplified = mesh.simplify_vertex_clustering(voxel_size)
        
        return simplified
    except:
        return mesh


def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int = 3,
    lamb: float = 0.5
) -> trimesh.Trimesh:
    """
    Сглаживание меша (Laplacian smoothing).
    """
    try:
        smoothed = trimesh.smoothing.filter_laplacian(
            mesh, 
            iterations=iterations,
            lamb=lamb
        )
        return smoothed
    except:
        return mesh


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Попытка починить меш (заполнить дыры, убрать дубликаты).
    """
    try:
        # Удаляем дубликаты вершин
        mesh.merge_vertices()
        
        # Удаляем вырожденные грани
        mesh.remove_degenerate_faces()
        
        # Удаляем дубликаты граней
        mesh.remove_duplicate_faces()
        
        # Пытаемся сделать watertight
        trimesh.repair.fill_holes(mesh)
        
        return mesh
    except:
        return mesh


def save_mesh(
    mesh: trimesh.Trimesh, 
    path: str,
    include_normals: bool = True
) -> bool:
    """
    Сохранение меша в файл.
    
    Args:
        mesh: меш для сохранения
        path: путь к файлу
        include_normals: включать нормали
        
    Returns:
        True если успешно
    """
    try:
        # Определяем формат по расширению
        ext = path.lower().split('.')[-1]
        
        if ext == 'obj':
            mesh.export(path, file_type='obj', include_normals=include_normals)
        elif ext == 'ply':
            mesh.export(path, file_type='ply')
        elif ext == 'stl':
            mesh.export(path, file_type='stl')
        elif ext == 'glb' or ext == 'gltf':
            mesh.export(path, file_type='glb')
        else:
            mesh.export(path)
        
        print(f"[mesh_utils] ✓ Сохранено: {path} "
              f"({len(mesh.vertices)} вершин, {len(mesh.faces)} граней)")
        return True
        
    except Exception as e:
        print(f"[mesh_utils] ✗ Ошибка сохранения: {e}")
        return False


def mesh_to_pointcloud(
    mesh: trimesh.Trimesh,
    num_points: int = 10000
) -> np.ndarray:
    """
    Конвертация меша в облако точек.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.array(points, dtype=np.float32)


def pointcloud_to_mesh(
    points: np.ndarray,
    method: str = 'ball_pivot',
    **kwargs
) -> Optional[trimesh.Trimesh]:
    """
    Реконструкция меша из облака точек.
    
    Требует open3d для некоторых методов.
    """
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Оценка нормалей
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=10)
        
        if method == 'ball_pivot':
            radii = kwargs.get('radii', [0.01, 0.02, 0.04])
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        elif method == 'poisson':
            depth = kwargs.get('depth', 9)
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
        else:
            return None
        
        # Конвертация в trimesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
        
    except ImportError:
        print("[mesh_utils] open3d не установлен для реконструкции из облака точек")
        return None
    except Exception as e:
        print(f"[mesh_utils] Ошибка реконструкции: {e}")
        return None