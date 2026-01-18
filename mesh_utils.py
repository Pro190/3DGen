"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Утилиты для работы с 3D мешами
Дата: 2026
================================================================================

Модуль содержит функции для:
    1. Загрузки и сохранения мешей (.obj, .ply, .stl, .glb)
    2. Нормализации мешей (центрирование, масштабирование)
    3. Извлечения поверхности через Marching Cubes
    4. Сэмплирования точек на поверхности
    5. Упрощения и сглаживания мешей
    6. Конвертации между форматами

Зависимости:
    - trimesh: работа с мешами
    - scikit-image: Marching Cubes алгоритм
    - numpy: математические операции
    - torch: интеграция с моделью

Примеры использования:
    # Загрузка и нормализация меша
    mesh = load_mesh('model.obj', normalize=True)
    
    # Извлечение меша из occupancy функции
    mesh = extract_mesh_marching_cubes(occupancy_fn, resolution=128)
    
    # Упрощение меша
    simplified = simplify_mesh(mesh, target_faces=10000)
    
    # Сохранение
    save_mesh(mesh, 'output.obj')
"""

import numpy as np
import trimesh
from typing import Tuple, Optional, Callable, List
import torch
from skimage import measure
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И СОХРАНЕНИЕ
# ═══════════════════════════════════════════════════════════════════════════════

def load_mesh(
    path: str,
    normalize: bool = True,
    scale: float = 0.45
) -> trimesh.Trimesh:
    """
    Загрузка меша из файла.
    
    Поддерживаемые форматы:
        - .obj (Wavefront OBJ)
        - .ply (Stanford PLY)
        - .stl (Stereolithography)
        - .off (Object File Format)
        - .glb/.gltf (GL Transmission Format)
    
    Args:
        path: Путь к файлу меша
        normalize: Нормализовать меш в единичный куб
        scale: Масштаб нормализации (0.45 = куб [-0.45, 0.45])
    
    Returns:
        trimesh.Trimesh: загруженный меш
    
    Raises:
        ValueError: если файл пустой или повреждённый
        FileNotFoundError: если файл не существует
    
    Пример:
        mesh = load_mesh('chair.obj')
        print(f"Vertices: {len(mesh.vertices)}")
        print(f"Faces: {len(mesh.faces)}")
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Загрузка файла
    # ─────────────────────────────────────────────────────────────────────────
    #
    # force='mesh': принудительно загружаем как меш (не как Scene)
    # process=False: отключаем автоматическую обработку (merging и т.д.)
    # ─────────────────────────────────────────────────────────────────────────
    
    mesh = trimesh.load(path, force='mesh', process=False)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Обработка Scene (если файл содержит несколько объектов)
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Некоторые 3D файлы содержат несколько мешей в одной "сцене".
    # Объединяем их в один меш.
    # ─────────────────────────────────────────────────────────────────────────
    
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Empty scene in file: {path}")
        
        # Объединяем все геометрии в одну
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Нормализация (опционально)
    # ─────────────────────────────────────────────────────────────────────────
    
    if normalize:
        mesh = normalize_mesh(mesh, scale=scale)
    
    return mesh


def save_mesh(
    mesh: trimesh.Trimesh,
    path: str,
    include_normals: bool = True
) -> bool:
    """
    Сохранение меша в файл.
    
    Формат определяется автоматически по расширению файла:
        - .obj → Wavefront OBJ (текстовый, с нормалями)
        - .ply → Stanford PLY (бинарный, компактный)
        - .stl → STL (для 3D печати)
        - .glb → glTF Binary (для web/игр)
    
    Args:
        mesh: Меш для сохранения
        path: Путь к выходному файлу
        include_normals: Включать нормали вершин (для .obj)
    
    Returns:
        True если сохранение успешно, False иначе
    
    Пример:
        mesh = load_mesh('input.obj')
        save_mesh(mesh, 'output.ply')  # Конвертация в PLY
    """
    
    try:
        # Определяем формат по расширению
        ext = path.lower().split('.')[-1]
        
        if ext == 'obj':
            # OBJ с опциональными нормалями
            mesh.export(path, file_type='obj', include_normals=include_normals)
        elif ext == 'ply':
            # PLY (бинарный по умолчанию)
            mesh.export(path, file_type='ply')
        elif ext == 'stl':
            # STL для 3D печати
            mesh.export(path, file_type='stl')
        elif ext in ['glb', 'gltf']:
            # glTF для web
            mesh.export(path, file_type='glb')
        else:
            # Автоопределение
            mesh.export(path)
        
        print(f"[mesh_utils.py] ✓ Saved: {path} "
              f"({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
        return True
        
    except Exception as e:
        print(f"[mesh_utils.py] ✗ Save failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# НОРМАЛИЗАЦИЯ И ТРАНСФОРМАЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_mesh(
    mesh: trimesh.Trimesh,
    scale: float = 0.45
) -> trimesh.Trimesh:
    """
    Нормализация меша в единичный куб.
    
    Процесс:
        1. Центрирование: перемещаем центроид в начало координат
        2. Масштабирование: вписываем в куб [-scale, scale]³
    
    Args:
        mesh: Входной меш
        scale: Половина размера куба (0.45 → куб [-0.45, 0.45])
              Используем 0.45, а не 0.5, чтобы оставить зазор
              для сэмплирования точек "снаружи"
    
    Returns:
        Нормализованный меш (изменяет исходный объект!)
    
    Пример:
        mesh = load_mesh('chair.obj', normalize=False)
        print(f"Before: {mesh.vertices.min()} to {mesh.vertices.max()}")
        
        mesh = normalize_mesh(mesh)
        print(f"After: {mesh.vertices.min()} to {mesh.vertices.max()}")
        # After: -0.45 to 0.45
    """
    
    vertices = mesh.vertices.copy()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 1: Центрирование
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Вычисляем центроид (среднее всех вершин) и вычитаем его.
    # После этого центр объекта находится в точке (0, 0, 0).
    # ─────────────────────────────────────────────────────────────────────────
    
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    
    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 2: Масштабирование
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Находим максимальную абсолютную координату и масштабируем так,
    # чтобы объект вписался в куб [-scale, scale]³.
    #
    # Используем max(abs) вместо max-min для сохранения пропорций
    # относительно центра.
    # ─────────────────────────────────────────────────────────────────────────
    
    max_dist = np.abs(vertices).max()
    
    if max_dist > 1e-6:  # Защита от деления на ноль
        vertices = vertices / max_dist * scale
    
    mesh.vertices = vertices
    
    return mesh


def transform_mesh(
    mesh: trimesh.Trimesh,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    scale: Optional[float] = None
) -> trimesh.Trimesh:
    """
    Применение трансформаций к мешу.
    
    Args:
        mesh: Входной меш
        rotation: Матрица вращения [3, 3] или углы Эйлера [3]
        translation: Вектор смещения [3]
        scale: Коэффициент масштабирования
    
    Returns:
        Трансформированный меш
    
    Пример:
        # Поворот на 90° вокруг оси Y
        import numpy as np
        angle = np.pi / 2
        rotation = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        rotated_mesh = transform_mesh(mesh, rotation=rotation)
    """
    
    vertices = mesh.vertices.copy()
    
    # Масштабирование
    if scale is not None:
        vertices = vertices * scale
    
    # Вращение
    if rotation is not None:
        rotation = np.array(rotation)
        
        if rotation.shape == (3,):
            # Углы Эйлера → матрица вращения
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_euler('xyz', rotation).as_matrix()
        
        vertices = vertices @ rotation.T
    
    # Смещение
    if translation is not None:
        vertices = vertices + np.array(translation)
    
    mesh_copy = mesh.copy()
    mesh_copy.vertices = vertices
    
    return mesh_copy


# ═══════════════════════════════════════════════════════════════════════════════
# MARCHING CUBES
# ═══════════════════════════════════════════════════════════════════════════════

def create_grid_points(
    resolution: int,
    bounds: Tuple[float, float] = (-0.5, 0.5)
) -> np.ndarray:
    """
    Создание равномерной 3D сетки точек.
    
    Args:
        resolution: Количество точек по каждой оси (N)
        bounds: Границы пространства (min, max)
    
    Returns:
        np.ndarray [N³, 3]: массив координат всех точек сетки
    
    Пример:
        # Сетка 64³ в кубе [-0.5, 0.5]³
        points = create_grid_points(64)
        print(points.shape)  # (262144, 3)
    """
    
    min_bound, max_bound = bounds
    
    # Координаты по каждой оси
    x = np.linspace(min_bound, max_bound, resolution)
    y = np.linspace(min_bound, max_bound, resolution)
    z = np.linspace(min_bound, max_bound, resolution)
    
    # 3D сетка
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Преобразуем в массив точек [N³, 3]
    grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    return grid_points.astype(np.float32)


def extract_mesh_marching_cubes(
    occupancy_fn: Callable[[torch.Tensor], torch.Tensor],
    resolution: int = 128,
    threshold: float = 0.5,
    bounds: Tuple[float, float] = (-0.5, 0.5),
    batch_size: int = 100000,
    device: str = 'cuda',
    verbose: bool = True
) -> Optional[trimesh.Trimesh]:
    """
    Извлечение меша из occupancy функции через Marching Cubes.
    
    Marching Cubes — классический алгоритм извлечения изоповерхности
    из скалярного 3D поля. Он проходит по всем "кубикам" сетки и
    определяет, как поверхность пересекает каждый кубик.
    
    Алгоритм:
        1. Создаём 3D сетку точек
        2. Для каждой точки вычисляем occupancy (вероятность "внутри")
        3. Применяем Marching Cubes с заданным порогом
        4. Получаем треугольный меш
    
    Args:
        occupancy_fn: Функция (points: Tensor[N, 3]) → occupancy: Tensor[N]
                     Возвращает вероятности "внутри объекта" для каждой точки
        resolution: Разрешение сетки (N точек по каждой оси)
                   Всего N³ точек. 128³ ≈ 2 миллиона точек.
        threshold: Порог для изоповерхности (обычно 0.5)
                  Точки с occupancy > threshold считаются "внутри"
        bounds: Границы пространства (min, max)
        batch_size: Размер батча для предсказания (ограничение GPU памяти)
        device: Устройство для вычислений ('cuda' или 'cpu')
        verbose: Показывать прогресс
    
    Returns:
        trimesh.Trimesh: извлечённый меш
        None: если извлечение не удалось
    
    Пример:
        # Определяем функцию occupancy
        def occupancy_fn(points):
            points = points.unsqueeze(0)  # [1, N, 3]
            logits = model.decode(latent, points)
            return torch.sigmoid(logits).squeeze(0)  # [N]
        
        mesh = extract_mesh_marching_cubes(occupancy_fn, resolution=128)
    """
    
    min_bound, max_bound = bounds
    
    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 1: Создание сетки точек
    # ─────────────────────────────────────────────────────────────────────────
    
    if verbose:
        print(f"[mesh_utils.py] Creating {resolution}³ grid...")
    
    grid_points = create_grid_points(resolution, bounds)
    total_points = len(grid_points)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 2: Вычисление occupancy батчами
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Все точки не влезут в GPU память, поэтому обрабатываем батчами.
    # Типичный batch_size = 100K для 16GB GPU.
    # ─────────────────────────────────────────────────────────────────────────
    
    occupancy_values = []
    
    iterator = range(0, total_points, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Computing occupancy", leave=False)
    
    with torch.no_grad():
        for i in iterator:
            # Берём батч точек
            batch_points = torch.from_numpy(
                grid_points[i:i + batch_size]
            ).float().to(device)
            
            # Предсказываем occupancy
            batch_occ = occupancy_fn(batch_points)
            
            # Сохраняем на CPU
            occupancy_values.append(batch_occ.cpu().numpy())
    
    # Объединяем и преобразуем в 3D volume
    occupancy_grid = np.concatenate(occupancy_values).reshape(
        resolution, resolution, resolution
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 2.5: Применяем Gaussian фильтр для уменьшения шума
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Сглаживание occupancy grid перед Marching Cubes уменьшает шум
    # и артефакты на поверхности меша.
    # ─────────────────────────────────────────────────────────────────────────
    
    try:
        from scipy.ndimage import gaussian_filter, median_filter
        
        # Сначала median filter для удаления выбросов (шумных точек)
        occupancy_grid = median_filter(occupancy_grid, size=3)
        
        # Затем Gaussian filter для мягкого сглаживания
        occupancy_grid = gaussian_filter(occupancy_grid, sigma=0.5)
        
        if verbose:
            print(f"[mesh_utils.py] Applied Median + Gaussian filters to reduce noise")
    except ImportError:
        if verbose:
            print(f"[mesh_utils.py] scipy not available, skipping filters")
    except Exception as e:
        if verbose:
            print(f"[mesh_utils.py] Warning: Filter failed: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 3: Проверка occupancy
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Если почти все точки снаружи (< 0.1%) или внутри (> 99.9%),
    # Marching Cubes не сможет извлечь поверхность.
    # Пробуем адаптивный порог.
    # ─────────────────────────────────────────────────────────────────────────
    
    occ_ratio = (occupancy_grid > threshold).mean()
    
    if verbose:
        print(f"[mesh_utils.py] Occupancy ratio: {occ_ratio:.2%}")
    
    if occ_ratio < 0.001:
        # Почти пустой объём — пробуем понизить порог
        if verbose:
            print("[mesh_utils.py] ⚠️ Very low occupancy, trying adaptive threshold")
        adaptive_threshold = np.percentile(occupancy_grid, 99)
        threshold = max(adaptive_threshold, 0.1)
        if verbose:
            print(f"[mesh_utils.py] Adaptive threshold: {threshold:.3f}")
    
    elif occ_ratio > 0.999:
        # Почти полный объём — пробуем повысить порог
        if verbose:
            print("[mesh_utils.py] ⚠️ Very high occupancy, trying adaptive threshold")
        adaptive_threshold = np.percentile(occupancy_grid, 1)
        threshold = min(adaptive_threshold, 0.9)
        if verbose:
            print(f"[mesh_utils.py] Adaptive threshold: {threshold:.3f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 4: Marching Cubes
    # ─────────────────────────────────────────────────────────────────────────
    #
    # scikit-image реализация marching_cubes:
    #   - level: значение изоповерхности (наш threshold)
    #   - spacing: размер вокселя в мировых координатах
    #
    # Возвращает:
    #   - vertices: координаты вершин
    #   - faces: индексы вершин для каждого треугольника
    #   - normals: нормали вершин
    #   - values: интерполированные значения (не используем)
    # ─────────────────────────────────────────────────────────────────────────
    
    if verbose:
        print(f"[mesh_utils.py] Running Marching Cubes (threshold={threshold:.3f})...")
    
    try:
        # Размер вокселя
        spacing = (max_bound - min_bound) / resolution
        
        vertices, faces, normals, _ = measure.marching_cubes(
            occupancy_grid,
            level=threshold,
            spacing=(spacing, spacing, spacing)
        )
        
        # Смещаем вершины в правильную систему координат
        # marching_cubes возвращает координаты в [0, size], нам нужно [min, max]
        vertices = vertices + min_bound
        
        # Создаём trimesh объект
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals
        )
        
        if verbose:
            print(f"[mesh_utils.py] Mesh extracted: {len(mesh.vertices)} vertices, "
                  f"{len(mesh.faces)} faces")
        
        # ─────────────────────────────────────────────────────────────────────────
        # Шаг 5: Базовая постобработка (удаление мелких компонентов)
        # ─────────────────────────────────────────────────────────────────────────
        
        try:
            # Исправление геометрии перед удалением компонентов
            # (используем встроенную функцию repair_mesh напрямую)
            try:
                mesh.merge_vertices()
                mesh.remove_degenerate_faces()
                mesh.remove_duplicate_faces()
            except:
                pass
            
            components = mesh.split(only_watertight=False)
            
            if len(components) > 1:
                # Фильтруем компоненты по размеру
                valid_components = [
                    c for c in components 
                    if len(c.faces) >= 100
                ]
                
                if valid_components:
                    # Берём самый большой компонент
                    mesh = max(valid_components, key=lambda x: len(x.vertices))
                    if verbose:
                        removed = len(components) - len(valid_components)
                        print(f"[mesh_utils.py] Removed {removed} small components: "
                              f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        except Exception:
            pass  # Если не удалось, продолжаем с исходным мешем
        
        return mesh
        
    except Exception as e:
        if verbose:
            print(f"[mesh_utils.py] Marching Cubes error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# СЭМПЛИРОВАНИЕ ТОЧЕК
# ═══════════════════════════════════════════════════════════════════════════════

def sample_points_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: int = 10000
) -> np.ndarray:
    """
    Сэмплирование точек на поверхности меша.
    
    Использует взвешенное сэмплирование по площади треугольников:
    большие грани получают пропорционально больше точек.
    
    Args:
        mesh: Входной меш
        num_points: Количество точек для сэмплирования
    
    Returns:
        np.ndarray [num_points, 3]: координаты точек на поверхности
    
    Пример:
        mesh = load_mesh('chair.obj')
        points = sample_points_from_mesh(mesh, num_points=10000)
        
        # Визуализация (требует open3d или matplotlib)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        plt.show()
    """
    
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.array(points, dtype=np.float32)


def sample_points_with_normals(
    mesh: trimesh.Trimesh,
    num_points: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Сэмплирование точек на поверхности с нормалями.
    
    Args:
        mesh: Входной меш
        num_points: Количество точек
    
    Returns:
        Tuple[points, normals]:
            - points: np.ndarray [num_points, 3]
            - normals: np.ndarray [num_points, 3]
    
    Пример:
        points, normals = sample_points_with_normals(mesh, 10000)
        
        # Точка немного внутри объекта
        inside_point = points[0] - normals[0] * 0.01
        
        # Точка немного снаружи объекта
        outside_point = points[0] + normals[0] * 0.01
    """
    
    # Сэмплируем точки и получаем индексы граней
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    points = np.array(points, dtype=np.float32)
    
    # Получаем нормали граней для каждой точки
    normals = mesh.face_normals[face_indices].astype(np.float32)
    
    return points, normals


def mesh_to_pointcloud(
    mesh: trimesh.Trimesh,
    num_points: int = 10000
) -> np.ndarray:
    """
    Конвертация меша в облако точек.
    
    Синоним для sample_points_from_mesh, для совместимости.
    
    Args:
        mesh: Входной меш
        num_points: Количество точек
    
    Returns:
        np.ndarray [num_points, 3]: облако точек
    """
    
    return sample_points_from_mesh(mesh, num_points)


# ═══════════════════════════════════════════════════════════════════════════════
# УПРОЩЕНИЕ И ОБРАБОТКА МЕШЕЙ
# ═══════════════════════════════════════════════════════════════════════════════

def simplify_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 10000
) -> trimesh.Trimesh:
    """
    Упрощение меша через Quadric Decimation.
    
    Quadric Decimation — алгоритм уменьшения количества полигонов
    с минимальной потерей визуального качества. Он итеративно
    объединяет рёбра, выбирая те, удаление которых меньше всего
    изменит форму объекта.
    
    Args:
        mesh: Входной меш
        target_faces: Целевое количество граней
    
    Returns:
        Упрощённый меш
    
    Пример:
        mesh = load_mesh('high_poly.obj')
        print(f"Before: {len(mesh.faces)} faces")  # 500000
        
        simplified = simplify_mesh(mesh, target_faces=10000)
        print(f"After: {len(simplified.faces)} faces")  # ~10000
    """
    
    # Если меш уже достаточно простой, возвращаем как есть
    if len(mesh.faces) <= target_faces:
        return mesh
    
    try:
        simplified = mesh.simplify_quadric_decimation(target_faces)
        print(f"[mesh_utils.py] Simplified: {len(mesh.faces)} → {len(simplified.faces)} faces")
        return simplified
    except Exception as e:
        print(f"[mesh_utils.py] Simplification failed: {e}")
        return mesh


def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int = 3,
    lamb: float = 0.5
) -> trimesh.Trimesh:
    """
    Сглаживание меша (Laplacian Smoothing).
    
    Laplacian Smoothing перемещает каждую вершину к среднему
    положению её соседей. Это уменьшает шум и острые углы,
    но может "сдувать" объект при большом количестве итераций.
    
    Args:
        mesh: Входной меш
        iterations: Количество итераций сглаживания
        lamb: Коэффициент сглаживания (0-1)
              0 = без изменений, 1 = сильное сглаживание
    
    Returns:
        Сглаженный меш
    
    Пример:
        noisy_mesh = load_mesh('noisy.obj')
        smooth = smooth_mesh(noisy_mesh, iterations=5)
    """
    
    try:
        smoothed = trimesh.smoothing.filter_laplacian(
            mesh,
            iterations=iterations,
            lamb=lamb
        )
        return smoothed
    except Exception:
        return mesh


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Попытка починить меш.
    
    Выполняет:
        1. Удаление дублирующихся вершин
        2. Удаление вырожденных граней (нулевой площади)
        3. Удаление дублирующихся граней
        4. Заполнение дыр (если возможно)
    
    Args:
        mesh: Входной меш
    
    Returns:
        Починенный меш
    """
    
    try:
        # Удаляем дубликаты вершин
        mesh.merge_vertices()
        
        # Удаляем вырожденные грани
        mesh.remove_degenerate_faces()
        
        # Удаляем дубликаты граней
        mesh.remove_duplicate_faces()
        
        # Заполняем дыры
        trimesh.repair.fill_holes(mesh)
        
        return mesh
    except Exception:
        return mesh


def remove_small_components(
    mesh: trimesh.Trimesh,
    min_faces: int = 50
) -> trimesh.Trimesh:
    """
    Удаление мелких несвязанных компонентов.
    
    После Marching Cubes могут появиться мелкие "островки" (шум).
    Эта функция оставляет только крупные компоненты.
    
    Args:
        mesh: Входной меш
        min_faces: Минимальное количество граней для сохранения компонента
    
    Returns:
        Меш без мелких компонентов
    
    Пример:
        mesh = extract_mesh_marching_cubes(...)
        clean = remove_small_components(mesh, min_faces=100)
    """
    
    try:
        components = mesh.split(only_watertight=False)
        
        if len(components) <= 1:
            return mesh
        
        # Фильтруем по количеству граней
        large_components = [c for c in components if len(c.faces) >= min_faces]
        
        if len(large_components) == 0:
            # Если все компоненты маленькие, оставляем самый большой
            return max(components, key=lambda x: len(x.faces))
        
        # Объединяем крупные компоненты
        if len(large_components) == 1:
            return large_components[0]
        else:
            return trimesh.util.concatenate(large_components)
            
    except Exception:
        return mesh


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Оставить только самый большой связный компонент.
    
    Args:
        mesh: Входной меш
    
    Returns:
        Самый большой компонент
    """
    
    try:
        components = mesh.split(only_watertight=False)
        
        if len(components) <= 1:
            return mesh
        
        largest = max(components, key=lambda x: len(x.vertices))
        
        print(f"[mesh_utils.py] Kept largest of {len(components)} components")
        
        return largest
        
    except Exception:
        return mesh


# ═══════════════════════════════════════════════════════════════════════════════
# АНАЛИЗ МЕША
# ═══════════════════════════════════════════════════════════════════════════════

def get_mesh_stats(mesh: trimesh.Trimesh) -> dict:
    """
    Получение статистики меша.
    
    Args:
        mesh: Входной меш
    
    Returns:
        Dict со статистикой:
            - vertices: количество вершин
            - faces: количество граней
            - is_watertight: замкнутый ли меш
            - volume: объём (если watertight)
            - surface_area: площадь поверхности
            - bounds: границы (min, max)
    """
    
    stats = {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'is_watertight': mesh.is_watertight,
        'surface_area': mesh.area,
        'bounds_min': mesh.vertices.min(axis=0).tolist(),
        'bounds_max': mesh.vertices.max(axis=0).tolist(),
    }
    
    # Объём только для watertight мешей
    if mesh.is_watertight:
        try:
            stats['volume'] = mesh.volume
        except Exception:
            stats['volume'] = None
    else:
        stats['volume'] = None
    
    return stats


def print_mesh_stats(mesh: trimesh.Trimesh) -> None:
    """
    Вывод статистики меша в консоль.
    
    Args:
        mesh: Входной меш
    """
    
    stats = get_mesh_stats(mesh)
    
    print("\n" + "=" * 40)
    print("MESH STATISTICS")
    print("=" * 40)
    print(f"Vertices:     {stats['vertices']:,}")
    print(f"Faces:        {stats['faces']:,}")
    print(f"Watertight:   {stats['is_watertight']}")
    print(f"Surface area: {stats['surface_area']:.4f}")
    if stats['volume'] is not None:
        print(f"Volume:       {stats['volume']:.4f}")
    print(f"Bounds min:   {stats['bounds_min']}")
    print(f"Bounds max:   {stats['bounds_max']}")
    print("=" * 40)


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Тест утилит при запуске как скрипта:
        python mesh_utils.py
    """
    
    print("=" * 60)
    print("MESH UTILS TEST")
    print("=" * 60)
    
    # Тест создания сетки точек
    print("\n[Test 1] Create grid points")
    points = create_grid_points(32)
    print(f"  Grid shape: {points.shape}")  # (32768, 3)
    print(f"  Min: {points.min():.2f}, Max: {points.max():.2f}")
    
    # Тест создания простого меша (куб)
    print("\n[Test 2] Create and analyze cube mesh")
    cube = trimesh.primitives.Box()
    cube = normalize_mesh(cube)
    print_mesh_stats(cube)
    
    # Тест сэмплирования
    print("\n[Test 3] Sample points from mesh")
    points = sample_points_from_mesh(cube, num_points=1000)
    print(f"  Sampled points shape: {points.shape}")
    print(f"  Points range: [{points.min():.3f}, {points.max():.3f}]")
    
    # Тест сэмплирования с нормалями
    print("\n[Test 4] Sample points with normals")
    points, normals = sample_points_with_normals(cube, num_points=1000)
    print(f"  Points shape: {points.shape}")
    print(f"  Normals shape: {normals.shape}")
    print(f"  Normal lengths: {np.linalg.norm(normals, axis=1).mean():.3f}")  # ~1.0
    
    # Тест упрощения
    print("\n[Test 5] Simplify mesh")
    sphere = trimesh.primitives.Sphere(subdivisions=4)  # Высокополигональная сфера
    print(f"  Before: {len(sphere.faces)} faces")
    simplified = simplify_mesh(sphere, target_faces=500)
    print(f"  After: {len(simplified.faces)} faces")
    
    # Тест сохранения
    print("\n[Test 6] Save and load mesh")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        temp_path = f.name
    
    save_mesh(cube, temp_path)
    loaded = load_mesh(temp_path, normalize=False)
    print(f"  Loaded mesh: {len(loaded.vertices)} vertices, {len(loaded.faces)} faces")
    
    # Удаляем временный файл
    import os
    os.remove(temp_path)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)