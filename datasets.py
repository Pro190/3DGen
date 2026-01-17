"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Датасеты для обучения Occupancy Network
Дата: 2025
================================================================================

Модуль содержит два датасета:

1. Pix3DDataset (оригинальный):
   - Загружает данные напрямую из PIX3D
   - Медленнее, но не требует препроцессинга
   - Подходит для небольших экспериментов

2. PreprocessedPix3DDataset (новый):
   - Использует препроцессированные .npz файлы
   - Быстрее в 10-20 раз
   - Требует предварительного запуска preprocessing.py

Структура PIX3D:
    PIX3D_DATA/
    ├── pix3d.json          # Аннотации
    ├── img/                # Изображения мебели
    ├── model/              # 3D модели (.obj)
    └── mask/               # Маски сегментации

Структура препроцессированных данных:
    cache/preprocessed/
    ├── index.json          # Индекс: sample_id → path
    └── *.npz               # Препроцессированные образцы
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Dict, Optional, List, Any, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import trimesh
import warnings

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# ТРАНСФОРМАЦИИ ИЗОБРАЖЕНИЙ
# ═══════════════════════════════════════════════════════════════════════════════

def get_image_transform(is_train: bool = True) -> transforms.Compose:
    """
    Создание трансформаций для изображений.
    
    Args:
        is_train: Режим обучения (включает аугментации)
    
    Returns:
        transforms.Compose: композиция трансформаций
    """
    transform_list = []
    
    if is_train:
        # Аугментация цвета только при обучении
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
        )
    
    transform_list.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    
    return transforms.Compose(transform_list)


# ═══════════════════════════════════════════════════════════════════════════════
# ОРИГИНАЛЬНЫЙ ДАТАСЕТ (без препроцессинга)
# ═══════════════════════════════════════════════════════════════════════════════

class Pix3DDataset(Dataset):
    """
    Датасет PIX3D для обучения Occupancy Network.
    
    Загружает данные напрямую из PIX3D без предварительной обработки.
    Медленнее, но не требует запуска preprocessing.py.
    
    Алгоритм сэмплирования точек:
        1. Сэмплируем точки на поверхности меша
        2. Получаем нормали в этих точках
        3. Сдвигаем часть точек ВНУТРЬ (против нормали) → occupancy = 1
        4. Сдвигаем часть точек НАРУЖУ (по нормали) → occupancy = 0
        5. Добавляем точки глубоко внутри/снаружи bbox
    
    Args:
        root_dir: Путь к папке PIX3D_DATA
        json_path: Путь к файлу pix3d.json
        num_points: Количество точек для сэмплирования
        is_train: Режим обучения (включает аугментации)
        category: Фильтр по категории или None для всех
    """
    
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_points: int = 4096,
        is_train: bool = True,
        category: Optional[str] = None
    ):
        self.root_dir = root_dir
        self.num_points = num_points
        self.is_train = is_train
        
        # Создание трансформаций
        self.transform = get_image_transform(is_train)
        
        # Загрузка и фильтрация аннотаций
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        skipped = 0
        
        for item in data:
            # Фильтрация по категории
            if category and item.get('category') != category:
                continue
            
            # Проверка существования файлов
            img_path = os.path.join(root_dir, item['img'])
            model_path = os.path.join(root_dir, item['model'])
            
            if not os.path.exists(img_path) or not os.path.exists(model_path):
                skipped += 1
                continue
            
            # Путь к маске (опционально)
            mask_path = None
            if 'mask' in item:
                mask_path = os.path.join(root_dir, item['mask'])
                if not os.path.exists(mask_path):
                    mask_path = None
            
            self.samples.append({
                'img': img_path,
                'model': model_path,
                'mask': mask_path,
                'category': item.get('category', 'unknown')
            })
        
        # Логирование
        mode = "train" if is_train else "val"
        cat_str = f" ({category})" if category else ""
        print(f"[datasets.py] Pix3DDataset: {len(self.samples)} samples{cat_str} [{mode}]")
        if skipped > 0:
            print(f"[datasets.py] Пропущено {skipped} образцов (файлы не найдены)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_mesh(self, path: str) -> Optional[trimesh.Trimesh]:
        """Загрузка и нормализация 3D меша."""
        try:
            mesh = trimesh.load(path, force='mesh', process=False)
            
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    return None
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            
            if len(mesh.vertices) < 100 or len(mesh.faces) < 50:
                return None
            
            # Нормализация в [-0.45, 0.45]
            vertices = mesh.vertices.copy()
            center = vertices.mean(axis=0)
            vertices -= center
            
            scale = np.abs(vertices).max()
            if scale > 1e-6:
                vertices = vertices / scale * 0.45
            
            mesh.vertices = vertices
            return mesh
            
        except Exception:
            return None
    
    def _sample_points(
        self,
        mesh: trimesh.Trimesh
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Сэмплирование точек с использованием нормалей поверхности.
        
        Гарантирует правильные метки occupancy даже для non-watertight мешей.
        """
        n_surface = self.num_points // 2
        n_space = self.num_points - n_surface
        
        try:
            # Сэмплируем точки на поверхности
            surface_pts, face_idx = trimesh.sample.sample_surface(mesh, n_surface * 2)
            surface_pts = np.array(surface_pts, dtype=np.float32)
            normals = mesh.face_normals[face_idx].astype(np.float32)
            
        except Exception:
            return None, None
        
        # Нормализуем нормали
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals = normals / norm_len
        
        points_list = []
        occ_list = []
        
        # Точки ВНУТРИ (сдвиг против нормали)
        n_in = n_surface // 2
        offset = np.random.uniform(0.008, 0.025, (n_in, 1)).astype(np.float32)
        inside_pts = surface_pts[:n_in] - normals[:n_in] * offset
        points_list.append(inside_pts)
        occ_list.append(np.ones(n_in, dtype=np.float32))
        
        # Точки СНАРУЖИ (сдвиг по нормали)
        n_out = n_surface - n_in
        offset = np.random.uniform(0.008, 0.025, (n_out, 1)).astype(np.float32)
        outside_pts = surface_pts[n_in:n_in + n_out] + normals[n_in:n_in + n_out] * offset
        points_list.append(outside_pts)
        occ_list.append(np.zeros(n_out, dtype=np.float32))
        
        # Точки глубоко внутри bbox
        bbox_min = mesh.vertices.min(axis=0)
        bbox_max = mesh.vertices.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min
        
        n_deep = n_space // 3
        deep_pts = np.random.uniform(
            bbox_center - bbox_size * 0.15,
            bbox_center + bbox_size * 0.15,
            (n_deep, 3)
        ).astype(np.float32)
        points_list.append(deep_pts)
        occ_list.append(np.ones(n_deep, dtype=np.float32))
        
        # Точки далеко снаружи bbox
        n_far = n_space // 3
        far_pts = []
        for _ in range(10):
            pts = np.random.uniform(-0.5, 0.5, (n_far * 3, 3))
            dist = np.abs(pts - bbox_center) - bbox_size / 2
            is_far = np.any(dist > 0.08, axis=1)
            far_pts.extend(pts[is_far].tolist())
            if len(far_pts) >= n_far:
                break
        
        far_pts = np.array(far_pts[:n_far], dtype=np.float32)
        points_list.append(far_pts)
        occ_list.append(np.zeros(len(far_pts), dtype=np.float32))
        
        # Точки около поверхности
        n_near = n_space - n_deep - len(far_pts)
        idx = np.random.choice(len(surface_pts), n_near, replace=True)
        near_pts = surface_pts[idx].copy()
        near_norms = normals[idx]
        tiny_offset = (np.random.rand(n_near, 1) - 0.5) * 0.01
        near_pts = (near_pts + near_norms * tiny_offset).astype(np.float32)
        near_occ = (tiny_offset.flatten() < 0).astype(np.float32)
        points_list.append(near_pts)
        occ_list.append(near_occ)
        
        # Объединение и перемешивание
        points = np.concatenate(points_list, axis=0)
        occupancies = np.concatenate(occ_list, axis=0)
        
        perm = np.random.permutation(len(points))
        points = points[perm][:self.num_points]
        occupancies = occupancies[perm][:self.num_points]
        
        return points, occupancies
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Получение одного образца."""
        sample = self.samples[idx]
        
        try:
            # Загрузка изображения
            img = Image.open(sample['img']).convert('RGB')
            
            # Применение маски (если есть)
            if sample['mask'] and os.path.exists(sample['mask']):
                try:
                    mask = Image.open(sample['mask']).convert('L')
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    img = Image.composite(img, bg, mask)
                except Exception:
                    pass
            
            img_tensor = self.transform(img)
            
            # Загрузка и обработка 3D модели
            mesh = self._load_mesh(sample['model'])
            if mesh is None:
                return None
            
            # Сэмплирование точек
            points, occupancies = self._sample_points(mesh)
            if points is None:
                return None
            
            return {
                'image': img_tensor,
                'points': torch.from_numpy(points),
                'occupancies': torch.from_numpy(occupancies),
                'category': sample['category']
            }
            
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# ПРЕПРОЦЕССИРОВАННЫЙ ДАТАСЕТ (быстрый)
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocessedPix3DDataset(Dataset):
    """
    Быстрый датасет, использующий препроцессированные данные.
    
    Требует предварительного запуска:
        python preprocessing.py
    
    Преимущества:
        - Загрузка в 10-20 раз быстрее
        - Точки уже сэмплированы
        - Нормали предвычислены
    
    Args:
        preprocessed_dir: Папка с препроцессированными данными
        num_points: Количество точек для сэмплирования
        is_train: Режим обучения
        category: Фильтр по категории
    """
    
    def __init__(
        self,
        preprocessed_dir: str,
        num_points: int = 4096,
        is_train: bool = True,
        category: Optional[str] = None
    ):
        self.preprocessed_dir = preprocessed_dir
        self.num_points = num_points
        self.is_train = is_train
        
        # Создание трансформаций
        self.transform = get_image_transform(is_train)
        
        # Загрузка индекса
        index_path = os.path.join(preprocessed_dir, 'index.json')
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Индекс не найден: {index_path}\n"
                f"Запустите: python preprocessing.py"
            )
        
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        
        # Фильтрация по категории и валидация файлов
        self.samples = []
        for sample_id, path in self.index.items():
            if not os.path.exists(path):
                continue
            
            # Быстрая проверка категории из имени файла или загрузка
            if category:
                try:
                    data = np.load(path, allow_pickle=True)
                    if str(data['category']) != category:
                        continue
                except Exception:
                    continue
            
            self.samples.append((sample_id, path))
        
        mode = "train" if is_train else "val"
        cat_str = f" ({category})" if category else ""
        print(f"[datasets.py] PreprocessedPix3DDataset: {len(self.samples)} samples{cat_str} [{mode}]")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _sample_from_preprocessed(
        self,
        surface_points: np.ndarray,
        surface_normals: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сэмплирование точек из препроцессированных данных.
        
        Использует предвычисленные точки на поверхности и нормали.
        """
        n_surface = self.num_points // 2
        n_space = self.num_points - n_surface
        
        # Случайный выбор точек поверхности
        n_available = len(surface_points)
        indices = np.random.choice(n_available, min(n_surface * 2, n_available), replace=True)
        
        pts = surface_points[indices].astype(np.float32)
        norms = surface_normals[indices].astype(np.float32)
        
        # Нормализуем нормали
        norm_len = np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8
        norms = norms / norm_len
        
        points_list = []
        occ_list = []
        
        # Точки внутри (против нормали)
        n_in = n_surface // 2
        offset = np.random.uniform(0.008, 0.025, (n_in, 1)).astype(np.float32)
        inside_pts = pts[:n_in] - norms[:n_in] * offset
        points_list.append(inside_pts)
        occ_list.append(np.ones(n_in, dtype=np.float32))
        
        # Точки снаружи (по нормали)
        n_out = n_surface - n_in
        offset = np.random.uniform(0.008, 0.025, (n_out, 1)).astype(np.float32)
        outside_pts = pts[n_in:n_in + n_out] + norms[n_in:n_in + n_out] * offset
        points_list.append(outside_pts)
        occ_list.append(np.zeros(n_out, dtype=np.float32))
        
        # Точки в пространстве
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min
        
        # Глубоко внутри
        n_deep = n_space // 3
        deep_pts = np.random.uniform(
            bbox_center - bbox_size * 0.15,
            bbox_center + bbox_size * 0.15,
            (n_deep, 3)
        ).astype(np.float32)
        points_list.append(deep_pts)
        occ_list.append(np.ones(n_deep, dtype=np.float32))
        
        # Далеко снаружи
        n_far = n_space // 3
        far_pts = np.random.uniform(-0.5, 0.5, (n_far, 3)).astype(np.float32)
        points_list.append(far_pts)
        occ_list.append(np.zeros(n_far, dtype=np.float32))
        
        # Около поверхности
        n_near = n_space - n_deep - n_far
        idx = np.random.choice(len(pts), n_near, replace=True)
        near_pts = pts[idx].copy()
        near_norms = norms[idx]
        tiny_offset = (np.random.rand(n_near, 1) - 0.5) * 0.01
        near_pts = (near_pts + near_norms * tiny_offset).astype(np.float32)
        near_occ = (tiny_offset.flatten() < 0).astype(np.float32)
        points_list.append(near_pts)
        occ_list.append(near_occ)
        
        # Объединение и перемешивание
        points = np.concatenate(points_list, axis=0)
        occupancies = np.concatenate(occ_list, axis=0)
        
        perm = np.random.permutation(len(points))
        points = points[perm][:self.num_points]
        occupancies = occupancies[perm][:self.num_points]
        
        return points, occupancies
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Получение одного образца."""
        sample_id, path = self.samples[idx]
        
        try:
            # Загрузка препроцессированных данных
            data = np.load(path, allow_pickle=True)
            
            # Загрузка изображения
            img_path = str(data['img_path'])
            img = Image.open(img_path).convert('RGB')
            
            # Применение маски (если есть)
            mask_path = str(data['mask_path'])
            if mask_path and os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('L')
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    img = Image.composite(img, bg, mask)
                except Exception:
                    pass
            
            img_tensor = self.transform(img)
            
            # Сэмплирование точек из препроцессированных данных
            points, occupancies = self._sample_from_preprocessed(
                data['surface_points'],
                data['surface_normals'],
                data['bbox_min'],
                data['bbox_max']
            )
            
            return {
                'image': img_tensor,
                'points': torch.from_numpy(points),
                'occupancies': torch.from_numpy(occupancies),
                'category': str(data['category'])
            }
            
        except Exception as e:
            print(f"[datasets.py] Ошибка загрузки {sample_id}: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# COLLATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, Any]]:
    """
    Функция сборки батча с фильтрацией None.
    
    Args:
        batch: Список образцов (или None для невалидных)
    
    Returns:
        Dict с батчами тензоров или None если все образцы невалидны
    """
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'points': torch.stack([b['points'] for b in batch]),
        'occupancies': torch.stack([b['occupancies'] for b in batch]),
        'category': [b['category'] for b in batch]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataset(
    cfg,
    is_train: bool = True
) -> Dataset:
    """
    Создание датасета на основе конфигурации.
    
    Автоматически выбирает между обычным и препроцессированным датасетом.
    
    Args:
        cfg: Объект конфигурации
        is_train: Режим обучения
    
    Returns:
        Dataset: Pix3DDataset или PreprocessedPix3DDataset
    """
    if cfg.train.use_preprocessed:
        # Проверяем наличие препроцессированных данных
        if os.path.exists(cfg.paths.preprocessed_index):
            print("[datasets.py] Использую препроцессированные данные")
            return PreprocessedPix3DDataset(
                preprocessed_dir=cfg.paths.preprocessed_dir,
                num_points=cfg.train.num_points,
                is_train=is_train,
                category=cfg.train.category_filter
            )
        else:
            print("[datasets.py] ⚠️ Препроцессированные данные не найдены, использую обычный датасет")
    
    return Pix3DDataset(
        root_dir=cfg.paths.data_root,
        json_path=cfg.paths.json_path,
        num_points=cfg.train.num_points,
        is_train=is_train,
        category=cfg.train.category_filter
    )


def create_data_loaders(
    cfg,
    train_dataset: Dataset,
    val_dataset: Dataset
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Создание DataLoader'ов для обучения и валидации.
    
    Args:
        cfg: Объект конфигурации
        train_dataset: Датасет для обучения
        val_dataset: Датасет для валидации
    
    Returns:
        Tuple[train_loader, val_loader]
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=True if cfg.train.num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=max(cfg.train.num_workers // 2, 1),
        collate_fn=collate_fn,
        pin_memory=cfg.train.pin_memory
    )
    
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """Тест датасетов."""
    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)
    
    # Тест обычного датасета
    dataset = Pix3DDataset(
        root_dir='./PIX3D_DATA',
        json_path='./PIX3D_DATA/pix3d.json',
        num_points=4096,
        is_train=True
    )
    
    print(f"\nВсего образцов: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        if sample is not None:
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Points shape: {sample['points'].shape}")
            print(f"  Occupancies shape: {sample['occupancies'].shape}")
            print(f"  Category: {sample['category']}")
            
            occ = sample['occupancies'].numpy()
            print(f"  Inside: {(occ == 1).sum()} ({(occ == 1).mean()*100:.1f}%)")
            print(f"  Outside: {(occ == 0).sum()} ({(occ == 0).mean()*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)