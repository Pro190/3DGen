"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Датасет PIX3D для обучения Occupancy Network
Дата: 2025
================================================================================

Ключевые особенности датасета:
    1. Загрузка изображений мебели из PIX3D
    2. Загрузка и нормализация 3D моделей (meshes)
    3. Сэмплирование точек с использованием НОРМАЛЕЙ поверхности
       (гарантирует правильные метки inside/outside)
    4. Применение масок для удаления фона

Структура PIX3D:
    PIX3D_DATA/
    ├── pix3d.json          # Аннотации (список всех образцов)
    ├── img/                # Изображения мебели
    │   ├── bed/
    │   ├── chair/
    │   └── ...
    ├── model/              # 3D модели (.obj)
    │   ├── bed/
    │   ├── chair/
    │   └── ...
    └── mask/               # Маски сегментации
        ├── bed/
        ├── chair/
        └── ...

Формат аннотации (pix3d.json):
    [
        {
            "img": "img/chair/0001.png",
            "model": "model/chair/model.obj",
            "mask": "mask/chair/0001.png",
            "category": "chair",
            ...
        },
        ...
    ]
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

# Подавляем предупреждения от trimesh и других библиотек
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ДАТАСЕТ
# ═══════════════════════════════════════════════════════════════════════════════

class Pix3DDataset(Dataset):
    """
    Датасет PIX3D для обучения Occupancy Network.
    
    Ключевая особенность: сэмплирование точек с использованием НОРМАЛЕЙ.
    Это гарантирует правильное разделение точек на inside/outside,
    даже для non-watertight мешей.
    
    Алгоритм сэмплирования:
        1. Сэмплируем точки на поверхности меша
        2. Получаем нормали в этих точках
        3. Сдвигаем часть точек ВНУТРЬ (против нормали) → occupancy = 1
        4. Сдвигаем часть точек НАРУЖУ (по нормали) → occupancy = 0
        5. Добавляем точки глубоко внутри bbox → occupancy = 1
        6. Добавляем точки далеко снаружи → occupancy = 0
    
    Args:
        root_dir: Путь к папке PIX3D_DATA
        json_path: Путь к файлу pix3d.json
        num_points: Количество точек для сэмплирования (по умолчанию 4096)
        is_train: Режим обучения (включает аугментации)
        category: Фильтр по категории ('chair', 'table', ...) или None для всех
    
    Пример использования:
        dataset = Pix3DDataset(
            root_dir='./PIX3D_DATA',
            json_path='./PIX3D_DATA/pix3d.json',
            num_points=4096,
            is_train=True,
            category='chair'
        )
        
        sample = dataset[0]
        # sample['image']      - torch.Tensor [3, 224, 224]
        # sample['points']     - torch.Tensor [4096, 3]
        # sample['occupancies'] - torch.Tensor [4096]
        # sample['category']   - str
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
        
        # ─────────────────────────────────────────────────────────────────────
        # Построение трансформаций изображения
        # ─────────────────────────────────────────────────────────────────────
        # 
        # Для обучения добавляем ColorJitter (аугментация цвета):
        #   - brightness: ±20% яркости
        #   - contrast: ±20% контраста  
        #   - saturation: ±20% насыщенности
        #   - hue: ±5% оттенка
        #
        # Для всех режимов:
        #   - Resize до 224x224 (вход ResNet)
        #   - ToTensor: [0, 255] → [0, 1]
        #   - Normalize: стандартизация ImageNet (mean, std)
        # ─────────────────────────────────────────────────────────────────────
        
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
        
        self.transform = transforms.Compose(transform_list)
        
        # ─────────────────────────────────────────────────────────────────────
        # Загрузка и фильтрация аннотаций
        # ─────────────────────────────────────────────────────────────────────
        
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
            
            if not os.path.exists(img_path):
                skipped += 1
                continue
            if not os.path.exists(model_path):
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
        print(f"[datasets.py] Loaded {len(self.samples)} samples{cat_str} [{mode}]")
        if skipped > 0:
            print(f"[datasets.py] Skipped {skipped} samples (missing files)")
    
    def __len__(self) -> int:
        """Количество образцов в датасете."""
        return len(self.samples)
    
    def _load_mesh(self, path: str) -> Optional[trimesh.Trimesh]:
        """
        Загрузка и нормализация 3D меша.
        
        Процесс:
            1. Загрузка .obj файла через trimesh
            2. Обработка Scene (если модель состоит из нескольких частей)
            3. Проверка валидности (минимум вершин и граней)
            4. Нормализация в диапазон [-0.45, 0.45]
        
        Args:
            path: Путь к .obj файлу
        
        Returns:
            trimesh.Trimesh или None если загрузка не удалась
        """
        try:
            # Загрузка без автоматической обработки
            mesh = trimesh.load(path, force='mesh', process=False)
            
            # Если загрузилась сцена (несколько объектов) - объединяем
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    return None
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            
            # Проверка минимального количества геометрии
            if len(mesh.vertices) < 100 or len(mesh.faces) < 50:
                return None
            
            # ─────────────────────────────────────────────────────────────────
            # Нормализация меша в единичный куб [-0.45, 0.45]
            # ─────────────────────────────────────────────────────────────────
            #
            # 1. Центрируем меш (перемещаем центроид в начало координат)
            # 2. Масштабируем так, чтобы максимальная координата = 0.45
            #
            # Почему 0.45, а не 0.5?
            #   Оставляем небольшой зазор для точек "снаружи" объекта
            #   при сэмплировании в диапазоне [-0.5, 0.5]
            # ─────────────────────────────────────────────────────────────────
            
            vertices = mesh.vertices.copy()
            
            # Центрирование
            center = vertices.mean(axis=0)
            vertices -= center
            
            # Масштабирование
            scale = np.abs(vertices).max()
            if scale > 1e-6:
                vertices = vertices / scale * 0.45
            
            mesh.vertices = vertices
            
            return mesh
            
        except Exception as e:
            # Молча возвращаем None при ошибке загрузки
            return None
    
    def _sample_points(
        self, 
        mesh: trimesh.Trimesh
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Сэмплирование точек с использованием НОРМАЛЕЙ поверхности.
        
        Это ключевой метод! Использование нормалей гарантирует правильные
        метки occupancy даже для non-watertight мешей.
        
        Стратегия сэмплирования (для num_points=4096):
            - 1024 точек: сдвинуты ВНУТРЬ по нормали → occ=1
            - 1024 точек: сдвинуты НАРУЖУ по нормали → occ=0
            - 682 точек: глубоко внутри bbox (центр) → occ=1
            - 682 точек: далеко снаружи bbox → occ=0
            - 684 точек: около поверхности (смешанные метки)
        
        Args:
            mesh: Нормализованный trimesh объект
        
        Returns:
            Tuple[points, occupancies] или (None, None) при ошибке
            points: np.ndarray [N, 3]
            occupancies: np.ndarray [N] (0 или 1)
        """
        
        n_surface = self.num_points // 2  # Половина - около поверхности
        n_space = self.num_points - n_surface  # Половина - в пространстве
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 1: Сэмплирование точек на поверхности и получение нормалей
        # ─────────────────────────────────────────────────────────────────────
        
        try:
            # Сэмплируем с запасом (x2) для надёжности
            surface_pts, face_idx = trimesh.sample.sample_surface(
                mesh, n_surface * 2
            )
            surface_pts = np.array(surface_pts, dtype=np.float32)
            
            # Получаем нормали граней для каждой точки
            normals = mesh.face_normals[face_idx].astype(np.float32)
            
        except Exception:
            return None, None
        
        # Нормализуем нормали (на случай если они не единичные)
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals = normals / norm_len
        
        # Списки для накопления точек и меток
        points_list = []
        occ_list = []
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 2: Точки ВНУТРИ объекта (сдвиг против нормали)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Нормаль направлена НАРУЖУ от поверхности.
        # Сдвиг ПРОТИВ нормали = движение ВНУТРЬ объекта.
        # Эти точки имеют occupancy = 1 (inside).
        #
        # Offset: случайное значение от 0.008 до 0.025
        # (8-25 мм в нормализованном пространстве)
        # ─────────────────────────────────────────────────────────────────────
        
        n_in = n_surface // 2
        offset = np.random.uniform(0.008, 0.025, (n_in, 1)).astype(np.float32)
        inside_pts = surface_pts[:n_in] - normals[:n_in] * offset
        
        points_list.append(inside_pts)
        occ_list.append(np.ones(n_in, dtype=np.float32))
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 3: Точки СНАРУЖИ объекта (сдвиг по нормали)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Сдвиг ПО нормали = движение НАРУЖУ объекта.
        # Эти точки имеют occupancy = 0 (outside).
        # ─────────────────────────────────────────────────────────────────────
        
        n_out = n_surface - n_in
        offset = np.random.uniform(0.008, 0.025, (n_out, 1)).astype(np.float32)
        outside_pts = surface_pts[n_in:n_in + n_out] + normals[n_in:n_in + n_out] * offset
        
        points_list.append(outside_pts)
        occ_list.append(np.zeros(n_out, dtype=np.float32))
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 4: Точки ГЛУБОКО ВНУТРИ (центр bounding box)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Эти точки находятся в центральной части объекта (±15% от центра).
        # Почти гарантированно внутри для выпуклых и большинства вогнутых объектов.
        # Occupancy = 1.
        # ─────────────────────────────────────────────────────────────────────
        
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
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 5: Точки ДАЛЕКО СНАРУЖИ (за пределами bounding box)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Генерируем точки в диапазоне [-0.5, 0.5] и отбираем только те,
        # которые находятся достаточно далеко от bounding box (> 0.08).
        # Гарантированно снаружи объекта. Occupancy = 0.
        # ─────────────────────────────────────────────────────────────────────
        
        n_far = n_space // 3
        far_pts = []
        
        for _ in range(10):  # Максимум 10 попыток
            pts = np.random.uniform(-0.5, 0.5, (n_far * 3, 3))
            
            # Вычисляем расстояние до bbox
            dist = np.abs(pts - bbox_center) - bbox_size / 2
            is_far = np.any(dist > 0.08, axis=1)  # Хотя бы по одной оси далеко
            
            far_pts.extend(pts[is_far].tolist())
            if len(far_pts) >= n_far:
                break
        
        far_pts = np.array(far_pts[:n_far], dtype=np.float32)
        points_list.append(far_pts)
        occ_list.append(np.zeros(len(far_pts), dtype=np.float32))
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 6: Точки ОКОЛО ПОВЕРХНОСТИ (смешанные метки)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Эти точки находятся очень близко к поверхности (±0.5 см).
        # Метка зависит от направления сдвига:
        #   - сдвиг < 0 (внутрь) → occupancy = 1
        #   - сдвиг > 0 (наружу) → occupancy = 0
        #
        # Эти "сложные" примеры помогают модели лучше определять границу.
        # ─────────────────────────────────────────────────────────────────────
        
        n_near = n_space - n_deep - len(far_pts)
        idx = np.random.choice(len(surface_pts), n_near, replace=True)
        near_pts = surface_pts[idx].copy()
        near_norms = normals[idx]
        
        # Tiny offset: от -0.005 до +0.005 (равномерно)
        tiny_offset = (np.random.rand(n_near, 1) - 0.5) * 0.01
        near_pts = (near_pts + near_norms * tiny_offset).astype(np.float32)
        
        # Если сдвиг отрицательный (внутрь) → occupancy = 1
        near_occ = (tiny_offset.flatten() < 0).astype(np.float32)
        
        points_list.append(near_pts)
        occ_list.append(near_occ)
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 7: Объединение и перемешивание
        # ─────────────────────────────────────────────────────────────────────
        
        points = np.concatenate(points_list, axis=0)
        occupancies = np.concatenate(occ_list, axis=0)
        
        # Случайная перестановка
        perm = np.random.permutation(len(points))
        points = points[perm][:self.num_points]
        occupancies = occupancies[perm][:self.num_points]
        
        return points, occupancies
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Получение одного образца по индексу.
        
        Args:
            idx: Индекс образца
        
        Returns:
            Dict с ключами:
                - 'image': torch.Tensor [3, 224, 224]
                - 'points': torch.Tensor [num_points, 3]
                - 'occupancies': torch.Tensor [num_points]
                - 'category': str
            
            Или None если образец невалидный
        """
        sample = self.samples[idx]
        
        try:
            # ─────────────────────────────────────────────────────────────────
            # Загрузка и обработка изображения
            # ─────────────────────────────────────────────────────────────────
            
            img = Image.open(sample['img']).convert('RGB')
            
            # Применение маски (если есть)
            # Маска удаляет фон, заменяя его на белый цвет
            if sample['mask'] and os.path.exists(sample['mask']):
                try:
                    mask = Image.open(sample['mask']).convert('L')  # Grayscale
                    bg = Image.new('RGB', img.size, (255, 255, 255))  # Белый фон
                    img = Image.composite(img, bg, mask)
                except Exception:
                    pass  # Если маска не загрузилась - используем оригинал
            
            # Применение трансформаций
            img_tensor = self.transform(img)
            
            # ─────────────────────────────────────────────────────────────────
            # Загрузка и обработка 3D модели
            # ─────────────────────────────────────────────────────────────────
            
            mesh = self._load_mesh(sample['model'])
            if mesh is None:
                return None
            
            # ─────────────────────────────────────────────────────────────────
            # Сэмплирование точек
            # ─────────────────────────────────────────────────────────────────
            
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
            # При любой ошибке возвращаем None
            # collate_fn отфильтрует такие образцы
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# COLLATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, Any]]:
    """
    Функция сборки батча с фильтрацией None.
    
    DataLoader вызывает эту функцию для объединения отдельных образцов
    в батч. Мы фильтруем None (невалидные образцы) перед объединением.
    
    Args:
        batch: Список образцов (или None для невалидных)
    
    Returns:
        Dict с батчами тензоров или None если все образцы невалидны
    
    Пример:
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        for batch in loader:
            if batch is None:
                continue
            images = batch['image']  # [32, 3, 224, 224]
            points = batch['points']  # [32, 4096, 3]
    """
    # Фильтруем None
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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

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
        drop_last=True,  # Отбрасываем неполный последний батч
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
    """
    Тест датасета при запуске как скрипта:
        python datasets.py
    """
    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)
    
    # Создание датасета
    dataset = Pix3DDataset(
        root_dir='./PIX3D_DATA',
        json_path='./PIX3D_DATA/pix3d.json',
        num_points=4096,
        is_train=True,
        category=None  # Все категории
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Тест загрузки одного образца
    print("\n[Test] Loading sample 0...")
    sample = dataset[0]
    
    if sample is not None:
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Points shape: {sample['points'].shape}")
        print(f"  Occupancies shape: {sample['occupancies'].shape}")
        print(f"  Category: {sample['category']}")
        
        # Статистика occupancy
        occ = sample['occupancies'].numpy()
        print(f"\n  Occupancy stats:")
        print(f"    Inside (1):  {(occ == 1).sum()} ({(occ == 1).mean()*100:.1f}%)")
        print(f"    Outside (0): {(occ == 0).sum()} ({(occ == 0).mean()*100:.1f}%)")
        
        # Статистика координат
        pts = sample['points'].numpy()
        print(f"\n  Points stats:")
        print(f"    Min: [{pts.min(axis=0)}]")
        print(f"    Max: [{pts.max(axis=0)}]")
    else:
        print("  Failed to load sample!")
    
    # Тест DataLoader
    print("\n[Test] Creating DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(loader))
    if batch is not None:
        print(f"  Batch image shape: {batch['image'].shape}")
        print(f"  Batch points shape: {batch['points'].shape}")
        print(f"  Batch occupancies shape: {batch['occupancies'].shape}")
    else:
        print("  Empty batch!")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)