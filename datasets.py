"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".   
Описание: Логика загрузки и предобработки данных; подготовка изображений и соответствующих им 3D-моделей.
Дата: 2026
================================================================================
"""
import os
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import warnings

# Подавляем предупреждения
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import trimesh


class Pix3DOccupancyDataset(Dataset):
    """
    Датасет PIX3D для обучения Occupancy Networks.
    Работает с любыми мешами (включая non-watertight).
    """
    
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        img_size: int = 224,
        is_train: bool = True,
        num_points_surface: int = 1024,
        num_points_uniform: int = 1024,
        surface_noise: float = 0.05,
        category_filter: Optional[str] = None
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        self.num_points_surface = num_points_surface
        self.num_points_uniform = num_points_uniform
        self.surface_noise = surface_noise
        
        # Трансформации
        self.transform = self._build_transforms(is_train, img_size)
        
        # Загрузка данных
        self.samples = self._load_samples(json_path, category_filter)
        
        mode = "train" if is_train else "val"
        print(f"[datasets.py] Режим: {mode}, образцов: {len(self.samples)}")

    def _build_transforms(self, is_train: bool, img_size: int) -> transforms.Compose:
        """Создание трансформаций."""
        transform_list = []
        
        if is_train:
            transform_list.append(
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            )
        
        transform_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transform_list)

    def _load_samples(
        self, 
        json_path: str, 
        category_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Загрузка и фильтрация образцов."""
        print(f"[datasets.py] Загружаю {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        skipped = 0
        
        for item in data:
            # Фильтр по категории
            if category_filter and item.get('category') != category_filter:
                continue
            
            if 'img' not in item or 'model' not in item:
                skipped += 1
                continue
            
            img_path = os.path.join(self.root_dir, item['img'])
            model_path = os.path.join(self.root_dir, item['model'])
            mask_path = os.path.join(self.root_dir, item['mask']) if 'mask' in item else None
            
            if os.path.exists(img_path) and os.path.exists(model_path):
                samples.append({
                    'img': img_path,
                    'mask': mask_path,
                    'model': model_path,
                    'category': item.get('category', 'unknown')
                })
            else:
                skipped += 1
        
        if category_filter:
            print(f"[datasets.py] Категория '{category_filter}': {len(samples)} образцов")
        else:
            print(f"[datasets.py] Всего: {len(samples)}, пропущено: {skipped}")
        
        return samples

    def _load_mesh(self, model_path: str) -> Optional[trimesh.Trimesh]:
        """Безопасная загрузка меша."""
        try:
            mesh = trimesh.load(model_path, force='mesh', process=False)
            
            # Обработка Scene
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    return None
                # Объединяем все геометрии
                meshes = list(mesh.geometry.values())
                mesh = trimesh.util.concatenate(meshes)
            
            # Проверка на пустой меш
            if mesh.vertices is None or len(mesh.vertices) < 3:
                return None
            
            if mesh.faces is None or len(mesh.faces) < 1:
                return None
            
            # Нормализация в [-0.5, 0.5]
            vertices = np.array(mesh.vertices, dtype=np.float32)
            centroid = vertices.mean(axis=0)
            vertices -= centroid
            
            max_dist = np.abs(vertices).max()
            if max_dist > 1e-6:
                vertices = vertices / max_dist * 0.45  # Немного меньше 0.5 для запаса
            
            mesh.vertices = vertices
            
            return mesh
            
        except Exception as e:
            return None

    def _sample_points(
        self, 
        mesh: trimesh.Trimesh
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сэмплирование точек для обучения.
        Работает с любыми мешами (включая non-watertight).
        """
        points_list = []
        occupancies_list = []
        
        vertices = mesh.vertices
        
        # ═══════════════════════════════════════════════════════════
        # 1. Точки НА поверхности (occupancy = 0.5, но используем как 1)
        # ═══════════════════════════════════════════════════════════
        try:
            surface_points, _ = trimesh.sample.sample_surface(
                mesh, 
                self.num_points_surface
            )
            surface_points = np.array(surface_points, dtype=np.float32)
        except:
            # Fallback: случайные точки на вершинах
            indices = np.random.choice(len(vertices), self.num_points_surface, replace=True)
            surface_points = vertices[indices].astype(np.float32)
        
        # Добавляем небольшой шум
        noise = np.random.randn(*surface_points.shape).astype(np.float32) * self.surface_noise
        noisy_surface = surface_points + noise
        
        # Половина точек внутри (с шумом внутрь), половина снаружи (с шумом наружу)
        # Для точек на поверхности определяем направление через нормали или случайно
        n_half = len(noisy_surface) // 2
        
        # Точки, сдвинутые "внутрь" = 1
        points_list.append(noisy_surface[:n_half])
        occupancies_list.append(np.ones(n_half, dtype=np.float32))
        
        # Точки, сдвинутые "наружу" = 0
        points_list.append(noisy_surface[n_half:])
        occupancies_list.append(np.zeros(len(noisy_surface) - n_half, dtype=np.float32))
        
        # ═══════════════════════════════════════════════════════════
        # 2. Точки ВНУТРИ bounding box объекта (высокая вероятность = 1)
        # ═══════════════════════════════════════════════════════════
        n_inside = self.num_points_uniform // 2
        
        # Точки близко к центру объекта
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min
        
        # Точки в уменьшенном bbox (более вероятно внутри)
        inner_points = np.random.uniform(
            bbox_center - bbox_size * 0.3,
            bbox_center + bbox_size * 0.3,
            (n_inside, 3)
        ).astype(np.float32)
        
        points_list.append(inner_points)
        occupancies_list.append(np.ones(n_inside, dtype=np.float32))
        
        # ═══════════════════════════════════════════════════════════
        # 3. Точки СНАРУЖИ (далеко от объекта) = 0
        # ═══════════════════════════════════════════════════════════
        n_outside = self.num_points_uniform - n_inside
        
        # Точки в полном пространстве [-0.55, 0.55]
        outer_points = np.random.uniform(-0.55, 0.55, (n_outside * 2, 3)).astype(np.float32)
        
        # Фильтруем те, что далеко от объекта
        distances_to_center = np.linalg.norm(outer_points - bbox_center, axis=1)
        max_bbox_dist = np.linalg.norm(bbox_size) / 2
        
        # Берём точки, которые далеко от центра объекта
        far_mask = distances_to_center > max_bbox_dist * 1.2
        far_points = outer_points[far_mask][:n_outside]
        
        # Если не хватило, добираем случайные
        if len(far_points) < n_outside:
            extra = np.random.uniform(-0.55, 0.55, (n_outside - len(far_points), 3)).astype(np.float32)
            far_points = np.vstack([far_points, extra]) if len(far_points) > 0 else extra
        
        points_list.append(far_points)
        occupancies_list.append(np.zeros(len(far_points), dtype=np.float32))
        
        # ═══════════════════════════════════════════════════════════
        # Объединяем и перемешиваем
        # ═══════════════════════════════════════════════════════════
        points = np.concatenate(points_list, axis=0)
        occupancies = np.concatenate(occupancies_list, axis=0)
        
        # Перемешиваем
        perm = np.random.permutation(len(points))
        points = points[perm]
        occupancies = occupancies[perm]
        
        # Ограничиваем количество точек
        total_points = self.num_points_surface + self.num_points_uniform
        points = points[:total_points]
        occupancies = occupancies[:total_points]
        
        return points, occupancies

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Получение образца."""
        sample = self.samples[idx]
        
        try:
            # 1. Загрузка изображения
            img = Image.open(sample['img']).convert('RGB')
            
            # Применение маски
            if sample['mask'] and os.path.exists(sample['mask']):
                try:
                    mask = Image.open(sample['mask']).convert('L')
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    img = Image.composite(img, background, mask)
                except:
                    pass
            
            img_tensor = self.transform(img)
            
            # 2. Загрузка меша
            mesh = self._load_mesh(sample['model'])
            
            if mesh is None:
                return None
            
            # 3. Сэмплирование точек
            points, occupancies = self._sample_points(mesh)
            
            return {
                'image': img_tensor,
                'points': torch.from_numpy(points).float(),
                'occupancies': torch.from_numpy(occupancies).float(),
                'category': sample['category']
            }
            
        except Exception as e:
            return None


def create_train_val_split(
    root_dir: str,
    json_path: str,
    val_split: float = 0.1,
    seed: int = 42,
    category_filter: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[Dataset, Dataset]:
    """Создание train/val split."""
    
    train_dataset = Pix3DOccupancyDataset(
        root_dir, json_path,
        is_train=True,
        category_filter=category_filter,
        **dataset_kwargs
    )
    
    val_dataset = Pix3DOccupancyDataset(
        root_dir, json_path,
        is_train=False,
        category_filter=category_filter,
        **dataset_kwargs
    )
    
    num_samples = len(train_dataset)
    indices = list(range(num_samples))
    
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    val_size = int(num_samples * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"[datasets.py] Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    return train_subset, val_subset


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, Any]]:
    """Функция сборки батча с фильтрацией None."""
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'points': torch.stack([b['points'] for b in batch]),
        'occupancies': torch.stack([b['occupancies'] for b in batch]),
        'category': [b['category'] for b in batch]
    }