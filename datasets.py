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

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import trimesh

from preprocessing import PreprocessedSample


class OccupancySampler:
    """
    Улучшенный сэмплер точек для обучения Occupancy Network.
    """
    
    def __init__(
        self,
        num_points_surface: int = 2048,
        num_points_uniform: int = 2048,
        surface_noise_std: float = 0.02,
        use_normals: bool = True
    ):
        self.num_points_surface = num_points_surface
        self.num_points_uniform = num_points_uniform
        self.surface_noise_std = surface_noise_std
        self.use_normals = use_normals
    
    def sample_from_preprocessed(
        self,
        sample: PreprocessedSample,
        augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сэмплирование из препроцессированных данных.
        """
        points_list = []
        occupancies_list = []
        
        surface_points = sample.surface_points
        surface_normals = sample.surface_normals
        bbox_min = sample.bbox_min
        bbox_max = sample.bbox_max
        
        n_surface = len(surface_points)
        
        # ═══════════════════════════════════════════════════════════
        # 1. Точки на поверхности с шумом (50% внутрь, 50% наружу)
        # ═══════════════════════════════════════════════════════════
        indices = np.random.choice(n_surface, self.num_points_surface, replace=True)
        sampled_surface = surface_points[indices].copy()
        sampled_normals = surface_normals[indices].copy()
        
        # Шум вдоль нормали (более точный, чем случайный)
        noise_scale = np.random.randn(self.num_points_surface, 1) * self.surface_noise_std
        
        if self.use_normals and np.abs(sampled_normals).sum() > 0:
            noise = noise_scale * sampled_normals
        else:
            noise = np.random.randn(self.num_points_surface, 3) * self.surface_noise_std
        
        noisy_surface = sampled_surface + noise.astype(np.float32)
        
        # Половина внутрь (occupancy=1), половина наружу (occupancy=0)
        n_half = self.num_points_surface // 2
        
        # Внутренние точки (сдвиг против нормали)
        inner_points = sampled_surface[:n_half] - np.abs(noise[:n_half])
        points_list.append(inner_points.astype(np.float32))
        occupancies_list.append(np.ones(n_half, dtype=np.float32))
        
        # Внешние точки (сдвиг по нормали)
        outer_points = sampled_surface[n_half:] + np.abs(noise[n_half:])
        points_list.append(outer_points.astype(np.float32))
        occupancies_list.append(np.zeros(self.num_points_surface - n_half, dtype=np.float32))
        
        # ═══════════════════════════════════════════════════════════
        # 2. Точки внутри bbox (высокая вероятность = 1)
        # ═══════════════════════════════════════════════════════════
        n_inside = self.num_points_uniform // 2
        
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min
        
        # Точки в уменьшенном bbox (70% размера)
        inner_scale = 0.35
        inner_points = np.random.uniform(
            bbox_center - bbox_size * inner_scale,
            bbox_center + bbox_size * inner_scale,
            (n_inside, 3)
        ).astype(np.float32)
        
        points_list.append(inner_points)
        occupancies_list.append(np.ones(n_inside, dtype=np.float32))
        
        # ═══════════════════════════════════════════════════════════
        # 3. Точки снаружи (далеко от объекта) = 0
        # ═══════════════════════════════════════════════════════════
        n_outside = self.num_points_uniform - n_inside
        
        outer_points = np.random.uniform(-0.55, 0.55, (n_outside * 3, 3)).astype(np.float32)
        
        # Фильтруем точки далеко от bbox
        distances_to_center = np.linalg.norm(outer_points - bbox_center, axis=1)
        max_bbox_dist = np.linalg.norm(bbox_size) / 2
        
        far_mask = distances_to_center > max_bbox_dist * 1.3
        far_points = outer_points[far_mask][:n_outside]
        
        if len(far_points) < n_outside:
            extra = np.random.uniform(-0.55, 0.55, (n_outside - len(far_points), 3))
            far_points = np.vstack([far_points, extra]) if len(far_points) > 0 else extra
        
        points_list.append(far_points.astype(np.float32))
        occupancies_list.append(np.zeros(len(far_points), dtype=np.float32))
        
        # ═══════════════════════════════════════════════════════════
        # 4. Точки на границе (смесь 0 и 1)
        # ═══════════════════════════════════════════════════════════
        n_boundary = self.num_points_uniform // 4
        
        boundary_indices = np.random.choice(n_surface, n_boundary, replace=True)
        boundary_points = surface_points[boundary_indices].copy()
        
        # Маленький случайный шум
        tiny_noise = np.random.randn(n_boundary, 3) * 0.01
        boundary_points = (boundary_points + tiny_noise).astype(np.float32)
        
        # Случайные метки (примерно 50/50)
        boundary_occ = (np.random.rand(n_boundary) > 0.5).astype(np.float32)
        
        points_list.append(boundary_points)
        occupancies_list.append(boundary_occ)
        
        # ═══════════════════════════════════════════════════════════
        # Объединяем и перемешиваем
        # ═══════════════════════════════════════════════════════════
        points = np.concatenate(points_list, axis=0)
        occupancies = np.concatenate(occupancies_list, axis=0)
        
        # Перемешиваем
        perm = np.random.permutation(len(points))
        points = points[perm]
        occupancies = occupancies[perm]
        
        # Аугментация
        if augment:
            points, occupancies = self._augment(points, occupancies)
        
        # Ограничиваем количество точек
        total_points = self.num_points_surface + self.num_points_uniform
        points = points[:total_points]
        occupancies = occupancies[:total_points]
        
        return points, occupancies
    
    def sample_from_mesh(
        self,
        mesh: trimesh.Trimesh,
        augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сэмплирование напрямую из меша (для обратной совместимости).
        """
        # Сэмплируем поверхность
        try:
            surface_points, face_idx = trimesh.sample.sample_surface(
                mesh, max(10000, self.num_points_surface * 2)
            )
            surface_points = np.array(surface_points, dtype=np.float32)
            surface_normals = mesh.face_normals[face_idx].astype(np.float32)
        except:
            indices = np.random.choice(len(mesh.vertices), 10000, replace=True)
            surface_points = mesh.vertices[indices].astype(np.float32)
            surface_normals = np.zeros_like(surface_points)
        
        # Создаём временный PreprocessedSample
        temp_sample = PreprocessedSample(
            sample_id='temp',
            img_path='',
            mask_path=None,
            category='',
            vertices=mesh.vertices.astype(np.float32),
            faces=mesh.faces.astype(np.int32),
            surface_points=surface_points,
            surface_normals=surface_normals,
            bbox_min=mesh.vertices.min(axis=0).astype(np.float32),
            bbox_max=mesh.vertices.max(axis=0).astype(np.float32),
            is_watertight=False
        )
        
        return self.sample_from_preprocessed(temp_sample, augment)
    
    def _augment(
        self, 
        points: np.ndarray, 
        occupancies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Аугментация точек."""
        # Случайное масштабирование
        scale = np.random.uniform(0.9, 1.1)
        points = points * scale
        
        # Случайное вращение вокруг Y (вертикальной оси)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=np.float32)
            points = points @ rotation.T
        
        return points, occupancies


class Pix3DPreprocessedDataset(Dataset):
    """
    Датасет на основе препроцессированных данных.
    """
    
    def __init__(
        self,
        index_path: str,
        img_size: int = 224,
        is_train: bool = True,
        num_points_surface: int = 2048,
        num_points_uniform: int = 2048,
        surface_noise: float = 0.02,
        use_augmentation: bool = True
    ):
        self.img_size = img_size
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train
        
        # Загрузка индекса
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        
        self.sample_ids = list(self.index.keys())
        
        # Сэмплер
        self.sampler = OccupancySampler(
            num_points_surface=num_points_surface,
            num_points_uniform=num_points_uniform,
            surface_noise_std=surface_noise
        )
        
        # Трансформации
        self.transform = self._build_transforms(is_train, img_size)
        
        mode = "train" if is_train else "val"
        print(f"[datasets.py] Preprocessed Dataset: {mode}, образцов: {len(self.sample_ids)}")
    
    def _build_transforms(self, is_train: bool, img_size: int) -> transforms.Compose:
        transform_list = []
        
        if is_train:
            transform_list.append(
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
            )
            transform_list.append(
                transforms.RandomHorizontalFlip(p=0.5)
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
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample_id = self.sample_ids[idx]
        sample_path = self.index[sample_id]
        
        try:
            # Загрузка препроцессированных данных
            sample = PreprocessedSample.load(sample_path)
            
            # Загрузка изображения
            img = Image.open(sample.img_path).convert('RGB')
            
            # Применение маски
            if sample.mask_path and os.path.exists(sample.mask_path):
                try:
                    mask = Image.open(sample.mask_path).convert('L')
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    img = Image.composite(img, background, mask)
                except:
                    pass
            
            img_tensor = self.transform(img)
            
            # Сэмплирование точек
            points, occupancies = self.sampler.sample_from_preprocessed(
                sample, augment=self.use_augmentation
            )
            
            return {
                'image': img_tensor,
                'points': torch.from_numpy(points).float(),
                'occupancies': torch.from_numpy(occupancies).float(),
                'category': sample.category,
                'sample_id': sample_id
            }
            
        except Exception as e:
            return None


class Pix3DOccupancyDataset(Dataset):
    """
    Оригинальный датасет PIX3D (без препроцессинга).
    Для обратной совместимости.
    """
    
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        img_size: int = 224,
        is_train: bool = True,
        num_points_surface: int = 2048,
        num_points_uniform: int = 2048,
        surface_noise: float = 0.02,
        category_filter: Optional[str] = None,
        use_augmentation: bool = True
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train
        
        # Трансформации
        self.transform = self._build_transforms(is_train, img_size)
        
        # Сэмплер
        self.sampler = OccupancySampler(
            num_points_surface=num_points_surface,
            num_points_uniform=num_points_uniform,
            surface_noise_std=surface_noise
        )
        
        # Загрузка данных
        self.samples = self._load_samples(json_path, category_filter)
        
        mode = "train" if is_train else "val"
        print(f"[datasets.py] Original Dataset: {mode}, образцов: {len(self.samples)}")

    def _build_transforms(self, is_train: bool, img_size: int) -> transforms.Compose:
        transform_list = []
        
        if is_train:
            transform_list.append(
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
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
        print(f"[datasets.py] Загружаю {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        
        for item in data:
            if category_filter and item.get('category') != category_filter:
                continue
            
            if 'img' not in item or 'model' not in item:
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
        
        return samples

    def _load_mesh(self, model_path: str) -> Optional[trimesh.Trimesh]:
        try:
            mesh = trimesh.load(model_path, force='mesh', process=False)
            
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    return None
                meshes = list(mesh.geometry.values())
                mesh = trimesh.util.concatenate(meshes)
            
            if mesh.vertices is None or len(mesh.vertices) < 10:
                return None
            if mesh.faces is None or len(mesh.faces) < 10:
                return None
            
            # Нормализация
            vertices = np.array(mesh.vertices, dtype=np.float32)
            centroid = vertices.mean(axis=0)
            vertices -= centroid
            
            max_dist = np.abs(vertices).max()
            if max_dist > 1e-6:
                vertices = vertices / max_dist * 0.45
            
            mesh.vertices = vertices
            
            return mesh
            
        except:
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = self.samples[idx]
        
        try:
            # Изображение
            img = Image.open(sample['img']).convert('RGB')
            
            if sample['mask'] and os.path.exists(sample['mask']):
                try:
                    mask = Image.open(sample['mask']).convert('L')
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    img = Image.composite(img, background, mask)
                except:
                    pass
            
            img_tensor = self.transform(img)
            
            # Меш
            mesh = self._load_mesh(sample['model'])
            if mesh is None:
                return None
            
            # Сэмплирование
            points, occupancies = self.sampler.sample_from_mesh(
                mesh, augment=self.use_augmentation
            )
            
            return {
                'image': img_tensor,
                'points': torch.from_numpy(points).float(),
                'occupancies': torch.from_numpy(occupancies).float(),
                'category': sample['category']
            }
            
        except:
            return None


def create_datasets(
    root_dir: str,
    json_path: str,
    preprocessed_index: Optional[str] = None,
    val_split: float = 0.1,
    seed: int = 42,
    category_filter: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[Dataset, Dataset]:
    """
    Создание train/val датасетов.
    
    Если preprocessed_index указан, использует препроцессированные данные.
    """
    
    if preprocessed_index and os.path.exists(preprocessed_index):
        print(f"[datasets.py] Использую препроцессированные данные: {preprocessed_index}")
        
        train_dataset = Pix3DPreprocessedDataset(
            preprocessed_index,
            is_train=True,
            **dataset_kwargs
        )
        
        val_dataset = Pix3DPreprocessedDataset(
            preprocessed_index,
            is_train=False,
            **dataset_kwargs
        )
    else:
        print(f"[datasets.py] Использую оригинальный датасет")
        
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
    
    # Разделение
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


# Для обратной совместимости
def create_train_val_split(*args, **kwargs):
    return create_datasets(*args, **kwargs)


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, Any]]:
    """Функция сборки батча с фильтрацией None."""
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    result = {
        'image': torch.stack([b['image'] for b in batch]),
        'points': torch.stack([b['points'] for b in batch]),
        'occupancies': torch.stack([b['occupancies'] for b in batch]),
        'category': [b['category'] for b in batch]
    }
    
    if 'sample_id' in batch[0]:
        result['sample_id'] = [b['sample_id'] for b in batch]
    
    return result