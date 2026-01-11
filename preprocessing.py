"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".
Описание: Предварительная обработка мешей и генерация occupancy данных.
Дата: 2026
================================================================================
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import pickle
import hashlib

warnings.filterwarnings('ignore')

import trimesh
from PIL import Image


@dataclass
class PreprocessedSample:
    """Предобработанный образец."""
    sample_id: str
    img_path: str
    mask_path: Optional[str]
    category: str
    
    # Нормализованные данные меша
    vertices: np.ndarray  # [V, 3]
    faces: np.ndarray     # [F, 3]
    
    # Предвычисленные точки
    surface_points: np.ndarray   # [N, 3]
    surface_normals: np.ndarray  # [N, 3]
    
    # Bounding box
    bbox_min: np.ndarray  # [3]
    bbox_max: np.ndarray  # [3]
    
    # Флаги
    is_watertight: bool
    
    def save(self, path: str):
        """Сохранение в файл."""
        np.savez_compressed(
            path,
            sample_id=self.sample_id,
            img_path=self.img_path,
            mask_path=self.mask_path if self.mask_path else '',
            category=self.category,
            vertices=self.vertices,
            faces=self.faces,
            surface_points=self.surface_points,
            surface_normals=self.surface_normals,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            is_watertight=self.is_watertight
        )
    
    @staticmethod
    def load(path: str) -> 'PreprocessedSample':
        """Загрузка из файла."""
        data = np.load(path, allow_pickle=True)
        return PreprocessedSample(
            sample_id=str(data['sample_id']),
            img_path=str(data['img_path']),
            mask_path=str(data['mask_path']) if data['mask_path'] != '' else None,
            category=str(data['category']),
            vertices=data['vertices'],
            faces=data['faces'],
            surface_points=data['surface_points'],
            surface_normals=data['surface_normals'],
            bbox_min=data['bbox_min'],
            bbox_max=data['bbox_max'],
            is_watertight=bool(data['is_watertight'])
        )


class MeshProcessor:
    """Обработчик мешей."""
    
    def __init__(
        self,
        num_surface_samples: int = 100000,
        normalize_scale: float = 0.45
    ):
        self.num_surface_samples = num_surface_samples
        self.normalize_scale = normalize_scale
    
    def load_and_normalize(self, mesh_path: str) -> Optional[trimesh.Trimesh]:
        """Загрузка и нормализация меша."""
        try:
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            
            # Обработка Scene
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    return None
                meshes = list(mesh.geometry.values())
                mesh = trimesh.util.concatenate(meshes)
            
            # Проверки
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
                vertices = vertices / max_dist * self.normalize_scale
            
            mesh.vertices = vertices
            
            return mesh
            
        except Exception as e:
            return None
    
    def sample_surface(
        self, 
        mesh: trimesh.Trimesh
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сэмплирование точек на поверхности."""
        try:
            points, face_indices = trimesh.sample.sample_surface(
                mesh, self.num_surface_samples
            )
            points = np.array(points, dtype=np.float32)
            
            # Получаем нормали
            normals = mesh.face_normals[face_indices].astype(np.float32)
            
            return points, normals
        except:
            # Fallback
            indices = np.random.choice(
                len(mesh.vertices), 
                min(self.num_surface_samples, len(mesh.vertices)), 
                replace=True
            )
            points = mesh.vertices[indices].astype(np.float32)
            normals = np.zeros_like(points)
            
            return points, normals
    
    def check_watertight(self, mesh: trimesh.Trimesh) -> bool:
        """Проверка на watertight."""
        try:
            return mesh.is_watertight
        except:
            return False
    
    def process(self, mesh_path: str) -> Optional[Dict[str, Any]]:
        """Полная обработка меша."""
        mesh = self.load_and_normalize(mesh_path)
        if mesh is None:
            return None
        
        surface_points, surface_normals = self.sample_surface(mesh)
        is_watertight = self.check_watertight(mesh)
        
        return {
            'vertices': mesh.vertices.astype(np.float32),
            'faces': mesh.faces.astype(np.int32),
            'surface_points': surface_points,
            'surface_normals': surface_normals,
            'bbox_min': mesh.vertices.min(axis=0).astype(np.float32),
            'bbox_max': mesh.vertices.max(axis=0).astype(np.float32),
            'is_watertight': is_watertight
        }


def _process_single_sample(args) -> Optional[Tuple[str, PreprocessedSample]]:
    """Обработка одного образца (для multiprocessing)."""
    idx, item, root_dir, processor_kwargs = args
    
    try:
        img_path = os.path.join(root_dir, item['img'])
        model_path = os.path.join(root_dir, item['model'])
        mask_path = os.path.join(root_dir, item['mask']) if 'mask' in item else None
        
        if not os.path.exists(img_path) or not os.path.exists(model_path):
            return None
        
        # Создаём процессор
        processor = MeshProcessor(**processor_kwargs)
        result = processor.process(model_path)
        
        if result is None:
            return None
        
        # Создаём ID
        sample_id = hashlib.md5(f"{item['img']}_{item['model']}".encode()).hexdigest()[:16]
        
        sample = PreprocessedSample(
            sample_id=sample_id,
            img_path=img_path,
            mask_path=mask_path,
            category=item.get('category', 'unknown'),
            **result
        )
        
        return (sample_id, sample)
        
    except Exception as e:
        return None


class DatasetPreprocessor:
    """Препроцессор датасета PIX3D."""
    
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        output_dir: str,
        num_surface_samples: int = 100000,
        num_workers: int = 8,
        category_filter: Optional[str] = None
    ):
        self.root_dir = root_dir
        self.json_path = json_path
        self.output_dir = output_dir
        self.num_surface_samples = num_surface_samples
        self.num_workers = num_workers
        self.category_filter = category_filter
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_json(self) -> List[Dict]:
        """Загрузка JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Фильтрация
        if self.category_filter:
            data = [d for d in data if d.get('category') == self.category_filter]
        
        return data
    
    def preprocess(self, force: bool = False) -> Dict[str, str]:
        """
        Основной метод препроцессинга.
        
        Returns:
            Словарь {sample_id: path_to_preprocessed}
        """
        index_path = os.path.join(self.output_dir, 'index.json')
        
        # Проверяем существующий индекс
        if not force and os.path.exists(index_path):
            print(f"[preprocessing.py] Загружаю существующий индекс: {index_path}")
            with open(index_path, 'r') as f:
                return json.load(f)
        
        print(f"[preprocessing.py] Начинаю препроцессинг...")
        
        data = self._load_json()
        print(f"[preprocessing.py] Всего образцов: {len(data)}")
        
        processor_kwargs = {
            'num_surface_samples': self.num_surface_samples
        }
        
        # Подготовка аргументов
        args_list = [
            (i, item, self.root_dir, processor_kwargs)
            for i, item in enumerate(data)
        ]
        
        # Обработка
        index = {}
        successful = 0
        failed = 0
        
        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(_process_single_sample, args): args[0]
                    for args in args_list
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                    result = future.result()
                    if result is not None:
                        sample_id, sample = result
                        save_path = os.path.join(self.output_dir, f"{sample_id}.npz")
                        sample.save(save_path)
                        index[sample_id] = save_path
                        successful += 1
                    else:
                        failed += 1
        else:
            for args in tqdm(args_list, desc="Preprocessing"):
                result = _process_single_sample(args)
                if result is not None:
                    sample_id, sample = result
                    save_path = os.path.join(self.output_dir, f"{sample_id}.npz")
                    sample.save(save_path)
                    index[sample_id] = save_path
                    successful += 1
                else:
                    failed += 1
        
        # Сохраняем индекс
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"[preprocessing.py] Готово! Успешно: {successful}, Ошибок: {failed}")
        print(f"[preprocessing.py] Индекс сохранён: {index_path}")
        
        return index
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики датасета."""
        index_path = os.path.join(self.output_dir, 'index.json')
        
        if not os.path.exists(index_path):
            return {'error': 'Индекс не найден'}
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        categories = {}
        watertight_count = 0
        total_vertices = 0
        total_faces = 0
        
        for sample_id, path in tqdm(index.items(), desc="Computing stats"):
            try:
                sample = PreprocessedSample.load(path)
                
                cat = sample.category
                categories[cat] = categories.get(cat, 0) + 1
                
                if sample.is_watertight:
                    watertight_count += 1
                
                total_vertices += len(sample.vertices)
                total_faces += len(sample.faces)
                
            except:
                continue
        
        n = len(index)
        return {
            'total_samples': n,
            'categories': categories,
            'watertight_ratio': watertight_count / n if n > 0 else 0,
            'avg_vertices': total_vertices / n if n > 0 else 0,
            'avg_faces': total_faces / n if n > 0 else 0
        }


def run_preprocessing(
    root_dir: str = './PIX3D_DATA',
    json_path: str = './PIX3D_DATA/pix3d.json',
    output_dir: str = './cache/preprocessed',
    num_workers: int = 8,
    category: Optional[str] = None,
    force: bool = False
):
    """Запуск препроцессинга из командной строки."""
    
    preprocessor = DatasetPreprocessor(
        root_dir=root_dir,
        json_path=json_path,
        output_dir=output_dir,
        num_workers=num_workers,
        category_filter=category
    )
    
    index = preprocessor.preprocess(force=force)
    
    print("\n" + "="*50)
    stats = preprocessor.get_statistics()
    print("Статистика датасета:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    return index


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess PIX3D dataset')
    parser.add_argument('--root', type=str, default='./PIX3D_DATA')
    parser.add_argument('--output', type=str, default='./cache/preprocessed')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    
    args = parser.parse_args()
    
    run_preprocessing(
        root_dir=args.root,
        json_path=os.path.join(args.root, 'pix3d.json'),
        output_dir=args.output,
        num_workers=args.workers,
        category=args.category,
        force=args.force
    )