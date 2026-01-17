"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Препроцессинг датасета PIX3D для ускорения обучения
Дата: 2025
================================================================================

Зачем нужен препроцессинг?
    При обучении Occupancy Network для каждого образца нужно:
    1. Загрузить 3D модель (.obj файл)
    2. Нормализовать меш
    3. Сэмплировать точки на поверхности
    4. Вычислить нормали
    
    Эти операции занимают много времени.
    Препроцессинг выполняет их заранее и сохраняет результаты.

Что сохраняется:
    - Нормализованные вершины и грани меша
    - Точки на поверхности (100K точек)
    - Нормали в этих точках
    - Bounding box
    - Метаданные

Ускорение: загрузка данных в 10-20 раз быстрее.

Запуск:
    python preprocessing.py
    python preprocessing.py --category chair
    python preprocessing.py --workers 16 --force
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import hashlib

warnings.filterwarnings('ignore')

import trimesh


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASS ДЛЯ ПРЕПРОЦЕССИРОВАННЫХ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessedSample:
    """
    Препроцессированный образец датасета.
    
    Содержит все данные для обучения в предвычисленном виде.
    """
    
    sample_id: str
    img_path: str
    mask_path: Optional[str]
    category: str
    
    vertices: np.ndarray
    faces: np.ndarray
    
    surface_points: np.ndarray
    surface_normals: np.ndarray
    
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    
    is_watertight: bool
    
    def save(self, path: str) -> None:
        """Сохранение в .npz файл."""
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
        """Загрузка из .npz файла."""
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики образца."""
        return {
            'sample_id': self.sample_id,
            'category': self.category,
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces),
            'num_surface_points': len(self.surface_points),
            'is_watertight': self.is_watertight
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ОБРАБОТЧИК МЕШЕЙ
# ═══════════════════════════════════════════════════════════════════════════════

class MeshProcessor:
    """
    Класс для обработки 3D мешей.
    
    Выполняет:
        - Загрузку .obj файлов
        - Нормализацию
        - Сэмплирование точек на поверхности
        - Вычисление нормалей
    """
    
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
                vertices = vertices / max_dist * self.normalize_scale
            
            mesh.vertices = vertices
            
            return mesh
            
        except Exception:
            return None
    
    def sample_surface(
        self,
        mesh: trimesh.Trimesh
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сэмплирование точек на поверхности с нормалями."""
        
        try:
            points, face_indices = trimesh.sample.sample_surface(
                mesh,
                self.num_surface_samples
            )
            points = np.array(points, dtype=np.float32)
            normals = mesh.face_normals[face_indices].astype(np.float32)
            
            return points, normals
            
        except Exception:
            # Fallback
            num_samples = min(self.num_surface_samples, len(mesh.vertices))
            indices = np.random.choice(len(mesh.vertices), num_samples, replace=True)
            points = mesh.vertices[indices].astype(np.float32)
            normals = np.zeros_like(points)
            
            return points, normals
    
    def check_watertight(self, mesh: trimesh.Trimesh) -> bool:
        """Проверка, является ли меш замкнутым."""
        try:
            return mesh.is_watertight
        except Exception:
            return False
    
    def process(self, mesh_path: str) -> Optional[Dict[str, Any]]:
        """Полная обработка меша."""
        
        mesh = self.load_and_normalize(mesh_path)
        if mesh is None:
            return None
        
        surface_points, surface_normals = self.sample_surface(mesh)
        is_watertight = self.check_watertight(mesh)
        
        bbox_min = mesh.vertices.min(axis=0).astype(np.float32)
        bbox_max = mesh.vertices.max(axis=0).astype(np.float32)
        
        return {
            'vertices': mesh.vertices.astype(np.float32),
            'faces': mesh.faces.astype(np.int32),
            'surface_points': surface_points,
            'surface_normals': surface_normals,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'is_watertight': is_watertight
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИЯ ДЛЯ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ
# ═══════════════════════════════════════════════════════════════════════════════

def _process_single_sample(
    args: Tuple[int, Dict, str, Dict]
) -> Optional[Tuple[str, PreprocessedSample]]:
    """Обработка одного образца (для multiprocessing)."""
    
    idx, item, root_dir, processor_kwargs = args
    
    try:
        img_path = os.path.join(root_dir, item['img'])
        model_path = os.path.join(root_dir, item['model'])
        mask_path = os.path.join(root_dir, item['mask']) if 'mask' in item else None
        
        if not os.path.exists(img_path) or not os.path.exists(model_path):
            return None
        
        processor = MeshProcessor(**processor_kwargs)
        result = processor.process(model_path)
        
        if result is None:
            return None
        
        # Создание уникального ID
        id_string = f"{item['img']}_{item['model']}"
        sample_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
        
        sample = PreprocessedSample(
            sample_id=sample_id,
            img_path=img_path,
            mask_path=mask_path,
            category=item.get('category', 'unknown'),
            vertices=result['vertices'],
            faces=result['faces'],
            surface_points=result['surface_points'],
            surface_normals=result['surface_normals'],
            bbox_min=result['bbox_min'],
            bbox_max=result['bbox_max'],
            is_watertight=result['is_watertight']
        )
        
        return (sample_id, sample)
        
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# КЛАСС ПРЕПРОЦЕССОРА
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetPreprocessor:
    """
    Препроцессор датасета PIX3D.
    
    Выполняет параллельную обработку всех образцов
    и сохраняет результаты для быстрой загрузки.
    """
    
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
        
        print(f"[preprocessing.py] DatasetPreprocessor initialized")
        print(f"[preprocessing.py] Output: {output_dir}")
        print(f"[preprocessing.py] Workers: {num_workers}")
        if category_filter:
            print(f"[preprocessing.py] Category filter: {category_filter}")
    
    def _load_json(self) -> List[Dict]:
        """Загрузка и фильтрация аннотаций."""
        
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        if self.category_filter:
            data = [d for d in data if d.get('category') == self.category_filter]
            print(f"[preprocessing.py] Filtered: {len(data)} samples ({self.category_filter})")
        
        return data
    
    def preprocess(self, force: bool = False) -> Dict[str, str]:
        """
        Основной метод препроцессинга.
        
        Args:
            force: Перезаписать существующие данные
        
        Returns:
            Dict[sample_id → path]: индекс препроцессированных файлов
        """
        
        index_path = os.path.join(self.output_dir, 'index.json')
        
        # Проверка существующего индекса
        if not force and os.path.exists(index_path):
            print(f"[preprocessing.py] Loading existing index: {index_path}")
            with open(index_path, 'r') as f:
                return json.load(f)
        
        print(f"[preprocessing.py] Starting preprocessing...")
        
        data = self._load_json()
        print(f"[preprocessing.py] Total samples: {len(data)}")
        
        processor_kwargs = {
            'num_surface_samples': self.num_surface_samples
        }
        
        args_list = [
            (i, item, self.root_dir, processor_kwargs)
            for i, item in enumerate(data)
        ]
        
        index = {}
        successful = 0
        failed = 0
        
        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(_process_single_sample, args): args[0]
                    for args in args_list
                }
                
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Preprocessing"
                ):
                    result = future.result()
                    
                    if result is not None:
                        sample_id, sample = result
                        
                        save_path = os.path.join(
                            self.output_dir,
                            f"{sample_id}.npz"
                        )
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
                    
                    save_path = os.path.join(
                        self.output_dir,
                        f"{sample_id}.npz"
                    )
                    sample.save(save_path)
                    
                    index[sample_id] = save_path
                    successful += 1
                else:
                    failed += 1
        
        # Сохранение индекса
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"\n[preprocessing.py] Preprocessing complete!")
        print(f"[preprocessing.py] Successful: {successful}")
        print(f"[preprocessing.py] Failed: {failed}")
        print(f"[preprocessing.py] Index saved: {index_path}")
        
        return index
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики препроцессированного датасета."""
        
        index_path = os.path.join(self.output_dir, 'index.json')
        
        if not os.path.exists(index_path):
            return {'error': 'Index not found. Run preprocess() first.'}
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        categories = {}
        watertight_count = 0
        total_vertices = 0
        total_faces = 0
        total_surface_points = 0
        
        for sample_id, path in tqdm(index.items(), desc="Computing stats", leave=False):
            try:
                sample = PreprocessedSample.load(path)
                
                cat = sample.category
                categories[cat] = categories.get(cat, 0) + 1
                
                if sample.is_watertight:
                    watertight_count += 1
                
                total_vertices += len(sample.vertices)
                total_faces += len(sample.faces)
                total_surface_points += len(sample.surface_points)
                
            except Exception:
                continue
        
        n = len(index)
        
        return {
            'total_samples': n,
            'categories': categories,
            'watertight_ratio': watertight_count / n if n > 0 else 0,
            'avg_vertices': int(total_vertices / n) if n > 0 else 0,
            'avg_faces': int(total_faces / n) if n > 0 else 0,
            'avg_surface_points': int(total_surface_points / n) if n > 0 else 0
        }
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Проверка целостности препроцессированных данных."""
        
        index_path = os.path.join(self.output_dir, 'index.json')
        
        if not os.path.exists(index_path):
            return {'error': 'Index not found'}
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        valid = 0
        missing = 0
        corrupted = 0
        
        for sample_id, path in tqdm(index.items(), desc="Verifying", leave=False):
            if not os.path.exists(path):
                missing += 1
                continue
            
            try:
                sample = PreprocessedSample.load(path)
                
                if sample.surface_points is None or len(sample.surface_points) == 0:
                    corrupted += 1
                    continue
                
                valid += 1
                
            except Exception:
                corrupted += 1
        
        return {
            'total': len(index),
            'valid': valid,
            'missing': missing,
            'corrupted': corrupted,
            'integrity_ok': (missing == 0 and corrupted == 0)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИЯ ДЛЯ КОМАНДНОЙ СТРОКИ
# ═══════════════════════════════════════════════════════════════════════════════

def run_preprocessing(
    root_dir: str = './PIX3D_DATA',
    json_path: str = None,
    output_dir: str = './cache/preprocessed',
    num_workers: int = 8,
    category: Optional[str] = None,
    force: bool = False
) -> Dict[str, str]:
    """Запуск препроцессинга."""
    
    if json_path is None:
        json_path = os.path.join(root_dir, 'pix3d.json')
    
    print("=" * 60)
    print("PIX3D DATASET PREPROCESSING")
    print("=" * 60)
    print(f"Root dir: {root_dir}")
    print(f"JSON: {json_path}")
    print(f"Output: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"Category: {category or 'all'}")
    print(f"Force: {force}")
    print("=" * 60)
    
    preprocessor = DatasetPreprocessor(
        root_dir=root_dir,
        json_path=json_path,
        output_dir=output_dir,
        num_workers=num_workers,
        category_filter=category
    )
    
    index = preprocessor.preprocess(force=force)
    
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    stats = preprocessor.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 60)
    
    return index


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess PIX3D dataset for Occupancy Network training'
    )
    
    parser.add_argument('--root', type=str, default='./PIX3D_DATA',
                        help='Root directory of PIX3D dataset')
    parser.add_argument('--output', type=str, default='./cache/preprocessed',
                        help='Output directory for preprocessed data')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--category', type=str, default=None,
                        help='Process only this category')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing preprocessed data')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify existing preprocessed data')
    
    args = parser.parse_args()
    
    if args.verify:
        print("=" * 60)
        print("VERIFYING PREPROCESSED DATA")
        print("=" * 60)
        
        preprocessor = DatasetPreprocessor(
            root_dir=args.root,
            json_path=os.path.join(args.root, 'pix3d.json'),
            output_dir=args.output,
            num_workers=1
        )
        
        result = preprocessor.verify_integrity()
        
        for key, value in result.items():
            print(f"{key}: {value}")
        
        if result.get('integrity_ok'):
            print("\n✓ All data is valid!")
        else:
            print("\n✗ Some data is missing or corrupted!")
    else:
        run_preprocessing(
            root_dir=args.root,
            json_path=os.path.join(args.root, 'pix3d.json'),
            output_dir=args.output,
            num_workers=args.workers,
            category=args.category,
            force=args.force
        )