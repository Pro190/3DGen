"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Инференс - генерация 3D модели мебели из фотографии
Дата: 2025
================================================================================

Процесс генерации 3D модели (инференс):

    1. ЗАГРУЗКА ИЗОБРАЖЕНИЯ
       - Чтение изображения мебели
       - Маска ОПЦИОНАЛЬНА (по умолчанию не используется)
       - Resize до 224x224, нормализация ImageNet

    2. ENCODING
       - Пропускаем изображение через ResNet50
       - Получаем латентный вектор [1, 512]

    3. СОЗДАНИЕ 3D СЕТКИ
       - Создаём равномерную сетку точек в пространстве [-0.5, 0.5]³
       - Разрешение N×N×N (например, 128³ = 2,097,152 точек)

    4. ПРЕДСКАЗАНИЕ OCCUPANCY
       - Для каждой точки предсказываем вероятность "внутри объекта"
       - Батчевая обработка для эффективности

    5. MARCHING CUBES
       - Извлечение изоповерхности из 3D volume
       - Результат: треугольный меш

    6. ПОСТОБРАБОТКА
       - Удаление мелких компонентов
       - Опциональное упрощение меша
       - Сохранение в .obj/.ply/.stl файл

Запуск:
    python infer.py --image photo.jpg --output result.obj
    python infer.py --image photo.jpg --resolution 256 --threshold 0.4
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import argparse
from datetime import datetime
from typing import Optional, Dict, Any
from tqdm import tqdm

from skimage import measure
import trimesh

from model import create_model
from config import get_config
from mesh_utils import smooth_mesh


# ═══════════════════════════════════════════════════════════════════════════════
# КЛАСС INFERENCER
# ═══════════════════════════════════════════════════════════════════════════════

class Inferencer:
    """
    Класс для генерации 3D моделей из изображений.
    
    Инкапсулирует:
        - Загрузку модели из чекпоинта
        - Предобработку изображений
        - Генерацию 3D меша через Marching Cubes
        - Постобработку и сохранение результата
    
    Args:
        checkpoint_path: Путь к файлу чекпоинта (.pth)
        device: Устройство ('cuda' или 'cpu')
        resolution: Разрешение 3D сетки (по умолчанию 128)
        threshold: Порог для Marching Cubes (по умолчанию 0.5)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        resolution: int = 128,
        threshold: float = 0.5
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.resolution = resolution
        self.threshold = threshold
        
        # Загрузка модели
        self.model, self.config = self._load_model(checkpoint_path)
        
        # Трансформации для изображений (без аугментаций)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"[infer.py] Inferencer ready")
        print(f"[infer.py] Resolution: {resolution}, Threshold: {threshold}")
    
    def _load_model(self, checkpoint_path: str) -> tuple:
        """Загрузка модели из чекпоинта."""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[infer.py] Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        config = checkpoint.get('config', {})
        latent_dim = config.get('latent_dim', 512)
        num_frequencies = config.get('num_frequencies', 10)
        
        model = create_model(
            latent_dim=latent_dim,
            num_frequencies=num_frequencies
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        epoch = checkpoint.get('epoch', '?')
        best_iou = checkpoint.get('best_iou', 0)
        
        print(f"[infer.py] Model loaded successfully")
        print(f"[infer.py] Epoch: {epoch}, Best IoU: {best_iou:.4f}")
        print(f"[infer.py] Latent dim: {latent_dim}")
        
        return model, config
    
    def load_image(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        use_mask: bool = False
    ) -> torch.Tensor:
        """
        Загрузка и предобработка изображения.
        
        Args:
            image_path: Путь к изображению
            mask_path: Путь к маске сегментации (опционально)
            use_mask: Применять ли маску (по умолчанию False)
        
        Returns:
            torch.Tensor [1, 3, 224, 224]: подготовленное изображение
        """
        
        img = Image.open(image_path).convert('RGB')
        
        # Маска применяется ТОЛЬКО если use_mask=True и маска существует
        if use_mask and mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert('L')
                background = Image.new('RGB', img.size, (255, 255, 255))
                img = Image.composite(img, background, mask)
                print(f"[infer.py] Mask applied: {mask_path}")
            except Exception as e:
                print(f"[infer.py] Warning: failed to apply mask: {e}")
        else:
            print(f"[infer.py] Generating from image only (no mask)")
        
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    @torch.no_grad()
    def generate(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        use_mask: bool = False,
        resolution: Optional[int] = None,
        threshold: Optional[float] = None,
        verbose: bool = True
    ) -> Optional[trimesh.Trimesh]:
        """
        Генерация 3D меша из изображения.
        
        Args:
            image_path: Путь к входному изображению
            mask_path: Путь к маске (опционально)
            use_mask: Использовать ли маску (по умолчанию False)
            resolution: Разрешение сетки
            threshold: Порог для Marching Cubes
            verbose: Выводить прогресс в консоль
        
        Returns:
            trimesh.Trimesh: сгенерированный меш
            None: если генерация не удалась
        """
        
        resolution = resolution or self.resolution
        threshold = threshold or self.threshold
        
        start_time = datetime.now()
        
        # Загрузка изображения
        if verbose:
            print(f"\n[infer.py] Loading image: {image_path}")
        
        img_tensor = self.load_image(image_path, mask_path, use_mask)
        
        # Encoding
        if verbose:
            print("[infer.py] Encoding image...")
        
        latent = self.model.encode(img_tensor)
        
        # Создание 3D сетки точек
        if verbose:
            print(f"[infer.py] Creating {resolution}³ grid ({resolution**3:,} points)...")
        
        coords = np.linspace(-0.5, 0.5, resolution).astype(np.float32)
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
        grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        # Предсказание occupancy
        if verbose:
            print("[infer.py] Predicting occupancy...")
        
        occupancy_values = []
        batch_size = 100000
        
        iterator = range(0, len(grid_points), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Occupancy", leave=False)
        
        for i in iterator:
            batch_points = grid_points[i:i + batch_size]
            points_tensor = torch.from_numpy(batch_points).to(self.device)
            points_tensor = points_tensor.unsqueeze(0)
            
            logits = self.model.decode(latent, points_tensor)
            # Temperature scaling для смягчения предсказаний (меньше шума)
            # Более низкая температура = более уверенные предсказания
            temperature = 0.8  # Значения 0.7-1.0 дают хорошие результаты
            scaled_logits = logits / temperature
            probs = torch.sigmoid(scaled_logits).squeeze(0)
            occupancy_values.append(probs.cpu().numpy())
        
        occupancy = np.concatenate(occupancy_values)
        occupancy_grid = occupancy.reshape(resolution, resolution, resolution)
        
        # Применяем фильтры для уменьшения шума (median + gaussian)
        if verbose:
            print("[infer.py] Applying filters to reduce noise...")
        try:
            from scipy.ndimage import gaussian_filter, median_filter
            
            # Median filter удаляет выбросы (шумные точки)
            occupancy_grid = median_filter(occupancy_grid, size=3)
            
            # Gaussian filter мягко сглаживает результат
            occupancy_grid = gaussian_filter(occupancy_grid, sigma=0.5)
            
            if verbose:
                print("[infer.py] Applied Median + Gaussian filters")
        except ImportError:
            if verbose:
                print("[infer.py] scipy not available, skipping filters")
        except Exception as e:
            if verbose:
                print(f"[infer.py] Warning: Filter failed: {e}")
        
        # Статистика occupancy
        if verbose:
            print(f"\n[infer.py] Occupancy statistics:")
            print(f"  Mean: {occupancy_grid.mean():.3f}")
            print(f"  Std:  {occupancy_grid.std():.3f}")
            print(f"  Min:  {occupancy_grid.min():.3f}")
            print(f"  Max:  {occupancy_grid.max():.3f}")
            print(f"  > 0.5: {(occupancy_grid > 0.5).mean() * 100:.1f}%")
        
        # Адаптивный порог при необходимости
        occ_ratio = (occupancy_grid > threshold).mean()
        
        if occ_ratio < 0.001:
            if verbose:
                print("[infer.py] ⚠️ Very low occupancy, trying adaptive threshold")
            adaptive_threshold = np.percentile(occupancy_grid, 99)
            threshold = max(adaptive_threshold, 0.1)
            if verbose:
                print(f"[infer.py] Adaptive threshold: {threshold:.3f}")
        
        # Marching Cubes
        if verbose:
            print(f"\n[infer.py] Running Marching Cubes (threshold={threshold:.3f})...")
        
        try:
            spacing = 1.0 / resolution
            
            vertices, faces, normals, _ = measure.marching_cubes(
                occupancy_grid,
                level=threshold,
                spacing=(spacing, spacing, spacing)
            )
            
            vertices = vertices - 0.5
            
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals
            )
            
            if verbose:
                print(f"[infer.py] Raw mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
        except Exception as e:
            print(f"[infer.py] Marching Cubes failed: {e}")
            return None
        
        # Постобработка с сглаживанием для уменьшения шума
        mesh = self._postprocess_mesh(mesh, verbose=verbose, smooth=True, min_faces=100)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if verbose:
            print(f"\n[infer.py] Generation complete in {elapsed:.1f}s")
            print(f"[infer.py] Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
    
    def _postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        verbose: bool = True,
        smooth: bool = True,
        min_faces: int = 100
    ) -> trimesh.Trimesh:
        """
        Постобработка меша - удаление шума и сглаживание.
        
        Args:
            mesh: Входной меш
            verbose: Выводить информацию
            smooth: Применять сглаживание
            min_faces: Минимальное количество граней для сохранения компонента
        """
        
        try:
            # Шаг 1: Исправление геометрических ошибок
            if verbose:
                print(f"[infer.py] Repairing mesh geometry...")
            try:
                from mesh_utils import repair_mesh
                mesh = repair_mesh(mesh)
                if verbose:
                    print(f"[infer.py] Mesh repaired")
            except Exception as e:
                if verbose:
                    print(f"[infer.py] Warning: repair failed: {e}")
            
            # Шаг 2: Удаление мелких компонентов (более агрессивная фильтрация)
            components = mesh.split(only_watertight=False)
            
            if len(components) > 1:
                # Фильтруем компоненты по размеру
                valid_components = [
                    c for c in components 
                    if len(c.faces) >= min_faces
                ]
                
                if valid_components:
                    # Берём самый большой компонент
                    mesh = max(valid_components, key=lambda x: len(x.vertices))
                    
                    if verbose:
                        removed = len(components) - len(valid_components)
                        print(f"[infer.py] Removed {removed} small components")
                        print(f"[infer.py] Kept largest component: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                else:
                    # Если все компоненты слишком маленькие, берём самый большой
                    mesh = max(components, key=lambda x: len(x.vertices))
                    if verbose:
                        print(f"[infer.py] All components small, kept largest")
        except Exception as e:
            if verbose:
                print(f"[infer.py] Warning: component filtering failed: {e}")
        
        # Шаг 3: Двухэтапное сглаживание для лучшего качества
        if smooth:
            try:
                if verbose:
                    print(f"[infer.py] Applying two-stage Laplacian smoothing...")
                
                # Первый этап: более агрессивное сглаживание (убирает основной шум)
                mesh = smooth_mesh(mesh, iterations=2, lamb=0.6)
                
                # Второй этап: мягкое сглаживание (сохраняет детали)
                mesh = smooth_mesh(mesh, iterations=2, lamb=0.3)
                
                if verbose:
                    print(f"[infer.py] Mesh smoothed (two-stage)")
            except Exception as e:
                if verbose:
                    print(f"[infer.py] Warning: smoothing failed: {e}")
        
        # Шаг 4: Нормализация нормалей для лучшего визуального качества
        try:
            mesh.vertex_normals
            # Принудительно пересчитываем нормали после сглаживания
            mesh.fix_normals()
            if verbose:
                print(f"[infer.py] Normals normalized")
        except Exception:
            pass
        
        return mesh
    
    def generate_and_save(
        self,
        image_path: str,
        output_path: str,
        mask_path: Optional[str] = None,
        use_mask: bool = False,
        resolution: Optional[int] = None,
        threshold: Optional[float] = None,
        simplify: bool = False,
        target_faces: int = 10000
    ) -> Dict[str, Any]:
        """
        Генерация и сохранение 3D модели.
        
        Args:
            image_path: Путь к входному изображению
            output_path: Путь для сохранения .obj файла
            mask_path: Путь к маске (опционально)
            use_mask: Использовать ли маску (по умолчанию False)
            resolution: Разрешение сетки
            threshold: Порог Marching Cubes
            simplify: Упростить меш
            target_faces: Целевое количество граней при упрощении
        
        Returns:
            Dict с информацией о результате
        """
        
        start_time = datetime.now()
        
        mesh = self.generate(
            image_path=image_path,
            mask_path=mask_path,
            use_mask=use_mask,
            resolution=resolution or self.resolution,
            threshold=threshold or self.threshold,
            verbose=True
        )
        
        if mesh is None:
            return {
                'success': False,
                'error': 'Failed to generate mesh'
            }
        
        # Упрощение меша
        if simplify and len(mesh.faces) > target_faces:
            print(f"[infer.py] Simplifying: {len(mesh.faces)} → {target_faces} faces...")
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                print(f"[infer.py] Simplified to {len(mesh.faces)} faces")
            except Exception as e:
                print(f"[infer.py] Warning: simplification failed: {e}")
        
        # Сохранение
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        mesh.export(output_path)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n[infer.py] ✓ Saved: {output_path}")
        
        return {
            'success': True,
            'path': output_path,
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'time': elapsed
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИЯ ДЛЯ БЫСТРОГО ИСПОЛЬЗОВАНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def generate_mesh(
    checkpoint_path: str,
    image_path: str,
    output_path: str,
    resolution: int = 128,
    threshold: float = 0.5,
    mask_path: Optional[str] = None,
    use_mask: bool = False,
    simplify: bool = False,
    target_faces: int = 10000
) -> Optional[trimesh.Trimesh]:
    """
    Функция для быстрой генерации 3D модели.
    
    Args:
        checkpoint_path: Путь к чекпоинту модели
        image_path: Путь к изображению
        output_path: Путь для сохранения результата
        resolution: Разрешение 3D сетки
        threshold: Порог Marching Cubes
        mask_path: Путь к маске (опционально)
        use_mask: Использовать ли маску (по умолчанию False)
        simplify: Упростить меш
        target_faces: Целевое количество граней
    
    Returns:
        trimesh.Trimesh или None
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    inferencer = Inferencer(
        checkpoint_path=checkpoint_path,
        device=device,
        resolution=resolution,
        threshold=threshold
    )
    
    result = inferencer.generate_and_save(
        image_path=image_path,
        output_path=output_path,
        mask_path=mask_path,
        use_mask=use_mask,
        simplify=simplify,
        target_faces=target_faces
    )
    
    if result['success']:
        return trimesh.load(output_path)
    else:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Главная функция для запуска из командной строки."""
    
    parser = argparse.ArgumentParser(
        description='Generate 3D mesh from image using Occupancy Network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python infer.py --image chair.jpg
    python infer.py --image chair.jpg --output ./results/chair.obj
    python infer.py --image chair.jpg --resolution 256 --threshold 0.4
    python infer.py --image chair.jpg --mask chair_mask.png --use_mask
        """
    )
    
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='./checkpoints/best.pth',
        help='Path to model checkpoint (default: ./checkpoints/best.pth)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for 3D mesh (default: auto-generated)'
    )
    parser.add_argument(
        '--resolution', type=int, default=128,
        help='3D grid resolution (default: 128)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Marching Cubes threshold (default: 0.5)'
    )
    # Маска теперь опциональна и по умолчанию НЕ используется
    parser.add_argument(
        '--mask', type=str, default=None,
        help='Path to segmentation mask (optional)'
    )
    parser.add_argument(
        '--use_mask', action='store_true',
        help='Apply mask to remove background (default: False)'
    )
    parser.add_argument(
        '--simplify', action='store_true',
        help='Simplify mesh after generation'
    )
    parser.add_argument(
        '--target-faces', type=int, default=10000,
        help='Target number of faces for simplification (default: 10000)'
    )
    parser.add_argument(
        '--format', type=str, default='obj', choices=['obj', 'ply', 'stl', 'glb'],
        help='Output format (default: obj)'
    )
    
    args = parser.parse_args()
    
    # Определение пути для сохранения
    if args.output is None:
        cfg = get_config()
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join(
            cfg.paths.output_dir,
            f"{base_name}_3d.{args.format}"
        )
    
    # Вывод параметров
    print("=" * 60)
    print("OCCUPANCY NETWORK INFERENCE")
    print("=" * 60)
    print(f"Image:      {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output}")
    print(f"Resolution: {args.resolution}")
    print(f"Threshold:  {args.threshold}")
    print(f"Use mask:   {args.use_mask}")
    if args.mask and args.use_mask:
        print(f"Mask:       {args.mask}")
    if args.simplify:
        print(f"Simplify:   Yes (target: {args.target_faces} faces)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        inferencer = Inferencer(
            checkpoint_path=args.checkpoint,
            device=device,
            resolution=args.resolution,
            threshold=args.threshold
        )
        
        result = inferencer.generate_and_save(
            image_path=args.image,
            output_path=args.output,
            mask_path=args.mask,
            use_mask=args.use_mask,
            resolution=args.resolution,
            threshold=args.threshold,
            simplify=args.simplify,
            target_faces=args.target_faces
        )
        
        if result['success']:
            print("\n" + "=" * 60)
            print("✓ GENERATION COMPLETE")
            print("=" * 60)
            print(f"Output:   {result['path']}")
            print(f"Vertices: {result['vertices']}")
            print(f"Faces:    {result['faces']}")
            print(f"Time:     {result['time']:.1f}s")
            print("=" * 60)
        else:
            print(f"\n✗ Generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()