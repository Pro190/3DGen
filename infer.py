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
       - Применение маски (опционально) для удаления фона
       - Resize до 224x224, нормализация ImageNet

    2. ENCODING
       - Пропускаем изображение через ResNet50
       - Получаем латентный вектор [1, 512]
       - Этот вектор "описывает" 3D форму объекта

    3. СОЗДАНИЕ 3D СЕТКИ
       - Создаём равномерную сетку точек в пространстве [-0.5, 0.5]³
       - Разрешение N×N×N (например, 128³ = 2,097,152 точек)

    4. ПРЕДСКАЗАНИЕ OCCUPANCY
       - Для каждой точки предсказываем вероятность "внутри объекта"
       - Батчевая обработка для эффективности (по 100K точек)
       - Получаем 3D volume с вероятностями

    5. MARCHING CUBES
       - Алгоритм извлечения изоповерхности из 3D volume
       - Порог (threshold) определяет границу объекта
       - Результат: треугольный меш (vertices + faces)

    6. ПОСТОБРАБОТКА
       - Удаление мелких компонентов (шум)
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

# Marching Cubes из scikit-image
from skimage import measure

# Работа с мешами
import trimesh

# Наши модули
from model import create_model
from config import get_config


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
    
    Пример:
        inferencer = Inferencer(
            checkpoint_path='./checkpoints/best.pth',
            resolution=128
        )
        
        mesh = inferencer.generate('photo.jpg')
        mesh.export('result.obj')
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
        
        # ─────────────────────────────────────────────────────────────────────
        # Загрузка модели
        # ─────────────────────────────────────────────────────────────────────
        
        self.model, self.config = self._load_model(checkpoint_path)
        
        # ─────────────────────────────────────────────────────────────────────
        # Трансформации для изображений
        # ─────────────────────────────────────────────────────────────────────
        #
        # Такие же как при обучении (без аугментаций):
        #   1. Resize до 224x224 (вход ResNet)
        #   2. ToTensor: [0, 255] → [0, 1]
        #   3. Normalize: стандартизация ImageNet
        # ─────────────────────────────────────────────────────────────────────
        
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
        """
        Загрузка модели из чекпоинта.
        
        Args:
            checkpoint_path: Путь к .pth файлу
        
        Returns:
            Tuple[model, config]: загруженная модель и её конфигурация
        
        Raises:
            FileNotFoundError: если чекпоинт не найден
        """
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[infer.py] Loading model from: {checkpoint_path}")
        
        # Загружаем чекпоинт
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Извлекаем конфигурацию модели
        config = checkpoint.get('config', {})
        latent_dim = config.get('latent_dim', 512)
        num_frequencies = config.get('num_frequencies', 10)
        
        # Создаём модель с такой же архитектурой
        model = create_model(
            latent_dim=latent_dim,
            num_frequencies=num_frequencies
        ).to(self.device)
        
        # Загружаем веса
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Переводим в режим оценки (отключает dropout и т.д.)
        model.eval()
        
        # Логирование информации
        epoch = checkpoint.get('epoch', '?')
        best_iou = checkpoint.get('best_iou', 0)
        
        print(f"[infer.py] Model loaded successfully")
        print(f"[infer.py] Epoch: {epoch}, Best IoU: {best_iou:.4f}")
        print(f"[infer.py] Latent dim: {latent_dim}")
        
        return model, config
    
    def load_image(
        self,
        image_path: str,
        mask_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Загрузка и предобработка изображения.
        
        Args:
            image_path: Путь к изображению
            mask_path: Путь к маске сегментации (опционально)
        
        Returns:
            torch.Tensor [1, 3, 224, 224]: подготовленное изображение
        """
        
        # Загружаем изображение
        img = Image.open(image_path).convert('RGB')
        
        # ─────────────────────────────────────────────────────────────────────
        # Применение маски (если указана)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Маска удаляет фон, заменяя его на белый цвет.
        # Это помогает модели фокусироваться на объекте.
        # ─────────────────────────────────────────────────────────────────────
        
        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert('L')  # Grayscale
                
                # Создаём белый фон
                background = Image.new('RGB', img.size, (255, 255, 255))
                
                # Композитинг: объект на белом фоне
                img = Image.composite(img, background, mask)
                
                print(f"[infer.py] Mask applied: {mask_path}")
            except Exception as e:
                print(f"[infer.py] Warning: failed to apply mask: {e}")
        
        # Применяем трансформации
        img_tensor = self.transform(img)
        
        # Добавляем batch dimension: [3, 224, 224] → [1, 3, 224, 224]
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    @torch.no_grad()  # Отключаем градиенты для инференса
    def generate(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        resolution: Optional[int] = None,
        threshold: Optional[float] = None,
        verbose: bool = True
    ) -> Optional[trimesh.Trimesh]:
        """
        Генерация 3D меша из изображения.
        
        Основной метод класса. Выполняет полный pipeline:
            1. Загрузка изображения
            2. Encoding в латентный вектор
            3. Создание 3D сетки точек
            4. Предсказание occupancy
            5. Marching Cubes
            6. Постобработка
        
        Args:
            image_path: Путь к входному изображению
            mask_path: Путь к маске (опционально)
            resolution: Разрешение сетки (None = использовать значение по умолчанию)
            threshold: Порог для Marching Cubes (None = использовать значение по умолчанию)
            verbose: Выводить прогресс в консоль
        
        Returns:
            trimesh.Trimesh: сгенерированный меш
            None: если генерация не удалась
        """
        
        resolution = resolution or self.resolution
        threshold = threshold or self.threshold
        
        start_time = datetime.now()
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 1: Загрузка изображения
        # ─────────────────────────────────────────────────────────────────────
        
        if verbose:
            print(f"\n[infer.py] Loading image: {image_path}")
        
        img_tensor = self.load_image(image_path, mask_path)
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 2: Encoding
        # ─────────────────────────────────────────────────────────────────────
        #
        # ResNet50 преобразует изображение [1, 3, 224, 224]
        # в латентный вектор [1, 512], который "описывает" 3D форму.
        # ─────────────────────────────────────────────────────────────────────
        
        if verbose:
            print("[infer.py] Encoding image...")
        
        latent = self.model.encode(img_tensor)
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 3: Создание 3D сетки точек
        # ─────────────────────────────────────────────────────────────────────
        #
        # Создаём равномерную сетку в кубе [-0.5, 0.5]³
        # Количество точек = resolution³
        #
        # Например, при resolution=128:
        #   128³ = 2,097,152 точек
        # ─────────────────────────────────────────────────────────────────────
        
        if verbose:
            print(f"[infer.py] Creating {resolution}³ grid ({resolution**3:,} points)...")
        
        # Координаты по каждой оси
        coords = np.linspace(-0.5, 0.5, resolution).astype(np.float32)
        
        # 3D сетка (meshgrid)
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
        
        # Преобразуем в массив точек [N, 3]
        grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 4: Предсказание occupancy
        # ─────────────────────────────────────────────────────────────────────
        #
        # Для каждой точки предсказываем вероятность "внутри объекта".
        # Обрабатываем батчами, т.к. все точки не влезут в GPU память.
        # ─────────────────────────────────────────────────────────────────────
        
        if verbose:
            print("[infer.py] Predicting occupancy...")
        
        occupancy_values = []
        batch_size = 100000  # 100K точек за раз
        
        # Итератор по батчам
        iterator = range(0, len(grid_points), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Occupancy", leave=False)
        
        for i in iterator:
            # Берём батч точек
            batch_points = grid_points[i:i + batch_size]
            
            # Конвертируем в тензор
            points_tensor = torch.from_numpy(batch_points).to(self.device)
            
            # Добавляем batch dimension: [N, 3] → [1, N, 3]
            points_tensor = points_tensor.unsqueeze(0)
            
            # Forward pass через декодер
            logits = self.model.decode(latent, points_tensor)
            
            # Применяем sigmoid для получения вероятностей
            probs = torch.sigmoid(logits).squeeze(0)
            
            # Сохраняем на CPU
            occupancy_values.append(probs.cpu().numpy())
        
        # Объединяем все батчи
        occupancy = np.concatenate(occupancy_values)
        
        # Преобразуем в 3D volume
        occupancy_grid = occupancy.reshape(resolution, resolution, resolution)
        
        # ─────────────────────────────────────────────────────────────────────
        # Статистика occupancy (для отладки)
        # ─────────────────────────────────────────────────────────────────────
        
        if verbose:
            print(f"\n[infer.py] Occupancy statistics:")
            print(f"  Mean: {occupancy_grid.mean():.3f}")
            print(f"  Std:  {occupancy_grid.std():.3f}")
            print(f"  Min:  {occupancy_grid.min():.3f}")
            print(f"  Max:  {occupancy_grid.max():.3f}")
            print(f"  > 0.5: {(occupancy_grid > 0.5).mean() * 100:.1f}%")
            print(f"  > 0.7: {(occupancy_grid > 0.7).mean() * 100:.1f}%")
            print(f"  > 0.9: {(occupancy_grid > 0.9).mean() * 100:.1f}%")
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 5: Marching Cubes
        # ─────────────────────────────────────────────────────────────────────
        #
        # Алгоритм Marching Cubes извлекает изоповерхность из 3D volume.
        #
        # Принцип работы:
        #   1. Проходим по всем "кубикам" (8 соседних вокселей)
        #   2. Для каждого кубика определяем, какие вершины "внутри" (> threshold)
        #   3. По таблице определяем, как провести треугольники через кубик
        #   4. Результат: треугольный меш
        #
        # Параметры:
        #   - level: порог (threshold) для изоповерхности
        #   - spacing: размер вокселя в мировых координатах
        # ─────────────────────────────────────────────────────────────────────
        
        if verbose:
            print(f"\n[infer.py] Running Marching Cubes (threshold={threshold})...")
        
        try:
            # Размер вокселя
            spacing = 1.0 / resolution
            
            # Marching Cubes
            vertices, faces, normals, _ = measure.marching_cubes(
                occupancy_grid,
                level=threshold,
                spacing=(spacing, spacing, spacing)
            )
            
            # Центрируем меш (сетка была в [0, 1], нужно [-0.5, 0.5])
            vertices = vertices - 0.5
            
            # Создаём trimesh объект
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
        
        # ─────────────────────────────────────────────────────────────────────
        # Шаг 6: Постобработка
        # ─────────────────────────────────────────────────────────────────────
        
        mesh = self._postprocess_mesh(mesh, verbose=verbose)
        
        # Время генерации
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if verbose:
            print(f"\n[infer.py] Generation complete in {elapsed:.1f}s")
            print(f"[infer.py] Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
    
    def _postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        verbose: bool = True
    ) -> trimesh.Trimesh:
        """
        Постобработка меша.
        
        Выполняет:
            1. Разделение на компоненты связности
            2. Сохранение только самого большого компонента
               (удаление мелкого "шума")
        
        Args:
            mesh: Входной меш
            verbose: Выводить информацию
        
        Returns:
            Обработанный меш
        """
        
        # ─────────────────────────────────────────────────────────────────────
        # Удаление мелких компонентов
        # ─────────────────────────────────────────────────────────────────────
        #
        # Marching Cubes может создать мелкие "островки" (артефакты).
        # Оставляем только самый большой компонент связности.
        # ─────────────────────────────────────────────────────────────────────
        
        try:
            # Разбиваем на компоненты связности
            components = mesh.split(only_watertight=False)
            
            if len(components) > 1:
                # Находим самый большой по количеству вершин
                largest = max(components, key=lambda x: len(x.vertices))
                
                if verbose:
                    print(f"[infer.py] Kept largest of {len(components)} components")
                
                mesh = largest
        except Exception:
            pass  # Если split не удался, оставляем как есть
        
        return mesh
    
    def generate_and_save(
        self,
        image_path: str,
        output_path: str,
        mask_path: Optional[str] = None,
        resolution: Optional[int] = None,
        threshold: Optional[float] = None,
        simplify: bool = False,
        target_faces: int = 10000
    ) -> Dict[str, Any]:
        """
        Генерация и сохранение 3D модели.
        
        Удобный метод, объединяющий generate() и export().
        
        Args:
            image_path: Путь к входному изображению
            output_path: Путь для сохранения .obj файла
            mask_path: Путь к маске (опционально)
            resolution: Разрешение сетки
            threshold: Порог Marching Cubes
            simplify: Упростить меш
            target_faces: Целевое количество граней при упрощении
        
        Returns:
            Dict с информацией о результате:
                - 'success': bool
                - 'path': str (путь к файлу)
                - 'vertices': int
                - 'faces': int
                - 'time': float (секунды)
        """
        
        start_time = datetime.now()
        
        # Генерация
        mesh = self.generate(
            image_path=image_path,
            mask_path=mask_path,
            resolution=resolution,
            threshold=threshold,
            verbose=True
        )
        
        if mesh is None:
            return {
                'success': False,
                'error': 'Failed to generate mesh'
            }
        
        # ─────────────────────────────────────────────────────────────────────
        # Упрощение меша (опционально)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Quadric decimation уменьшает количество граней,
        # сохраняя общую форму объекта.
        # ─────────────────────────────────────────────────────────────────────
        
        if simplify and len(mesh.faces) > target_faces:
            print(f"[infer.py] Simplifying: {len(mesh.faces)} → {target_faces} faces...")
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                print(f"[infer.py] Simplified to {len(mesh.faces)} faces")
            except Exception as e:
                print(f"[infer.py] Warning: simplification failed: {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Сохранение
        # ─────────────────────────────────────────────────────────────────────
        
        # Создаём папку если не существует
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Экспорт (формат определяется по расширению)
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
    simplify: bool = False,
    target_faces: int = 10000
) -> Optional[trimesh.Trimesh]:
    """
    Функция для быстрой генерации 3D модели.
    
    Создаёт Inferencer, генерирует меш и сохраняет его.
    Удобна для использования из других скриптов.
    
    Args:
        checkpoint_path: Путь к чекпоинту модели
        image_path: Путь к изображению
        output_path: Путь для сохранения результата
        resolution: Разрешение 3D сетки
        threshold: Порог Marching Cubes
        mask_path: Путь к маске (опционально)
        simplify: Упростить меш
        target_faces: Целевое количество граней
    
    Returns:
        trimesh.Trimesh или None
    
    Пример:
        from infer import generate_mesh
        
        mesh = generate_mesh(
            checkpoint_path='./checkpoints/best.pth',
            image_path='chair.jpg',
            output_path='chair_3d.obj'
        )
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
        simplify=simplify,
        target_faces=target_faces
    )
    
    if result['success']:
        # Загружаем и возвращаем меш
        return trimesh.load(output_path)
    else:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Главная функция для запуска из командной строки.
    
    Использование:
        python infer.py --image photo.jpg
        python infer.py --image photo.jpg --output result.obj
        python infer.py --image photo.jpg --resolution 256 --threshold 0.4
        python infer.py --image photo.jpg --mask mask.png --simplify
    """
    
    parser = argparse.ArgumentParser(
        description='Generate 3D mesh from image using Occupancy Network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python infer.py --image chair.jpg
    python infer.py --image chair.jpg --output ./results/chair.obj
    python infer.py --image chair.jpg --resolution 256 --threshold 0.4
    python infer.py --image chair.jpg --mask chair_mask.png --simplify
        """
    )
    
    # Обязательные аргументы
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to input image'
    )
    
    # Опциональные аргументы
    parser.add_argument(
        '--checkpoint', type=str, default='./checkpoints/best.pth',
        help='Path to model checkpoint (default: ./checkpoints/best.pth)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for 3D mesh (default: auto-generated in ./inference_results/)'
    )
    parser.add_argument(
        '--resolution', type=int, default=128,
        help='3D grid resolution (default: 128)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Marching Cubes threshold (default: 0.5)'
    )
    parser.add_argument(
        '--mask', type=str, default=None,
        help='Path to segmentation mask (optional)'
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # Определение пути для сохранения
    # ─────────────────────────────────────────────────────────────────────────
    
    if args.output is None:
        # Автоматическое имя файла
        cfg = get_config()
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join(
            cfg.paths.output_dir,
            f"{base_name}_3d.{args.format}"
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Вывод параметров
    # ─────────────────────────────────────────────────────────────────────────
    
    print("=" * 60)
    print("OCCUPANCY NETWORK INFERENCE")
    print("=" * 60)
    print(f"Image:      {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output}")
    print(f"Resolution: {args.resolution}")
    print(f"Threshold:  {args.threshold}")
    if args.mask:
        print(f"Mask:       {args.mask}")
    if args.simplify:
        print(f"Simplify:   Yes (target: {args.target_faces} faces)")
    print("=" * 60)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Генерация
    # ─────────────────────────────────────────────────────────────────────────
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()