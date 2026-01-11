"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".
Описание: Скрипт для запуска инференса: генерация 3D-сетки мебели на основе произвольного изображения.
Дата: 2026
================================================================================
"""
import torch
import numpy as np
from PIL import Image
import os
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from model import OccupancyNetwork, create_model, AVAILABLE_ENCODERS
from mesh_utils import (
    extract_mesh_marching_cubes, 
    simplify_mesh, 
    smooth_mesh,
    save_mesh,
    repair_mesh
)


class Inferencer:
    """Класс для инференса модели."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        resolution: int = 128,
        threshold: float = 0.5
    ):
        self.device = device
        self.resolution = resolution
        self.threshold = threshold
        
        # Загрузка модели
        self.model, self.config = self._load_model(checkpoint_path)
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"[infer.py] Модель загружена, готов к инференсу")
    
    def _load_model(self, checkpoint_path: str) -> tuple:
        """Загрузка модели из чекпоинта."""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
        
        print(f"[infer.py] Загружаю: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Получаем конфигурацию модели из чекпоинта
        model_config = checkpoint.get('config', {})
        
        encoder_type = model_config.get('encoder_type', 'resnet50')
        latent_dim = model_config.get('latent_dim', 512)
        hidden_dims = model_config.get('hidden_dims', (512, 512, 512, 256, 256))
        
        print(f"[infer.py] Encoder: {encoder_type}, Latent: {latent_dim}")
        
        # Создаём модель
        model = create_model(
            encoder_type=encoder_type,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            use_positional_encoding=True,
            pretrained=False  # Загружаем веса из чекпоинта
        ).to(self.device)
        
        # Загружаем веса
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Информация о чекпоинте
        epoch = checkpoint.get('epoch', 'unknown')
        best_iou = checkpoint.get('best_val_iou', 'unknown')
        print(f"[infer.py] Epoch: {epoch}, Best IoU: {best_iou}")
        
        return model, model_config
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Загрузка и предобработка изображения."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def load_image_with_mask(
        self, 
        image_path: str, 
        mask_path: Optional[str] = None
    ) -> torch.Tensor:
        """Загрузка изображения с применением маски."""
        img = Image.open(image_path).convert('RGB')
        
        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert('L')
                background = Image.new('RGB', img.size, (255, 255, 255))
                img = Image.composite(img, background, mask)
            except Exception as e:
                print(f"[infer.py] ⚠️ Ошибка применения маски: {e}")
        
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor.to(self.device)
    
    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Получение latent code из изображения."""
        return self.model.encode(image_tensor)
    
    @torch.no_grad()
    def query_occupancy(
        self, 
        latent: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        """Запрос occupancy для точек."""
        if points.dim() == 2:
            points = points.unsqueeze(0)
        
        logits = self.model.decode(latent, points)
        probs = torch.sigmoid(logits)
        
        return probs.squeeze(0)
    
    def generate_mesh(
        self,
        image_tensor: torch.Tensor,
        resolution: Optional[int] = None,
        threshold: Optional[float] = None,
        simplify: bool = False,
        target_faces: int = 10000,
        smooth: bool = False,
        smooth_iterations: int = 2
    ):
        """
        Генерация меша из изображения.
        
        Args:
            image_tensor: тензор изображения [1, 3, 224, 224]
            resolution: разрешение Marching Cubes
            threshold: порог occupancy
            simplify: упростить меш
            target_faces: целевое количество граней
            smooth: сгладить меш
            smooth_iterations: количество итераций сглаживания
            
        Returns:
            trimesh.Trimesh или None
        """
        resolution = resolution or self.resolution
        threshold = threshold or self.threshold
        
        # Encode
        latent = self.encode_image(image_tensor)
        
        # Функция для Marching Cubes
        def occupancy_fn(points: torch.Tensor) -> torch.Tensor:
            points = points.unsqueeze(0)
            with torch.no_grad():
                logits = self.model.decode(latent, points)
                probs = torch.sigmoid(logits)
            return probs.squeeze(0)
        
        # Marching Cubes
        print(f"[infer.py] Marching Cubes (resolution={resolution})...")
        mesh = extract_mesh_marching_cubes(
            occupancy_fn,
            resolution=resolution,
            threshold=threshold,
            device=self.device,
            verbose=True
        )
        
        if mesh is None:
            return None
        
        # Post-processing
        if smooth and len(mesh.faces) > 0:
            print(f"[infer.py] Сглаживание...")
            mesh = smooth_mesh(mesh, iterations=smooth_iterations)
        
        if simplify and len(mesh.faces) > target_faces:
            print(f"[infer.py] Упрощение до {target_faces} граней...")
            mesh = simplify_mesh(mesh, target_faces)
        
        # Repair
        mesh = repair_mesh(mesh)
        
        return mesh
    
    def process_single(
        self,
        image_path: str,
        output_path: str,
        mask_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Обработка одного изображения.
        
        Returns:
            Словарь с результатами
        """
        start_time = datetime.now()
        
        # Загрузка
        image_tensor = self.load_image_with_mask(image_path, mask_path)
        
        # Генерация
        mesh = self.generate_mesh(image_tensor, **kwargs)
        
        if mesh is None:
            return {
                'success': False,
                'error': 'Failed to extract mesh',
                'input': image_path
            }
        
        # Сохранение
        save_mesh(mesh, output_path)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': True,
            'input': image_path,
            'output': output_path,
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'time': elapsed
        }
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Батчевая обработка изображений.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for image_path in tqdm(image_paths, desc="Processing"):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_3d.obj")
            
            result = self.process_single(image_path, output_path, **kwargs)
            results.append(result)
        
        # Статистика
        successful = sum(1 for r in results if r['success'])
        print(f"\n[infer.py] Обработано: {successful}/{len(results)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D mesh from image using Occupancy Network'
    )
    
    # Основные аргументы
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./inference_results/',
                        help='Output directory')
    
    # Параметры генерации
    parser.add_argument('--resolution', type=int, default=128,
                        help='Marching Cubes resolution')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Occupancy threshold')
    
    # Post-processing
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify mesh')
    parser.add_argument('--target-faces', type=int, default=10000,
                        help='Target faces for simplification')
    parser.add_argument('--smooth', action='store_true',
                        help='Smooth mesh')
    parser.add_argument('--smooth-iterations', type=int, default=2,
                        help='Smoothing iterations')
    
    # Формат вывода
    parser.add_argument('--format', type=str, default='obj',
                        choices=['obj', 'ply', 'stl', 'glb'],
                        help='Output format')
    
    # Маска
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to mask image')
    
    args = parser.parse_args()
    
    cfg = get_config()
    device = cfg.device
    
    print("="*60)
    print("OCCUPANCY NETWORK INFERENCE")
    print("="*60)
    print(f"Input: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resolution: {args.resolution}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {device}")
    print("="*60)
    
    # Создаём inferencer
    inferencer = Inferencer(
        checkpoint_path=args.checkpoint,
        device=device,
        resolution=args.resolution,
        threshold=args.threshold
    )
    
    # Определяем тип входа (файл или директория)
    if os.path.isdir(args.image):
        # Батчевая обработка
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        image_paths = [
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.lower().endswith(extensions)
        ]
        
        print(f"\n[infer.py] Найдено {len(image_paths)} изображений")
        
        results = inferencer.process_batch(
            image_paths,
            args.output,
            simplify=args.simplify,
            target_faces=args.target_faces,
            smooth=args.smooth,
            smooth_iterations=args.smooth_iterations
        )
        
    else:
        # Одиночное изображение
        os.makedirs(args.output, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(args.output, f"{base_name}_3d.{args.format}")
        
        result = inferencer.process_single(
            args.image,
            output_path,
            mask_path=args.mask,
            simplify=args.simplify,
            target_faces=args.target_faces,
            smooth=args.smooth,
            smooth_iterations=args.smooth_iterations
        )
        
        if result['success']:
            print(f"\n[infer.py] ✓ Готово!")
            print(f"  Файл: {result['output']}")
            print(f"  Вершин: {result['vertices']}")
            print(f"  Граней: {result['faces']}")
            print(f"  Время: {result['time']:.2f}s")
        else:
            print(f"\n[infer.py] ✗ Ошибка: {result.get('error', 'Unknown')}")


if __name__ == '__main__':
    main()