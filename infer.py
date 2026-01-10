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
from torchvision import transforms

from config import get_config
from model import OccupancyNetwork
from mesh_utils import extract_mesh_marching_cubes, simplify_mesh, save_mesh


def load_model(checkpoint_path: str, cfg, device: str):
    """Загрузка обученной модели."""
    model = OccupancyNetwork(
        backbone=cfg.model.encoder_type,
        latent_dim=cfg.model.latent_dim,
        hidden_dims=cfg.model.decoder_hidden_dims
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"[infer.py] Загружаю: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"[infer.py] ⚠️ Чекпоинт не найден, используются случайные веса")
    
    model.eval()
    return model


def load_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """Загрузка и предобработка изображения."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor


def generate_mesh(
    model: OccupancyNetwork,
    image_tensor: torch.Tensor,
    device: str,
    resolution: int = 128,
    threshold: float = 0.5
):
    """Генерация меша из изображения."""
    
    image_tensor = image_tensor.to(device)
    
    # Получаем latent code
    with torch.no_grad():
        latent = model.encode(image_tensor)  # [1, latent_dim]
    
    # Функция для Marching Cubes
    def occupancy_fn(points: torch.Tensor) -> torch.Tensor:
        """Query occupancy для точек."""
        points = points.unsqueeze(0)  # [1, N, 3]
        latent_expanded = latent  # [1, latent_dim]
        
        with torch.no_grad():
            logits = model.decode(latent_expanded, points)
            probs = torch.sigmoid(logits)
        
        return probs.squeeze(0)  # [N]
    
    # Извлекаем mesh
    print(f"[infer.py] Marching Cubes (resolution={resolution})...")
    mesh = extract_mesh_marching_cubes(
        occupancy_fn,
        resolution=resolution,
        threshold=threshold,
        device=device
    )
    
    return mesh


def main():
    parser = argparse.ArgumentParser(description='Generate 3D mesh from image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pth')
    parser.add_argument('--output', type=str, default='./inference_results/')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--simplify', action='store_true', help='Simplify mesh')
    parser.add_argument('--target_faces', type=int, default=10000)
    
    args = parser.parse_args()
    
    cfg = get_config()
    device = cfg.device
    
    print("="*60)
    print("OCCUPANCY NETWORK INFERENCE")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resolution: {args.resolution}")
    print(f"Device: {device}")
    print("="*60)
    
    # Загрузка модели
    model = load_model(args.checkpoint, cfg, device)
    
    # Загрузка изображения
    print(f"\n[infer.py] Загрузка изображения...")
    image_tensor = load_image(args.image)
    
    # Генерация меша
    print(f"[infer.py] Генерация 3D модели...")
    mesh = generate_mesh(
        model, image_tensor, device,
        resolution=args.resolution,
        threshold=args.threshold
    )
    
    if mesh is None:
        print("[infer.py] ❌ Не удалось извлечь меш")
        return
    
    print(f"[infer.py] Меш создан: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
    
    # Упрощение
    if args.simplify and len(mesh.faces) > args.target_faces:
        print(f"[infer.py] Упрощение до {args.target_faces} граней...")
        mesh = simplify_mesh(mesh, args.target_faces)
    
    # Сохранение
    os.makedirs(args.output, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_path = os.path.join(args.output, f"{base_name}_3d.obj")
    
    save_mesh(mesh, output_path)
    
    print(f"\n[infer.py] ✓ Готово: {output_path}")


if __name__ == '__main__':
    main()