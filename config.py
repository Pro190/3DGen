"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".   
Описание: Валидация: расчет численных показателей качества генерации.
Дата: 2026
================================================================================
"""
from dataclasses import dataclass, field
from typing import Optional
import torch
import os


@dataclass
class PathConfig:
    """Пути к данным."""
    data_root: str = './PIX3D_DATA'
    checkpoint_dir: str = './checkpoints'
    output_dir: str = './inference_results'
    
    @property
    def json_path(self) -> str:
        return os.path.join(self.data_root, 'pix3d.json')


@dataclass
class ModelConfig:
    """Конфигурация модели."""
    # Encoder
    encoder_type: str = 'resnet18'
    latent_dim: int = 512
    
    # Decoder (MLP)
    decoder_hidden_dims: tuple = (256, 256, 256, 256)
    decoder_dropout: float = 0.0
    
    # Для Marching Cubes
    mc_resolution: int = 64  # Разрешение сетки для извлечения mesh
    mc_threshold: float = 0.5  # Порог occupancy


@dataclass
class TrainConfig:
    """Конфигурация обучения."""
    # Батч и эпохи
    batch_size: int = 32  # Можно больше - модель легче
    num_epochs: int = 200
    
    # Оптимизатор
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Сэмплирование точек
    num_points_surface: int = 1024  # Точек на поверхности
    num_points_uniform: int = 1024  # Случайных точек в объёме
    
    # Добавление шума к поверхностным точкам
    surface_noise_std: float = 0.05
    
    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    
    # Логирование
    log_interval: int = 50
    save_interval: int = 10
    val_interval: int = 5
    
    # Разделение данных
    val_split: float = 0.1
    
    # Воспроизводимость
    seed: int = 42
    
    # Категория (None = все)
    category_filter: Optional[str] = None


@dataclass
class InferConfig:
    """Конфигурация инференса."""
    mc_resolution: int = 128  # Выше разрешение для финального mesh
    mc_threshold: float = 0.5
    simplify_mesh: bool = True
    target_faces: int = 10000  # Целевое количество граней после упрощения


@dataclass
class Config:
    """Главный конфиг."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    
    @property
    def device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def use_amp(self) -> bool:
        return torch.cuda.is_available()


# Глобальный экземпляр
_config = Config()


def get_config() -> Config:
    return _config