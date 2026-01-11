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
from typing import Optional, List, Tuple
import torch
import os


@dataclass
class PathConfig:
    """Пути к данным."""
    data_root: str = './PIX3D_DATA'
    checkpoint_dir: str = './checkpoints'
    output_dir: str = './inference_results'
    cache_dir: str = './cache'  # Для препроцессинга
    
    @property
    def json_path(self) -> str:
        return os.path.join(self.data_root, 'pix3d.json')
    
    @property
    def preprocessed_dir(self) -> str:
        return os.path.join(self.cache_dir, 'preprocessed')


@dataclass
class ModelConfig:
    """Конфигурация модели."""
    # Encoder - теперь поддерживаем больше вариантов
    encoder_type: str = 'resnet50'  # resnet18, resnet34, resnet50, resnet101, efficientnet_b0, efficientnet_b3, convnext_tiny, convnext_small
    latent_dim: int = 512
    encoder_pretrained: bool = True
    encoder_freeze_bn: bool = False  # Заморозить BatchNorm
    
    # Decoder (MLP с улучшениями)
    decoder_hidden_dims: Tuple[int, ...] = (512, 512, 512, 256, 256)  # Глубже
    decoder_dropout: float = 0.1
    decoder_use_residual: bool = True  # Residual connections
    decoder_use_layer_norm: bool = True  # LayerNorm вместо BatchNorm
    
    # Для Marching Cubes
    mc_resolution: int = 64
    mc_threshold: float = 0.5


@dataclass
class TrainConfig:
    """Конфигурация обучения."""
    # Батч и эпохи
    batch_size: int = 32
    num_epochs: int = 300  # Больше эпох
    
    # Оптимизатор
    learning_rate: float = 3e-4  # Немного выше
    weight_decay: float = 1e-4
    
    # Warmup
    warmup_epochs: int = 10
    warmup_lr: float = 1e-6
    
    # Gradient clipping
    grad_clip: float = 1.0
    
    # Сэмплирование точек - увеличено
    num_points_surface: int = 2048
    num_points_uniform: int = 2048
    
    # Добавление шума к поверхностным точкам
    surface_noise_std: float = 0.02  # Меньше шума
    
    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    
    # Логирование и сохранение
    log_interval: int = 50
    save_interval: int = 10  # Сохранять каждые 10 эпох
    val_interval: int = 5
    
    # Разделение данных
    val_split: float = 0.1
    
    # Воспроизводимость
    seed: int = 42
    
    # Категория (None = все)
    category_filter: Optional[str] = None
    
    # Loss weights
    bce_weight: float = 1.0
    iou_weight: float = 0.5
    
    # Аугментации
    use_augmentation: bool = True
    color_jitter: float = 0.3
    random_rotation: bool = True  # Случайное вращение 3D точек
    random_scale: Tuple[float, float] = (0.9, 1.1)


@dataclass 
class PreprocessConfig:
    """Конфигурация препроцессинга."""
    # Voxelization
    voxel_resolution: int = 64
    
    # Сэмплирование
    num_surface_samples: int = 100000  # Много точек для кэша
    num_uniform_samples: int = 100000
    
    # Параллелизм
    num_workers: int = 8
    
    # Фильтрация
    min_vertices: int = 100
    min_faces: int = 50


@dataclass
class InferConfig:
    """Конфигурация инференса."""
    mc_resolution: int = 128
    mc_threshold: float = 0.5
    simplify_mesh: bool = True
    target_faces: int = 10000
    
    # Батчевый инференс
    batch_size: int = 65536  # Для GPU с большой памятью


@dataclass
class Config:
    """Главный конфиг."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    
    @property
    def device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def use_amp(self) -> bool:
        return torch.cuda.is_available()
    
    def get_encoder_list(self) -> List[str]:
        """Список доступных энкодеров."""
        return [
            'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b5',
            'convnext_tiny', 'convnext_small', 'convnext_base'
        ]


# Глобальный экземпляр
_config = Config()


def get_config() -> Config:
    return _config


def update_config(**kwargs) -> Config:
    """Обновление конфига."""
    global _config
    
    for key, value in kwargs.items():
        if hasattr(_config.model, key):
            setattr(_config.model, key, value)
        elif hasattr(_config.train, key):
            setattr(_config.train, key, value)
        elif hasattr(_config.paths, key):
            setattr(_config.paths, key, value)
    
    return _config