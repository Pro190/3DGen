"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Конфигурация проекта - все настройки в одном месте
Дата: 2025
================================================================================

Структура конфигурации:
    Config (главный класс)
    ├── PathConfig      - пути к данным, чекпоинтам, результатам
    ├── ModelConfig     - параметры архитектуры нейросети
    ├── TrainConfig     - параметры обучения (lr, batch_size, epochs...)
    └── InferConfig     - параметры инференса (resolution, threshold...)

Использование:
    from config import get_config
    cfg = get_config()
    
    # Доступ к параметрам:
    cfg.paths.data_root      # './PIX3D_DATA'
    cfg.model.latent_dim     # 512
    cfg.train.batch_size     # 32
    cfg.device               # 'cuda' или 'cpu'
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import os


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ПУТЕЙ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    """
    Пути к данным и результатам.
    
    Attributes:
        data_root: Корневая папка датасета PIX3D
        checkpoint_dir: Папка для сохранения чекпоинтов модели
        output_dir: Папка для результатов инференса (3D модели)
        cache_dir: Папка для кэшированных/препроцессированных данных
    """
    
    # Основные пути
    data_root: str = './PIX3D_DATA'
    checkpoint_dir: str = './checkpoints'
    output_dir: str = './inference_results'
    cache_dir: str = './cache'
    
    @property
    def json_path(self) -> str:
        """
        Путь к файлу аннотаций PIX3D.
        
        Файл pix3d.json содержит список всех образцов с путями к:
        - изображениям (img)
        - 3D моделям (model)  
        - маскам сегментации (mask)
        - категориям мебели (category)
        """
        return os.path.join(self.data_root, 'pix3d.json')
    
    @property
    def preprocessed_dir(self) -> str:
        """
        Путь к папке с препроцессированными данными.
        
        Препроцессинг ускоряет обучение за счёт предварительного:
        - Нормализации мешей
        - Сэмплирования точек на поверхности
        - Вычисления нормалей
        """
        return os.path.join(self.cache_dir, 'preprocessed')


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """
    Параметры архитектуры Occupancy Network.
    
    Архитектура Global Latent:
        1. Encoder (ResNet50): Изображение [B,3,224,224] → Latent [B,512]
        2. PositionalEncoding: Координаты [B,N,3] → [B,N,63]
        3. Decoder (MLP): [Latent + Points] → Occupancy [B,N]
    
    Attributes:
        latent_dim: Размерность латентного вектора (выход энкодера)
                   Больше = больше информации, но медленнее обучение
                   Рекомендуется: 256-512
        
        num_frequencies: Количество частот для positional encoding
                        Формула выходной размерности: 3 + 6*num_frequencies
                        При 10: 3 + 60 = 63
                        Больше частот = лучше мелкие детали
    """
    
    latent_dim: int = 512
    num_frequencies: int = 10


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ОБУЧЕНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    Параметры обучения модели.
    
    Процесс обучения:
        1. Загрузка изображения и 3D модели
        2. Сэмплирование точек (часть внутри, часть снаружи меша)
        3. Forward pass: image → latent → occupancy predictions
        4. Loss: BCE между предсказаниями и ground truth
        5. Backward pass: обновление весов
    
    Attributes:
        batch_size: Количество образцов в одном батче
                   Больше = стабильнее градиенты, но больше памяти GPU
                   Для RTX 5080 (16GB): 32-64
        
        num_epochs: Количество полных проходов по датасету
                   Обычно 100-300 достаточно для сходимости
        
        learning_rate: Начальная скорость обучения
                      3e-4 хорошо работает с AdamW
        
        weight_decay: L2 регуляризация весов
                     Помогает предотвратить переобучение
        
        warmup_epochs: Эпохи с линейным увеличением LR от 0.01*lr до lr
                      Стабилизирует начало обучения
        
        grad_clip: Максимальная норма градиента
                  Предотвращает "взрыв" градиентов
        
        num_points: Количество точек для сэмплирования на один образец
                   Половина inside, половина outside
                   Больше = точнее, но медленнее
        
        val_split: Доля данных для валидации (0.1 = 10%)
        
        save_interval: Сохранять чекпоинт каждые N эпох
        
        category_filter: Обучать только на одной категории мебели
                        None = все категории
                        'chair', 'table', 'bed', 'sofa', 'desk'...
    """
    
    # Размер батча и количество эпох
    batch_size: int = 32
    num_epochs: int = 200
    
    # Оптимизатор AdamW
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
    # Warmup scheduler
    # Первые warmup_epochs эпох LR растёт линейно: 0.01*lr → lr
    # Затем CosineAnnealing: lr → 1e-6
    warmup_epochs: int = 10
    
    # Gradient clipping для стабильности
    grad_clip: float = 1.0
    
    # Сэмплирование точек
    # 4096 = 2048 inside + 2048 outside (примерно)
    num_points: int = 4096
    
    # DataLoader
    num_workers: int = 8      # Потоки для загрузки данных
    pin_memory: bool = True   # Ускоряет передачу на GPU
    
    # Сохранение и валидация
    save_interval: int = 10   # Сохранять каждые 10 эпох
    val_interval: int = 1     # Валидация каждую эпоху
    
    # Разделение данных
    val_split: float = 0.1    # 10% на валидацию
    
    # Воспроизводимость
    seed: int = 42
    
    # Фильтр категории (None = все)
    # Доступные: 'bed', 'bookcase', 'chair', 'desk', 'misc', 
    #            'sofa', 'table', 'tool', 'wardrobe'
    category_filter: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ИНФЕРЕНСА
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InferConfig:
    """
    Параметры генерации 3D моделей (инференс).
    
    Процесс инференса:
        1. Загрузка изображения
        2. Encoder: image → latent vector
        3. Создание 3D сетки точек (resolution³)
        4. Decoder: предсказание occupancy для каждой точки
        5. Marching Cubes: извлечение поверхности (isosurface)
        6. Сохранение меша в .obj файл
    
    Attributes:
        resolution: Разрешение 3D сетки для Marching Cubes
                   64  = быстро, грубая модель
                   128 = баланс качества и скорости
                   256 = высокое качество, медленно
                   
                   Количество точек = resolution³
                   128³ = 2,097,152 точек
        
        threshold: Порог для Marching Cubes (isosurface level)
                  0.5 = стандартный (граница inside/outside)
                  0.3 = больше объём (для тонких объектов)
                  0.7 = меньше объём (убрать артефакты)
        
        batch_size: Размер батча для предсказания occupancy
                   Больше = быстрее, но больше памяти GPU
                   100000 хорошо для 16GB GPU
    """
    
    resolution: int = 128
    threshold: float = 0.5
    batch_size: int = 100000  # Точек за один forward pass


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ КЛАСС КОНФИГУРАЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    Главный класс конфигурации.
    
    Объединяет все подконфиги и предоставляет удобные свойства
    для определения устройства и режима обучения.
    
    Пример использования:
        cfg = get_config()
        
        # Создание модели
        model = create_model(latent_dim=cfg.model.latent_dim)
        model = model.to(cfg.device)
        
        # Создание оптимизатора
        optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate)
        
        # Создание DataLoader
        loader = DataLoader(dataset, batch_size=cfg.train.batch_size)
    """
    
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    
    @property
    def device(self) -> str:
        """
        Автоматическое определение устройства.
        
        Returns:
            'cuda' если доступна GPU, иначе 'cpu'
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def use_amp(self) -> bool:
        """
        Использовать ли Automatic Mixed Precision (FP16).
        
        AMP ускоряет обучение ~2x на современных GPU
        с минимальной потерей точности.
        
        Returns:
            True если доступна CUDA (AMP требует GPU)
        """
        return torch.cuda.is_available()
    
    def print_config(self) -> None:
        """Вывод текущей конфигурации в консоль."""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        
        print("\n[Paths]")
        print(f"  Data root:     {self.paths.data_root}")
        print(f"  Checkpoints:   {self.paths.checkpoint_dir}")
        print(f"  Output:        {self.paths.output_dir}")
        
        print("\n[Model]")
        print(f"  Latent dim:    {self.model.latent_dim}")
        print(f"  Frequencies:   {self.model.num_frequencies}")
        
        print("\n[Training]")
        print(f"  Batch size:    {self.train.batch_size}")
        print(f"  Epochs:        {self.train.num_epochs}")
        print(f"  Learning rate: {self.train.learning_rate}")
        print(f"  Num points:    {self.train.num_points}")
        print(f"  Category:      {self.train.category_filter or 'all'}")
        
        print("\n[Inference]")
        print(f"  Resolution:    {self.infer.resolution}")
        print(f"  Threshold:     {self.infer.threshold}")
        
        print("\n[System]")
        print(f"  Device:        {self.device}")
        print(f"  AMP:           {self.use_amp}")
        
        if self.device == 'cuda':
            print(f"  GPU:           {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory:    {memory_gb:.1f} GB")
        
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР И ФУНКЦИИ ДОСТУПА
# ═══════════════════════════════════════════════════════════════════════════════

# Глобальный синглтон конфигурации
_config = Config()


def get_config() -> Config:
    """
    Получить глобальный экземпляр конфигурации.
    
    Returns:
        Config: глобальный объект конфигурации
    
    Пример:
        cfg = get_config()
        print(cfg.train.batch_size)  # 32
    """
    return _config


def update_config(**kwargs) -> Config:
    """
    Обновить параметры конфигурации.
    
    Автоматически определяет, к какому подконфигу относится параметр.
    
    Args:
        **kwargs: параметры для обновления
    
    Returns:
        Config: обновлённый объект конфигурации
    
    Пример:
        update_config(batch_size=64, latent_dim=256)
        cfg = get_config()
        print(cfg.train.batch_size)  # 64
        print(cfg.model.latent_dim)  # 256
    """
    global _config
    
    for key, value in kwargs.items():
        # Ищем параметр в подконфигах
        if hasattr(_config.model, key):
            setattr(_config.model, key, value)
            print(f"[config.py] model.{key} = {value}")
        elif hasattr(_config.train, key):
            setattr(_config.train, key, value)
            print(f"[config.py] train.{key} = {value}")
        elif hasattr(_config.paths, key):
            setattr(_config.paths, key, value)
            print(f"[config.py] paths.{key} = {value}")
        elif hasattr(_config.infer, key):
            setattr(_config.infer, key, value)
            print(f"[config.py] infer.{key} = {value}")
        else:
            print(f"[config.py] Warning: неизвестный параметр '{key}'")
    
    return _config


def reset_config() -> Config:
    """
    Сбросить конфигурацию к значениям по умолчанию.
    
    Returns:
        Config: новый объект конфигурации с дефолтными значениями
    """
    global _config
    _config = Config()
    return _config


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Тест конфигурации при запуске как скрипта:
        python config.py
    """
    cfg = get_config()
    cfg.print_config()
    
    print("\n[Test] Обновление конфигурации:")
    update_config(batch_size=64, latent_dim=256)
    
    print(f"\n[Test] Новые значения:")
    print(f"  batch_size = {cfg.train.batch_size}")
    print(f"  latent_dim = {cfg.model.latent_dim}")