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
    ├── PathConfig   - пути к данным, чекпоинтам, результатам
    ├── ModelConfig  - параметры архитектуры нейросети
    ├── TrainConfig  - параметры обучения (lr, batch_size, epochs...)
    └── InferConfig  - параметры инференса (resolution, threshold...)

Использование:
    from config import get_config, update_config
    
    # Получение текущей конфигурации
    cfg = get_config()
    
    # Обновление параметров (из GUI или командной строки)
    update_config(batch_size=64, learning_rate=1e-4)
    
    # Доступ к параметрам:
    cfg.paths.data_root      # './PIX3D_DATA'
    cfg.model.latent_dim     # 512
    cfg.train.batch_size     # 64 (обновлённое значение)
    cfg.device               # 'cuda' или 'cpu'
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
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
    
    @property
    def preprocessed_index(self) -> str:
        """Путь к индексному файлу препроцессированных данных."""
        return os.path.join(self.preprocessed_dir, 'index.json')


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
        
        use_preprocessed: Использовать препроцессированные данные
                         Ускоряет загрузку данных в ~10-20 раз
    """
    
    # Размер батча и количество эпох
    batch_size: int = 32
    num_epochs: int = 200
    
    # Оптимизатор AdamW
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
    # Warmup scheduler
    warmup_epochs: int = 10
    
    # Gradient clipping для стабильности
    grad_clip: float = 1.0
    
    # Сэмплирование точек
    num_points: int = 4096
    
    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    
    # Сохранение и валидация
    save_interval: int = 10
    val_interval: int = 1
    
    # Разделение данных
    val_split: float = 0.1
    
    # Воспроизводимость
    seed: int = 42
    
    # Фильтр категории (None = все)
    category_filter: Optional[str] = None
    
    # Использовать препроцессированные данные
    use_preprocessed: bool = False


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
        
        threshold: Порог для Marching Cubes (isosurface level)
                  0.5 = стандартный (граница inside/outside)
                  0.3 = больше объём (для тонких объектов)
                  0.7 = меньше объём (убрать артефакты)
        
        batch_size: Размер батча для предсказания occupancy
                   Больше = быстрее, но больше памяти GPU
                   100000 хорошо для 16GB GPU
        
        use_mask: Использовать маску для удаления фона
                 По умолчанию False - генерация только по фото
    """
    
    resolution: int = 128
    threshold: float = 0.5
    batch_size: int = 100000
    use_mask: bool = False  # По умолчанию генерация только по фото


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
        print(f"  Data root:       {self.paths.data_root}")
        print(f"  Checkpoints:     {self.paths.checkpoint_dir}")
        print(f"  Output:          {self.paths.output_dir}")
        print(f"  Preprocessed:    {self.paths.preprocessed_dir}")
        
        print("\n[Model]")
        print(f"  Latent dim:      {self.model.latent_dim}")
        print(f"  Frequencies:     {self.model.num_frequencies}")
        
        print("\n[Training]")
        print(f"  Batch size:      {self.train.batch_size}")
        print(f"  Epochs:          {self.train.num_epochs}")
        print(f"  Learning rate:   {self.train.learning_rate}")
        print(f"  Num points:      {self.train.num_points}")
        print(f"  Category:        {self.train.category_filter or 'all'}")
        print(f"  Use preprocessed:{self.train.use_preprocessed}")
        
        print("\n[Inference]")
        print(f"  Resolution:      {self.infer.resolution}")
        print(f"  Threshold:       {self.infer.threshold}")
        print(f"  Use mask:        {self.infer.use_mask}")
        
        print("\n[System]")
        print(f"  Device:          {self.device}")
        print(f"  AMP:             {self.use_amp}")
        
        if self.device == 'cuda':
            print(f"  GPU:             {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory:      {memory_gb:.1f} GB")
        
        print("=" * 60)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертация конфигурации в словарь.
        
        Полезно для сохранения конфигурации в JSON или передачи в subprocess.
        
        Returns:
            Dict со всеми параметрами
        """
        return {
            'paths': {
                'data_root': self.paths.data_root,
                'checkpoint_dir': self.paths.checkpoint_dir,
                'output_dir': self.paths.output_dir,
                'cache_dir': self.paths.cache_dir,
            },
            'model': {
                'latent_dim': self.model.latent_dim,
                'num_frequencies': self.model.num_frequencies,
            },
            'train': {
                'batch_size': self.train.batch_size,
                'num_epochs': self.train.num_epochs,
                'learning_rate': self.train.learning_rate,
                'weight_decay': self.train.weight_decay,
                'warmup_epochs': self.train.warmup_epochs,
                'grad_clip': self.train.grad_clip,
                'num_points': self.train.num_points,
                'num_workers': self.train.num_workers,
                'save_interval': self.train.save_interval,
                'val_split': self.train.val_split,
                'seed': self.train.seed,
                'category_filter': self.train.category_filter,
                'use_preprocessed': self.train.use_preprocessed,
            },
            'infer': {
                'resolution': self.infer.resolution,
                'threshold': self.infer.threshold,
                'batch_size': self.infer.batch_size,
                'use_mask': self.infer.use_mask,
            }
        }


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
    Эта функция ДОЛЖНА вызываться из GUI перед запуском train.py
    для применения пользовательских параметров.
    
    Args:
        **kwargs: параметры для обновления
    
    Returns:
        Config: обновлённый объект конфигурации
    
    Пример:
        # Из GUI перед запуском обучения:
        update_config(
            batch_size=64,
            learning_rate=1e-4,
            num_epochs=300,
            category_filter='chair'
        )
        
        cfg = get_config()
        print(cfg.train.batch_size)  # 64
    """
    global _config
    
    updated = []
    
    for key, value in kwargs.items():
        # Пропускаем None значения
        if value is None:
            continue
            
        # Ищем параметр в подконфигах
        if hasattr(_config.model, key):
            setattr(_config.model, key, value)
            updated.append(f"model.{key}={value}")
        elif hasattr(_config.train, key):
            setattr(_config.train, key, value)
            updated.append(f"train.{key}={value}")
        elif hasattr(_config.paths, key):
            setattr(_config.paths, key, value)
            updated.append(f"paths.{key}={value}")
        elif hasattr(_config.infer, key):
            setattr(_config.infer, key, value)
            updated.append(f"infer.{key}={value}")
        else:
            print(f"[config.py] ⚠️ Неизвестный параметр: '{key}'")
    
    if updated:
        print(f"[config.py] ✓ Обновлено: {', '.join(updated)}")
    
    return _config


def reset_config() -> Config:
    """
    Сбросить конфигурацию к значениям по умолчанию.
    
    Returns:
        Config: новый объект конфигурации с дефолтными значениями
    """
    global _config
    _config = Config()
    print("[config.py] ✓ Конфигурация сброшена к значениям по умолчанию")
    return _config


def apply_gui_config(gui_config: Dict[str, Any]) -> Config:
    """
    Применить конфигурацию из GUI.
    
    Специальная функция для интеграции с main.py.
    Принимает словарь с параметрами и применяет их.
    
    Args:
        gui_config: Словарь с параметрами из GUI
    
    Returns:
        Config: обновлённый объект конфигурации
    
    Пример (из main.py):
        gui_config = {
            'batch_size': self.batch_spin.value(),
            'num_epochs': self.epochs_spin.value(),
            'learning_rate': float(self.lr_combo.currentText()),
            'category_filter': category if category != 'all' else None,
        }
        apply_gui_config(gui_config)
    """
    return update_config(**gui_config)


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
    
    print("\n[Test] Обновление конфигурации из GUI:")
    update_config(
        batch_size=64,
        latent_dim=256,
        learning_rate=1e-4,
        num_epochs=300,
        category_filter='chair'
    )
    
    print(f"\n[Test] Новые значения:")
    print(f"  batch_size = {cfg.train.batch_size}")
    print(f"  latent_dim = {cfg.model.latent_dim}")
    print(f"  learning_rate = {cfg.train.learning_rate}")
    print(f"  num_epochs = {cfg.train.num_epochs}")
    print(f"  category_filter = {cfg.train.category_filter}")
    
    print("\n[Test] Конвертация в словарь:")
    config_dict = cfg.to_dict()
    print(f"  Ключи: {list(config_dict.keys())}")