"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Обучение Occupancy Network на датасете PIX3D
Дата: 2026
================================================================================

Процесс обучения Occupancy Network:

    1. ЗАГРУЗКА ДАННЫХ
       - Изображение мебели [B, 3, 224, 224]
       - 3D точки в пространстве [B, N, 3]
       - Ground truth occupancy [B, N] (0=снаружи, 1=внутри)

    2. FORWARD PASS
       - Encoder: изображение → латентный вектор [B, 512]
       - PositionalEncoding: точки [B, N, 3] → [B, N, 63]
       - Decoder: [latent, points_enc] → logits [B, N]

    3. LOSS COMPUTATION
       - BCE Loss: бинарная классификация каждой точки
       - IoU Loss: оптимизация метрики IoU
       - Total = BCE + 0.5 * IoU

    4. BACKWARD PASS
       - Вычисление градиентов
       - Gradient clipping
       - Обновление весов через AdamW

Запуск:
    python train.py
    
    # Или с параметрами (обновляют глобальную конфигурацию):
    python train.py --batch_size 64 --num_epochs 300 --category chair
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import os
import sys
import signal
import argparse
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Импорты из наших модулей
from config import get_config, update_config
from model import create_model
from datasets import Pix3DDataset, PreprocessedPix3DDataset, collate_fn, create_dataset
from loss import OccupancyLoss


# ═══════════════════════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════════

STOP_TRAINING = False


def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown."""
    global STOP_TRAINING
    print("\n" + "=" * 60)
    print("[train.py] Получен сигнал остановки (Ctrl+C или SIGTERM)")
    print("[train.py] Завершаю текущую эпоху и сохраняю чекпоинт...")
    print("=" * 60)
    STOP_TRAINING = True


# Регистрируем обработчики
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# КЛАСС TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Класс для обучения Occupancy Network.
    
    Инкапсулирует:
        - Модель и оптимизатор
        - Цикл обучения и валидации
        - Сохранение/загрузку чекпоинтов
        - Логирование метрик
    
    Args:
        cfg: Объект конфигурации (из config.py)
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        
        # ─────────────────────────────────────────────────────────────────────
        # Логирование информации о системе
        # ─────────────────────────────────────────────────────────────────────
        
        print(f"[train.py] Device: {self.device}")
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[train.py] GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # ─────────────────────────────────────────────────────────────────────
        # Создание модели
        # ─────────────────────────────────────────────────────────────────────
        
        self.model = create_model(
            latent_dim=cfg.model.latent_dim,
            num_frequencies=cfg.model.num_frequencies
        ).to(self.device)
        
        # ─────────────────────────────────────────────────────────────────────
        # Создание Loss функции
        # ─────────────────────────────────────────────────────────────────────
        
        self.criterion = OccupancyLoss(
            bce_weight=1.0,
            iou_weight=0.5
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Создание оптимизатора AdamW
        # ─────────────────────────────────────────────────────────────────────
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.999)
        )
        
        print(f"[train.py] Learning rate: {cfg.train.learning_rate}")
        print(f"[train.py] Batch size: {cfg.train.batch_size}")
        print(f"[train.py] Epochs: {cfg.train.num_epochs}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Создание Learning Rate Scheduler
        # ─────────────────────────────────────────────────────────────────────
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=cfg.train.warmup_epochs
        )
        
        # Cosine scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(cfg.train.num_epochs - cfg.train.warmup_epochs, 1),
            eta_min=1e-6
        )
        
        # Объединяем
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.train.warmup_epochs]
        )
        
        print(f"[train.py] Scheduler: Warmup({cfg.train.warmup_epochs}) + Cosine")
        
        # ─────────────────────────────────────────────────────────────────────
        # Automatic Mixed Precision (AMP)
        # ─────────────────────────────────────────────────────────────────────
        
        self.use_amp = cfg.use_amp
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print("[train.py] AMP (FP16) enabled")
        else:
            self.scaler = None
        
        # ─────────────────────────────────────────────────────────────────────
        # Tracking переменные
        # ─────────────────────────────────────────────────────────────────────
        
        self.best_iou = 0.0
        self.start_epoch = 0
        self.current_epoch = 0
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        is_periodic: bool = False,
        reason: str = ""
    ) -> None:
        """Сохранение чекпоинта модели."""
        
        os.makedirs(self.cfg.paths.checkpoint_dir, exist_ok=True)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'epoch': epoch,
            'best_iou': self.best_iou,
            'config': {
                'latent_dim': self.cfg.model.latent_dim,
                'num_frequencies': self.cfg.model.num_frequencies,
                'type': 'global'
            },
            # Сохраняем полную конфигурацию обучения для воспроизводимости
            'train_config': {
                'batch_size': self.cfg.train.batch_size,
                'learning_rate': self.cfg.train.learning_rate,
                'num_epochs': self.cfg.train.num_epochs,
                'num_points': self.cfg.train.num_points,
                'category_filter': self.cfg.train.category_filter,
            }
        }
        
        # Всегда сохраняем latest
        latest_path = os.path.join(self.cfg.paths.checkpoint_dir, 'latest.pth')
        torch.save(state, latest_path)
        
        if reason:
            print(f"[train.py] 💾 Checkpoint saved: {reason}")
        
        if is_best:
            best_path = os.path.join(self.cfg.paths.checkpoint_dir, 'model.pth')
            torch.save(state, best_path)
            print(f"[train.py] ⭐ Best model saved (IoU: {self.best_iou:.4f})")
        
        if is_periodic:
            periodic_path = os.path.join(
                self.cfg.paths.checkpoint_dir,
                f'epoch_{epoch + 1:03d}.pth'
            )
            torch.save(state, periodic_path)
            print(f"[train.py] 💾 Periodic checkpoint: epoch_{epoch + 1:03d}.pth")
    
    def load_checkpoint(self, path: str) -> bool:
        """Загрузка чекпоинта для возобновления обучения."""
        
        if not os.path.exists(path):
            print(f"[train.py] Checkpoint not found: {path}")
            return False
        
        print(f"[train.py] Loading checkpoint: {path}")
        
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception:
                    print("[train.py] Warning: scheduler state incompatible, resetting")
            
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_iou = checkpoint.get('best_iou', 0.0)
            
            print(f"[train.py] Resuming from epoch {self.start_epoch}")
            print(f"[train.py] Best IoU so far: {self.best_iou:.4f}")
            
            return True
            
        except Exception as e:
            print(f"[train.py] Error loading checkpoint: {e}")
            return False
    
    def train_epoch(self, loader: DataLoader, epoch: int) -> tuple:
        """Обучение одной эпохи."""
        global STOP_TRAINING
        
        self.model.train()
        
        total_loss = 0.0
        total_iou = 0.0
        n_batches = 0
        
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.train.num_epochs}",
            ncols=100,
            leave=False
        )
        
        for batch in pbar:
            if STOP_TRAINING:
                print("\n[train.py] Stopping training loop...")
                break
            
            if batch is None:
                continue
            
            images = batch['image'].to(self.device, non_blocking=True)
            points = batch['points'].to(self.device, non_blocking=True)
            targets = batch['occupancies'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images, points)
                loss_dict = self.criterion(logits, targets)
                loss = loss_dict['total']
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.train.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.train.grad_clip
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            total_iou += loss_dict['iou'].item()
            n_batches += 1
            
            pbar.set_postfix({
                'L': f"{loss.item():.3f}",
                'IoU': f"{loss_dict['iou'].item():.3f}"
            })
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_iou = total_iou / max(n_batches, 1)
        
        return avg_loss, avg_iou
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> tuple:
        """Валидация модели."""
        
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        n_batches = 0
        
        for batch in tqdm(loader, desc="Validation", ncols=100, leave=False):
            if batch is None:
                continue
            
            images = batch['image'].to(self.device, non_blocking=True)
            points = batch['points'].to(self.device, non_blocking=True)
            targets = batch['occupancies'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images, points)
                loss_dict = self.criterion(logits, targets)
            
            total_loss += loss_dict['total'].item()
            total_iou += loss_dict['iou'].item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_iou = total_iou / max(n_batches, 1)
        
        return avg_loss, avg_iou
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> None:
        """Основной цикл обучения."""
        global STOP_TRAINING
        
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print(f"Epochs: {self.start_epoch + 1} → {self.cfg.train.num_epochs}")
        print(f"Batch size: {self.cfg.train.batch_size}")
        print(f"Learning rate: {self.cfg.train.learning_rate}")
        print(f"Category: {self.cfg.train.category_filter or 'all'}")
        print("=" * 60)
        
        for epoch in range(self.start_epoch, self.cfg.train.num_epochs):
            self.current_epoch = epoch
            epoch_start = datetime.now()
            
            if STOP_TRAINING:
                print(f"\n[train.py] Stopping before epoch {epoch + 1}")
                self.save_checkpoint(epoch - 1, reason="Stopped by user")
                break
            
            # Обучение
            train_loss, train_iou = self.train_epoch(train_loader, epoch)
            
            if STOP_TRAINING:
                print(f"\n[train.py] Stopping after epoch {epoch + 1}")
                self.save_checkpoint(epoch, reason="Stopped by user")
                break
            
            # Валидация
            val_loss, val_iou = self.validate(val_loader)
            
            # Логирование
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch + 1}/{self.cfg.train.num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
            print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Сохранение чекпоинтов
            is_best = val_iou > self.best_iou
            if is_best:
                self.best_iou = val_iou
            
            is_periodic = (epoch + 1) % self.cfg.train.save_interval == 0
            
            self.save_checkpoint(
                epoch,
                is_best=is_best,
                is_periodic=is_periodic
            )
            
            # Обновление scheduler
            self.scheduler.step()
            
            sys.stdout.flush()
        
        if not STOP_TRAINING:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print(f"Best Val IoU: {self.best_iou:.4f}")
            print(f"Checkpoints saved to: {self.cfg.paths.checkpoint_dir}")
            print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# ПАРСЕР АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Позволяет переопределять параметры из config.py через командную строку.
    """
    parser = argparse.ArgumentParser(
        description='Train Occupancy Network on PIX3D dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py
    python train.py --batch_size 64 --num_epochs 300
    python train.py --category chair --learning_rate 1e-4
    python train.py --use_preprocessed
        """
    )
    
    # Параметры обучения
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--num_points', type=int, default=None,
                        help='Number of points per sample (default: from config)')
    
    # Параметры модели
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent dimension (default: from config)')
    
    # Данные
    parser.add_argument('--category', type=str, default=None,
                        help='Category filter (chair, table, etc.)')
    parser.add_argument('--use_preprocessed', action='store_true',
                        help='Use preprocessed data for faster loading')
    
    # Пути
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to PIX3D data')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Path to save checkpoints')
    
    # Разное
    parser.add_argument('--save_interval', type=int, default=None,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Главная функция запуска обучения."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Парсинг аргументов и обновление конфигурации
    # ─────────────────────────────────────────────────────────────────────────
    
    args = parse_args()
    
    # Собираем параметры для обновления (только не-None)
    config_updates = {}
    
    if args.batch_size is not None:
        config_updates['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config_updates['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config_updates['learning_rate'] = args.learning_rate
    if args.num_points is not None:
        config_updates['num_points'] = args.num_points
    if args.latent_dim is not None:
        config_updates['latent_dim'] = args.latent_dim
    if args.category is not None:
        config_updates['category_filter'] = args.category if args.category != 'all' else None
    if args.use_preprocessed:
        config_updates['use_preprocessed'] = True
    if args.data_root is not None:
        config_updates['data_root'] = args.data_root
    if args.checkpoint_dir is not None:
        config_updates['checkpoint_dir'] = args.checkpoint_dir
    if args.save_interval is not None:
        config_updates['save_interval'] = args.save_interval
    
    # Применяем обновления
    if config_updates:
        print("[train.py] Применяю параметры командной строки:")
        update_config(**config_updates)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Загрузка конфигурации
    # ─────────────────────────────────────────────────────────────────────────
    
    cfg = get_config()
    
    print("=" * 60)
    print("OCCUPANCY NETWORK TRAINING")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    cfg.print_config()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Установка random seed
    # ─────────────────────────────────────────────────────────────────────────
    
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.train.seed)
        torch.backends.cudnn.benchmark = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Создание датасета
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[1/3] Loading data...")
    
    # Используем factory функцию
    full_dataset = create_dataset(cfg, is_train=True)
    
    # Разделение на train/val
    n_total = len(full_dataset)
    n_val = int(n_total * cfg.train.val_split)
    n_train = n_total - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed)
    )
    
    print(f"[train.py] Train samples: {n_train}")
    print(f"[train.py] Val samples: {n_val}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Создание DataLoader'ов
    # ─────────────────────────────────────────────────────────────────────────
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=max(cfg.train.num_workers // 2, 1),
        collate_fn=collate_fn,
        pin_memory=cfg.train.pin_memory
    )
    
    print(f"[train.py] Train batches: {len(train_loader)}")
    print(f"[train.py] Val batches: {len(val_loader)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Инициализация Trainer
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[2/3] Creating model...")
    trainer = Trainer(cfg)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Загрузка чекпоинта
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[3/3] Checking for existing checkpoint...")
    
    if args.resume:
        # Явно указанный чекпоинт
        trainer.load_checkpoint(args.resume)
    else:
        # Автоматическое возобновление из latest.pth
        checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, 'latest.pth')
        trainer.load_checkpoint(checkpoint_path)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Запуск обучения
    # ─────────────────────────────────────────────────────────────────────────
    
    try:
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"\n[train.py] Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        trainer.save_checkpoint(
            trainer.current_epoch,
            reason=f"Error: {str(e)[:50]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()