"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".
Описание: Обучение системы: основной цикл тренировки нейросети и работа с весами.
Дата: 2026
================================================================================
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import sys
import signal
from datetime import datetime
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

from config import get_config, Config
from datasets import create_datasets, collate_fn
from model import OccupancyNetwork, create_model, AVAILABLE_ENCODERS
from loss import CombinedLoss, create_loss
from metrics import compute_occupancy_metrics, MetricsTracker


# Глобальный флаг для graceful shutdown
STOP_TRAINING = False


def signal_handler(signum, frame):
    """Обработчик сигнала для graceful shutdown."""
    global STOP_TRAINING
    print("\n[train.py] Получен сигнал остановки, завершаю текущую эпоху...")
    STOP_TRAINING = True


# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_gui_config() -> Optional[Dict]:
    """Загрузка конфигурации из GUI."""
    config_path = os.environ.get('TRAIN_CONFIG')
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                gui_config = json.load(f)
            print(f"[train.py] Загружена конфигурация из GUI: {config_path}")
            return gui_config
        except Exception as e:
            print(f"[train.py] Ошибка загрузки GUI конфига: {e}")
    
    return None


def apply_gui_config(cfg: Config, gui_config: Dict) -> Config:
    """Применение настроек из GUI к конфигу."""
    
    # Model settings
    if 'encoder_type' in gui_config:
        cfg.model.encoder_type = gui_config['encoder_type']
        print(f"[train.py] Encoder: {cfg.model.encoder_type}")
    
    if 'latent_dim' in gui_config:
        cfg.model.latent_dim = gui_config['latent_dim']
        print(f"[train.py] Latent dim: {cfg.model.latent_dim}")
    
    # Training settings
    if 'num_epochs' in gui_config:
        cfg.train.num_epochs = gui_config['num_epochs']
        print(f"[train.py] Epochs: {cfg.train.num_epochs}")
    
    if 'batch_size' in gui_config:
        cfg.train.batch_size = gui_config['batch_size']
        print(f"[train.py] Batch size: {cfg.train.batch_size}")
    
    if 'learning_rate' in gui_config:
        cfg.train.learning_rate = float(gui_config['learning_rate'])
        print(f"[train.py] Learning rate: {cfg.train.learning_rate}")
    
    if 'category' in gui_config:
        cfg.train.category_filter = gui_config['category']
        print(f"[train.py] Category: {cfg.train.category_filter or 'all'}")
    
    if 'save_interval' in gui_config:
        cfg.train.save_interval = gui_config['save_interval']
        print(f"[train.py] Save interval: {cfg.train.save_interval}")
    
    if 'use_augmentation' in gui_config:
        cfg.train.use_augmentation = gui_config['use_augmentation']
        print(f"[train.py] Augmentation: {cfg.train.use_augmentation}")
    
    return cfg


class Trainer:
    """Класс для обучения модели."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        
        print(f"[train.py] Устройство: {self.device}")
        
        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[train.py] GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Модель
        self.model = create_model(
            encoder_type=cfg.model.encoder_type,
            latent_dim=cfg.model.latent_dim,
            hidden_dims=cfg.model.decoder_hidden_dims,
            dropout=cfg.model.decoder_dropout,
            use_residual=cfg.model.decoder_use_residual,
            use_layer_norm=cfg.model.decoder_use_layer_norm,
            use_positional_encoding=True,
            pretrained=cfg.model.encoder_pretrained,
            freeze_bn=cfg.model.encoder_freeze_bn
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[train.py] Параметров: {num_params:,}")
        
        # Loss
        self.criterion = create_loss(
            loss_type='combined',
            bce_weight=cfg.train.bce_weight,
            iou_weight=cfg.train.iou_weight,
            pos_weight=1.0,
            label_smoothing=0.01
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler с warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=cfg.train.warmup_lr / cfg.train.learning_rate,
            end_factor=1.0,
            total_iters=cfg.train.warmup_epochs
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.train.num_epochs - cfg.train.warmup_epochs,
            eta_min=1e-6
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[cfg.train.warmup_epochs]
        )
        
        print(f"[train.py] Warmup: {cfg.train.warmup_epochs} эпох")
        
        # AMP
        self.use_amp = cfg.use_amp and self.device == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print("[train.py] AMP (FP16) включён")
        else:
            self.scaler = None
        
        # Gradient clipping
        self.grad_clip = cfg.train.grad_clip
        if self.grad_clip > 0:
            print(f"[train.py] Gradient clipping: {self.grad_clip}")
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.start_epoch = 0
        self.current_epoch = 0
        self.train_history = []
        self.val_history = []

    def load_checkpoint(self, path: str) -> bool:
        """Загрузка чекпоинта."""
        if not os.path.exists(path):
            print(f"[train.py] Чекпоинт не найден: {path}")
            return False
        
        print(f"[train.py] Загружаю: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("[train.py] ⚠️ Не удалось загрузить scheduler state")
        
        if 'scaler_state_dict' in checkpoint and self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_iou = checkpoint.get('best_val_iou', 0.0)
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        print(f"[train.py] Возобновление с эпохи {self.start_epoch}")
        return True

    def save_checkpoint(
        self, 
        epoch: int, 
        is_best: bool = False,
        is_periodic: bool = False,
        reason: str = ""
    ) -> None:
        """Сохранение чекпоинта."""
        os.makedirs(self.cfg.paths.checkpoint_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': {
                'encoder_type': self.cfg.model.encoder_type,
                'latent_dim': self.cfg.model.latent_dim,
                'hidden_dims': self.cfg.model.decoder_hidden_dims
            }
        }
        
        # Всегда сохраняем latest
        path = os.path.join(self.cfg.paths.checkpoint_dir, 'latest.pth')
        torch.save(state, path)
        
        if reason:
            print(f"[train.py] 💾 Сохранён чекпоинт: {reason}")
        
        # Сохраняем лучшую модель
        if is_best:
            best_path = os.path.join(self.cfg.paths.checkpoint_dir, 'best.pth')
            torch.save(state, best_path)
            print(f"[train.py] ✓ Лучшая модель (IoU: {self.best_val_iou:.4f})")
        
        # Периодическое сохранение (каждые N эпох)
        if is_periodic:
            periodic_path = os.path.join(
                self.cfg.paths.checkpoint_dir, 
                f'epoch_{epoch + 1:03d}.pth'
            )
            torch.save(state, periodic_path)
            print(f"[train.py] 💾 Сохранён чекпоинт эпохи {epoch + 1}")
        
        # Сохраняем историю обучения
        history_path = os.path.join(self.cfg.paths.checkpoint_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)

    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Обучение одной эпохи."""
        global STOP_TRAINING
        
        self.model.train()
        
        tracker = MetricsTracker()
        num_batches = 0
        num_skipped = 0
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{self.cfg.train.num_epochs}", 
            leave=False,
            ncols=100
        )
        
        for batch in pbar:
            # Проверка на остановку
            if STOP_TRAINING:
                print("\n[train.py] Остановка обучения...")
                break
            
            if batch is None:
                num_skipped += 1
                continue
            
            images = batch['image'].to(self.device, non_blocking=True)
            points = batch['points'].to(self.device, non_blocking=True)
            occupancies = batch['occupancies'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images, points)
                loss_dict = self.criterion(logits, occupancies)
                loss = loss_dict['total']
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.grad_clip
                    )
                
                self.optimizer.step()
            
            # Метрики
            metrics = compute_occupancy_metrics(logits, occupancies)
            tracker.update(metrics, loss.item())
            num_batches += 1
            
            # Прогресс
            pbar.set_postfix({
                'L': f"{loss.item():.3f}",
                'Acc': f"{metrics.accuracy:.3f}",
                'IoU': f"{metrics.iou:.3f}"
            })
        
        if num_batches == 0:
            print(f"[train.py] ⚠️ Все батчи пустые!")
            return 0.0, {'accuracy': 0, 'iou': 0}
        
        avg_metrics, avg_loss = tracker.compute()
        
        return avg_loss, avg_metrics.to_dict()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Валидация."""
        self.model.eval()
        
        tracker = MetricsTracker()
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validation", leave=False, ncols=100):
            if batch is None:
                continue
            
            images = batch['image'].to(self.device, non_blocking=True)
            points = batch['points'].to(self.device, non_blocking=True)
            occupancies = batch['occupancies'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images, points)
                loss_dict = self.criterion(logits, occupancies)
            
            metrics = compute_occupancy_metrics(logits, occupancies)
            tracker.update(metrics, loss_dict['total'].item())
            num_batches += 1
        
        if num_batches == 0:
            return 0.0, {'accuracy': 0, 'iou': 0}
        
        avg_metrics, avg_loss = tracker.compute()
        return avg_loss, avg_metrics.to_dict()

    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> None:
        """Основной цикл обучения."""
        global STOP_TRAINING
        
        print("\n" + "="*60)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print(f"Encoder: {self.cfg.model.encoder_type}")
        print(f"Latent dim: {self.cfg.model.latent_dim}")
        print(f"Эпохи: {self.start_epoch + 1} → {self.cfg.train.num_epochs}")
        print("="*60)
        
        save_interval = self.cfg.train.save_interval
        
        for epoch in range(self.start_epoch, self.cfg.train.num_epochs):
            self.current_epoch = epoch
            
            # Проверка на остановку перед эпохой
            if STOP_TRAINING:
                print(f"\n[train.py] Остановка перед эпохой {epoch + 1}")
                self.save_checkpoint(epoch, is_best=False, reason="Остановлено пользователем")
                break
            
            epoch_start = datetime.now()
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Проверка на остановку после эпохи
            if STOP_TRAINING:
                print(f"\n[train.py] Остановка после эпохи {epoch + 1}")
                self.save_checkpoint(epoch, is_best=False, reason="Остановлено пользователем")
                break
            
            # Сохраняем историю
            self.train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                **train_metrics
            })
            
            print(f"\nEpoch {epoch + 1}/{self.cfg.train.num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"IoU: {train_metrics['iou']:.4f}")
            
            # Validation
            is_val_epoch = (epoch + 1) % self.cfg.train.val_interval == 0
            is_last_epoch = (epoch + 1) == self.cfg.train.num_epochs
            
            if is_val_epoch or is_last_epoch:
                val_loss, val_metrics = self.validate(val_loader)
                
                self.val_history.append({
                    'epoch': epoch + 1,
                    'loss': val_loss,
                    **val_metrics
                })
                
                print(f"  Val   - Loss: {val_loss:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"IoU: {val_metrics['iou']:.4f}")
                
                is_best = val_metrics['iou'] > self.best_val_iou
                if is_best:
                    self.best_val_iou = val_metrics['iou']
                    self.best_val_loss = val_loss
            else:
                is_best = False
            
            # Периодическое сохранение каждые N эпох
            is_periodic = (epoch + 1) % save_interval == 0
            
            self.save_checkpoint(epoch, is_best=is_best, is_periodic=is_periodic)
            
            # Scheduler step
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            
            print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Flush stdout для GUI
            sys.stdout.flush()
        
        if not STOP_TRAINING:
            print("\n" + "="*60)
            print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
            print(f"Лучший Val IoU: {self.best_val_iou:.4f}")
            print(f"Чекпоинты сохранены в: {self.cfg.paths.checkpoint_dir}")
            print("="*60)


def main():
    # Загружаем базовый конфиг
    cfg = get_config()
    
    # Проверяем наличие конфига из GUI
    gui_config = load_gui_config()
    if gui_config:
        cfg = apply_gui_config(cfg, gui_config)
    
    print("="*60)
    print("OCCUPANCY NETWORK TRAINING")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Device: {cfg.device}")
    print(f"Encoder: {cfg.model.encoder_type}")
    print(f"Latent dim: {cfg.model.latent_dim}")
    print(f"Decoder: {cfg.model.decoder_hidden_dims}")
    print(f"Category: {cfg.train.category_filter or 'all'}")
    print(f"Batch size: {cfg.train.batch_size}")
    print(f"Learning rate: {cfg.train.learning_rate}")
    print(f"Epochs: {cfg.train.num_epochs}")
    print(f"Save interval: every {cfg.train.save_interval} epochs")
    print("="*60)
    sys.stdout.flush()
    
    # Seed
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.train.seed)
        torch.backends.cudnn.benchmark = True
    
    # Проверка на препроцессированные данные
    preprocessed_index = os.path.join(cfg.paths.preprocessed_dir, 'index.json')
    use_preprocessed = gui_config.get('use_preprocessed', True) if gui_config else True
    
    # Данные
    print("\n[1/3] Загрузка данных...")
    sys.stdout.flush()
    
    train_dataset, val_dataset = create_datasets(
        root_dir=cfg.paths.data_root,
        json_path=cfg.paths.json_path,
        preprocessed_index=preprocessed_index if (use_preprocessed and os.path.exists(preprocessed_index)) else None,
        val_split=cfg.train.val_split,
        seed=cfg.train.seed,
        category_filter=cfg.train.category_filter,
        num_points_surface=cfg.train.num_points_surface,
        num_points_uniform=cfg.train.num_points_uniform,
        surface_noise=cfg.train.surface_noise_std,
        use_augmentation=cfg.train.use_augmentation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True if cfg.train.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.train.num_workers > 0 else False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    sys.stdout.flush()
    
    # Trainer
    print("\n[2/3] Инициализация модели...")
    sys.stdout.flush()
    
    trainer = Trainer(cfg)
    
    # Чекпоинт (только если не новая конфигурация)
    print("\n[3/3] Проверка чекпоинтов...")
    sys.stdout.flush()
    
    checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, 'latest.pth')
    
    # Если конфиг из GUI и encoder изменился - не загружаем старый чекпоинт
    if gui_config and os.path.exists(checkpoint_path):
        try:
            old_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            old_encoder = old_checkpoint.get('config', {}).get('encoder_type', 'resnet18')
            new_encoder = cfg.model.encoder_type
            
            if old_encoder != new_encoder:
                print(f"[train.py] Encoder изменился ({old_encoder} → {new_encoder}), начинаю с нуля")
            else:
                trainer.load_checkpoint(checkpoint_path)
        except:
            pass
    else:
        trainer.load_checkpoint(checkpoint_path)
    
    # Обучение
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n[train.py] Прервано пользователем (Ctrl+C)")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False, reason="Прервано (Ctrl+C)")
    except Exception as e:
        print(f"\n[train.py] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint(trainer.current_epoch, is_best=False, reason=f"Ошибка: {e}")
    finally:
        # Удаляем временный конфиг
        config_path = os.environ.get('TRAIN_CONFIG')
        if config_path and os.path.exists(config_path):
            try:
                os.remove(config_path)
            except:
                pass


if __name__ == '__main__':
    main()