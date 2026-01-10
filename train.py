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
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
from datetime import datetime
from typing import Dict, Tuple
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from config import get_config, Config
from datasets import create_train_val_split, collate_fn
from model import OccupancyNetwork
from loss import CombinedLoss
from metrics import compute_occupancy_metrics, MetricsTracker


class Trainer:
    """Класс для обучения модели."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        
        print(f"[train.py] Устройство: {self.device}")
        
        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[train.py] GPU: {gpu_name}")
        
        # Модель
        self.model = OccupancyNetwork(
            backbone=cfg.model.encoder_type,
            latent_dim=cfg.model.latent_dim,
            hidden_dims=cfg.model.decoder_hidden_dims,
            dropout=cfg.model.decoder_dropout
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[train.py] Параметров: {num_params:,}")
        
        # Loss
        self.criterion = CombinedLoss(
            bce_weight=1.0,
            iou_weight=0.5,
            pos_weight=1.0
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.train.num_epochs,
            eta_min=1e-6
        )
        
        # AMP
        self.use_amp = cfg.use_amp and self.device == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print("[train.py] AMP (FP16) включён")
        else:
            self.scaler = None
        
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.start_epoch = 0

    def load_checkpoint(self, path: str) -> None:
        """Загрузка чекпоинта."""
        if not os.path.exists(path):
            print(f"[train.py] Чекпоинт не найден: {path}")
            return
        
        print(f"[train.py] Загружаю: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_iou = checkpoint.get('best_val_iou', 0.0)
        
        print(f"[train.py] Возобновление с эпохи {self.start_epoch + 1}")

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
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
        }
        
        path = os.path.join(self.cfg.paths.checkpoint_dir, 'latest.pth')
        torch.save(state, path)
        
        if is_best:
            best_path = os.path.join(self.cfg.paths.checkpoint_dir, 'best.pth')
            torch.save(state, best_path)
            print(f"[train.py] ✓ Лучшая модель (IoU: {self.best_val_iou:.4f})")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Обучение одной эпохи."""
        self.model.train()
        
        tracker = MetricsTracker()
        num_batches = 0
        num_skipped = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch in pbar:
            # Пропускаем пустые батчи
            if batch is None:
                num_skipped += 1
                continue
            
            images = batch['image'].to(self.device)
            points = batch['points'].to(self.device)
            occupancies = batch['occupancies'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images, points)
                loss_dict = self.criterion(logits, occupancies)
                loss = loss_dict['total']
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Метрики
            metrics = compute_occupancy_metrics(logits, occupancies)
            tracker.update(metrics, loss.item())
            num_batches += 1
            
            # Прогресс
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics.accuracy:.3f}",
                'iou': f"{metrics.iou:.3f}",
                'skip': num_skipped
            })
        
        if num_batches == 0:
            print(f"[train.py] ⚠️ Все батчи пустые! Проверьте данные.")
            return 0.0, {'accuracy': 0, 'iou': 0}
        
        avg_metrics, avg_loss = tracker.compute()
        
        if num_skipped > 0:
            print(f"[train.py] Пропущено батчей: {num_skipped}/{num_skipped + num_batches}")
        
        return avg_loss, avg_metrics.to_dict()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Валидация."""
        self.model.eval()
        
        tracker = MetricsTracker()
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if batch is None:
                continue
            
            images = batch['image'].to(self.device)
            points = batch['points'].to(self.device)
            occupancies = batch['occupancies'].to(self.device)
            
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

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Основной цикл обучения."""
        
        print("\n" + "="*60)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("="*60)
        
        for epoch in range(self.start_epoch, self.cfg.train.num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            print(f"\nEpoch {epoch + 1}/{self.cfg.train.num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"IoU: {train_metrics['iou']:.4f}")
            
            # Validation каждые N эпох
            if (epoch + 1) % self.cfg.train.val_interval == 0:
                val_loss, val_metrics = self.validate(val_loader)
                
                print(f"  Val   - Loss: {val_loss:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"IoU: {val_metrics['iou']:.4f}")
                
                is_best = val_metrics['iou'] > self.best_val_iou
                if is_best:
                    self.best_val_iou = val_metrics['iou']
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(epoch, is_best)
            else:
                self.save_checkpoint(epoch)
            
            self.scheduler.step()
            
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"  LR: {current_lr:.2e}")
        
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"Лучший Val IoU: {self.best_val_iou:.4f}")
        print("="*60)


def main():
    cfg = get_config()
    
    print("="*60)
    print(f"OCCUPANCY NETWORK TRAINING")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Device: {cfg.device}")
    print(f"Category: {cfg.train.category_filter or 'all'}")
    print(f"Batch size: {cfg.train.batch_size}")
    print(f"Epochs: {cfg.train.num_epochs}")
    print("="*60)
    
    # Seed
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.train.seed)
    
    # Данные
    print("\n[1/3] Загрузка данных...")
    train_dataset, val_dataset = create_train_val_split(
        root_dir=cfg.paths.data_root,
        json_path=cfg.paths.json_path,
        val_split=cfg.train.val_split,
        seed=cfg.train.seed,
        category_filter=cfg.train.category_filter,
        num_points_surface=cfg.train.num_points_surface,
        num_points_uniform=cfg.train.num_points_uniform,
        surface_noise=cfg.train.surface_noise_std
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
    
    # Тест загрузки данных
    print("\n[datasets.py] Тест загрузки...")
    test_count = 0
    for i in range(min(10, len(train_dataset.dataset))):
        sample = train_dataset.dataset[i]
        if sample is not None:
            test_count += 1
    print(f"[datasets.py] Тест: {test_count}/10 образцов загружены успешно")
    
    # Trainer
    print("\n[2/3] Инициализация модели...")
    trainer = Trainer(cfg)
    
    # Чекпоинт
    print("\n[3/3] Проверка чекпоинтов...")
    checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, 'latest.pth')
    trainer.load_checkpoint(checkpoint_path)
    
    # Обучение
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n[train.py] Прервано пользователем")
        trainer.save_checkpoint(trainer.start_epoch, is_best=False)
        print("[train.py] Чекпоинт сохранён")


if __name__ == '__main__':
    main()