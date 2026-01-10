"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".
Описание: Функции потерь для обучения, включая метрики соответствия формы и сглаживания сетки.  
Дата: 2026
================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class OccupancyLoss(nn.Module):
    """
    Loss для обучения Occupancy Network.
    
    Основной loss: Binary Cross Entropy
    + опциональная регуляризация
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,  # Вес для позитивного класса
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        
        print(f"[loss.py] OccupancyLoss: pos_weight={pos_weight}, smoothing={label_smoothing}")

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, N] предсказанные logits
            targets: [B, N] ground truth occupancy (0 или 1)
            
        Returns:
            dict с 'total' и отдельными компонентами loss
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # BCE Loss с pos_weight
        pos_weight = torch.tensor([self.pos_weight], device=logits.device)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets,
            pos_weight=pos_weight
        )
        
        # Метрики для логирования
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == targets).float().mean()
            
            # Баланс классов в батче
            pos_ratio = targets.mean()
        
        return {
            'total': bce_loss,
            'bce': bce_loss,
            'accuracy': accuracy,
            'pos_ratio': pos_ratio
        }


class IoULoss(nn.Module):
    """
    Intersection over Union loss.
    Дифференцируемая версия IoU.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, N]
            targets: [B, N]
        """
        probs = torch.sigmoid(logits)
        
        # Intersection
        intersection = (probs * targets).sum(dim=1)
        
        # Union
        union = probs.sum(dim=1) + targets.sum(dim=1) - intersection
        
        # IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Loss = 1 - IoU
        return 1 - iou.mean()


class CombinedLoss(nn.Module):
    """
    Комбинированный loss: BCE + IoU.
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        iou_weight: float = 0.5,
        pos_weight: float = 1.0
    ):
        super().__init__()
        
        self.bce_loss = OccupancyLoss(pos_weight=pos_weight)
        self.iou_loss = IoULoss()
        
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        
        print(f"[loss.py] CombinedLoss: BCE×{bce_weight} + IoU×{iou_weight}")

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        bce_dict = self.bce_loss(logits, targets)
        iou = self.iou_loss(logits, targets)
        
        total = self.bce_weight * bce_dict['bce'] + self.iou_weight * iou
        
        return {
            'total': total,
            'bce': bce_dict['bce'],
            'iou': iou,
            'accuracy': bce_dict['accuracy'],
            'pos_ratio': bce_dict['pos_ratio']
        }