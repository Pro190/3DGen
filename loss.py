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
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss для работы с несбалансированными классами.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss (мягкий IoU).
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class IoULoss(nn.Module):
    """
    Intersection over Union loss.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Per-sample IoU
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou.mean()


class OccupancyLoss(nn.Module):
    """
    Основной Loss для обучения Occupancy Network.
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
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
        
        # Метрики
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == targets).float().mean()
            pos_ratio = targets.mean()
        
        return {
            'total': bce_loss,
            'bce': bce_loss,
            'accuracy': accuracy,
            'pos_ratio': pos_ratio
        }


class CombinedLoss(nn.Module):
    """
    Комбинированный loss: BCE + IoU + Focal (опционально).
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        iou_weight: float = 0.5,
        focal_weight: float = 0.0,
        dice_weight: float = 0.0,
        pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        
        self.bce_loss = OccupancyLoss(
            pos_weight=pos_weight,
            label_smoothing=label_smoothing
        )
        self.iou_loss = IoULoss()
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.focal_loss = None
        
        if dice_weight > 0:
            self.dice_loss = DiceLoss()
        else:
            self.dice_loss = None
        
        components = []
        if bce_weight > 0:
            components.append(f"BCE×{bce_weight}")
        if iou_weight > 0:
            components.append(f"IoU×{iou_weight}")
        if focal_weight > 0:
            components.append(f"Focal×{focal_weight}")
        if dice_weight > 0:
            components.append(f"Dice×{dice_weight}")
        
        print(f"[loss.py] CombinedLoss: {' + '.join(components)}")

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        bce_dict = self.bce_loss(logits, targets)
        iou = self.iou_loss(logits, targets)
        
        total = self.bce_weight * bce_dict['bce'] + self.iou_weight * iou
        
        result = {
            'bce': bce_dict['bce'],
            'iou': iou,
            'accuracy': bce_dict['accuracy'],
            'pos_ratio': bce_dict['pos_ratio']
        }
        
        if self.focal_loss is not None:
            focal = self.focal_loss(logits, targets)
            total = total + self.focal_weight * focal
            result['focal'] = focal
        
        if self.dice_loss is not None:
            dice = self.dice_loss(logits, targets)
            total = total + self.dice_weight * dice
            result['dice'] = dice
        
        result['total'] = total
        
        return result


class AdaptiveLoss(nn.Module):
    """
    Адаптивный loss с автоматической настройкой весов.
    """
    
    def __init__(
        self,
        num_losses: int = 3,
        init_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        if init_weights is None:
            init_weights = torch.ones(num_losses)
        
        # Learnable log-weights
        self.log_weights = nn.Parameter(torch.log(init_weights))
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.iou_loss = IoULoss()
        self.dice_loss = DiceLoss()
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        bce = self.bce_loss(logits, targets)
        iou = self.iou_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        weights = torch.exp(-self.log_weights)
        
        total = (
            weights[0] * bce + self.log_weights[0] +
            weights[1] * iou + self.log_weights[1] +
            weights[2] * dice + self.log_weights[2]
        )
        
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == targets).float().mean()
        
        return {
            'total': total,
            'bce': bce,
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'weights': weights.detach()
        }


def create_loss(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """Фабричная функция для создания loss."""
    
    if loss_type == 'bce':
        return OccupancyLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveLoss(**kwargs)
    else:
        raise ValueError(f"Неизвестный тип loss: {loss_type}")