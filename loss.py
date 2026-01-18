"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Функции потерь для обучения Occupancy Network
Дата: 2026
================================================================================

Теория Occupancy Networks:
    Occupancy Network предсказывает вероятность того, что точка находится
    ВНУТРИ 3D объекта: o(p) ∈ [0, 1], где p = (x, y, z).
    
    Это задача БИНАРНОЙ КЛАССИФИКАЦИИ для каждой точки:
        - Класс 1 (inside):  точка внутри объекта
        - Класс 0 (outside): точка снаружи объекта

Выбор функции потерь:
    1. Binary Cross-Entropy (BCE) - основной loss
       BCE = -[y·log(p) + (1-y)·log(1-p)]
       где y - ground truth (0 или 1), p - предсказание модели
    
    2. IoU Loss - дополнительный loss для улучшения метрики IoU
       IoU = intersection / union
       IoU_loss = 1 - IoU
    
    Комбинация: total_loss = BCE + λ·IoU_loss
    где λ обычно 0.5

Почему BCE, а не MSE?
    - BCE специально разработан для бинарной классификации
    - Градиенты BCE более стабильны около 0 и 1
    - BCE штрафует уверенные неправильные предсказания сильнее
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class OccupancyLoss(nn.Module):
    """
    Функция потерь для Occupancy Network.
    
    Комбинирует BCE Loss и IoU Loss для лучшей сходимости.
    
    BCE Loss:
        Стандартный loss для бинарной классификации.
        Хорошо работает для обучения, но не напрямую оптимизирует IoU.
    
    IoU Loss:
        Напрямую оптимизирует метрику IoU (Intersection over Union).
        Помогает модели лучше определять границы объекта.
    
    Args:
        bce_weight: Вес BCE loss (по умолчанию 1.0)
        iou_weight: Вес IoU loss (по умолчанию 0.5)
        label_smoothing: Сглаживание меток (0.0 = выключено)
                        Помогает предотвратить overconfident predictions
    
    Пример:
        criterion = OccupancyLoss(bce_weight=1.0, iou_weight=0.5)
        
        logits = model(images, points)  # [B, N]
        targets = batch['occupancies']   # [B, N]
        
        loss_dict = criterion(logits, targets)
        loss = loss_dict['total']
        loss.backward()
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        iou_weight: float = 0.5,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.label_smoothing = label_smoothing
        
        print(f"[loss.py] OccupancyLoss: BCE×{bce_weight} + IoU×{iou_weight}")
        if label_smoothing > 0:
            print(f"[loss.py] Label smoothing: {label_smoothing}")
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисление loss и метрик.
        
        Args:
            logits: [B, N] предсказания модели (ДО sigmoid!)
                   Модель выдаёт logits, а не вероятности.
                   sigmoid применяется внутри BCE loss.
            
            targets: [B, N] ground truth occupancy (0 или 1)
        
        Returns:
            Dict с ключами:
                - 'total': общий loss для backward()
                - 'bce': значение BCE loss
                - 'iou_loss': значение IoU loss
                - 'accuracy': точность классификации
                - 'iou': метрика IoU (для логирования)
        """
        
        # ─────────────────────────────────────────────────────────────────────
        # Label Smoothing (опционально)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Заменяет жёсткие метки (0, 1) на мягкие:
        #   0 → label_smoothing / 2
        #   1 → 1 - label_smoothing / 2
        #
        # Например, при label_smoothing=0.1:
        #   0 → 0.05
        #   1 → 0.95
        #
        # Это помогает модели не быть "слишком уверенной" и улучшает
        # генерализацию.
        # ─────────────────────────────────────────────────────────────────────
        
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # ─────────────────────────────────────────────────────────────────────
        # BCE Loss
        # ─────────────────────────────────────────────────────────────────────
        #
        # binary_cross_entropy_with_logits объединяет sigmoid + BCE:
        #   loss = -[y·log(σ(x)) + (1-y)·log(1-σ(x))]
        #
        # Это численно более стабильно, чем sigmoid + BCE отдельно.
        # ─────────────────────────────────────────────────────────────────────
        
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth)
        
        # ─────────────────────────────────────────────────────────────────────
        # IoU Loss (Soft IoU / Differentiable IoU)
        # ─────────────────────────────────────────────────────────────────────
        #
        # Стандартный IoU не дифференцируем (из-за пороговой операции).
        # Soft IoU использует вероятности вместо бинарных предсказаний:
        #
        #   intersection = Σ(p_i · y_i)        # "мягкое" пересечение
        #   union = Σp_i + Σy_i - intersection  # "мягкое" объединение
        #   soft_iou = intersection / union
        #   iou_loss = 1 - soft_iou
        #
        # Это позволяет градиентам течь и напрямую оптимизировать IoU.
        # ─────────────────────────────────────────────────────────────────────
        
        probs = torch.sigmoid(logits)
        
        # Вычисляем по последней размерности (точки)
        intersection = (probs * targets).sum(dim=-1)
        union = probs.sum(dim=-1) + targets.sum(dim=-1) - intersection
        
        # Усредняем по батчу
        soft_iou = (intersection / (union + 1e-8)).mean()
        iou_loss = 1 - soft_iou
        
        # ─────────────────────────────────────────────────────────────────────
        # Общий Loss
        # ─────────────────────────────────────────────────────────────────────
        
        total = self.bce_weight * bce + self.iou_weight * iou_loss
        
        # ─────────────────────────────────────────────────────────────────────
        # Метрики (без градиентов, только для логирования)
        # ─────────────────────────────────────────────────────────────────────
        
        with torch.no_grad():
            # Бинарные предсказания (порог 0.5)
            preds = (probs > 0.5).float()
            
            # Accuracy: доля правильных предсказаний
            accuracy = (preds == targets).float().mean()
            
            # IoU (Intersection over Union) - основная метрика
            # True Positives: предсказано 1 И ground truth 1
            tp = ((preds == 1) & (targets == 1)).sum().float()
            # False Positives: предсказано 1 НО ground truth 0
            fp = ((preds == 1) & (targets == 0)).sum().float()
            # False Negatives: предсказано 0 НО ground truth 1
            fn = ((preds == 0) & (targets == 1)).sum().float()
            
            iou_metric = tp / (tp + fp + fn + 1e-8)
        
        return {
            'total': total,
            'bce': bce,
            'iou_loss': iou_loss,
            'accuracy': accuracy,
            'iou': iou_metric
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ПРОСТОЙ BCE LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleBCELoss(nn.Module):
    """
    Простой BCE Loss без дополнительных компонентов.
    
    Используется когда нужен минимальный overhead или для отладки.
    
    Args:
        pos_weight: Вес положительного класса (для несбалансированных данных)
                   > 1.0 увеличивает важность inside точек
                   < 1.0 увеличивает важность outside точек
    """
    
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        print(f"[loss.py] SimpleBCELoss: pos_weight={pos_weight}")
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, N] предсказания модели (до sigmoid)
            targets: [B, N] ground truth (0 или 1)
        
        Returns:
            Dict с 'total', 'accuracy', 'iou'
        """
        
        # BCE с весом положительного класса
        if self.pos_weight != 1.0:
            pos_weight = torch.tensor([self.pos_weight], device=logits.device)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Метрики
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            accuracy = (preds == targets).float().mean()
            
            tp = ((preds == 1) & (targets == 1)).sum().float()
            fp = ((preds == 1) & (targets == 0)).sum().float()
            fn = ((preds == 0) & (targets == 1)).sum().float()
            iou = tp / (tp + fp + fn + 1e-8)
        
        return {
            'total': bce,
            'bce': bce,
            'accuracy': accuracy,
            'iou': iou
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss для работы с несбалансированными классами.
    
    Focal Loss уменьшает вклад "лёгких" примеров и фокусируется на "сложных":
        FL(p) = -α · (1-p)^γ · log(p)
    
    где:
        - α (alpha): вес класса
        - γ (gamma): focusing parameter
          γ = 0: эквивалентно BCE
          γ = 2: стандартное значение (хорошо работает)
          γ > 2: ещё сильнее фокусируется на сложных примерах
    
    Когда использовать:
        - Много "лёгких" outside точек (далеко от поверхности)
        - Мало "сложных" точек около границы
        - Несбалансированные классы (больше outside чем inside)
    
    Args:
        alpha: Вес положительного класса (0.25 по умолчанию)
        gamma: Focusing parameter (2.0 по умолчанию)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        print(f"[loss.py] FocalLoss: alpha={alpha}, gamma={gamma}")
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, N] предсказания модели (до sigmoid)
            targets: [B, N] ground truth (0 или 1)
        
        Returns:
            Dict с 'total', 'accuracy', 'iou'
        """
        
        probs = torch.sigmoid(logits)
        
        # ─────────────────────────────────────────────────────────────────────
        # Focal Loss формула:
        #   FL = -α_t · (1 - p_t)^γ · log(p_t)
        #
        # где:
        #   p_t = p если y=1, иначе (1-p)
        #   α_t = α если y=1, иначе (1-α)
        # ─────────────────────────────────────────────────────────────────────
        
        # p_t: вероятность правильного класса
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^γ
        # Чем выше p_t (уверенность в правильном классе), тем меньше вес
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE без редукции
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss
        focal_loss = alpha_t * focal_weight * bce
        focal_loss = focal_loss.mean()
        
        # Метрики
        with torch.no_grad():
            preds = (probs > 0.5).float()
            accuracy = (preds == targets).float().mean()
            
            tp = ((preds == 1) & (targets == 1)).sum().float()
            fp = ((preds == 1) & (targets == 0)).sum().float()
            fn = ((preds == 0) & (targets == 1)).sum().float()
            iou = tp / (tp + fp + fn + 1e-8)
        
        return {
            'total': focal_loss,
            'focal': focal_loss,
            'accuracy': accuracy,
            'iou': iou
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def create_loss(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """
    Factory функция для создания loss.
    
    Args:
        loss_type: Тип loss функции
            - 'combined': BCE + IoU (рекомендуется)
            - 'bce': Только BCE
            - 'focal': Focal Loss
        **kwargs: Дополнительные параметры для конкретного loss
    
    Returns:
        nn.Module: функция потерь
    
    Пример:
        # Стандартный loss
        criterion = create_loss('combined', bce_weight=1.0, iou_weight=0.5)
        
        # Для несбалансированных данных
        criterion = create_loss('focal', alpha=0.25, gamma=2.0)
    """
    
    if loss_type == 'combined':
        return OccupancyLoss(**kwargs)
    elif loss_type == 'bce':
        return SimpleBCELoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    else:
        print(f"[loss.py] Warning: неизвестный тип '{loss_type}', использую 'combined'")
        return OccupancyLoss(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Вычисление метрик для occupancy predictions.
    
    Args:
        logits: [B, N] или [N] предсказания (до sigmoid)
        targets: [B, N] или [N] ground truth
        threshold: Порог для бинаризации (по умолчанию 0.5)
    
    Returns:
        Dict с метриками: accuracy, precision, recall, f1, iou
    """
    
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Flatten для упрощения
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # True/False Positives/Negatives
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        eps = 1e-8
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item()
        }


def compute_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Быстрое вычисление IoU.
    
    Args:
        logits: [B, N] предсказания (до sigmoid)
        targets: [B, N] ground truth
        threshold: Порог бинаризации
    
    Returns:
        torch.Tensor: скалярное значение IoU
    """
    
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        iou = tp / (tp + fp + fn + 1e-8)
        
        return iou


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Тест функций потерь при запуске как скрипта:
        python loss.py
    """
    print("=" * 60)
    print("LOSS FUNCTIONS TEST")
    print("=" * 60)
    
    # Создаём тестовые данные
    torch.manual_seed(42)
    batch_size = 4
    num_points = 1000
    
    # Случайные logits (до sigmoid)
    logits = torch.randn(batch_size, num_points)
    
    # Случайные targets (0 или 1)
    targets = torch.randint(0, 2, (batch_size, num_points)).float()
    
    print(f"\nTest data:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Targets mean: {targets.mean():.3f} (balance)")
    
    # Тест OccupancyLoss
    print("\n[Test 1] OccupancyLoss (BCE + IoU)")
    print("-" * 40)
    
    criterion = OccupancyLoss(bce_weight=1.0, iou_weight=0.5)
    loss_dict = criterion(logits, targets)
    
    print(f"  Total loss:  {loss_dict['total'].item():.4f}")
    print(f"  BCE loss:    {loss_dict['bce'].item():.4f}")
    print(f"  IoU loss:    {loss_dict['iou_loss'].item():.4f}")
    print(f"  Accuracy:    {loss_dict['accuracy'].item():.4f}")
    print(f"  IoU metric:  {loss_dict['iou'].item():.4f}")
    
    # Тест SimpleBCELoss
    print("\n[Test 2] SimpleBCELoss")
    print("-" * 40)
    
    criterion_bce = SimpleBCELoss()
    loss_dict_bce = criterion_bce(logits, targets)
    
    print(f"  Total loss:  {loss_dict_bce['total'].item():.4f}")
    print(f"  Accuracy:    {loss_dict_bce['accuracy'].item():.4f}")
    print(f"  IoU metric:  {loss_dict_bce['iou'].item():.4f}")
    
    # Тест FocalLoss
    print("\n[Test 3] FocalLoss")
    print("-" * 40)
    
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss_dict_focal = criterion_focal(logits, targets)
    
    print(f"  Total loss:  {loss_dict_focal['total'].item():.4f}")
    print(f"  Accuracy:    {loss_dict_focal['accuracy'].item():.4f}")
    print(f"  IoU metric:  {loss_dict_focal['iou'].item():.4f}")
    
    # Тест градиентов
    print("\n[Test 4] Gradient check")
    print("-" * 40)
    
    logits_grad = logits.clone().requires_grad_(True)
    loss = criterion(logits_grad, targets)['total']
    loss.backward()
    
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Gradient shape: {logits_grad.grad.shape}")
    print(f"  Gradient mean: {logits_grad.grad.mean().item():.6f}")
    print(f"  Gradient std: {logits_grad.grad.std().item():.6f}")
    
    # Тест с "идеальными" предсказаниями
    print("\n[Test 5] Perfect predictions")
    print("-" * 40)
    
    # Если модель идеально предсказывает, IoU должен быть ~1
    perfect_logits = targets * 10 - 5  # 1 → 5, 0 → -5
    loss_dict_perfect = criterion(perfect_logits, targets)
    
    print(f"  Total loss:  {loss_dict_perfect['total'].item():.4f}")
    print(f"  Accuracy:    {loss_dict_perfect['accuracy'].item():.4f}")
    print(f"  IoU metric:  {loss_dict_perfect['iou'].item():.4f}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)