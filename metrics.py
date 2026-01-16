"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Метрики для оценки качества Occupancy Network
Дата: 2025
================================================================================

Модуль содержит метрики для оценки качества на двух уровнях:

1. OCCUPANCY METRICS (уровень точек):
   Оценивают, насколько хорошо модель предсказывает occupancy для отдельных точек.
   
   - Accuracy: доля правильно классифицированных точек
   - Precision: доля правильных среди предсказанных "внутри"
   - Recall: доля найденных среди реально "внутри"
   - F1-Score: гармоническое среднее precision и recall
   - IoU: Intersection over Union (главная метрика)

2. MESH METRICS (уровень 3D моделей):
   Оценивают качество сгенерированного 3D меша по сравнению с ground truth.
   
   - Chamfer Distance: среднее расстояние между поверхностями
   - F-Score: доля точек в пределах порога
   - Volume IoU: пересечение объёмов

Использование:
    from metrics import compute_occupancy_metrics, OccupancyMetrics
    
    # Вычисление метрик
    metrics = compute_occupancy_metrics(logits, targets)
    print(f"IoU: {metrics.iou:.4f}")
    
    # Накопление метрик за эпоху
    tracker = MetricsTracker()
    for batch in loader:
        metrics = compute_occupancy_metrics(logits, targets)
        tracker.update(metrics, loss.item())
    avg_metrics, avg_loss = tracker.compute()
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASS ДЛЯ МЕТРИК
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OccupancyMetrics:
    """
    Контейнер для метрик occupancy prediction.
    
    Все метрики находятся в диапазоне [0, 1], где 1 = идеально.
    
    Attributes:
        accuracy: Доля правильно классифицированных точек
                 (TP + TN) / (TP + TN + FP + FN)
        
        precision: Доля правильных среди предсказанных как "внутри"
                  TP / (TP + FP)
                  Высокий precision = мало ложных срабатываний
        
        recall: Доля найденных среди реально находящихся "внутри"
               TP / (TP + FN)
               Высокий recall = мало пропущенных точек
        
        f1_score: Гармоническое среднее precision и recall
                 2 * P * R / (P + R)
                 Баланс между precision и recall
        
        iou: Intersection over Union (Jaccard Index)
            TP / (TP + FP + FN)
            Основная метрика для occupancy prediction
            Более строгая, чем accuracy (не учитывает TN)
    
    Пример:
        metrics = OccupancyMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            iou=0.82
        )
        print(metrics)
        # Acc: 0.9500, Prec: 0.9200, Rec: 0.8800, F1: 0.9000, IoU: 0.8200
    """
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    iou: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Конвертация в словарь."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'iou': self.iou
        }
    
    def __str__(self) -> str:
        """Строковое представление."""
        return (
            f"Acc: {self.accuracy:.4f}, "
            f"Prec: {self.precision:.4f}, "
            f"Rec: {self.recall:.4f}, "
            f"F1: {self.f1_score:.4f}, "
            f"IoU: {self.iou:.4f}"
        )
    
    def __add__(self, other: 'OccupancyMetrics') -> 'OccupancyMetrics':
        """Сложение метрик (для накопления)."""
        return OccupancyMetrics(
            accuracy=self.accuracy + other.accuracy,
            precision=self.precision + other.precision,
            recall=self.recall + other.recall,
            f1_score=self.f1_score + other.f1_score,
            iou=self.iou + other.iou
        )
    
    def __truediv__(self, n: int) -> 'OccupancyMetrics':
        """Деление метрик (для усреднения)."""
        return OccupancyMetrics(
            accuracy=self.accuracy / n,
            precision=self.precision / n,
            recall=self.recall / n,
            f1_score=self.f1_score / n,
            iou=self.iou / n
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ВЫЧИСЛЕНИЕ OCCUPANCY МЕТРИК
# ═══════════════════════════════════════════════════════════════════════════════

def compute_occupancy_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> OccupancyMetrics:
    """
    Вычисление метрик для occupancy prediction.
    
    Принимает logits (до sigmoid) и бинарные targets,
    вычисляет все стандартные метрики бинарной классификации.
    
    Args:
        logits: [B, N] или [N] — предсказания модели (до sigmoid!)
        targets: [B, N] или [N] — ground truth (0 или 1)
        threshold: Порог для бинаризации (по умолчанию 0.5)
    
    Returns:
        OccupancyMetrics со всеми вычисленными значениями
    
    Пример:
        logits = model(images, points)  # [32, 4096]
        targets = batch['occupancies']   # [32, 4096]
        
        metrics = compute_occupancy_metrics(logits, targets)
        print(f"IoU: {metrics.iou:.4f}")
    
    Примечание:
        Функция использует torch.no_grad() для эффективности,
        так как метрики не участвуют в backpropagation.
    """
    
    with torch.no_grad():
        # ─────────────────────────────────────────────────────────────────────
        # Преобразование logits → predictions
        # ─────────────────────────────────────────────────────────────────────
        #
        # sigmoid(logits) → вероятности [0, 1]
        # (probs > threshold) → бинарные предсказания {0, 1}
        # ─────────────────────────────────────────────────────────────────────
        
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Flatten для упрощения вычислений
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # ─────────────────────────────────────────────────────────────────────
        # Confusion Matrix компоненты
        # ─────────────────────────────────────────────────────────────────────
        #
        # True Positive (TP): предсказано 1, реально 1 (правильно внутри)
        # False Positive (FP): предсказано 1, реально 0 (ложное срабатывание)
        # True Negative (TN): предсказано 0, реально 0 (правильно снаружи)
        # False Negative (FN): предсказано 0, реально 1 (пропущенная точка)
        #
        #                  Predicted
        #                  0      1
        #           0     TN     FP
        # Actual
        #           1     FN     TP
        # ─────────────────────────────────────────────────────────────────────
        
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        # Малое число для предотвращения деления на ноль
        eps = 1e-8
        
        # ─────────────────────────────────────────────────────────────────────
        # Вычисление метрик
        # ─────────────────────────────────────────────────────────────────────
        
        # Accuracy: общая доля правильных предсказаний
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        
        # Precision: точность положительных предсказаний
        # "Из всех точек, которые модель назвала 'внутри', сколько реально внутри?"
        precision = tp / (tp + fp + eps)
        
        # Recall (Sensitivity): полнота
        # "Из всех точек, которые реально внутри, сколько модель нашла?"
        recall = tp / (tp + fn + eps)
        
        # F1-Score: гармоническое среднее precision и recall
        # Используется когда важны оба показателя
        f1_score = 2 * precision * recall / (precision + recall + eps)
        
        # IoU (Intersection over Union, Jaccard Index)
        # Самая важная метрика для occupancy prediction
        # IoU = Area of Overlap / Area of Union
        # Не учитывает True Negatives, поэтому более строгая
        iou = tp / (tp + fp + fn + eps)
        
        return OccupancyMetrics(
            accuracy=accuracy.item(),
            precision=precision.item(),
            recall=recall.item(),
            f1_score=f1_score.item(),
            iou=iou.item()
        )


def compute_per_sample_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> List[OccupancyMetrics]:
    """
    Вычисление метрик для каждого образца в батче отдельно.
    
    Полезно для анализа, какие образцы модель обрабатывает хуже.
    
    Args:
        logits: [B, N] — предсказания для батча
        targets: [B, N] — ground truth для батча
        threshold: Порог бинаризации
    
    Returns:
        List[OccupancyMetrics] — метрики для каждого образца
    
    Пример:
        metrics_list = compute_per_sample_metrics(logits, targets)
        
        # Найти худший образец по IoU
        worst_idx = min(range(len(metrics_list)), 
                       key=lambda i: metrics_list[i].iou)
        print(f"Worst sample: {worst_idx}, IoU: {metrics_list[worst_idx].iou}")
    """
    
    batch_size = logits.shape[0]
    metrics_list = []
    
    for i in range(batch_size):
        metrics = compute_occupancy_metrics(
            logits[i],
            targets[i],
            threshold
        )
        metrics_list.append(metrics)
    
    return metrics_list


# ═══════════════════════════════════════════════════════════════════════════════
# БЫСТРЫЕ ФУНКЦИИ ДЛЯ ОТДЕЛЬНЫХ МЕТРИК
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Быстрое вычисление только IoU.
    
    Используется когда нужна только одна метрика (например, в loss).
    
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


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Быстрое вычисление только accuracy.
    
    Args:
        logits: [B, N] предсказания (до sigmoid)
        targets: [B, N] ground truth
        threshold: Порог бинаризации
    
    Returns:
        torch.Tensor: скалярное значение accuracy
    """
    
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        accuracy = (preds == targets).float().mean()
        return accuracy


def compute_precision_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Быстрое вычисление precision и recall.
    
    Args:
        logits: [B, N] предсказания (до sigmoid)
        targets: [B, N] ground truth
        threshold: Порог бинаризации
    
    Returns:
        Tuple[precision, recall]
    """
    
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return precision, recall


# ═══════════════════════════════════════════════════════════════════════════════
# ТРЕКЕР МЕТРИК
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """
    Накопитель метрик для эпохи или валидации.
    
    Позволяет накапливать метрики по батчам и вычислять
    средние значения в конце эпохи.
    
    Пример:
        tracker = MetricsTracker()
        
        for batch in train_loader:
            logits = model(images, points)
            loss = criterion(logits, targets)
            
            metrics = compute_occupancy_metrics(logits, targets)
            tracker.update(metrics, loss.item())
        
        avg_metrics, avg_loss = tracker.compute()
        print(f"Epoch avg - Loss: {avg_loss:.4f}, {avg_metrics}")
        
        tracker.reset()  # Для следующей эпохи
    """
    
    def __init__(self):
        """Инициализация трекера."""
        self.reset()
    
    def reset(self) -> None:
        """
        Сброс накопленных значений.
        
        Вызывается в начале каждой эпохи.
        """
        self._metrics_sum = OccupancyMetrics()
        self._loss_sum = 0.0
        self._count = 0
    
    def update(
        self,
        metrics: OccupancyMetrics,
        loss: float = 0.0
    ) -> None:
        """
        Добавление метрик одного батча.
        
        Args:
            metrics: Метрики текущего батча
            loss: Значение loss текущего батча (опционально)
        """
        self._metrics_sum = self._metrics_sum + metrics
        self._loss_sum += loss
        self._count += 1
    
    def compute(self) -> Tuple[OccupancyMetrics, float]:
        """
        Вычисление средних значений.
        
        Returns:
            Tuple[avg_metrics, avg_loss]
        """
        if self._count == 0:
            return OccupancyMetrics(), 0.0
        
        avg_metrics = self._metrics_sum / self._count
        avg_loss = self._loss_sum / self._count
        
        return avg_metrics, avg_loss
    
    @property
    def count(self) -> int:
        """Количество накопленных батчей."""
        return self._count


class EMAMetrics:
    """
    Exponential Moving Average для метрик.
    
    Сглаживает метрики для более стабильного отображения прогресса.
    Полезно для отображения в progress bar.
    
    Args:
        alpha: Коэффициент сглаживания (0-1)
              Меньше alpha = более сильное сглаживание
              0.1 = учитываем 10% нового значения
    
    Пример:
        ema = EMAMetrics(alpha=0.1)
        
        for batch in loader:
            metrics = compute_occupancy_metrics(logits, targets)
            smoothed = ema.update(metrics)
            
            pbar.set_postfix({'IoU': f'{smoothed.iou:.3f}'})
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Коэффициент сглаживания (0-1)
        """
        self.alpha = alpha
        self._ema: Optional[OccupancyMetrics] = None
    
    def update(self, metrics: OccupancyMetrics) -> OccupancyMetrics:
        """
        Обновление EMA новыми метриками.
        
        Формула: EMA_new = alpha * value + (1 - alpha) * EMA_old
        
        Args:
            metrics: Новые метрики
        
        Returns:
            Сглаженные метрики
        """
        if self._ema is None:
            # Первое значение — инициализация
            self._ema = metrics
        else:
            # Экспоненциальное усреднение
            self._ema = OccupancyMetrics(
                accuracy=self.alpha * metrics.accuracy + (1 - self.alpha) * self._ema.accuracy,
                precision=self.alpha * metrics.precision + (1 - self.alpha) * self._ema.precision,
                recall=self.alpha * metrics.recall + (1 - self.alpha) * self._ema.recall,
                f1_score=self.alpha * metrics.f1_score + (1 - self.alpha) * self._ema.f1_score,
                iou=self.alpha * metrics.iou + (1 - self.alpha) * self._ema.iou
            )
        
        return self._ema
    
    def reset(self) -> None:
        """Сброс EMA."""
        self._ema = None
    
    @property
    def value(self) -> Optional[OccupancyMetrics]:
        """Текущее значение EMA."""
        return self._ema


# ═══════════════════════════════════════════════════════════════════════════════
# МЕТРИКИ ДЛЯ 3D МЕШЕЙ
# ═══════════════════════════════════════════════════════════════════════════════

def compute_chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray
) -> Dict[str, float]:
    """
    Вычисление Chamfer Distance между двумя облаками точек.
    
    Chamfer Distance — симметричная мера расстояния между двумя
    множествами точек. Для каждой точки одного множества находится
    ближайшая точка другого множества.
    
    CD = mean(dist(pred→gt)) + mean(dist(gt→pred))
    
    Args:
        pred_points: [N, 3] предсказанные точки
        gt_points: [M, 3] ground truth точки
    
    Returns:
        Dict с:
            - chamfer_l1: сумма средних L1 расстояний
            - chamfer_l2: сумма средних L2 расстояний (квадраты)
    
    Пример:
        pred_pts = sample_points_from_mesh(pred_mesh, 10000)
        gt_pts = sample_points_from_mesh(gt_mesh, 10000)
        
        cd = compute_chamfer_distance(pred_pts, gt_pts)
        print(f"Chamfer L1: {cd['chamfer_l1']:.4f}")
    """
    from scipy.spatial import cKDTree
    
    # ─────────────────────────────────────────────────────────────────────────
    # Построение KD-деревьев
    # ─────────────────────────────────────────────────────────────────────────
    #
    # KD-дерево — структура данных для быстрого поиска ближайших соседей.
    # Поиск за O(log N) вместо O(N).
    # ─────────────────────────────────────────────────────────────────────────
    
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Вычисление расстояний
    # ─────────────────────────────────────────────────────────────────────────
    
    # Для каждой pred точки находим расстояние до ближайшей gt точки
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    
    # Для каждой gt точки находим расстояние до ближайшей pred точки
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Chamfer Distance
    # ─────────────────────────────────────────────────────────────────────────
    
    # L1: сумма средних расстояний
    chamfer_l1 = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()
    
    # L2: сумма средних квадратов расстояний
    # Более чувствителен к выбросам (большим ошибкам)
    chamfer_l2 = (dist_pred_to_gt ** 2).mean() + (dist_gt_to_pred ** 2).mean()
    
    return {
        'chamfer_l1': float(chamfer_l1),
        'chamfer_l2': float(chamfer_l2)
    }


def compute_f_score(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    thresholds: Tuple[float, ...] = (0.01, 0.02, 0.05)
) -> Dict[str, float]:
    """
    Вычисление F-Score для разных порогов расстояния.
    
    F-Score показывает, какая доля точек находится в пределах
    заданного порогового расстояния.
    
    - Precision: доля pred точек близких к gt
    - Recall: доля gt точек близких к pred
    - F-Score: гармоническое среднее
    
    Args:
        pred_points: [N, 3] предсказанные точки
        gt_points: [M, 3] ground truth точки
        thresholds: Пороговые расстояния (например, 0.01 = 1% от размера)
    
    Returns:
        Dict с f_score, precision, recall для каждого порога
    
    Пример:
        metrics = compute_f_score(pred_pts, gt_pts)
        print(f"F-Score@0.02: {metrics['f_score@0.02']:.4f}")
    """
    from scipy.spatial import cKDTree
    
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)
    
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    metrics = {}
    
    for t in thresholds:
        # Precision: какая доля pred точек близка к gt
        precision = (dist_pred_to_gt < t).mean()
        
        # Recall: какая доля gt точек близка к pred
        recall = (dist_gt_to_pred < t).mean()
        
        # F-Score
        f_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics[f'f_score@{t}'] = float(f_score)
        metrics[f'precision@{t}'] = float(precision)
        metrics[f'recall@{t}'] = float(recall)
    
    return metrics


def compute_mesh_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    thresholds: Tuple[float, ...] = (0.01, 0.02, 0.05)
) -> Dict[str, float]:
    """
    Полный набор метрик для сравнения мешей.
    
    Объединяет Chamfer Distance и F-Score.
    
    Args:
        pred_points: [N, 3] точки с предсказанного меша
        gt_points: [M, 3] точки с ground truth меша
        thresholds: Пороги для F-Score
    
    Returns:
        Dict со всеми метриками
    """
    
    metrics = {}
    
    # Chamfer Distance
    chamfer = compute_chamfer_distance(pred_points, gt_points)
    metrics.update(chamfer)
    
    # F-Score
    f_scores = compute_f_score(pred_points, gt_points, thresholds)
    metrics.update(f_scores)
    
    return metrics


def compute_volume_iou(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Вычисление Volume IoU между двумя occupancy grids.
    
    Воксельное представление объектов сравнивается по IoU.
    
    Args:
        pred_occupancy: [D, H, W] предсказанный occupancy volume
        gt_occupancy: [D, H, W] ground truth occupancy volume
        threshold: Порог для бинаризации
    
    Returns:
        Volume IoU (0 - 1)
    
    Пример:
        # Создаём воксельные представления
        pred_voxels = voxelize_mesh(pred_mesh, resolution=32)
        gt_voxels = voxelize_mesh(gt_mesh, resolution=32)
        
        vol_iou = compute_volume_iou(pred_voxels, gt_voxels)
        print(f"Volume IoU: {vol_iou:.4f}")
    """
    
    pred_binary = pred_occupancy > threshold
    gt_binary = gt_occupancy > threshold
    
    intersection = (pred_binary & gt_binary).sum()
    union = (pred_binary | gt_binary).sum()
    
    if union == 0:
        # Оба пустые — считаем идеальное совпадение
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Тест метрик при запуске как скрипта:
        python metrics.py
    """
    
    print("=" * 60)
    print("METRICS TEST")
    print("=" * 60)
    
    # Создаём тестовые данные
    torch.manual_seed(42)
    batch_size = 4
    num_points = 1000
    
    # Случайные logits и targets
    logits = torch.randn(batch_size, num_points)
    targets = torch.randint(0, 2, (batch_size, num_points)).float()
    
    print(f"\nTest data:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Targets balance: {targets.mean():.2%} inside")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Тест compute_occupancy_metrics
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[Test 1] compute_occupancy_metrics")
    print("-" * 40)
    
    metrics = compute_occupancy_metrics(logits, targets)
    print(f"  {metrics}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Тест с идеальными предсказаниями
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[Test 2] Perfect predictions")
    print("-" * 40)
    
    # Если logits точно соответствуют targets, IoU ≈ 1
    perfect_logits = targets * 10 - 5  # 1 → 5, 0 → -5
    perfect_metrics = compute_occupancy_metrics(perfect_logits, targets)
    print(f"  {perfect_metrics}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Тест MetricsTracker
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[Test 3] MetricsTracker")
    print("-" * 40)
    
    tracker = MetricsTracker()
    
    for i in range(5):
        batch_logits = torch.randn(4, 1000)
        batch_targets = torch.randint(0, 2, (4, 1000)).float()
        batch_metrics = compute_occupancy_metrics(batch_logits, batch_targets)
        tracker.update(batch_metrics, loss=0.5 + i * 0.1)
    
    avg_metrics, avg_loss = tracker.compute()
    print(f"  Batches: {tracker.count}")
    print(f"  Avg loss: {avg_loss:.4f}")
    print(f"  Avg metrics: {avg_metrics}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Тест per-sample metrics
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[Test 4] Per-sample metrics")
    print("-" * 40)
    
    per_sample = compute_per_sample_metrics(logits, targets)
    print(f"  Samples: {len(per_sample)}")
    for i, m in enumerate(per_sample):
        print(f"  Sample {i}: IoU={m.iou:.4f}, Acc={m.accuracy:.4f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Тест EMAMetrics
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[Test 5] EMAMetrics")
    print("-" * 40)
    
    ema = EMAMetrics(alpha=0.3)
    
    for i in range(5):
        batch_metrics = compute_occupancy_metrics(
            torch.randn(4, 1000),
            torch.randint(0, 2, (4, 1000)).float()
        )
        smoothed = ema.update(batch_metrics)
        print(f"  Step {i + 1}: IoU={smoothed.iou:.4f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Тест mesh metrics
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[Test 6] Mesh metrics (Chamfer Distance)")
    print("-" * 40)
    
    # Случайные облака точек
    np.random.seed(42)
    pred_points = np.random.randn(1000, 3).astype(np.float32) * 0.1
    gt_points = np.random.randn(1000, 3).astype(np.float32) * 0.1 + 0.05
    
    mesh_metrics = compute_mesh_metrics(pred_points, gt_points)
    print(f"  Chamfer L1: {mesh_metrics['chamfer_l1']:.4f}")
    print(f"  Chamfer L2: {mesh_metrics['chamfer_l2']:.4f}")
    print(f"  F-Score@0.01: {mesh_metrics['f_score@0.01']:.4f}")
    print(f"  F-Score@0.02: {mesh_metrics['f_score@0.02']:.4f}")
    print(f"  F-Score@0.05: {mesh_metrics['f_score@0.05']:.4f}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)