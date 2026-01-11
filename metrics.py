"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".
Описание: Метрики для оценки качества Occupancy Networks.
Дата: 2026
================================================================================
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class OccupancyMetrics:
    """Контейнер для метрик occupancy."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    iou: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'iou': self.iou
        }
    
    def __str__(self) -> str:
        return (f"Acc: {self.accuracy:.4f}, "
                f"Prec: {self.precision:.4f}, "
                f"Rec: {self.recall:.4f}, "
                f"F1: {self.f1_score:.4f}, "
                f"IoU: {self.iou:.4f}")
    
    def __add__(self, other: 'OccupancyMetrics') -> 'OccupancyMetrics':
        return OccupancyMetrics(
            accuracy=self.accuracy + other.accuracy,
            precision=self.precision + other.precision,
            recall=self.recall + other.recall,
            f1_score=self.f1_score + other.f1_score,
            iou=self.iou + other.iou
        )
    
    def __truediv__(self, n: int) -> 'OccupancyMetrics':
        return OccupancyMetrics(
            accuracy=self.accuracy / n,
            precision=self.precision / n,
            recall=self.recall / n,
            f1_score=self.f1_score / n,
            iou=self.iou / n
        )


def compute_occupancy_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> OccupancyMetrics:
    """
    Вычисление метрик для occupancy prediction.
    
    Args:
        logits: [B, N] или [N] предсказанные logits
        targets: [B, N] или [N] ground truth (0 или 1)
        threshold: порог для бинаризации
        
    Returns:
        OccupancyMetrics с вычисленными значениями
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # True/False Positives/Negatives
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        eps = 1e-8
        
        # Метрики
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * precision * recall / (precision + recall + eps)
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
    Вычисление метрик для каждого образца в батче.
    
    Args:
        logits: [B, N]
        targets: [B, N]
        
    Returns:
        Список OccupancyMetrics для каждого образца
    """
    batch_size = logits.shape[0]
    metrics_list = []
    
    for i in range(batch_size):
        metrics = compute_occupancy_metrics(logits[i], targets[i], threshold)
        metrics_list.append(metrics)
    
    return metrics_list


class MetricsTracker:
    """Накопитель метрик для эпохи."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._metrics_sum = OccupancyMetrics()
        self._loss_sum = 0.0
        self._count = 0
    
    def update(self, metrics: OccupancyMetrics, loss: float = 0.0):
        self._metrics_sum = self._metrics_sum + metrics
        self._loss_sum += loss
        self._count += 1
    
    def compute(self) -> Tuple[OccupancyMetrics, float]:
        if self._count == 0:
            return OccupancyMetrics(), 0.0
        
        avg_metrics = self._metrics_sum / self._count
        avg_loss = self._loss_sum / self._count
        
        return avg_metrics, avg_loss
    
    @property
    def count(self) -> int:
        return self._count


class EMAMetrics:
    """Exponential Moving Average для метрик."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._ema = None
    
    def update(self, metrics: OccupancyMetrics) -> OccupancyMetrics:
        if self._ema is None:
            self._ema = metrics
        else:
            self._ema = OccupancyMetrics(
                accuracy=self.alpha * metrics.accuracy + (1 - self.alpha) * self._ema.accuracy,
                precision=self.alpha * metrics.precision + (1 - self.alpha) * self._ema.precision,
                recall=self.alpha * metrics.recall + (1 - self.alpha) * self._ema.recall,
                f1_score=self.alpha * metrics.f1_score + (1 - self.alpha) * self._ema.f1_score,
                iou=self.alpha * metrics.iou + (1 - self.alpha) * self._ema.iou
            )
        return self._ema
    
    @property
    def value(self) -> Optional[OccupancyMetrics]:
        return self._ema


# ═══════════════════════════════════════════════════════════════
# Метрики для 3D мешей (после Marching Cubes)
# ═══════════════════════════════════════════════════════════════

def compute_chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray
) -> Dict[str, float]:
    """
    Вычисление Chamfer Distance между двумя облаками точек.
    
    Args:
        pred_points: [N, 3] предсказанные точки
        gt_points: [M, 3] ground truth точки
        
    Returns:
        Словарь с chamfer_l1 и chamfer_l2
    """
    from scipy.spatial import cKDTree
    
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)
    
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    chamfer_l1 = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()
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
    Вычисление F-Score для разных порогов.
    
    Args:
        pred_points: [N, 3]
        gt_points: [M, 3]
        thresholds: пороги расстояния
        
    Returns:
        Словарь с f_score, precision, recall для каждого порога
    """
    from scipy.spatial import cKDTree
    
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)
    
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    metrics = {}
    
    for t in thresholds:
        precision = (dist_pred_to_gt < t).mean()
        recall = (dist_gt_to_pred < t).mean()
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
    
    Args:
        pred_occupancy: [D, H, W] предсказанный occupancy
        gt_occupancy: [D, H, W] ground truth occupancy
        threshold: порог для бинаризации
        
    Returns:
        Volume IoU
    """
    pred_binary = pred_occupancy > threshold
    gt_binary = gt_occupancy > threshold
    
    intersection = (pred_binary & gt_binary).sum()
    union = (pred_binary | gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)