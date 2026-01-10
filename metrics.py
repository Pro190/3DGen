"""
Метрики для оценки качества Occupancy Networks.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OccupancyMetrics:
    """Контейнер для метрик occupancy."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    iou: float
    
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
        
        # Flatten для простоты
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # True/False Positives/Negatives
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        # Метрики
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        
        return OccupancyMetrics(
            accuracy=accuracy.item(),
            precision=precision.item(),
            recall=recall.item(),
            f1_score=f1_score.item(),
            iou=iou.item()
        )


class MetricsTracker:
    """Накопитель метрик для эпохи."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._accuracy_sum = 0.0
        self._precision_sum = 0.0
        self._recall_sum = 0.0
        self._f1_sum = 0.0
        self._iou_sum = 0.0
        self._loss_sum = 0.0
        self._count = 0
    
    def update(self, metrics: OccupancyMetrics, loss: float = 0.0):
        self._accuracy_sum += metrics.accuracy
        self._precision_sum += metrics.precision
        self._recall_sum += metrics.recall
        self._f1_sum += metrics.f1_score
        self._iou_sum += metrics.iou
        self._loss_sum += loss
        self._count += 1
    
    def compute(self) -> Tuple[OccupancyMetrics, float]:
        if self._count == 0:
            return OccupancyMetrics(0, 0, 0, 0, 0), 0.0
        
        metrics = OccupancyMetrics(
            accuracy=self._accuracy_sum / self._count,
            precision=self._precision_sum / self._count,
            recall=self._recall_sum / self._count,
            f1_score=self._f1_sum / self._count,
            iou=self._iou_sum / self._count
        )
        
        avg_loss = self._loss_sum / self._count
        
        return metrics, avg_loss


# ═══════════════════════════════════════════════════════════════
# Метрики для 3D мешей (после Marching Cubes)
# ═══════════════════════════════════════════════════════════════

def compute_mesh_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    thresholds: Tuple[float, ...] = (0.01, 0.02, 0.05)
) -> Dict[str, float]:
    """
    Вычисление метрик между двумя облаками точек.
    
    Args:
        pred_points: [N, 3] предсказанные точки
        gt_points: [M, 3] ground truth точки
        thresholds: пороги для F-Score
        
    Returns:
        Словарь с метриками
    """
    from scipy.spatial import cKDTree
    
    # KD-деревья для быстрого поиска
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)
    
    # Расстояния
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    # Chamfer Distance
    chamfer_l1 = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()
    chamfer_l2 = (dist_pred_to_gt ** 2).mean() + (dist_gt_to_pred ** 2).mean()
    
    metrics = {
        'chamfer_l1': chamfer_l1,
        'chamfer_l2': chamfer_l2,
    }
    
    # F-Score для разных порогов
    for t in thresholds:
        precision = (dist_pred_to_gt < t).mean()
        recall = (dist_gt_to_pred < t).mean()
        f_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics[f'f_score@{t}'] = f_score
        metrics[f'precision@{t}'] = precision
        metrics[f'recall@{t}'] = recall
    
    return metrics