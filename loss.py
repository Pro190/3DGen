import torch
import torch.nn as nn

def chamfer_distance(p1, p2):
    """
    Chamfer Distance между двумя облаками точек.
    Измеряет расстояние между предсказанной и истинной формой.
    
    Args:
        p1: [B, N, 3] - Предсказанные вершины
        p2: [B, M, 3] - Ground Truth вершины
    
    Returns:
        loss: scalar - Среднее расстояние Chamfer
    """
    # Расширяем размерности для вычисления попарных расстояний
    x = p1.unsqueeze(2)  # [B, N, 1, 3]
    y = p2.unsqueeze(1)  # [B, 1, M, 3]
    
    # Вычисляем матрицу расстояний
    dist = torch.norm(x - y, dim=3)  # [B, N, M]
    
    # Для каждой точки p1 находим ближайшую в p2
    min_dist_p1, _ = torch.min(dist, dim=2)  # [B, N]
    
    # Для каждой точки p2 находим ближайшую в p1
    min_dist_p2, _ = torch.min(dist, dim=1)  # [B, M]
    
    # Chamfer Distance = среднее по обоим направлениям
    return torch.mean(min_dist_p1) + torch.mean(min_dist_p2)


def edge_length_regularization(vertices, edges):
    """
    Регуляризация длины ребер для предотвращения деформации сетки.
    Минимизирует вариацию длин ребер для более равномерной сетки.
    
    Args:
        vertices: [B, V, 3] - Координаты вершин
        edges: [E, 2] - Индексы ребер
    
    Returns:
        loss: scalar - Вариация длин ребер
    """
    if not isinstance(edges, torch.Tensor):
        edges = torch.from_numpy(edges).to(vertices.device)
    
    # Получаем вершины на концах каждого ребра
    v1 = vertices[:, edges[:, 0]]  # [B, E, 3]
    v2 = vertices[:, edges[:, 1]]  # [B, E, 3]
    
    # Вычисляем длины всех ребер
    lengths = torch.norm(v1 - v2, dim=2)  # [B, E]
    
    # Минимизируем вариацию (делает длины более однородными)
    return torch.var(lengths)


class CompositeLoss(nn.Module):
    """
    Комбинированная функция потерь для Pixel2Mesh.
    Состоит из:
    1. Chamfer Distance - основная метрика формы
    2. Edge Regularization - сглаживание сетки
    """
    
    def __init__(self, edges=None, lambda_chamfer=1.0, lambda_edge=0.1):
        """
        Args:
            edges: [E, 2] numpy array с индексами ребер
            lambda_chamfer: Вес Chamfer Distance (ИЗМЕНЕНО: 1.0 вместо 10.0!)
            lambda_edge: Вес регуляризации ребер
        """
        super().__init__()
        self.edges = edges
        self.lambda_chamfer = lambda_chamfer
        self.lambda_edge = lambda_edge
        
        print(f"[loss.py] CompositeLoss инициализирован:")
        print(f"  - Chamfer Distance (λ={lambda_chamfer})")
        if edges is not None:
            print(f"  - Edge Regularization (λ={lambda_edge}, {len(edges)} ребер)")

    def forward(self, pred_vertices, gt_vertices):
        """
        Args:
            pred_vertices: [B, V, 3] - Предсказанные вершины
            gt_vertices: [B, M, 3] - Ground Truth вершины
        
        Returns:
            total_loss: scalar
        """
        # 1. Chamfer Distance (главная метрика)
        cd_loss = chamfer_distance(pred_vertices, gt_vertices)
        total_loss = self.lambda_chamfer * cd_loss
        
        # 2. Edge Regularization (опционально)
        if self.edges is not None and self.lambda_edge > 0:
            edge_loss = edge_length_regularization(pred_vertices, self.edges)
            total_loss = total_loss + self.lambda_edge * edge_loss
        
        return total_loss