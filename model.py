import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from geometry import create_high_res_icosphere

class ResNetEncoder(nn.Module):
    """Энкодер на базе ResNet18 для извлечения фичей из изображения"""
    
    def __init__(self):
        super().__init__()
        # Загрузка предобученного ResNet18
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Убираем последние слои (fc и avgpool), чтобы получить пространственные фичи
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 512  # ResNet18 выдает 512 каналов

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            features: [B, 512, 7, 7]
        """
        return self.features(x)


class GraphConv(nn.Module):
    """Слой графовой свертки (GCN)"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Args:
            x: [B, V, F_in] - фичи вершин
            adj: [V, V] - матрица смежности
        Returns:
            x: [B, V, F_out] - трансформированные фичи
        """
        # Операция GCN: H' = σ(AHW)
        # 1. Агрегация соседей: AH
        x = torch.matmul(adj, x) 
        # 2. Линейная трансформация: HW
        x = self.fc(x)
        return x


class MeshDecoder(nn.Module):
    """Декодер на базе Graph Convolutional Networks"""
    
    def __init__(self, in_features, hidden_dim=192, out_dim=3, num_layers=6):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Входная размерность: (512 каналов * 7 * 7 пикселей) + 3 координаты = 25091
        current_dim = in_features * 49 + 3 
        
        # Создаем 6 слоев GCN (как в оригинальном Pixel2Mesh)
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else out_dim
            self.layers.append(GraphConv(current_dim, next_dim))
            current_dim = next_dim
        
    def forward(self, img_features, initial_mesh_coord, adj):
        """
        Args:
            img_features: [B, 512, 7, 7] - фичи из ResNet
            initial_mesh_coord: [B, V, 3] - начальные координаты сферы
            adj: [V, V] - матрица смежности
        Returns:
            offset: [B, V, 3] - смещение для каждой вершины
        """
        batch_size = img_features.size(0)
        
        # 1. Сжатие пространственных фичей изображения
        global_features = img_features.view(batch_size, -1)  # [B, 25088]
        
        # 2. Объединение глобальных фичей и координат сетки
        # Расширяем глобальные фичи для каждой вершины
        num_vertices = initial_mesh_coord.size(1)
        expanded_features = global_features.unsqueeze(1).repeat(1, num_vertices, 1)
        
        # Конкатенация: [B, V, 25088 + 3] = [B, V, 25091]
        x = torch.cat([expanded_features, initial_mesh_coord], dim=2)
        
        # 3. Проход через слои GCN
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            # ReLU после всех слоев кроме последнего
            if i < len(self.layers) - 1:
                x = F.relu(x)
        
        # Возвращаем СМЕЩЕНИЕ (offset), а не абсолютные координаты
        return x


class Pixel2Mesh(nn.Module):
    """Основная модель Pixel2Mesh"""
    
    def __init__(self, subdivisions=3):
        super().__init__()
        
        self.subdivisions = subdivisions
        
        # Создание начальной сетки (сфера)
        vertices, faces, adjacency, edges = create_high_res_icosphere(subdivisions=subdivisions)
        
        print(f"[model.py] Инициализация: {vertices.shape[0]} вершин")
        
        # Сохранение геометрии как нетренируемых буферов
        self.register_buffer('initial_mesh', torch.from_numpy(vertices))
        self.register_buffer('adjacency', torch.from_numpy(adjacency))
        self.faces = faces  # numpy array
        self.edges = edges  # numpy array
        
        # Энкодер и декодер
        self.encoder = ResNetEncoder()
        self.decoder = MeshDecoder(in_features=self.encoder.out_channels) 

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] - батч изображений
        Returns:
            pred_vertices: [B, V, 3] - предсказанные координаты вершин
        """
        # 1. Извлечение фичей из изображения
        img_features = self.encoder(x)  # [B, 512, 7, 7]
        
        batch_size = x.size(0)
        
        # 2. Клонирование начальной сферы для батча
        mesh = self.initial_mesh.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        adj = self.adjacency.to(x.device)
        
        # 3. Предсказание смещения через декодер
        offset = self.decoder(img_features, mesh, adj)
        
        # 4. Применение смещения к начальной сфере
        pred_vertices = mesh + offset
        
        return pred_vertices