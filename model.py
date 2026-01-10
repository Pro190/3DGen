"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".
Описание: Ядро нейросети: архитектура Occupancy Network, включающая энкодер, декодер.
Дата: 2026
================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from typing import Tuple, Optional


class ImageEncoder(nn.Module):
    """
    Encoder изображения на базе ResNet.
    Извлекает глобальный feature vector.
    """
    
    def __init__(
        self, 
        backbone: str = 'resnet18',
        latent_dim: int = 512,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Выбор backbone
        if backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            resnet_out_dim = 512
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            resnet_out_dim = 512
        else:
            raise ValueError(f"Неизвестный backbone: {backbone}")
        
        # Убираем последний FC слой
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Проекция в latent space
        self.fc = nn.Linear(resnet_out_dim, latent_dim)
        
        self.latent_dim = latent_dim
        
        print(f"[model.py] Encoder: {backbone}, latent_dim={latent_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
            
        Returns:
            latent: [B, latent_dim]
        """
        features = self.features(x)         # [B, 512, 1, 1]
        features = features.flatten(1)      # [B, 512]
        latent = self.fc(features)          # [B, latent_dim]
        
        return latent


class OccupancyDecoder(nn.Module):
    """
    Decoder для предсказания occupancy.
    MLP который принимает (latent, point) и выдаёт occupancy.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        point_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Входная размерность: latent + point coordinates
        input_dim = latent_dim + point_dim
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Выходной слой
        layers.append(nn.Linear(current_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        print(f"[model.py] Decoder: {input_dim} → {hidden_dims} → 1")

    def forward(
        self, 
        latent: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, latent_dim]
            points: [B, N, 3] query points
            
        Returns:
            occupancy: [B, N] значения occupancy (logits)
        """
        batch_size, num_points, _ = points.shape
        
        # Расширяем latent для каждой точки
        latent_expanded = latent.unsqueeze(1).expand(-1, num_points, -1)  # [B, N, latent_dim]
        
        # Конкатенируем с координатами точек
        x = torch.cat([latent_expanded, points], dim=-1)  # [B, N, latent_dim + 3]
        
        # Reshape для MLP
        x = x.reshape(batch_size * num_points, -1)  # [B*N, latent_dim + 3]
        
        # MLP
        x = self.mlp(x)  # [B*N, 1]
        
        # Reshape обратно
        x = x.reshape(batch_size, num_points)  # [B, N]
        
        return x


class OccupancyNetwork(nn.Module):
    """
    Полная модель Occupancy Network.
    
    Вход: изображение + query points
    Выход: occupancy для каждой точки
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        latent_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.encoder = ImageEncoder(
            backbone=backbone,
            latent_dim=latent_dim
        )
        
        self.decoder = OccupancyDecoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.latent_dim = latent_dim
        
        print(f"[model.py] OccupancyNetwork инициализирована")

    def forward(
        self, 
        images: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224]
            points: [B, N, 3]
            
        Returns:
            occupancy_logits: [B, N]
        """
        # Encode image
        latent = self.encoder(images)  # [B, latent_dim]
        
        # Decode occupancy
        logits = self.decoder(latent, points)  # [B, N]
        
        return logits

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Только encoding (для инференса)."""
        return self.encoder(images)

    def decode(
        self, 
        latent: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        """Только decoding (для инференса)."""
        return self.decoder(latent, points)

    def predict_occupancy(
        self, 
        images: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Предсказание occupancy с sigmoid.
        
        Returns:
            occupancy: [B, N] значения в [0, 1]
        """
        logits = self.forward(images, points)
        return torch.sigmoid(logits)