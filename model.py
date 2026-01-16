"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Occupancy Network с Global Latent архитектурой
Дата: 2025
================================================================================
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from typing import Dict
import math


class PositionalEncoding(nn.Module):
    """Positional encoding для 3D координат."""
    
    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer('freqs', freqs)
        self.output_dim = 3 + 6 * num_frequencies  # 3 + 60 = 63
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 3] координаты
        Returns:
            [B, N, output_dim] закодированные координаты
        """
        # x: [B, N, 3]
        x_proj = x.unsqueeze(-1) * self.freqs * math.pi  # [B, N, 3, L]
        sin_enc = torch.sin(x_proj)
        cos_enc = torch.cos(x_proj)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)  # [B, N, 3, 2L]
        enc = enc.reshape(*x.shape[:-1], -1)  # [B, N, 6L]
        return torch.cat([x, enc], dim=-1)  # [B, N, 3 + 6L]


class Encoder(nn.Module):
    """
    ResNet50 encoder → global latent vector.
    
    ВАЖНО: Структура полностью совместима с train_global.py!
    Имена: self.features, self.fc
    """
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(2048, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] изображения
        Returns:
            [B, latent_dim] латентные векторы
        """
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    """
    MLP decoder для предсказания occupancy.
    
    ВАЖНО: Структура полностью совместима с train_global.py!
    Имена: self.net
    """
    
    def __init__(self, latent_dim: int = 512, point_dim: int = 63):
        super().__init__()
        
        input_dim = latent_dim + point_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    
    def forward(
        self,
        latent: torch.Tensor,
        points_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, latent_dim]
            points_encoded: [B, N, point_dim]
        Returns:
            [B, N] logits
        """
        B, N, _ = points_encoded.shape
        
        latent = latent.unsqueeze(1).expand(-1, N, -1)  # [B, N, latent_dim]
        x = torch.cat([latent, points_encoded], dim=-1)  # [B, N, input_dim]
        
        x = x.reshape(B * N, -1)
        x = self.net(x)
        x = x.reshape(B, N)
        
        return x


class OccupancyNetwork(nn.Module):
    """
    Occupancy Network с Global Latent архитектурой.
    
    ВАЖНО: Имена атрибутов совместимы с train_global.py:
    - self.encoder
    - self.pos_enc (НЕ pos_encoding!)
    - self.decoder
    """
    
    def __init__(self, latent_dim: int = 512, num_frequencies: int = 10):
        super().__init__()
        
        self.encoder = Encoder(latent_dim)
        self.pos_enc = PositionalEncoding(num_frequencies)
        self.decoder = Decoder(latent_dim, self.pos_enc.output_dim)
        
        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[model.py] OccupancyNetwork: {n_params:,} params")
        print(f"[model.py] Latent dim: {latent_dim}, Frequencies: {num_frequencies}")
    
    def forward(
        self,
        images: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224]
            points: [B, N, 3] в диапазоне [-0.5, 0.5]
        Returns:
            [B, N] logits (до sigmoid)
        """
        latent = self.encoder(images)
        points_enc = self.pos_enc(points)
        logits = self.decoder(latent, points_enc)
        return logits
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Получить latent vector из изображения."""
        return self.encoder(images)
    
    def decode(
        self,
        latent: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """Предсказать occupancy для точек."""
        points_enc = self.pos_enc(points)
        return self.decoder(latent, points_enc)
    
    def get_config(self) -> Dict:
        """Получить конфигурацию модели для сохранения."""
        return {
            'latent_dim': self.latent_dim,
            'num_frequencies': self.num_frequencies,
            'type': 'global'
        }


# Алиас для совместимости с train_global.py
OccupancyNetworkGlobal = OccupancyNetwork


def create_model(
    latent_dim: int = 512,
    num_frequencies: int = 10,
    **kwargs  # Игнорируем лишние аргументы для совместимости
) -> OccupancyNetwork:
    """
    Factory функция для создания модели.
    
    Args:
        latent_dim: размерность латентного вектора
        num_frequencies: число частот для positional encoding
        **kwargs: игнорируемые аргументы (для совместимости)
    
    Returns:
        OccupancyNetwork модель
    """
    # Логируем игнорируемые параметры для отладки
    ignored_keys = ['encoder_type', 'hidden_dims', 'dropout', 'pretrained',
                    'use_pixel_aligned', 'feature_dim', 'use_positional_encoding',
                    'use_residual', 'use_layer_norm', 'freeze_bn', 'decoder_hidden_dims',
                    'decoder_dropout', 'backbone', 'decoder_use_residual', 
                    'decoder_use_layer_norm', 'encoder_pretrained', 'encoder_freeze_bn']
    
    for key in list(kwargs.keys()):
        if key in ignored_keys:
            del kwargs[key]
    
    if kwargs:
        print(f"[model.py] Warning: неизвестные параметры: {list(kwargs.keys())}")
    
    return OccupancyNetwork(latent_dim=latent_dim, num_frequencies=num_frequencies)


# Список доступных энкодеров (для GUI)
AVAILABLE_ENCODERS = ['resnet50']