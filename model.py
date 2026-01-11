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
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
    EfficientNet_B0_Weights, EfficientNet_B3_Weights, EfficientNet_B5_Weights,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
)
from typing import Tuple, Optional, Dict, List
import math


class ImageEncoder(nn.Module):
    """
    Encoder изображения с поддержкой различных backbone.
    """
    
    SUPPORTED_BACKBONES = {
        'resnet18': (models.resnet18, ResNet18_Weights.DEFAULT, 512),
        'resnet34': (models.resnet34, ResNet34_Weights.DEFAULT, 512),
        'resnet50': (models.resnet50, ResNet50_Weights.DEFAULT, 2048),
        'resnet101': (models.resnet101, ResNet101_Weights.DEFAULT, 2048),
        'efficientnet_b0': (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
        'efficientnet_b3': (models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT, 1536),
        'efficientnet_b5': (models.efficientnet_b5, EfficientNet_B5_Weights.DEFAULT, 2048),
        'convnext_tiny': (models.convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, 768),
        'convnext_small': (models.convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
        'convnext_base': (models.convnext_base, ConvNeXt_Base_Weights.DEFAULT, 1024),
    }
    
    def __init__(
        self, 
        backbone: str = 'resnet50',
        latent_dim: int = 512,
        pretrained: bool = True,
        freeze_bn: bool = False
    ):
        super().__init__()
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Неизвестный backbone: {backbone}. "
                           f"Доступные: {list(self.SUPPORTED_BACKBONES.keys())}")
        
        model_fn, weights, out_dim = self.SUPPORTED_BACKBONES[backbone]
        
        # Загрузка модели
        if pretrained:
            base_model = model_fn(weights=weights)
        else:
            base_model = model_fn(weights=None)
        
        # Извлечение feature extractor
        if backbone.startswith('resnet'):
            self.features = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone.startswith('efficientnet'):
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif backbone.startswith('convnext'):
            self.features = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1)
            )
        
        self.backbone_name = backbone
        self.backbone_out_dim = out_dim
        
        # Проекция в latent space
        self.projection = nn.Sequential(
            nn.Linear(out_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.latent_dim = latent_dim
        
        # Заморозка BatchNorm
        if freeze_bn:
            self._freeze_bn()
        
        print(f"[model.py] Encoder: {backbone} ({out_dim}D) → {latent_dim}D")

    def _freeze_bn(self):
        """Заморозка BatchNorm слоёв."""
        for module in self.features.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
            
        Returns:
            latent: [B, latent_dim]
        """
        features = self.features(x)
        
        if self.backbone_name.startswith('efficientnet'):
            features = self.pool(features)
        
        features = features.flatten(1)
        latent = self.projection(features)
        
        return latent


class ResidualBlock(nn.Module):
    """Residual block для декодера."""
    
    def __init__(
        self, 
        dim: int, 
        dropout: float = 0.0,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim) if use_layer_norm else nn.Identity(),
        )
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class OccupancyDecoder(nn.Module):
    """
    Decoder для предсказания occupancy.
    Улучшенный MLP с residual connections.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        point_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (512, 512, 512, 256, 256),
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual
        input_dim = latent_dim + point_dim
        
        # Входной слой
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity(),
            nn.GELU()
        )
        
        # Скрытые слои
        self.hidden_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            
            if use_residual and in_dim == out_dim:
                self.hidden_layers.append(
                    ResidualBlock(in_dim, dropout, use_layer_norm)
                )
            else:
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity(),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                ))
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Инициализация весов
        self._init_weights()
        
        print(f"[model.py] Decoder: {input_dim} → {hidden_dims} → 1 "
              f"(residual={use_residual}, layernorm={use_layer_norm})")

    def _init_weights(self):
        """Инициализация весов."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        latent_expanded = latent.unsqueeze(1).expand(-1, num_points, -1)
        
        # Конкатенируем с координатами точек
        x = torch.cat([latent_expanded, points], dim=-1)
        
        # Reshape для MLP
        x = x.reshape(batch_size * num_points, -1)
        
        # Forward
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        # Reshape обратно
        x = x.reshape(batch_size, num_points)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding для 3D координат.
    Улучшает способность модели различать близкие точки.
    """
    
    def __init__(self, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Частоты: 2^0, 2^1, ..., 2^(L-1)
        frequencies = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer('frequencies', frequencies)
        
        # Выходная размерность
        self.output_dim = 3 * (1 + 2 * num_frequencies) if include_input else 3 * 2 * num_frequencies
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 3] координаты
            
        Returns:
            encoded: [B, N, output_dim]
        """
        # x: [B, N, 3]
        # frequencies: [L]
        
        # [B, N, 3, 1] * [L] -> [B, N, 3, L]
        x_freq = x.unsqueeze(-1) * self.frequencies * math.pi
        
        # Синусы и косинусы
        sin_enc = torch.sin(x_freq)  # [B, N, 3, L]
        cos_enc = torch.cos(x_freq)  # [B, N, 3, L]
        
        # Объединяем
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)  # [B, N, 3, 2L]
        encoded = encoded.reshape(*x.shape[:-1], -1)  # [B, N, 6L]
        
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)  # [B, N, 3 + 6L]
        
        return encoded


class OccupancyDecoderWithPE(nn.Module):
    """
    Decoder с Positional Encoding для лучшего разрешения деталей.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (512, 512, 512, 256, 256),
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        num_frequencies: int = 6
    ):
        super().__init__()
        
        self.positional_encoding = PositionalEncoding(
            num_frequencies=num_frequencies,
            include_input=True
        )
        
        point_dim = self.positional_encoding.output_dim
        
        self.decoder = OccupancyDecoder(
            latent_dim=latent_dim,
            point_dim=point_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm
        )
        
        print(f"[model.py] Positional Encoding: {num_frequencies} frequencies, "
              f"point_dim: 3 → {point_dim}")

    def forward(
        self, 
        latent: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        # Применяем positional encoding к точкам
        points_encoded = self.positional_encoding(points)
        
        return self.decoder(latent, points_encoded)


class OccupancyNetwork(nn.Module):
    """
    Полная модель Occupancy Network.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        latent_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (512, 512, 512, 256, 256),
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        use_positional_encoding: bool = True,
        num_frequencies: int = 6,
        pretrained: bool = True,
        freeze_bn: bool = False
    ):
        super().__init__()
        
        self.encoder = ImageEncoder(
            backbone=backbone,
            latent_dim=latent_dim,
            pretrained=pretrained,
            freeze_bn=freeze_bn
        )
        
        if use_positional_encoding:
            self.decoder = OccupancyDecoderWithPE(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
                num_frequencies=num_frequencies
            )
        else:
            self.decoder = OccupancyDecoder(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm
            )
        
        self.latent_dim = latent_dim
        self.backbone = backbone
        
        # Подсчёт параметров
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"[model.py] OccupancyNetwork: {total_params:,} параметров "
              f"({trainable_params:,} обучаемых)")

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
        latent = self.encoder(images)
        logits = self.decoder(latent, points)
        
        return logits

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Только encoding."""
        return self.encoder(images)

    def decode(
        self, 
        latent: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        """Только decoding."""
        return self.decoder(latent, points)

    def predict_occupancy(
        self, 
        images: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        """Предсказание occupancy с sigmoid."""
        logits = self.forward(images, points)
        return torch.sigmoid(logits)

    def get_config(self) -> Dict:
        """Получение конфигурации модели."""
        return {
            'backbone': self.backbone,
            'latent_dim': self.latent_dim,
        }


def create_model(
    encoder_type: str = 'resnet50',
    latent_dim: int = 512,
    hidden_dims: Tuple[int, ...] = (512, 512, 512, 256, 256),
    use_positional_encoding: bool = True,
    **kwargs
) -> OccupancyNetwork:
    """Фабричная функция для создания модели."""
    
    return OccupancyNetwork(
        backbone=encoder_type,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        use_positional_encoding=use_positional_encoding,
        **kwargs
    )


# Список доступных моделей
AVAILABLE_ENCODERS = list(ImageEncoder.SUPPORTED_BACKBONES.keys())