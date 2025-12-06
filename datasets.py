import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import trimesh
from torchvision import transforms 

class Pix3DDataset(Dataset):
    def __init__(self, root_dir, json_path, transform=None, img_size=224):
        """
        Датасет для Pix3D.
        
        Args:
            root_dir (str): Корневая директория датасета
            json_path (str): Путь к pix3d.json
            img_size (int): Размер изображения для ResNet
        """
        self.root_dir = root_dir
        self.img_size = img_size
        
        # Transform для изображений (БЕЗ RandomHorizontalFlip!)
        self.transform = transforms.Compose([
            # Аугментация цвета (улучшает обобщение)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            # Изменение размера
            transforms.Resize((img_size, img_size)),
            # Преобразование в тензор
            transforms.ToTensor(),
            # Нормализация ImageNet (требуется для ResNet)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Загрузка JSON
        print(f"[datasets.py] Загружаю {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        # Фильтрация и парсинг путей
        self.samples = []
        skipped = 0
        
        for item in self.data:
            if 'img' in item and 'model' in item:
                img_path = os.path.join(root_dir, item['img'])
                model_path = os.path.join(root_dir, item['model'])
                mask_path = os.path.join(root_dir, item['mask']) if 'mask' in item else None
                
                # Проверка существования файлов
                if os.path.exists(img_path) and os.path.exists(model_path):
                    self.samples.append({
                        'img': img_path,
                        'mask': mask_path,
                        'model': model_path,
                        'category': item.get('category', 'unknown')
                    })
                else:
                    skipped += 1
        
        print(f"[datasets.py] Загружено {len(self.samples)} образцов (пропущено {skipped})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # 1. Загрузка и обработка изображения
            img = Image.open(sample['img']).convert('RGB')
            
            # Применение маски (если есть)
            if sample['mask'] and os.path.exists(sample['mask']):
                mask = Image.open(sample['mask']).convert('L')
                # Создаем композицию: объект на черном фоне
                img = Image.composite(img, Image.new('RGB', img.size, (0, 0, 0)), mask)

            # Применение трансформаций
            img_tensor = self.transform(img)
            
            # 2. Загрузка и обработка 3D-модели (Ground Truth)
            model_path = sample['model']
            mesh = trimesh.load(model_path, force='mesh')
            
            # Обработка Scene (если модель содержит несколько объектов)
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    mesh = next(iter(mesh.geometry.values()))
                else:
                    # Пустая сцена - берем следующий элемент
                    return self.__getitem__((idx + 1) % len(self))

            vertices = np.array(mesh.vertices, dtype=np.float32)
            
            # 3. Нормализация координат GT
            if vertices.shape[0] > 0:
                # А. Центрирование
                centroid = np.mean(vertices, axis=0)
                vertices -= centroid
                
                # Б. Масштабирование в [-0.5, 0.5]
                # (соответствует начальной сфере с radius=0.5)
                max_dist = np.max(np.abs(vertices))
                if max_dist > 1e-6:
                    vertices = vertices / max_dist * 0.5
            else:
                # Пустая модель - берем следующий элемент
                return self.__getitem__((idx + 1) % len(self))
                
            vertices_tensor = torch.from_numpy(vertices).float()

            return {
                'image': img_tensor,           # [3, 224, 224]
                'vertices': vertices_tensor,   # [M, 3]
                'category': sample['category']
            }
            
        except Exception as e:
            print(f"[datasets.py] Ошибка при загрузке индекса {idx}: {e}")
            # В случае ошибки возвращаем следующий элемент
            return self.__getitem__((idx + 1) % len(self))