import torch
from torch.utils.data import DataLoader
from datasets import Pix3DDataset
from model import Pixel2Mesh
from loss import CompositeLoss
import os
import glob
import sys
import re
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ (оптимизировано для RTX 3060 Laptop 6GB + i5-12500H)
# ═══════════════════════════════════════════════════════════════════

DATA_ROOT = './PIX3D_DATA'
JSON_PATH = os.path.join(DATA_ROOT, 'pix3d.json')
CHECKPOINT_DIR = './checkpoints/'

# Параметры обучения
BATCH_SIZE = 1          # RTX 3060 6GB - безопасный размер батча
ACCUM_STEPS = 8        # Эффективный batch = 8
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4

# Параметры DataLoader (для i5-12500H и 64GB RAM)
NUM_WORKERS = 6         # 6-8 оптимально для 12-ядерного CPU
PIN_MEMORY = False       # Ускорение передачи данных на GPU

# Частота логирования
LOG_INTERVAL = 10       # Каждые N батчей выводим loss
DEBUG_INTERVAL = 50     # Каждые N батчей детальная статистика
SAVE_INTERVAL = 5       # Сохранять чекпоинт каждые N эпох (в дополнение к последней)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════════

# Инициализация GradScaler для mixed precision (FP16)
if torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()
    print(f"[train.py] Mixed precision (FP16) включен")
else:
    scaler = None
    print(f"[train.py] CUDA недоступна, обучение на CPU")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_checkpoint(model, optimizer, scaler, checkpoint_dir):
    """
    Автоматически находит и загружает последний чекпоинт.
    
    Returns:
        start_epoch: С какой эпохи продолжать обучение
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model*.pth'))
    
    if not checkpoint_files:
        print("[train.py] Чекпоинты не найдены. Начинаю обучение с нуля.")
        return 0

    # Находим последний файл по времени создания
    latest_file = max(checkpoint_files, key=os.path.getctime)
    print(f"[train.py] Загружаю чекпоинт: {latest_file}")
    
    try:
        checkpoint = torch.load(latest_file, map_location=DEVICE)

        # Проверяем формат чекпоинта
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            # Полный формат (с optimizer и scaler)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch']
            print(f"[train.py] Возобновление с эпохи {start_epoch + 1}")
            return start_epoch
        
        else:
            # Старый формат (только веса модели)
            model.load_state_dict(checkpoint)
            
            # Пытаемся извлечь номер эпохи из имени файла
            match = re.search(r'model(\d+)\.pth', latest_file)
            start_epoch = int(match.group(1)) if match else 0
            
            print(f"[train.py] Загружены только веса. Возобновление с эпохи {start_epoch + 1}")
            print("[train.py] ВНИМАНИЕ: Optimizer и Scaler сброшены!")
            return start_epoch

    except Exception as e:
        print(f"[train.py] ОШИБКА при загрузке чекпоинта: {e}")
        print("[train.py] Начинаю обучение с нуля.")
        return 0


def save_checkpoint(epoch, model, optimizer, scaler, checkpoint_dir):
    """
    Сохраняет полное состояние для возобновления обучения.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Сохраняем с номером эпохи
    checkpoint_path = os.path.join(checkpoint_dir, f'model{epoch}.pth')
    torch.save(state, checkpoint_path)
    
    # Сохраняем как "последний"
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(state, latest_path)
    
    print(f"[train.py] ✓ Чекпоинт сохранен: epoch {epoch}")


def main():
    print("=" * 70)
    print(f"PIXEL2MESH TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Устройство: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} (аккумуляция: {ACCUM_STEPS}, эффективный: {BATCH_SIZE * ACCUM_STEPS})")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 70)
    
    # 1. Загрузка датасета
    print("\n[1/4] Загружаю датасет...")
    dataset = Pix3DDataset(DATA_ROOT, JSON_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    print(f"✓ Загружено {len(dataset)} образцов")

    # 2. Создание модели
    print("\n[2/4] Инициализирую модель...")
    model = Pixel2Mesh(subdivisions=3).to(DEVICE)
    
    # 3. Loss и оптимизатор
    print("\n[3/4] Настраиваю loss и optimizer...")
    # Передаем edges для регуляризации
    criterion = CompositeLoss(
        edges=model.edges,
        lambda_chamfer=10.0,
        lambda_edge=0.1
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Загрузка чекпоинта (если есть)
    print("\n[4/4] Проверяю наличие чекпоинтов...")
    start_epoch = load_checkpoint(model, optimizer, scaler, CHECKPOINT_DIR)
    
    print("\n" + "=" * 70)
    print("НАЧИНАЮ ОБУЧЕНИЕ")
    print("=" * 70 + "\n")
    
    model.train()
    
    # ═══════════════════════════════════════════════════════════════
    # ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
    # ═══════════════════════════════════════════════════════════════
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0.0
        running_loss = 0.0
        optimizer.zero_grad()
        
        try:
            for i, batch in enumerate(dataloader):
                
                images = batch['image'].to(DEVICE)
                gt_vertices = batch['vertices'].to(DEVICE)
                
                # Forward pass с автоматической FP16
                with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                    pred_vertices = model(images)
                    loss = criterion(pred_vertices, gt_vertices)
                    loss = loss / ACCUM_STEPS  # Нормализуем для аккумуляции
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                running_loss += loss.item() * ACCUM_STEPS
                epoch_loss += loss.item() * ACCUM_STEPS
                
                # Шаг оптимизатора (каждые ACCUM_STEPS)
                if (i + 1) % ACCUM_STEPS == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    # Логирование
                    if (i + 1) % (ACCUM_STEPS * LOG_INTERVAL) == 0:
                        avg_loss = running_loss / LOG_INTERVAL
                        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                              f"Step [{i+1}/{len(dataloader)}] "
                              f"Loss: {avg_loss:.4f}")
                        running_loss = 0.0
                
                # Детальная статистика
                if (i + 1) % DEBUG_INTERVAL == 0:
                    with torch.no_grad():
                        gt_min = gt_vertices.min().item()
                        gt_max = gt_vertices.max().item()
                        pred_min = pred_vertices.min().item()
                        pred_max = pred_vertices.max().item()
                        
                        print(f"\n{'─' * 50}")
                        print(f"DEBUG STEP {i+1}:")
                        print(f"  GT   range: [{gt_min:.3f}, {gt_max:.3f}]")
                        print(f"  PRED range: [{pred_min:.3f}, {pred_max:.3f}]")
                        print(f"{'─' * 50}\n")
        
        except KeyboardInterrupt:
            print("\n" + "!" * 70)
            print("ПРЕРЫВАНИЕ: Ctrl+C обнаружен")
            print("!" * 70)
            print("Сохраняю текущее состояние...")
            save_checkpoint(epoch, model, optimizer, scaler, CHECKPOINT_DIR)
            print("✓ Состояние сохранено. Выход.")
            sys.exit(0)
        
        # Статистика эпохи
        avg_epoch_loss = epoch_loss / len(dataloader)
        print("\n" + "=" * 70)
        print(f"ЭПОХА {epoch+1}/{NUM_EPOCHS} ЗАВЕРШЕНА")
        print(f"Средний loss: {avg_epoch_loss:.4f}")
        print("=" * 70 + "\n")
        
        # Сохранение чекпоинта
        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            save_checkpoint(epoch + 1, model, optimizer, scaler, CHECKPOINT_DIR)
    
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)


if __name__ == '__main__':
    main()