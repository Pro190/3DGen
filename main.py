"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович
Руководитель: Простомолотов Андрей Сергеевич
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения"
Описание: Графический интерфейс пользователя (GUI) на базе PyQt5
Дата: 2025
================================================================================

Приложение предоставляет удобный интерфейс для:
    1. Обучения модели Occupancy Network
    2. Генерации 3D моделей из фотографий мебели
    3. Препроцессинга датасета
    4. Управления чекпоинтами

Изменения:
    - Параметры из GUI передаются в train.py через командную строку
    - Генерация 3D работает только по фото (без обязательной маски)
    - Добавлена опция использования препроцессированных данных
"""

import sys
import os
import subprocess
import json
from datetime import datetime
from typing import Optional, Dict, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget, QComboBox,
    QMessageBox, QLineEdit, QCheckBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QPalette, QColor

import torch
from PIL import Image
from torchvision import transforms


# ═══════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ═══════════════════════════════════════════════════════════════════════════════

# Категории мебели в датасете PIX3D
CATEGORIES = [
    'all',       # Все категории
    'bed',       # Кровати
    'bookcase',  # Книжные шкафы
    'chair',     # Стулья
    'desk',      # Столы письменные
    'misc',      # Разное
    'sofa',      # Диваны
    'table',     # Столы
    'tool',      # Инструменты
    'wardrobe'   # Шкафы
]

# Доступные форматы экспорта
EXPORT_FORMATS = ['obj', 'ply', 'stl', 'glb']


# ═══════════════════════════════════════════════════════════════════════════════
# ПОТОК ОБУЧЕНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingThread(QThread):
    """
    Поток для запуска обучения в фоновом режиме.
    
    ВАЖНО: Параметры из GUI передаются через командную строку train.py,
    а не через update_config(), чтобы избежать проблем с subprocess.
    """
    
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str)
    metrics_update = pyqtSignal(dict)
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.process = None
        self.is_running = True
    
    def run(self):
        """Запуск обучения."""
        try:
            self.log_message.emit("=" * 50)
            self.log_message.emit("Запуск обучения...")
            self.log_message.emit("=" * 50)
            
            # ─────────────────────────────────────────────────────────────────
            # Формирование команды с параметрами
            # ─────────────────────────────────────────────────────────────────
            
            cmd = [sys.executable, 'train.py']
            
            # Добавляем параметры из GUI
            if self.config.get('batch_size'):
                cmd.extend(['--batch_size', str(self.config['batch_size'])])
            
            if self.config.get('num_epochs'):
                cmd.extend(['--num_epochs', str(self.config['num_epochs'])])
            
            if self.config.get('learning_rate'):
                cmd.extend(['--learning_rate', str(self.config['learning_rate'])])
            
            if self.config.get('latent_dim'):
                cmd.extend(['--latent_dim', str(self.config['latent_dim'])])
            
            if self.config.get('category') and self.config['category'] != 'all':
                cmd.extend(['--category', self.config['category']])
            
            if self.config.get('save_interval'):
                cmd.extend(['--save_interval', str(self.config['save_interval'])])
            
            if self.config.get('use_preprocessed'):
                cmd.append('--use_preprocessed')
            
            self.log_message.emit(f"Команда: {' '.join(cmd)}")
            self.log_message.emit("")
            
            # ─────────────────────────────────────────────────────────────────
            # Запуск subprocess
            # ─────────────────────────────────────────────────────────────────
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Чтение вывода в реальном времени
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                
                line = line.rstrip()
                if line:
                    self.log_message.emit(line)
                    self._parse_metrics(line)
            
            self.process.wait()
            
            if self.process.returncode == 0:
                self.finished.emit("✓ Обучение завершено успешно!")
            else:
                self.finished.emit(f"✗ Обучение завершено с кодом {self.process.returncode}")
                
        except Exception as e:
            self.log_message.emit(f"Ошибка: {str(e)}")
            self.finished.emit(f"✗ Ошибка: {str(e)}")
    
    def _parse_metrics(self, line: str):
        """Парсинг метрик из строки вывода."""
        try:
            if 'Loss:' in line and 'IoU:' in line:
                parts = line.split(',')
                metrics = {}
                
                for part in parts:
                    if 'Loss:' in part:
                        metrics['loss'] = float(part.split(':')[1].strip())
                    elif 'IoU:' in part:
                        metrics['iou'] = float(part.split(':')[1].strip())
                
                if metrics:
                    self.metrics_update.emit(metrics)
        except:
            pass
    
    def stop(self):
        """Остановка обучения."""
        self.is_running = False
        
        if self.process and self.process.poll() is None:
            self.log_message.emit("\n⏹ Остановка обучения...")
            
            if sys.platform == 'win32':
                self.process.terminate()
            else:
                import signal
                self.process.send_signal(signal.SIGTERM)
            
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.log_message.emit("Принудительное завершение...")
                self.process.kill()


# ═══════════════════════════════════════════════════════════════════════════════
# ПОТОК ГЕНЕРАЦИИ 3D
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceThread(QThread):
    """
    Поток для генерации 3D модели.
    
    ИЗМЕНЕНИЕ: Маска теперь опциональна, генерация работает только по фото.
    """
    
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str, object)
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
    
    def run(self):
        """Запуск генерации."""
        try:
            self.progress.emit(5, "Импорт модулей...")
            
            from model import create_model
            from mesh_utils import extract_mesh_marching_cubes, save_mesh, simplify_mesh, smooth_mesh
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.log_message.emit(f"Device: {device}")
            
            # ─────────────────────────────────────────────────────────────────
            # Загрузка модели
            # ─────────────────────────────────────────────────────────────────
            
            self.progress.emit(15, "Загрузка модели...")
            
            checkpoint_path = self.config['checkpoint']
            
            if not os.path.exists(checkpoint_path):
                self.finished.emit(f"Чекпоинт не найден: {checkpoint_path}", None)
                return
            
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False
            )
            
            model_config = checkpoint.get('config', {})
            latent_dim = model_config.get('latent_dim', 512)
            num_frequencies = model_config.get('num_frequencies', 10)
            
            self.log_message.emit(f"Latent dim: {latent_dim}")
            self.log_message.emit(f"Epoch: {checkpoint.get('epoch', '?')}")
            self.log_message.emit(f"Best IoU: {checkpoint.get('best_iou', 0):.4f}")
            
            model = create_model(
                latent_dim=latent_dim,
                num_frequencies=num_frequencies
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # ─────────────────────────────────────────────────────────────────
            # Загрузка изображения (БЕЗ ОБЯЗАТЕЛЬНОЙ МАСКИ)
            # ─────────────────────────────────────────────────────────────────
            
            self.progress.emit(30, "Загрузка изображения...")
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            img = Image.open(self.config['image']).convert('RGB')
            self.log_message.emit(f"Изображение: {self.config['image']}")
            
            # Маска опциональна - применяем только если указана и use_mask=True
            use_mask = self.config.get('use_mask', False)
            mask_path = self.config.get('mask')
            
            if use_mask and mask_path and os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('L')
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    img = Image.composite(img, background, mask)
                    self.log_message.emit(f"Маска применена: {mask_path}")
                except Exception as e:
                    self.log_message.emit(f"⚠️ Ошибка маски: {e}")
            else:
                self.log_message.emit("Генерация без маски (только по фото)")
            
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # ─────────────────────────────────────────────────────────────────
            # Encoding
            # ─────────────────────────────────────────────────────────────────
            
            self.progress.emit(45, "Encoding изображения...")
            
            with torch.no_grad():
                latent = model.encode(img_tensor)
            
            def occupancy_fn(points):
                points = points.unsqueeze(0)
                with torch.no_grad():
                    logits = model.decode(latent, points)
                    # Temperature scaling для смягчения предсказаний
                    temperature = 0.8
                    scaled_logits = logits / temperature
                    return torch.sigmoid(scaled_logits).squeeze(0)
            
            # ─────────────────────────────────────────────────────────────────
            # Marching Cubes
            # ─────────────────────────────────────────────────────────────────
            
            self.progress.emit(60, "Marching Cubes...")
            
            resolution = self.config.get('resolution', 128)
            threshold = self.config.get('threshold', 0.5)
            
            self.log_message.emit(f"Resolution: {resolution}")
            self.log_message.emit(f"Threshold: {threshold}")
            
            mesh = extract_mesh_marching_cubes(
                occupancy_fn,
                resolution=resolution,
                threshold=threshold,
                device=device,
                verbose=False
            )
            
            if mesh is None:
                self.finished.emit("Не удалось извлечь меш", None)
                return
            
            self.log_message.emit(f"Меш создан: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
            
            # ─────────────────────────────────────────────────────────────────
            # Постобработка с сглаживанием для уменьшения шума
            # ─────────────────────────────────────────────────────────────────
            
            # ─────────────────────────────────────────────────────────────────
            # Улучшенная постобработка
            # ─────────────────────────────────────────────────────────────────
            
            try:
                from mesh_utils import repair_mesh
                
                # Исправление геометрии
                self.log_message.emit("Исправление геометрии...")
                mesh = repair_mesh(mesh)
                
                # Двухэтапное сглаживание
                self.log_message.emit("Применение сглаживания (двухэтапное)...")
                mesh = smooth_mesh(mesh, iterations=2, lamb=0.6)  # Агрессивное
                mesh = smooth_mesh(mesh, iterations=2, lamb=0.3)  # Мягкое
                
                # Нормализация нормалей
                try:
                    mesh.fix_normals()
                except:
                    pass
                
                self.log_message.emit(f"Меш после постобработки: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
            except Exception as e:
                self.log_message.emit(f"⚠️ Ошибка постобработки: {e}")
            
            # ─────────────────────────────────────────────────────────────────
            # Упрощение (опционально)
            # ─────────────────────────────────────────────────────────────────
            
            if self.config.get('simplify', False):
                target_faces = self.config.get('target_faces', 10000)
                if len(mesh.faces) > target_faces:
                    self.progress.emit(80, "Упрощение меша...")
                    mesh = simplify_mesh(mesh, target_faces)
                    self.log_message.emit(f"Упрощено до {len(mesh.faces)} граней")
            
            # ─────────────────────────────────────────────────────────────────
            # Сохранение
            # ─────────────────────────────────────────────────────────────────
            
            self.progress.emit(90, "Сохранение...")
            
            output_dir = self.config['output']
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.config['image']))[0]
            output_format = self.config.get('format', 'obj')
            output_path = os.path.join(output_dir, f"{base_name}_3d.{output_format}")
            
            save_mesh(mesh, output_path)
            
            result = {
                'path': output_path,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces)
            }
            
            self.progress.emit(100, "Готово!")
            self.finished.emit(f"✓ Сохранено: {output_path}", result)
            
        except Exception as e:
            import traceback
            self.log_message.emit(traceback.format_exc())
            self.finished.emit(f"Ошибка: {str(e)}", None)


# ═══════════════════════════════════════════════════════════════════════════════
# ПОТОК ПРЕПРОЦЕССИНГА
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocessThread(QThread):
    """Поток для препроцессинга датасета."""
    
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str)
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
    
    def run(self):
        """Запуск препроцессинга."""
        try:
            self.log_message.emit("Запуск препроцессинга...")
            
            from preprocessing import DatasetPreprocessor
            
            preprocessor = DatasetPreprocessor(
                root_dir=self.config['root_dir'],
                json_path=self.config['json_path'],
                output_dir=self.config['output_dir'],
                num_workers=self.config.get('num_workers', 8),
                category_filter=self.config.get('category')
            )
            
            index = preprocessor.preprocess(force=self.config.get('force', False))
            
            self.log_message.emit(f"Обработано {len(index)} образцов")
            
            stats = preprocessor.get_statistics()
            for key, value in stats.items():
                if isinstance(value, dict):
                    self.log_message.emit(f"{key}:")
                    for k, v in value.items():
                        self.log_message.emit(f"  {k}: {v}")
                else:
                    self.log_message.emit(f"{key}: {value}")
            
            self.finished.emit(f"✓ Препроцессинг завершён! Образцов: {len(index)}")
            
        except Exception as e:
            import traceback
            self.log_message.emit(traceback.format_exc())
            self.finished.emit(f"Ошибка: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛАВНОЕ ОКНО
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Главное окно приложения."""
    
    DEFAULT_DATA_PATH = './PIX3D_DATA'
    DEFAULT_CKPT_PATH = './checkpoints'
    DEFAULT_OUTPUT_PATH = './inference_results'
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Occupancy Network - 3D Reconstruction")
        self.setGeometry(100, 100, 1200, 800)
        
        self._data_path = self.DEFAULT_DATA_PATH
        self._ckpt_path = self.DEFAULT_CKPT_PATH
        self._output_path = self.DEFAULT_OUTPUT_PATH
        
        self.current_image_path = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.tabs.addTab(self._create_training_tab(), "🎓 Обучение")
        self.tabs.addTab(self._create_inference_tab(), "✨ Генерация 3D")
        self.tabs.addTab(self._create_preprocessing_tab(), "⚙️ Препроцессинг")
        self.tabs.addTab(self._create_checkpoints_tab(), "📁 Чекпоинты")
        self.tabs.addTab(self._create_settings_tab(), "⚙️ Настройки")
        
        self.statusBar().showMessage("Готов к работе")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_ckpt_path(self) -> str:
        if hasattr(self, 'ckpt_path') and self.ckpt_path is not None:
            return self.ckpt_path.text()
        return self._ckpt_path
    
    def get_data_path(self) -> str:
        if hasattr(self, 'data_path') and self.data_path is not None:
            return self.data_path.text()
        return self._data_path
    
    def get_output_path(self) -> str:
        if hasattr(self, 'output_path') and self.output_path is not None:
            return self.output_path.text()
        return self._output_path
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ВКЛАДКА ОБУЧЕНИЯ
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_training_tab(self) -> QWidget:
        """Создание вкладки обучения."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Левая панель - параметры
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)
        
        # Группа: Модель
        model_group = QGroupBox("Модель")
        model_layout = QVBoxLayout()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Latent dim:"))
        self.latent_spin = QSpinBox()
        self.latent_spin.setRange(128, 1024)
        self.latent_spin.setValue(512)
        self.latent_spin.setSingleStep(128)
        row.addWidget(self.latent_spin)
        model_layout.addLayout(row)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Группа: Параметры обучения
        train_group = QGroupBox("Параметры обучения")
        train_layout = QVBoxLayout()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Эпохи:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(200)
        row.addWidget(self.epochs_spin)
        train_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Batch size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        row.addWidget(self.batch_spin)
        train_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Learning rate:"))
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(['1e-4', '3e-4', '5e-4', '1e-3'])
        self.lr_combo.setCurrentText('3e-4')
        row.addWidget(self.lr_combo)
        train_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Категория:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(CATEGORIES)
        row.addWidget(self.category_combo)
        train_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Сохранять каждые:"))
        self.save_interval_spin = QSpinBox()
        self.save_interval_spin.setRange(1, 100)
        self.save_interval_spin.setValue(10)
        row.addWidget(self.save_interval_spin)
        row.addWidget(QLabel("эпох"))
        train_layout.addLayout(row)
        
        # Чекбокс для препроцессированных данных
        self.use_preprocessed_cb = QCheckBox("Использовать препроцессированные данные")
        self.use_preprocessed_cb.setToolTip(
            "Ускоряет загрузку данных в 10-20 раз.\n"
            "Требует предварительного запуска препроцессинга."
        )
        train_layout.addWidget(self.use_preprocessed_cb)
        
        train_group.setLayout(train_layout)
        left_layout.addWidget(train_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.btn_train = QPushButton("▶ Начать обучение")
        self.btn_train.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px;"
        )
        self.btn_train.clicked.connect(self._start_training)
        buttons_layout.addWidget(self.btn_train)
        
        self.btn_stop = QPushButton("⏹ Остановить")
        self.btn_stop.setStyleSheet(
            "background-color: #f44336; color: white; padding: 10px;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        buttons_layout.addWidget(self.btn_stop)
        
        left_layout.addLayout(buttons_layout)
        
        # Прогресс
        self.train_progress = QProgressBar()
        self.train_progress.setTextVisible(True)
        left_layout.addWidget(self.train_progress)
        
        # Метрики
        metrics_group = QGroupBox("Текущие метрики")
        metrics_layout = QVBoxLayout()
        
        self.metric_loss_label = QLabel("Loss: --")
        self.metric_iou_label = QLabel("IoU: --")
        
        metrics_layout.addWidget(self.metric_loss_label)
        metrics_layout.addWidget(self.metric_iou_label)
        
        metrics_group.setLayout(metrics_layout)
        left_layout.addWidget(metrics_group)
        
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Правая панель - лог
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel("Лог обучения:"))
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("""
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
            background-color: #1e1e1e;
            color: #d4d4d4;
        """)
        right_layout.addWidget(self.train_log)
        
        btn_clear_log = QPushButton("🗑 Очистить лог")
        btn_clear_log.clicked.connect(self.train_log.clear)
        right_layout.addWidget(btn_clear_log)
        
        layout.addWidget(right_panel)
        
        return widget
    
    def _start_training(self):
        """Запуск обучения с параметрами из GUI."""
        
        # Собираем параметры из GUI
        config = {
            'latent_dim': self.latent_spin.value(),
            'num_epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': float(self.lr_combo.currentText()),
            'category': self.category_combo.currentText(),
            'save_interval': self.save_interval_spin.value(),
            'use_preprocessed': self.use_preprocessed_cb.isChecked()
        }
        
        self.train_thread = TrainingThread(config)
        self.train_thread.progress.connect(self._update_train_progress)
        self.train_thread.log_message.connect(self._append_train_log)
        self.train_thread.finished.connect(self._training_finished)
        self.train_thread.metrics_update.connect(self._update_metrics)
        self.train_thread.start()
        
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.train_log.clear()
        
        # Логируем параметры
        self._append_train_log("Параметры обучения:")
        self._append_train_log(f"  Latent dim: {config['latent_dim']}")
        self._append_train_log(f"  Epochs: {config['num_epochs']}")
        self._append_train_log(f"  Batch size: {config['batch_size']}")
        self._append_train_log(f"  Learning rate: {config['learning_rate']}")
        self._append_train_log(f"  Category: {config['category']}")
        self._append_train_log(f"  Use preprocessed: {config['use_preprocessed']}")
        self._append_train_log("")
    
    def _stop_training(self):
        if hasattr(self, 'train_thread'):
            self.train_thread.stop()
            self.statusBar().showMessage("Остановка обучения...")
    
    def _update_train_progress(self, value: int, message: str):
        self.train_progress.setValue(value)
        self.statusBar().showMessage(message)
    
    def _append_train_log(self, message: str):
        self.train_log.append(message)
        scrollbar = self.train_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _update_metrics(self, metrics: Dict):
        if 'loss' in metrics:
            self.metric_loss_label.setText(f"Loss: {metrics['loss']:.4f}")
        if 'iou' in metrics:
            self.metric_iou_label.setText(f"IoU: {metrics['iou']:.4f}")
    
    def _training_finished(self, message: str):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage(message)
        self._refresh_checkpoints()
        QMessageBox.information(self, "Обучение", message)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ВКЛАДКА ГЕНЕРАЦИИ 3D (БЕЗ ОБЯЗАТЕЛЬНОЙ МАСКИ)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_inference_tab(self) -> QWidget:
        """Создание вкладки генерации."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Левая панель
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(450)
        
        # Группа: Изображение
        img_group = QGroupBox("Входное изображение")
        img_layout = QVBoxLayout()
        
        # Только кнопка загрузки изображения (маска убрана)
        self.btn_load_image = QPushButton("📁 Загрузить изображение")
        self.btn_load_image.clicked.connect(self._load_image)
        img_layout.addWidget(self.btn_load_image)
        
        self.image_label = QLabel("Изображение не загружено")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 2px dashed #ccc;
            border-radius: 10px;
            background-color: #f5f5f5;
        """)
        img_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        img_group.setLayout(img_layout)
        left_layout.addWidget(img_group)
        
        # Группа: Параметры
        params_group = QGroupBox("Параметры генерации")
        params_layout = QVBoxLayout()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Чекпоинт:"))
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setMinimumWidth(200)
        row.addWidget(self.checkpoint_combo)
        
        btn_refresh = QPushButton("🔄")
        btn_refresh.setMaximumWidth(40)
        btn_refresh.clicked.connect(self._refresh_checkpoints)
        row.addWidget(btn_refresh)
        params_layout.addLayout(row)
        
        self._refresh_checkpoints()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Разрешение:"))
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(32, 256)
        self.resolution_spin.setValue(128)
        self.resolution_spin.setSingleStep(32)
        row.addWidget(self.resolution_spin)
        params_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.05)
        row.addWidget(self.threshold_spin)
        params_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Формат:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(EXPORT_FORMATS)
        row.addWidget(self.format_combo)
        params_layout.addLayout(row)
        
        self.simplify_cb = QCheckBox("Упростить меш")
        self.simplify_cb.setChecked(True)
        params_layout.addWidget(self.simplify_cb)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Макс. граней:"))
        self.target_faces_spin = QSpinBox()
        self.target_faces_spin.setRange(1000, 100000)
        self.target_faces_spin.setValue(10000)
        self.target_faces_spin.setSingleStep(1000)
        row.addWidget(self.target_faces_spin)
        params_layout.addLayout(row)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Кнопка генерации
        self.btn_generate = QPushButton("✨ Генерировать 3D модель")
        self.btn_generate.setStyleSheet("""
            background-color: #2196F3;
            color: white;
            padding: 15px;
            font-size: 14px;
            font-weight: bold;
        """)
        self.btn_generate.clicked.connect(self._generate_3d)
        self.btn_generate.setEnabled(False)
        left_layout.addWidget(self.btn_generate)
        
        self.infer_progress = QProgressBar()
        left_layout.addWidget(self.infer_progress)
        
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Правая панель - результат
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        result_group = QGroupBox("Результат")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel("Результат появится здесь")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("padding: 20px;")
        result_layout.addWidget(self.result_label)
        
        self.result_info = QTextEdit()
        self.result_info.setReadOnly(True)
        self.result_info.setMaximumHeight(200)
        result_layout.addWidget(self.result_info)
        
        self.btn_open_folder = QPushButton("📂 Открыть папку результатов")
        self.btn_open_folder.clicked.connect(self._open_results_folder)
        result_layout.addWidget(self.btn_open_folder)
        
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        right_layout.addStretch()
        
        layout.addWidget(right_panel)
        
        return widget
    
    def _load_image(self):
        """Загрузка изображения."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        
        if path:
            self.current_image_path = path
            
            pixmap = QPixmap(path)
            pixmap = pixmap.scaled(
                400, 400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            
            self.btn_generate.setEnabled(True)
            self.statusBar().showMessage(f"Загружено: {os.path.basename(path)}")
    
    def _refresh_checkpoints(self):
        """Обновление списка чекпоинтов."""
        self.checkpoint_combo.clear()
        
        ckpt_dir = self.get_ckpt_path()
        if os.path.exists(ckpt_dir):
            files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
            
            priority = {'best.pth': 0, 'latest.pth': 1}
            files.sort(key=lambda x: priority.get(x, 2))
            
            for f in files:
                self.checkpoint_combo.addItem(f)
    
    def _generate_3d(self):
        """Генерация 3D модели (только по фото, без маски)."""
        if not self.current_image_path:
            return
        
        checkpoint_file = self.checkpoint_combo.currentText()
        if not checkpoint_file:
            QMessageBox.warning(self, "Ошибка", "Выберите чекпоинт")
            return
        
        # ИЗМЕНЕНИЕ: маска не передаётся, use_mask=False
        config = {
            'image': self.current_image_path,
            'mask': None,  # Маска не используется
            'use_mask': False,  # Генерация только по фото
            'checkpoint': os.path.join(self.get_ckpt_path(), checkpoint_file),
            'output': self.get_output_path(),
            'resolution': self.resolution_spin.value(),
            'threshold': self.threshold_spin.value(),
            'format': self.format_combo.currentText(),
            'simplify': self.simplify_cb.isChecked(),
            'target_faces': self.target_faces_spin.value()
        }
        
        self.infer_thread = InferenceThread(config)
        self.infer_thread.progress.connect(self._update_infer_progress)
        self.infer_thread.log_message.connect(lambda m: self.result_info.append(m))
        self.infer_thread.finished.connect(self._inference_finished)
        self.infer_thread.start()
        
        self.btn_generate.setEnabled(False)
        self.infer_progress.setValue(0)
        self.result_info.clear()
    
    def _update_infer_progress(self, value: int, message: str):
        self.infer_progress.setValue(value)
        self.result_label.setText(message)
    
    def _inference_finished(self, message: str, result: Optional[Dict]):
        self.btn_generate.setEnabled(True)
        self.result_label.setText(message)
        
        if result:
            info = (
                f"\n✓ Генерация завершена!\n\n"
                f"Файл: {result['path']}\n"
                f"Вершин: {result['vertices']}\n"
                f"Граней: {result['faces']}"
            )
            self.result_info.append(info)
            
            QMessageBox.information(
                self, "Готово",
                f"3D модель сохранена:\n{result['path']}\n\n"
                f"Вершин: {result['vertices']}\n"
                f"Граней: {result['faces']}"
            )
    
    def _open_results_folder(self):
        self._open_folder(self.get_output_path())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ВКЛАДКА ПРЕПРОЦЕССИНГА
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_preprocessing_tab(self) -> QWidget:
        """Создание вкладки препроцессинга."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel(
            "Препроцессинг датасета позволяет ускорить обучение в 10-20 раз\n"
            "за счёт предварительного вычисления точек на поверхности мешей.\n\n"
            "После препроцессинга включите опцию 'Использовать препроцессированные данные'\n"
            "на вкладке обучения."
        )
        info_label.setStyleSheet(
            "padding: 10px; background-color: #e3f2fd; border-radius: 5px; color: #212121;"
        )
        layout.addWidget(info_label)
        
        params_group = QGroupBox("Параметры")
        params_layout = QVBoxLayout()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Категория:"))
        self.preprocess_category = QComboBox()
        self.preprocess_category.addItems(CATEGORIES)
        row.addWidget(self.preprocess_category)
        row.addStretch()
        params_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Workers:"))
        self.preprocess_workers = QSpinBox()
        self.preprocess_workers.setRange(1, 16)
        self.preprocess_workers.setValue(8)
        row.addWidget(self.preprocess_workers)
        row.addStretch()
        params_layout.addLayout(row)
        
        self.preprocess_force = QCheckBox("Перезаписать существующие данные")
        params_layout.addWidget(self.preprocess_force)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        self.btn_preprocess = QPushButton("⚙️ Запустить препроцессинг")
        self.btn_preprocess.setStyleSheet(
            "background-color: #FF9800; color: white; padding: 10px;"
        )
        self.btn_preprocess.clicked.connect(self._start_preprocessing)
        layout.addWidget(self.btn_preprocess)
        
        self.preprocess_progress = QProgressBar()
        layout.addWidget(self.preprocess_progress)
        
        self.preprocess_log = QTextEdit()
        self.preprocess_log.setReadOnly(True)
        layout.addWidget(self.preprocess_log)
        
        return widget
    
    def _start_preprocessing(self):
        category = self.preprocess_category.currentText()
        
        config = {
            'root_dir': self.get_data_path(),
            'json_path': os.path.join(self.get_data_path(), 'pix3d.json'),
            'output_dir': './cache/preprocessed',
            'num_workers': self.preprocess_workers.value(),
            'category': category if category != 'all' else None,
            'force': self.preprocess_force.isChecked()
        }
        
        self.preprocess_thread = PreprocessThread(config)
        self.preprocess_thread.progress.connect(
            lambda v, m: self.preprocess_progress.setValue(v)
        )
        self.preprocess_thread.log_message.connect(self.preprocess_log.append)
        self.preprocess_thread.finished.connect(self._preprocessing_finished)
        self.preprocess_thread.start()
        
        self.btn_preprocess.setEnabled(False)
        self.preprocess_log.clear()
    
    def _preprocessing_finished(self, message: str):
        self.btn_preprocess.setEnabled(True)
        self.preprocess_log.append(message)
        QMessageBox.information(self, "Препроцессинг", message)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ВКЛАДКА ЧЕКПОИНТОВ
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_checkpoints_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Доступные чекпоинты:"))
        
        self.checkpoints_list = QListWidget()
        self.checkpoints_list.itemDoubleClicked.connect(self._view_checkpoint_info)
        layout.addWidget(self.checkpoints_list)
        
        buttons = QHBoxLayout()
        
        btn_refresh = QPushButton("🔄 Обновить")
        btn_refresh.clicked.connect(self._refresh_checkpoints_list)
        buttons.addWidget(btn_refresh)
        
        btn_delete = QPushButton("🗑 Удалить выбранный")
        btn_delete.clicked.connect(self._delete_checkpoint)
        buttons.addWidget(btn_delete)
        
        btn_open = QPushButton("📂 Открыть папку")
        btn_open.clicked.connect(lambda: self._open_folder(self.get_ckpt_path()))
        buttons.addWidget(btn_open)
        
        layout.addLayout(buttons)
        
        self.checkpoint_info = QTextEdit()
        self.checkpoint_info.setReadOnly(True)
        self.checkpoint_info.setMaximumHeight(200)
        layout.addWidget(self.checkpoint_info)
        
        self._refresh_checkpoints_list()
        
        return widget
    
    def _refresh_checkpoints_list(self):
        self.checkpoints_list.clear()
        
        ckpt_dir = self.get_ckpt_path()
        if os.path.exists(ckpt_dir):
            files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
            
            for f in files:
                path = os.path.join(ckpt_dir, f)
                size = os.path.getsize(path) / 1e6
                item = QListWidgetItem(f"{f} ({size:.1f} MB)")
                self.checkpoints_list.addItem(item)
    
    def _view_checkpoint_info(self, item: QListWidgetItem):
        filename = item.text().split(' (')[0]
        path = os.path.join(self.get_ckpt_path(), filename)
        
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            info = []
            info.append(f"Файл: {filename}")
            info.append(f"Эпоха: {checkpoint.get('epoch', 'N/A')}")
            
            best_iou = checkpoint.get('best_iou', None)
            if best_iou is not None:
                info.append(f"Best IoU: {best_iou:.4f}")
            
            config = checkpoint.get('config', {})
            if config:
                info.append(f"\nКонфигурация модели:")
                info.append(f"  Latent dim: {config.get('latent_dim', 'N/A')}")
                info.append(f"  Type: {config.get('type', 'N/A')}")
            
            train_config = checkpoint.get('train_config', {})
            if train_config:
                info.append(f"\nПараметры обучения:")
                info.append(f"  Batch size: {train_config.get('batch_size', 'N/A')}")
                info.append(f"  Learning rate: {train_config.get('learning_rate', 'N/A')}")
                info.append(f"  Category: {train_config.get('category_filter', 'all')}")
            
            self.checkpoint_info.setText('\n'.join(info))
            
        except Exception as e:
            self.checkpoint_info.setText(f"Ошибка загрузки: {e}")
    
    def _delete_checkpoint(self):
        item = self.checkpoints_list.currentItem()
        if not item:
            return
        
        filename = item.text().split(' (')[0]
        
        if filename in ['best.pth', 'latest.pth']:
            QMessageBox.warning(
                self, "Предупреждение",
                f"Не рекомендуется удалять {filename}"
            )
            return
        
        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Удалить {filename}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            path = os.path.join(self.get_ckpt_path(), filename)
            try:
                os.remove(path)
                self._refresh_checkpoints_list()
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось удалить: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ВКЛАДКА НАСТРОЕК
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        paths_group = QGroupBox("Пути")
        paths_layout = QVBoxLayout()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Датасет:"))
        self.data_path = QLineEdit(self._data_path)
        row.addWidget(self.data_path)
        btn = QPushButton("📂")
        btn.setMaximumWidth(40)
        btn.clicked.connect(lambda: self._browse_folder(self.data_path))
        row.addWidget(btn)
        paths_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Чекпоинты:"))
        self.ckpt_path = QLineEdit(self._ckpt_path)
        row.addWidget(self.ckpt_path)
        btn = QPushButton("📂")
        btn.setMaximumWidth(40)
        btn.clicked.connect(lambda: self._browse_folder(self.ckpt_path))
        row.addWidget(btn)
        paths_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Результаты:"))
        self.output_path = QLineEdit(self._output_path)
        row.addWidget(self.output_path)
        btn = QPushButton("📂")
        btn.setMaximumWidth(40)
        btn.clicked.connect(lambda: self._browse_folder(self.output_path))
        row.addWidget(btn)
        paths_layout.addLayout(row)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        info_group = QGroupBox("Информация о системе")
        info_layout = QVBoxLayout()
        
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        info_layout.addWidget(QLabel(f"Устройство: {device}"))
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            info_layout.addWidget(QLabel(f"GPU: {gpu}"))
            info_layout.addWidget(QLabel(f"Память GPU: {memory:.1f} GB"))
        
        info_layout.addWidget(QLabel(f"PyTorch: {torch.__version__}"))
        info_layout.addWidget(QLabel(f"Python: {sys.version.split()[0]}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        return widget
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _browse_folder(self, line_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if folder:
            line_edit.setText(folder)
    
    def _open_folder(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        if sys.platform == 'win32':
            subprocess.run(['explorer', path])
        elif sys.platform == 'darwin':
            subprocess.run(['open', path])
        else:
            subprocess.run(['xdg-open', path])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Главная функция запуска приложения."""
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Тёмная тема
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()