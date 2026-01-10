"""
================================================================================
Автор: Бадрханов Аслан-бек Поладович.
Руководитель: Простомолотов Андрей Сергеевич.
Тема ВКР: "Генерация трехмерных моделей мебели на основе изображения".
Описание: Точка входа в приложение; реализация графического интерфейса пользователя на базе PyQt.
Дата: 2026
================================================================================
"""
import sys
import os
import subprocess
import re
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget, QComboBox,
    QMessageBox, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap

import torch
from PIL import Image
from torchvision import transforms


class TrainingThread(QThread):
    """Поток для обучения."""
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process = None
        self.is_running = True
    
    def run(self):
        try:
            python_exec = sys.executable
            train_script = os.path.join(os.path.dirname(__file__), 'train.py')
            
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            self.log_message.emit("Запуск обучения...")
            self.log_message.emit("="*50)
            
            self.process = subprocess.Popen(
                [python_exec, '-u', train_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                
                line = line.rstrip()
                if line:
                    self.log_message.emit(line)
                    
                    # Парсинг прогресса
                    epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                    if epoch_match:
                        current = int(epoch_match.group(1))
                        total = int(epoch_match.group(2))
                        progress = int((current / total) * 100)
                        self.progress.emit(progress, f"Эпоха {current}/{total}")
            
            self.process.wait()
            
            if self.process.returncode == 0:
                self.finished.emit("✓ Обучение завершено!")
            else:
                self.finished.emit(f"Обучение прервано (код {self.process.returncode})")
                
        except Exception as e:
            self.finished.emit(f"Ошибка: {str(e)}")
    
    def stop(self):
        self.is_running = False
        if self.process:
            self.process.terminate()


class InferenceThread(QThread):
    """Поток для генерации 3D модели."""
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str, object)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def run(self):
        try:
            self.progress.emit(10, "Загрузка модели...")
            
            from model import OccupancyNetwork
            from mesh_utils import extract_mesh_marching_cubes, save_mesh
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Загрузка модели
            model = OccupancyNetwork(latent_dim=512).to(device)
            
            checkpoint_path = self.config['checkpoint']
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.log_message.emit(f"Модель загружена: {checkpoint_path}")
            else:
                self.log_message.emit("⚠️ Чекпоинт не найден, используются случайные веса")
            
            model.eval()
            
            self.progress.emit(30, "Загрузка изображения...")
            
            # Загрузка изображения
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img = Image.open(self.config['image']).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            self.progress.emit(50, "Генерация occupancy...")
            
            # Encode
            with torch.no_grad():
                latent = model.encode(img_tensor)
            
            # Функция для Marching Cubes
            def occupancy_fn(points):
                points = points.unsqueeze(0)
                with torch.no_grad():
                    logits = model.decode(latent, points)
                    return torch.sigmoid(logits).squeeze(0)
            
            self.progress.emit(70, "Marching Cubes...")
            
            # Извлечение меша
            resolution = self.config.get('resolution', 64)
            mesh = extract_mesh_marching_cubes(
                occupancy_fn,
                resolution=resolution,
                threshold=0.5,
                device=device
            )
            
            if mesh is None:
                self.finished.emit("Не удалось извлечь меш", None)
                return
            
            self.progress.emit(90, "Сохранение...")
            
            # Сохранение
            output_dir = self.config['output']
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.config['image']))[0]
            output_path = os.path.join(output_dir, f"{base_name}_3d.obj")
            
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


class MainWindow(QMainWindow):
    """Главное окно приложения."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Occupancy Network - 3D реконструкция")
        self.setGeometry(100, 100, 1000, 700)
        
        self.current_image_path = None
        
        # Вкладки
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Вкладка обучения
        self.tabs.addTab(self.create_training_tab(), "Обучение")
        
        # Вкладка генерации
        self.tabs.addTab(self.create_inference_tab(), "Генерация 3D")
        
        # Вкладка настроек
        self.tabs.addTab(self.create_settings_tab(), "Настройки")
        
        self.statusBar().showMessage("Готов")
    
    def create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Параметры
        params_group = QGroupBox("Параметры обучения")
        params_layout = QVBoxLayout()
        
        # Эпохи
        row = QHBoxLayout()
        row.addWidget(QLabel("Эпохи:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(200)
        row.addWidget(self.epochs_spin)
        row.addStretch()
        params_layout.addLayout(row)
        
        # Batch size
        row = QHBoxLayout()
        row.addWidget(QLabel("Batch size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(32)
        row.addWidget(self.batch_spin)
        row.addStretch()
        params_layout.addLayout(row)
        
        # Категория
        row = QHBoxLayout()
        row.addWidget(QLabel("Категория:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(['all', 'bed', 'chair', 'sofa', 'table', 'desk'])
        row.addWidget(self.category_combo)
        row.addStretch()
        params_layout.addLayout(row)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Кнопки
        buttons = QHBoxLayout()
        
        self.btn_train = QPushButton("▶ Начать обучение")
        self.btn_train.clicked.connect(self.start_training)
        buttons.addWidget(self.btn_train)
        
        self.btn_stop = QPushButton("⏹ Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)
        buttons.addWidget(self.btn_stop)
        
        layout.addLayout(buttons)
        
        # Прогресс
        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)
        
        # Лог
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.train_log)
        
        widget.setLayout(layout)
        return widget
    
    def create_inference_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Выбор изображения
        img_group = QGroupBox("Изображение")
        img_layout = QVBoxLayout()
        
        self.btn_load_image = QPushButton("📁 Загрузить изображение")
        self.btn_load_image.clicked.connect(self.load_image)
        img_layout.addWidget(self.btn_load_image)
        
        self.image_label = QLabel("Изображение не загружено")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        img_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)
        
        # Параметры
        params_group = QGroupBox("Параметры генерации")
        params_layout = QVBoxLayout()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Разрешение:"))
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(32, 256)
        self.resolution_spin.setValue(64)
        row.addWidget(self.resolution_spin)
        row.addStretch()
        params_layout.addLayout(row)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Кнопка генерации
        self.btn_generate = QPushButton("✨ Генерировать 3D модель")
        self.btn_generate.clicked.connect(self.generate_3d)
        self.btn_generate.setEnabled(False)
        layout.addWidget(self.btn_generate)
        
        # Прогресс
        self.infer_progress = QProgressBar()
        layout.addWidget(self.infer_progress)
        
        # Результат
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Пути
        paths_group = QGroupBox("Пути")
        paths_layout = QVBoxLayout()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Датасет:"))
        self.data_path = QLineEdit("./PIX3D_DATA")
        row.addWidget(self.data_path)
        paths_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Чекпоинты:"))
        self.ckpt_path = QLineEdit("./checkpoints")
        row.addWidget(self.ckpt_path)
        paths_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Результаты:"))
        self.output_path = QLineEdit("./inference_results")
        row.addWidget(self.output_path)
        paths_layout.addLayout(row)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Информация
        info_group = QGroupBox("Система")
        info_layout = QVBoxLayout()
        
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        info_layout.addWidget(QLabel(f"Устройство: {device}"))
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            info_layout.addWidget(QLabel(f"GPU: {gpu}"))
        
        info_layout.addWidget(QLabel(f"PyTorch: {torch.__version__}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def start_training(self):
        config = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'category': self.category_combo.currentText()
        }
        
        self.train_thread = TrainingThread(config)
        self.train_thread.progress.connect(self.update_train_progress)
        self.train_thread.log_message.connect(self.append_train_log)
        self.train_thread.finished.connect(self.training_finished)
        self.train_thread.start()
        
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.train_log.clear()
    
    def stop_training(self):
        if hasattr(self, 'train_thread'):
            self.train_thread.stop()
    
    def update_train_progress(self, value, message):
        self.train_progress.setValue(value)
        self.statusBar().showMessage(message)
    
    def append_train_log(self, message):
        self.train_log.append(message)
        self.train_log.verticalScrollBar().setValue(
            self.train_log.verticalScrollBar().maximum()
        )
    
    def training_finished(self, message):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage(message)
        QMessageBox.information(self, "Обучение", message)
    
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if path:
            self.current_image_path = path
            
            pixmap = QPixmap(path)
            pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            
            self.btn_generate.setEnabled(True)
            self.statusBar().showMessage(f"Загружено: {os.path.basename(path)}")
    
    def generate_3d(self):
        if not self.current_image_path:
            return
        
        config = {
            'image': self.current_image_path,
            'checkpoint': os.path.join(self.ckpt_path.text(), 'best.pth'),
            'output': self.output_path.text(),
            'resolution': self.resolution_spin.value()
        }
        
        self.infer_thread = InferenceThread(config)
        self.infer_thread.progress.connect(self.update_infer_progress)
        self.infer_thread.log_message.connect(lambda m: print(m))
        self.infer_thread.finished.connect(self.inference_finished)
        self.infer_thread.start()
        
        self.btn_generate.setEnabled(False)
        self.infer_progress.setValue(0)
    
    def update_infer_progress(self, value, message):
        self.infer_progress.setValue(value)
        self.result_label.setText(message)
    
    def inference_finished(self, message, result):
        self.btn_generate.setEnabled(True)
        self.result_label.setText(message)
        
        if result:
            info = f"\n\nВершин: {result['vertices']}\nГраней: {result['faces']}"
            self.result_label.setText(message + info)
            
            QMessageBox.information(
                self, "Готово",
                f"3D модель сохранена:\n{result['path']}\n\n"
                f"Вершин: {result['vertices']}\n"
                f"Граней: {result['faces']}"
            )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()