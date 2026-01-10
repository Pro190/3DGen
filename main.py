import sys
import os
import torch
import subprocess
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget, QComboBox,
    QMessageBox, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from datetime import datetime

# Импорт модулей проекта
from model import Pixel2Mesh
from torchvision import transforms


class TrainingThread(QThread):
    """
    Поток для запуска train.py как отдельного процесса.
    Отслеживает вывод и парсит прогресс обучения.
    """
    progress = pyqtSignal(int, float, str)  # epoch, loss, message
    finished = pyqtSignal(str)  # final message
    log_message = pyqtSignal(str)  # raw log output
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process = None
        self.is_running = True
        
    def run(self):
        """Запуск train.py как subprocess"""
        try:
            import subprocess
            
            # Формирование команды запуска
            python_exec = sys.executable
            train_script = os.path.join(os.path.dirname(__file__), 'train.py')
            
            if not os.path.exists(train_script):
                self.finished.emit(f"Ошибка: train.py не найден по пути {train_script}")
                return
            
            # Подготовка переменных окружения
            env = os.environ.copy()
            env['PIX3D_DATA_ROOT'] = self.config['data_root']
            env['PIX3D_JSON_PATH'] = self.config['json_path']
            env['PIX3D_CHECKPOINT_DIR'] = self.config['checkpoint_dir']
            env['PIX3D_NUM_EPOCHS'] = str(self.config['num_epochs'])
            env['PIX3D_LEARNING_RATE'] = str(self.config['learning_rate'])
            env['PIX3D_BATCH_SIZE'] = str(self.config['batch_size'])
            env['PIX3D_SUBDIVISIONS'] = str(self.config['subdivisions'])
            env['PYTHONUNBUFFERED'] = '1'  # Отключение буферизации
            
            self.log_message.emit(f"Запуск обучения: {train_script}")
            self.log_message.emit(f"Python: {python_exec}")
            self.log_message.emit("=" * 70)
            
            # Запуск процесса с subprocess
            self.process = subprocess.Popen(
                [python_exec, '-u', train_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            # Чтение вывода построчно в реальном времени
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                
                if line:
                    line = line.rstrip()
                    # Отправка сырого лога
                    self.log_message.emit(line)
                    
                    # Парсинг прогресса
                    epoch_match = re.search(r'Epoch \[(\d+)/(\d+)\].*Loss: ([\d.]+)', line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        total_epochs = int(epoch_match.group(2))
                        loss = float(epoch_match.group(3))
                        
                        progress_pct = int((current_epoch / total_epochs) * 100)
                        message = f"Эпоха {current_epoch}/{total_epochs}: Loss = {loss:.4f}"
                        self.progress.emit(progress_pct, loss, message)
            
            # Ожидание завершения
            self.process.wait()
            
            if self.process.returncode == 0:
                self.finished.emit("✓ Обучение успешно завершено!")
            else:
                self.finished.emit(f"Обучение прервано с кодом {self.process.returncode}")
            
        except Exception as e:
            self.finished.emit(f"Ошибка при запуске обучения: {str(e)}")
    
    def stop(self):
        """Остановка обучения"""
        self.is_running = False
        if self.process:
            self.log_message.emit("\nОстановка процесса обучения...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
                self.process.wait()


class InferenceThread(QThread):
    """
    Поток для генерации 3D моделей через infer.py
    """
    progress = pyqtSignal(int, str)  # progress, message
    finished = pyqtSignal(str, object)  # message, result_data
    log_message = pyqtSignal(str)  # raw log output
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process = None
        self.is_running = True
        
    def run(self):
        """Запуск infer.py как subprocess"""
        try:
            import subprocess
            import tempfile
            import shutil
            
            # Формирование команды запуска
            python_exec = sys.executable
            infer_script = os.path.join(os.path.dirname(__file__), 'infer.py')
            
            if not os.path.exists(infer_script):
                self.finished.emit(f"Ошибка: infer.py не найден по пути {infer_script}", None)
                return
            
            self.progress.emit(10, "Подготовка к генерации...")
            
            # Создаем временную директорию для результата
            temp_dir = tempfile.mkdtemp(prefix='pixel2mesh_inference_')
            
            # Формируем аргументы командной строки для infer.py
            args = [
                python_exec, '-u', infer_script,
                '--checkpoint', self.config['checkpoint_path'],
                '--output', temp_dir,
                '--scale', str(self.config['scale']),
                '--subdivisions', str(self.config['subdivisions']),
                '--seed', str(self.config['seed'])
            ]
            
            # Если указан путь к изображению - режим одного файла
            if self.config.get('image_path'):
                args.extend(['--image', self.config['image_path']])
            # Иначе - режим датасета
            else:
                if self.config.get('data_root'):
                    args.extend(['--data_root', self.config['data_root']])
                if self.config.get('json_path'):
                    args.extend(['--json_path', self.config['json_path']])
                args.extend(['--num_samples', '1'])
            
            self.progress.emit(20, "Запуск inference...")
            self.log_message.emit(f"Команда: {' '.join(args)}")
            self.log_message.emit("=" * 70)
            
            # Подготовка переменных окружения
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Запуск процесса
            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            self.progress.emit(40, "Генерация модели...")
            
            # Чтение вывода
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                
                if line:
                    line = line.rstrip()
                    self.log_message.emit(line)
                    
                    # Обновление прогресса на основе вывода
                    if "Загружаю модель" in line:
                        self.progress.emit(50, "Загрузка модели...")
                    elif "Обработка образца" in line:
                        self.progress.emit(70, "Обработка изображения...")
                    elif "Сохранено" in line:
                        self.progress.emit(90, "Сохранение результата...")
            
            # Ожидание завершения
            self.process.wait()
            
            if self.process.returncode != 0:
                self.finished.emit(f"Ошибка при генерации (код {self.process.returncode})", None)
                return
            
            self.progress.emit(95, "Поиск результата...")
            
            # Поиск сгенерированного .obj файла
            obj_files = [f for f in os.listdir(temp_dir) if f.endswith('.obj')]
            
            if not obj_files:
                self.finished.emit("Ошибка: .obj файл не был создан", None)
                return
            
            # Копируем файл в целевую директорию
            src_path = os.path.join(temp_dir, obj_files[0])
            dst_dir = self.config['output_dir']
            os.makedirs(dst_dir, exist_ok=True)
            
            # Формируем имя файла на основе исходного изображения
            if self.config.get('image_path'):
                base_name = os.path.splitext(os.path.basename(self.config['image_path']))[0]
                dst_filename = f'{base_name}_3d.obj'
            else:
                dst_filename = obj_files[0]
            
            dst_path = os.path.join(dst_dir, dst_filename)
            shutil.copy2(src_path, dst_path)
            
            # Удаляем временную директорию
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Подсчет вершин и граней
            num_vertices = 0
            num_faces = 0
            with open(dst_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        num_vertices += 1
                    elif line.startswith('f '):
                        num_faces += 1
            
            result_data = {
                'output_path': dst_path,
                'num_vertices': num_vertices,
                'num_faces': num_faces
            }
            
            self.progress.emit(100, "Готово!")
            self.finished.emit(f"Модель сохранена: {dst_path}", result_data)
            
        except Exception as e:
            import traceback
            error_msg = f"Ошибка: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg)
            self.finished.emit(f"Ошибка: {str(e)}", None)
    
    def stop(self):
        """Остановка генерации"""
        self.is_running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
                self.process.wait()


class MainWindow(QMainWindow):
    """
    Главное окно приложения для работы с Pixel2Mesh.
    Содержит вкладки для обучения, тестирования и inference.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel2Mesh - 3D Реконструкция из изображений")
        self.setGeometry(100, 100, 1200, 800)
        
        # Инициализация переменных (ВАЖНО: до создания вкладок!)
        self.current_image_path = None
        self.loss_history = []
        
        # Создание вкладок
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # ВАЖНО: Настройки должны быть созданы ПЕРВЫМИ,
        # т.к. другие вкладки используют self.ckpt_path_edit
        self.tab_settings = self.create_settings_tab()
        self.tabs.addTab(self.tab_settings, "Настройки")
        
        # Вкладка 1: Обучение
        self.tab_train = self.create_training_tab()
        self.tabs.insertTab(0, self.tab_train, "Обучение модели")
        
        # Вкладка 2: Inference
        self.tab_infer = self.create_inference_tab()
        self.tabs.insertTab(1, self.tab_infer, "Генерация 3D")
        
        # Устанавливаем первую вкладку активной
        self.tabs.setCurrentIndex(0)
        
        # Статус бар
        self.statusBar().showMessage("Готов к работе")
        
    def create_training_tab(self):
        """Создание вкладки обучения"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Группа: Параметры обучения
        group_params = QGroupBox("Параметры обучения")
        params_layout = QVBoxLayout()
        
        # Количество эпох
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Количество эпох:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(150)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addStretch()
        params_layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 0.01)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setSingleStep(0.00001)
        lr_layout.addWidget(self.lr_spin)
        lr_layout.addStretch()
        params_layout.addLayout(lr_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16)
        self.batch_spin.setValue(1)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        params_layout.addLayout(batch_layout)
        
        # Subdivisions
        subdiv_layout = QHBoxLayout()
        subdiv_layout.addWidget(QLabel("Subdivisions:"))
        self.subdiv_combo = QComboBox()
        self.subdiv_combo.addItems(["3 (642 вершины)", "4 (2562 вершины)"])
        subdiv_layout.addWidget(self.subdiv_combo)
        subdiv_layout.addStretch()
        params_layout.addLayout(subdiv_layout)
        
        group_params.setLayout(params_layout)
        layout.addWidget(group_params)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        self.btn_start_train = QPushButton("▶ Начать обучение")
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_start_train.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        buttons_layout.addWidget(self.btn_start_train)
        
        self.btn_stop_train = QPushButton("⏹ Остановить")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self.stop_training)
        buttons_layout.addWidget(self.btn_stop_train)
        
        self.btn_clear_log = QPushButton("🗑 Очистить лог")
        self.btn_clear_log.clicked.connect(lambda: self.train_log.clear())
        buttons_layout.addWidget(self.btn_clear_log)
        
        layout.addLayout(buttons_layout)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Лог обучения
        log_label = QLabel("Лог обучения:")
        layout.addWidget(log_label)
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("QTextEdit { font-family: 'Consolas', 'Courier New', monospace; font-size: 10pt; }")
        layout.addWidget(self.train_log)
        
        widget.setLayout(layout)
        return widget
    
    def create_inference_tab(self):
        """Создание вкладки генерации"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Группа: Источник изображения
        source_group = QGroupBox("Источник изображения")
        source_layout = QVBoxLayout()
        
        # Радиокнопки для выбора источника
        self.source_custom = QPushButton("📁 Загрузить свое изображение")
        self.source_custom.clicked.connect(self.load_custom_image)
        source_layout.addWidget(self.source_custom)
        
        self.source_dataset = QPushButton("🎲 Случайное из датасета")
        self.source_dataset.clicked.connect(self.load_random_from_dataset)
        source_layout.addWidget(self.source_dataset)
        
        self.image_path_label = QLabel("Файл не выбран")
        self.image_path_label.setStyleSheet("QLabel { color: gray; }")
        source_layout.addWidget(self.image_path_label)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Превью изображения
        preview_group = QGroupBox("Предпросмотр изображения")
        preview_layout = QVBoxLayout()
        
        self.image_preview = QLabel()
        self.image_preview.setFixedSize(400, 400)
        self.image_preview.setStyleSheet("QLabel { border: 2px solid #ccc; background-color: #f0f0f0; }")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setText("Изображение не загружено")
        preview_layout.addWidget(self.image_preview, alignment=Qt.AlignCenter)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Параметры генерации
        params_group = QGroupBox("Параметры генерации")
        params_layout = QVBoxLayout()
        
        # Выбор чекпоинта
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(QLabel("Модель:"))
        self.checkpoint_combo = QComboBox()
        self.refresh_checkpoints()
        checkpoint_layout.addWidget(self.checkpoint_combo)
        
        self.btn_refresh_ckpt = QPushButton("🔄")
        self.btn_refresh_ckpt.setMaximumWidth(40)
        self.btn_refresh_ckpt.clicked.connect(self.refresh_checkpoints)
        checkpoint_layout.addWidget(self.btn_refresh_ckpt)
        params_layout.addLayout(checkpoint_layout)
        
        # Масштаб
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Масштаб:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 1000.0)
        self.scale_spin.setValue(100.0)
        self.scale_spin.setSingleStep(10.0)
        scale_layout.addWidget(self.scale_spin)
        scale_layout.addStretch()
        params_layout.addLayout(scale_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Кнопка генерации
        self.btn_generate = QPushButton("✨ Сгенерировать 3D модель")
        self.btn_generate.clicked.connect(self.generate_3d)
        self.btn_generate.setEnabled(False)
        self.btn_generate.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        layout.addWidget(self.btn_generate)
        
        # Прогресс
        self.infer_progress = QProgressBar()
        self.infer_progress.setTextVisible(True)
        layout.addWidget(self.infer_progress)
        
        # Лог inference
        log_label = QLabel("Лог генерации:")
        layout.addWidget(log_label)
        
        self.infer_log = QTextEdit()
        self.infer_log.setReadOnly(True)
        self.infer_log.setMaximumHeight(150)
        self.infer_log.setStyleSheet("QTextEdit { font-family: 'Consolas', 'Courier New', monospace; font-size: 9pt; }")
        layout.addWidget(self.infer_log)
        
        # Результат
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_settings_tab(self):
        """Создание вкладки настроек"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Группа: Пути к данным
        paths_group = QGroupBox("Пути к данным")
        paths_layout = QVBoxLayout()
        
        # Путь к датасету
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Датасет:"))
        self.data_path_edit = QLineEdit("./PIX3D_DATA")
        data_layout.addWidget(self.data_path_edit)
        btn_browse_data = QPushButton("📁")
        btn_browse_data.setMaximumWidth(40)
        btn_browse_data.clicked.connect(lambda: self.browse_folder(self.data_path_edit))
        data_layout.addWidget(btn_browse_data)
        paths_layout.addLayout(data_layout)
        
        # Путь к JSON
        json_layout = QHBoxLayout()
        json_layout.addWidget(QLabel("JSON файл:"))
        self.json_path_edit = QLineEdit("./PIX3D_DATA/pix3d.json")
        json_layout.addWidget(self.json_path_edit)
        btn_browse_json = QPushButton("📁")
        btn_browse_json.setMaximumWidth(40)
        btn_browse_json.clicked.connect(lambda: self.browse_file(self.json_path_edit))
        json_layout.addWidget(btn_browse_json)
        paths_layout.addLayout(json_layout)
        
        # Директория чекпоинтов
        ckpt_layout = QHBoxLayout()
        ckpt_layout.addWidget(QLabel("Модели:"))
        self.ckpt_path_edit = QLineEdit("./checkpoints")
        ckpt_layout.addWidget(self.ckpt_path_edit)
        btn_browse_ckpt = QPushButton("📁")
        btn_browse_ckpt.setMaximumWidth(40)
        btn_browse_ckpt.clicked.connect(lambda: self.browse_folder(self.ckpt_path_edit))
        ckpt_layout.addWidget(btn_browse_ckpt)
        paths_layout.addLayout(ckpt_layout)
        
        # Директория результатов
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Результаты:"))
        self.output_path_edit = QLineEdit("./inference_results")
        output_layout.addWidget(self.output_path_edit)
        btn_browse_output = QPushButton("📁")
        btn_browse_output.setMaximumWidth(40)
        btn_browse_output.clicked.connect(lambda: self.browse_folder(self.output_path_edit))
        output_layout.addWidget(btn_browse_output)
        paths_layout.addLayout(output_layout)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Информация о системе
        info_group = QGroupBox("Информация о системе")
        info_layout = QVBoxLayout()
        
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        info_layout.addWidget(QLabel(f"🖥 Устройство: {device}"))
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info_layout.addWidget(QLabel(f"🎮 GPU: {gpu_name}"))
            info_layout.addWidget(QLabel(f"💾 Память: {gpu_memory:.1f} GB"))
        
        info_layout.addWidget(QLabel(f"🐍 Python: {sys.version.split()[0]}"))
        info_layout.addWidget(QLabel(f"🔥 PyTorch: {torch.__version__}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Кнопка справки
        btn_help = QPushButton("ℹ О программе")
        btn_help.clicked.connect(self.show_about)
        layout.addWidget(btn_help)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def browse_folder(self, line_edit):
        """Выбор папки"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку", line_edit.text())
        if folder:
            line_edit.setText(folder)
    
    def browse_file(self, line_edit):
        """Выбор файла"""
        file, _ = QFileDialog.getOpenFileName(self, "Выберите файл", line_edit.text())
        if file:
            line_edit.setText(file)
    
    def refresh_checkpoints(self):
        """Обновление списка доступных чекпоинтов"""
        if not hasattr(self, 'checkpoint_combo'):
            return  # Виджет еще не создан
            
        self.checkpoint_combo.clear()
        
        # Используем значение по умолчанию, если виджет настроек еще не создан
        if hasattr(self, 'ckpt_path_edit'):
            ckpt_dir = self.ckpt_path_edit.text()
        else:
            ckpt_dir = './checkpoints'
        
        if not os.path.exists(ckpt_dir):
            self.checkpoint_combo.addItem("Нет доступных моделей")
            return
        
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        
        if not checkpoints:
            self.checkpoint_combo.addItem("Нет доступных моделей")
        else:
            # Сортировка по времени изменения (новые первые)
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
            for ckpt in checkpoints:
                self.checkpoint_combo.addItem(ckpt)
    
    def start_training(self):
        """Запуск обучения через train.py"""
        config = {
            'data_root': self.data_path_edit.text(),
            'json_path': self.json_path_edit.text(),
            'checkpoint_dir': self.ckpt_path_edit.text(),
            'num_epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'batch_size': self.batch_spin.value(),
            'subdivisions': 3 if "3" in self.subdiv_combo.currentText() else 4,
        }
        
        # Проверка путей
        if not os.path.exists(config['data_root']):
            QMessageBox.warning(self, "Ошибка", f"Директория датасета не найдена:\n{config['data_root']}")
            return
        
        if not os.path.exists(config['json_path']):
            QMessageBox.warning(self, "Ошибка", f"JSON файл не найден:\n{config['json_path']}")
            return
        
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Очистка истории loss
        self.loss_history = []
        
        # Запуск потока обучения
        self.train_thread = TrainingThread(config)
        self.train_thread.progress.connect(self.update_training_progress)
        self.train_thread.finished.connect(self.training_finished)
        self.train_thread.log_message.connect(self.append_training_log)
        self.train_thread.start()
        
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.progress_bar.setValue(0)
        self.train_log.clear()
        self.train_log.append("=" * 70)
        self.train_log.append("ЗАПУСК ОБУЧЕНИЯ")
        self.train_log.append("=" * 70)
        self.statusBar().showMessage("Обучение запущено...")
        
    def stop_training(self):
        """Остановка обучения"""
        if hasattr(self, 'train_thread'):
            reply = QMessageBox.question(
                self, 'Подтверждение',
                'Вы уверены что хотите остановить обучение?\nПрогресс будет сохранен.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.train_thread.stop()
                self.train_log.append("\n" + "!" * 70)
                self.train_log.append("ОСТАНОВКА ОБУЧЕНИЯ...")
                self.train_log.append("!" * 70)
    
    def append_training_log(self, message):
        """Добавление сообщения в лог обучения"""
        self.train_log.append(message.rstrip())
        # Автоскролл вниз
        self.train_log.verticalScrollBar().setValue(
            self.train_log.verticalScrollBar().maximum()
        )
    
    def update_training_progress(self, progress, loss, message):
        """Обновление прогресса обучения"""
        self.progress_bar.setValue(progress)
        self.loss_history.append(loss)
        self.statusBar().showMessage(message)
    
    def training_finished(self, message):
        """Завершение обучения"""
        self.train_log.append("\n" + "=" * 70)
        self.train_log.append(message)
        self.train_log.append("=" * 70)
        
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.statusBar().showMessage(message)
        
        # Обновление списка чекпоинтов
        self.refresh_checkpoints()
        
        QMessageBox.information(self, "Обучение завершено", message)
    
    def load_custom_image(self):
        """Загрузка пользовательского изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение",
            "", "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.current_image_source = 'custom'
            self.image_path_label.setText(f"Файл: {os.path.basename(file_path)}")
            self.image_path_label.setStyleSheet("QLabel { color: black; font-weight: bold; }")
            
            # Показать превью
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview.setPixmap(pixmap)
            self.image_preview.setText("")
            
            self.btn_generate.setEnabled(True)
            self.statusBar().showMessage(f"Загружено: {os.path.basename(file_path)}")
    
    def load_random_from_dataset(self):
        """Загрузка случайного изображения из датасета"""
        try:
            import random
            from datasets import Pix3DDataset
            
            data_root = self.data_path_edit.text()
            json_path = self.json_path_edit.text()
            
            if not os.path.exists(data_root) or not os.path.exists(json_path):
                QMessageBox.warning(self, "Ошибка", 
                                  "Датасет не найден! Проверьте пути в настройках.")
                return
            
            # Загрузка датасета
            dataset = Pix3DDataset(data_root, json_path)
            
            if len(dataset) == 0:
                QMessageBox.warning(self, "Ошибка", "Датасет пуст!")
                return
            
            # Выбор случайного образца
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset.samples[idx]
            
            self.current_image_path = sample['img']
            self.current_image_source = 'dataset'
            self.current_dataset_index = idx
            
            self.image_path_label.setText(f"Датасет: {sample['category']} (индекс {idx})")
            self.image_path_label.setStyleSheet("QLabel { color: black; font-weight: bold; }")
            
            # Показать превью
            pixmap = QPixmap(self.current_image_path)
            pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview.setPixmap(pixmap)
            self.image_preview.setText("")
            
            self.btn_generate.setEnabled(True)
            self.statusBar().showMessage(f"Загружен образец из датасета: {sample['category']}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить из датасета:\n{str(e)}")
    
    def generate_3d(self):
        """Генерация 3D модели через infer.py"""
        checkpoint_name = self.checkpoint_combo.currentText()
        
        if checkpoint_name == "Нет доступных моделей":
            QMessageBox.warning(self, "Ошибка", 
                              "Нет доступных моделей!\nСначала обучите модель или загрузите чекпоинт.")
            return
        
        checkpoint_path = os.path.join(self.ckpt_path_edit.text(), checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            QMessageBox.warning(self, "Ошибка", 
                              f"Чекпоинт не найден:\n{checkpoint_path}")
            return
        
        subdivisions = 3 if "3" in self.subdiv_combo.currentText() else 4
        output_dir = self.output_path_edit.text()
        
        config = {
            'checkpoint_path': checkpoint_path,
            'output_dir': output_dir,
            'scale': self.scale_spin.value(),
            'subdivisions': subdivisions,
            'seed': 42,
            'image_path': self.current_image_path,
            'data_root': self.data_path_edit.text(),
            'json_path': self.json_path_edit.text()
        }
        
        self.infer_thread = InferenceThread(config)
        self.infer_thread.progress.connect(self.update_inference_progress)
        self.infer_thread.finished.connect(self.inference_finished)
        self.infer_thread.log_message.connect(self.append_inference_log)
        self.infer_thread.start()
        
        self.btn_generate.setEnabled(False)
        self.infer_progress.setValue(0)
        self.result_label.setText("")
        self.infer_log.clear()
        self.statusBar().showMessage("Генерация 3D модели...")
    
    def append_inference_log(self, message):
        """Добавление сообщения в лог inference"""
        self.infer_log.append(message.rstrip())
        self.infer_log.verticalScrollBar().setValue(
            self.infer_log.verticalScrollBar().maximum()
        )
    
    def update_inference_progress(self, progress, message):
        """Обновление прогресса генерации"""
        self.infer_progress.setValue(progress)
        self.result_label.setText(message)
        self.result_label.setStyleSheet("QLabel { color: blue; }")
        self.statusBar().showMessage(message)
    
    def inference_finished(self, message, result_data):
        """Завершение генерации"""
        self.result_label.setText(message)
        self.btn_generate.setEnabled(True)
        
        if result_data:
            self.result_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            self.statusBar().showMessage("Генерация завершена успешно!")
            
            # Детальная информация
            details = (
                f"✓ 3D модель успешно создана!\n\n"
                f"📁 Файл: {result_data['output_path']}\n"
                f"📊 Вершин: {result_data['num_vertices']}\n"
                f"🔺 Граней: {result_data['num_faces']}\n\n"
                f"Модель можно открыть в:\n"
                f"• Blender (File → Import → Wavefront .obj)\n"
                f"• MeshLab\n"
                f"• CloudCompare\n"
                f"• 3D Viewer (Windows)"
            )
            
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Успех")
            msg_box.setText(details)
            msg_box.setIcon(QMessageBox.Information)
            
            # Кнопка для открытия папки
            btn_open_folder = msg_box.addButton("Открыть папку", QMessageBox.ActionRole)
            msg_box.addButton(QMessageBox.Ok)
            
            msg_box.exec_()
            
            if msg_box.clickedButton() == btn_open_folder:
                self.open_folder(os.path.dirname(result_data['output_path']))
        else:
            self.result_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            self.statusBar().showMessage("Ошибка при генерации")
    
    def open_folder(self, path):
        """Открытие папки в проводнике"""
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    
    def show_about(self):
        """Показать информацию о программе"""
        about_text = """
        <h2>Pixel2Mesh - 3D Реконструкция</h2>
        <p><b>Версия:</b> 1.0</p>
        <p><b>Описание:</b> Система генерации 3D моделей мебели из одиночных изображений 
        на основе глубоких нейронных сетей.</p>
        <p><b>Технологии:</b></p>
        <ul>
            <li>PyTorch - фреймворк глубокого обучения</li>
            <li>ResNet18 - энкодер изображений</li>
            <li>Graph Convolutional Networks - декодер сеток</li>
            <li>PyQt5 - графический интерфейс</li>
        </ul>
        <p><b>Датасет:</b> Pix3D (мебель)</p>
        <hr>
        <p><i>Разработано в рамках выпускной квалификационной работы</i></p>
        """
        
        QMessageBox.about(self, "О программе", about_text)


def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    
    # Установка стиля приложения
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()