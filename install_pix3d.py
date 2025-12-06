#!/usr/bin/env python3
"""
Скрипт для автоматической загрузки и установки датасета Pix3D
с сырыми отсканированными CAD моделями.

Источник: http://pix3d.csail.mit.edu/
Размер: ~5.8GB
"""

import os
import sys
import zipfile
import urllib.request
import hashlib
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Прогресс-бар для отслеживания загрузки."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_progress(url: str, output_path: str) -> None:
    """
    Загрузка файла с отображением прогресса.
    
    Args:
        url: URL для загрузки
        output_path: Путь для сохранения файла
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Загрузка") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Распаковка ZIP архива с отображением прогресса.
    
    Args:
        zip_path: Путь к ZIP файлу
        extract_to: Директория для распаковки
    """
    print(f"\n📦 Распаковка архива в: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        
        with tqdm(total=len(members), desc="Распаковка", unit="файл") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)


def check_disk_space(required_gb: float, path: str = ".") -> bool:
    """
    Проверка доступного места на диске.
    
    Args:
        required_gb: Требуемое место в гигабайтах
        path: Путь для проверки
        
    Returns:
        True если места достаточно
    """
    import shutil
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    
    print(f"💾 Свободно на диске: {free_gb:.2f} GB")
    print(f"📊 Требуется: ~{required_gb} GB (архив + распакованные данные)")
    
    return free_gb >= required_gb


def install_pix3d(
    install_dir: str = None,
    keep_zip: bool = False,
    force_download: bool = False
) -> str:
    """
    Основная функция установки датасета Pix3D.
    
    Args:
        install_dir: Директория для установки (по умолчанию - текущая)
        keep_zip: Сохранить ZIP архив после распаковки
        force_download: Принудительно перезагрузить даже если файл существует
        
    Returns:
        Путь к установленному датасету
    """
    
    # URL датасета Pix3D с CAD моделями
    DATASET_URL = "http://pix3d.csail.mit.edu/data/pix3d.zip"
    
    # Альтернативные URL (если основной недоступен)
    ALTERNATIVE_URLS = [
        "http://pix3d.csail.mit.edu/data/pix3d.zip",
    ]
    
    # Размер и требования
    ARCHIVE_SIZE_GB = 5.8
    REQUIRED_SPACE_GB = 15.0  # Архив + распакованные данные + запас
    
    # Определяем директорию установки
    if install_dir is None:
        install_dir = os.getcwd()
    
    install_path = Path(install_dir).resolve()
    install_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = install_path / "pix3d.zip"
    dataset_path = install_path / "pix3d"
    
    print("=" * 60)
    print("🎨 Установщик датасета Pix3D")
    print("=" * 60)
    print(f"📁 Директория установки: {install_path}")
    print(f"🔗 URL: {DATASET_URL}")
    print(f"📦 Размер архива: ~{ARCHIVE_SIZE_GB} GB")
    print("=" * 60)
    
    # Проверка места на диске
    if not check_disk_space(REQUIRED_SPACE_GB, str(install_path)):
        print("\n❌ Недостаточно места на диске!")
        print(f"   Освободите минимум {REQUIRED_SPACE_GB} GB и попробуйте снова.")
        sys.exit(1)
    
    # Проверяем, существует ли уже датасет
    if dataset_path.exists() and not force_download:
        print(f"\n✅ Датасет уже установлен: {dataset_path}")
        response = input("Переустановить? (y/n): ").strip().lower()
        if response != 'y':
            return str(dataset_path)
    
    # Загрузка архива
    if zip_path.exists() and not force_download:
        print(f"\n📥 Архив уже загружен: {zip_path}")
        response = input("Использовать существующий архив? (y/n): ").strip().lower()
        if response != 'y':
            os.remove(zip_path)
    
    if not zip_path.exists():
        print(f"\n⬇️  Начинаем загрузку датасета Pix3D...")
        print(f"   Это может занять некоторое время (~{ARCHIVE_SIZE_GB} GB)")
        
        try:
            download_with_progress(DATASET_URL, str(zip_path))
            print("\n✅ Загрузка завершена!")
            
        except Exception as e:
            print(f"\n❌ Ошибка загрузки: {e}")
            
            # Пробуем альтернативные URL
            for alt_url in ALTERNATIVE_URLS:
                if alt_url != DATASET_URL:
                    print(f"🔄 Пробуем альтернативный URL: {alt_url}")
                    try:
                        download_with_progress(alt_url, str(zip_path))
                        print("\n✅ Загрузка завершена!")
                        break
                    except Exception as e2:
                        print(f"❌ Ошибка: {e2}")
            else:
                print("\n❌ Не удалось загрузить датасет.")
                print("   Попробуйте скачать вручную с: http://pix3d.csail.mit.edu/")
                sys.exit(1)
    
    # Проверка целостности архива
    print("\n🔍 Проверка целостности архива...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            bad_file = zf.testzip()
            if bad_file:
                print(f"❌ Повреждённый файл в архиве: {bad_file}")
                sys.exit(1)
        print("✅ Архив в порядке!")
    except zipfile.BadZipFile:
        print("❌ Архив повреждён. Удалите его и запустите скрипт заново.")
        sys.exit(1)
    
    # Распаковка
    try:
        extract_zip(str(zip_path), str(install_path))
        print("\n✅ Распаковка завершена!")
        
    except Exception as e:
        print(f"\n❌ Ошибка распаковки: {e}")
        sys.exit(1)
    
    # Удаление архива (опционально)
    if not keep_zip and zip_path.exists():
        print("\n🗑️  Удаление архива для освобождения места...")
        os.remove(zip_path)
        print(f"   Освобождено ~{ARCHIVE_SIZE_GB} GB")
    
    # Информация об установленном датасете
    print("\n" + "=" * 60)
    print("✅ УСТАНОВКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print(f"📁 Путь к датасету: {dataset_path}")
    
    # Показываем структуру
    if dataset_path.exists():
        print("\n📂 Структура датасета:")
        for item in sorted(dataset_path.iterdir())[:10]:
            if item.is_dir():
                print(f"   📁 {item.name}/")
            else:
                print(f"   📄 {item.name}")
        
        # Считаем количество файлов
        total_files = sum(1 for _ in dataset_path.rglob("*") if _.is_file())
        print(f"\n   📊 Всего файлов: {total_files}")
    
    print("\n" + "=" * 60)
    
    return str(dataset_path)


def main():
    """Точка входа скрипта."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Загрузка и установка датасета Pix3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python install_pix3d.py                    # Установка в текущую директорию
  python install_pix3d.py -d ./datasets      # Установка в указанную директорию
  python install_pix3d.py --keep-zip         # Сохранить архив после распаковки
  python install_pix3d.py --force            # Принудительная переустановка
        """
    )
    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=None,
        help="Директория для установки датасета (по умолчанию: текущая)"
    )
    
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Сохранить ZIP архив после распаковки"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительно перезагрузить датасет"
    )
    
    args = parser.parse_args()
    
    # Проверяем наличие tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        print("📦 Установка зависимости tqdm...")
        os.system(f"{sys.executable} -m pip install tqdm")
        from tqdm import tqdm
    
    # Запускаем установку
    dataset_path = install_pix3d(
        install_dir=args.directory,
        keep_zip=args.keep_zip,
        force_download=args.force
    )
    
    print(f"\n🎉 Датасет готов к использованию: {dataset_path}")
    
    # Пример использования
    print("\n📝 Пример загрузки датасета в Python:")
    print("-" * 40)
    print(f'''
import json
from pathlib import Path

dataset_path = Path("{dataset_path}")

# Загрузка аннотаций
with open(dataset_path / "pix3d.json", "r") as f:
    annotations = json.load(f)

print(f"Количество образцов: {{len(annotations)}}")
    ''')


if __name__ == "__main__":
    main()