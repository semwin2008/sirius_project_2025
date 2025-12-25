import os
import json
import sys
from huggingface_hub import snapshot_download

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "ml", "model_config.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Куда качать


def get_model_name_from_config():
    """Читает имя модели из json файла."""
    if not os.path.exists(CONFIG_PATH):
        print(f"ОШИБКА: Файл конфигурации не найден: {CONFIG_PATH}")
        sys.exit(1)

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            model_name = data.get("model-name")

            if not model_name:
                print("ОШИБКА: В конфиге не найден ключ 'model-name'")
                sys.exit(1)

            return model_name.strip()

    except json.JSONDecodeError:
        print("ОШИБКА: Неверный формат JSON файла")
        sys.exit(1)


def main():
    model_name = get_model_name_from_config()
    local_model_path = os.path.join(MODELS_DIR, model_name.replace("/", "_"))

    print(f"--- Загрузчик Моделей ---")
    print(f"Читаю конфиг: {CONFIG_PATH}")
    print(f"Целевая модель: {model_name}")
    print(f"Папка назначения: {local_model_path}")
    print("-" * 30)

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_model_path,
            local_dir_use_symlinks=False
        )
        print("\nУСПЕХ: Модель скачана и готова к работе.")
    except Exception as e:
        print(f"\nОШИБКА при скачивании: {e}")


if __name__ == "__main__":
    main()