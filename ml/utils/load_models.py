import os
import json
import sys
from huggingface_hub import snapshot_download

# --- НАСТРОЙКА ПУТЕЙ ---
# Получаем абсолютный путь к папке, где лежит ЭТОТ скрипт (ml/utils)
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Поднимаемся на уровень выше, чтобы получить папку ml
ML_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)

# Поднимаемся еще на уровень выше, чтобы получить корень проекта
PROJECT_ROOT = os.path.dirname(ML_DIR)

# Собираем итоговые пути
# Конфиг лежит в папке ml/model_config.json
CONFIG_PATH = os.path.join(ML_DIR, "model_config.json")

# Модели сохраняем в корень проекта в папку models
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# -----------------------

def get_model_name_from_config():
    """Читает имя модели из json файла."""
    if not os.path.exists(CONFIG_PATH):
        print(f"ОШИБКА: Файл конфигурации не найден по пути: {CONFIG_PATH}")
        sys.exit(1)

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            model_name = data.get("model-name")

            if not model_name:
                print("ОШИБКА: В конфиге не найден ключ 'model-name'")
                sys.exit(1)

            return model_name.strip()[6:]

    except json.JSONDecodeError:
        print("ОШИБКА: Неверный формат JSON файла")
        sys.exit(1)


def main():
    model_name = get_model_name_from_config()
    os.system(f'echo loading ollama')
    os.system('curl -fsSL https://ollama.com/install.sh | sh')
    os.system(f'echo loading model from hf.co: {model_name}')
    os.system(f'ollama pull hf.co/{model_name}')



