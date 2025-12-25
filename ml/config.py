import json

from ml.utils.db_interface import create_database
from ml.model.model_input_formater import Formater

# инициализация конфигурации модели
try:
    with open('../ml/model_config.json', 'r', encoding='utf8') as model_config_file:
        model_config = json.load(model_config_file)
except FileNotFoundError:
    model_config = {}


# # инициализация БД
# db = create_database([])

# инициализация вспомогательного хранилища
formater = Formater(3)

