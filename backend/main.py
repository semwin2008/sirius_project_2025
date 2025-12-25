import random

import uvicorn
from fastapi import FastAPI, HTTPException

from backend.models.robot_input import RobotInput
from backend.models.robot_response import RobotResponse
from ml.model.model_interface import get_model_reply

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

@app.post(
    "/generate",
    response_model=RobotResponse,
    summary="Анализ телеметрии и генерация отчета",
    tags=["AI Analysis"],
    description="Этот метод принимает сырые данные с сенсоров робота и возвращает интерпретацию ситуации."
)


async def generate_response(inputs: RobotInput):
    """
    **Обрабатывает пакет данных с сенсоров робота (Lidar, Odometry) и генерирует текстовое описание поведения.**
    Функция выполняет следующие шаги:
    1.  **Валидация:** Проверяет структуру входящего JSON на соответствие схеме `RobotInput`.
    2.  **Обработка:** Передает данные во внутреннюю ML-модель (`get_model_reply`).
    3.  **Ответ:** Возвращает JSON с анализом аномалий и описанием движения.

    - **Входящие данные**:
        - `odometries`: Список координат, скоростей и ориентации.
        - `lidar_scans`: Массивы расстояний с лазерного дальномера.
        - `batch_id` и временные метки.

    - **Возвращаемые поля**:
        - `Movement info`: Текстовое описание того, что делает робот (например, "Движется прямо без препятствий").
        - `Anomalies`: Список обнаруженных странностей в данных.
        - `Possible Danger`: Предупреждения о столкновениях или опасных маневрах.
        - `interest`: Числовой коэффициент "интересности" события (0.0 - 1.0).
    """
    try:
        input_data = inputs.model_dump()
        outputs = get_model_reply(input_data)
        # outputs['Anomalies'] = ['Robot is dancing!!!']
        return RobotResponse(**outputs)

    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

