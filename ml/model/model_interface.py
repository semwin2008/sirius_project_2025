from typing import Any
from ml.utils import preprocess
from ml.config import model_config as cfg, formater
from time import time_ns
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json


def get_model_reply(robot_info) -> dict[str, list[Any] | str | int] | None | Any:
    if not cfg:
        return None

    batch_id = len(formater.array) + 1
    with open(f'../transported/batch_{batch_id}.json', 'r') as f:
        robot_info = json.load(f)

    print('Batch id: ', batch_id)
    # Агрегация данных с предыдущих запросов
    # print(robot_info)
    print('!!!!'*20)
    MAX_VALUE = 95
    formater.add(robot_info)
    robot_info = formater.get()
    odom_len = len(robot_info['odometries'])
    print('Odometry length: ', odom_len)
    # print(robot_info)
    # Если данных мало, не вызываем модель
    if odom_len < MAX_VALUE:
        print('Too less data to predict, returning mute signal.')
        return {
            "interest": 0,
            "joke": ""
        }

    # Обработка данных
    model_input = preprocess.preprocess_data(robot_info)
    # print('Model input: ', model_input)
    if not model_input:
        return {
            "interest": 0,
            "joke": ""
        }

    # Очищаем formater, чтобы не смотреть на эту ситуацию в ближайшее время
    formater.flush()

    # 3. Подключаем модель
    llm = ChatOllama(
        model=cfg['model-name'],
        temperature=cfg['temperature'],
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg['system-prompt']),
        ("human", cfg['basic-query'])
    ])


    chain = prompt | llm

    start_time = time_ns()
    # invoke теперь вернет уже готовый объект RobotAnalysisResponse, а не строку
    output_obj = chain.invoke({"model_input": model_input})
    end_time = time_ns()

    print(f'LLM prompt runned in\t\t\t {(end_time - start_time) * 1e-9:.3f}')

    ans_dict = {
        "interest": 1,
        "joke": output_obj.content.split('</think>')[1],
    }
    return ans_dict
