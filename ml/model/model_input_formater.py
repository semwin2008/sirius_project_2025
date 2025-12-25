def merge(data_list: list[dict]) -> dict:
    """
        Рекурсивно склеивает список словарей с вложенными списками в один словарь.

        Args:
            data_list (list): Список словарей одинаковой структуры.
                              Например: [{'a': [1,2]}, {'a': [3,4]}]
        Returns:
            dict: Объединенный словарь.
                  Например: {'a': [1,2,3,4]}
        """
    if not data_list:
        return {}

    # Берем первый элемент, чтобы понять структуру (какие есть ключи)
    template = data_list[0]

    # СЛУЧАЙ 1: Текущий элемент - это Словарь
    if isinstance(template, dict):
        result = {}
        for key in template:
            # Собираем значения для этого ключа из ВСЕХ элементов списка
            values_to_merge = [item[key] for item in data_list]
            # Рекурсивно вызываем функцию для этого списка значений
            result[key] = merge(values_to_merge)
        return result

    # СЛУЧАЙ 2: Текущий элемент - это Список (массив данных)
    elif isinstance(template, list):
        # Склеиваем списки (flatten)
        # [ [1,2], [3,4] ] -> [1, 2, 3, 4]
        merged_list = []
        for sublist in data_list:
            merged_list.extend(sublist)
        return merged_list

    # СЛУЧАЙ 3: Скалярное значение (число, строка, bool)
    else:
        # Если вдруг попалось просто число (не в списке), собираем их в список
        return data_list


class Formater:
    def __init__(self, num_batches: int):
        self.array = []
        self.num_batches = num_batches

    def add(self, other: dict):
        self.array.append(other)


    def get(self) -> dict:
        """
        Select last num_batches from array and merge them into one
        :return: sole dict with aggregated odometry
        """
        size = len(self.array)
        l = max(0, size - self.num_batches)
        return merge(self.array[l:])


    def flush(self):
        self.array = []





