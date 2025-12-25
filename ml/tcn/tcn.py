import torch
import torch.nn as nn
from pytorch_tcn import TCN


class RobotTCNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, tcn_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            input_dim (int): Размер входного вектора на каждом шаге (Одометрия + Лидар).
            num_classes (int): Количество классов поведения/ситуаций.
            tcn_channels (list): Список размеров скрытых слоев. Длина списка определяет глубину сети.
            kernel_size (int): Размер окна свертки.
            dropout (float): Вероятность дропаута.
        """
        super().__init__()

        # Основной блок TCN
        # num_inputs - количество признаков (каналов) на входе
        # num_channels - список, например [32, 64, 64], задает количество фильтров на каждом уровне
        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True  # Важно! Чтобы не подглядывать в будущее
        )

        # Классификатор (голова)
        # Принимает выход последнего слоя TCN и выдает классы
        self.classifier = nn.Sequential(
            nn.Linear(tcn_channels[-1], 64),  # Промежуточный слой (опционально)
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape на входе: (Batch_Size, Time_Steps, Features)
        # TCN ожидает:      (Batch_Size, Features, Time_Steps) -> нужно перевернуть

        x = x.transpose(1, 2)  # Меняем местами оси времени и признаков

        y = self.tcn(x)  # Выход TCN: (Batch, Last_Channel_Dim, Time_Steps)

        # Для классификации последовательности нам нужен только ПОСЛЕДНИЙ шаг времени
        # Это "сумма" всей истории
        last_step = y[:, :, -1]

        logits = self.classifier(last_step)
        return logits




