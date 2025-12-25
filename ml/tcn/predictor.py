import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from datetime import datetime
from ml.tcn.tcn import RobotTCNClassifier


class Predictor:
    def __init__(self, input_dim, num_classes, tcn_channels,
                 kernel_size=3, dropout=0.2, lr=0.001,
                 epochs=20, batch_size=32, device='auto',
                 log_dir='runs'):

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.tcn_channels = tcn_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = log_dir

        # Определение устройства
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Device: {self.device}")

        # Инициализация модели
        self.model = RobotTCNClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            tcn_channels=self.tcn_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

    def fit(self, X, y, eval_set=None, verbose=True, run_name=None):
        """
        Args:
            X, y: Обучающая выборка.
            eval_set: Кортеж (X_val, y_val) для валидации.
            verbose: Вывод логов в консоль.
            run_name: Имя эксперимента для TensorBoard.
        """
        # 1. Подготовка обучающего датасета
        self.model.train()

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        writer_path = os.path.join(self.log_dir, run_name)
        writer = SummaryWriter(writer_path)
        if verbose:
            print(f"Logging to: {writer_path}")

        train_tensor_x = torch.FloatTensor(X)
        train_tensor_y = torch.LongTensor(y)
        train_loader = DataLoader(TensorDataset(train_tensor_x, train_tensor_y),
                                  batch_size=self.batch_size, shuffle=True)

        # 2. Подготовка валидационного датасета (если есть)
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_tensor_x = torch.FloatTensor(X_val)
            val_tensor_y = torch.LongTensor(y_val)
            # Shuffle=False для валидации важно (хотя для метрик не критично, но порядок приятнее)
            val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y),
                                    batch_size=self.batch_size, shuffle=False)

        # --- ЦИКЛ ПО ЭПОХАМ ---
        for epoch in range(self.epochs):
            # === TRAIN LOOP ===
            self.model.train()  # Режим обучения (включает Dropout)
            train_loss = 0.0
            train_preds = []
            train_targets = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", disable=not verbose)

            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # Собираем данные для F1
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Метрики обучения за эпоху
            avg_train_loss = train_loss / len(train_loader)
            avg_train_f1 = f1_score(train_targets, train_preds, average='macro')

            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('F1/train', avg_train_f1, epoch)

            # === VALIDATION LOOP (если передан eval_set) ===
            val_info = ""
            if val_loader is not None:
                self.model.eval()  # Режим валидации (выключает Dropout)
                val_loss = 0.0
                val_preds = []
                val_targets = []

                with torch.no_grad():  # Отключаем расчет градиентов для скорости и памяти
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())

                avg_val_loss = val_loss / len(val_loader)
                avg_val_f1 = f1_score(val_targets, val_preds, average='macro')

                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('F1/val', avg_val_f1, epoch)

                val_info = f" | Val Loss: {avg_val_loss:.4f} | Val F1: {avg_val_f1:.4f}"

                # Возвращаем модель в режим обучения для следующей эпохи
                self.model.train()

                # Вывод итогов эпохи
            if verbose:
                print(
                    f"End Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Train F1: {avg_train_f1:.4f}{val_info}")

        writer.close()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.model(X_tensor).argmax(dim=1).cpu().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()