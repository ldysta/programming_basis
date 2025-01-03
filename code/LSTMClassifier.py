import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import make_scorer, accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SensorDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

        # 检查数据是否包含缺失值
        if self.data.isnull().values.any():
            raise ValueError("Data contains missing values")

        # 检查数据类型是否正确
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame")

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.seq_length, :-1].values  # 特征
        y = self.data.iloc[idx + self.seq_length - 1, -1]  # 标签

        # 转换为张量并指定数据类型
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        return out  # 返回形状为 (batch_size, output_size)


class LSTMClassifierWrapper:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0, bidirectional=False,
                 optimizer_params=None):
        """
        初始化LSTMClassifierWrapper类。

        :param input_size: 输入特征的数量。
        :param hidden_size: LSTM隐藏层的大小。
        :param output_size: 输出类别数量。
        :param num_layers: LSTM的层数。
        :param dropout: LSTM层之间的dropout率。
        :param bidirectional: 是否使用双向LSTM。
        :param optimizer_params: 字典，包含优化器的默认参数设置。
        """
        if optimizer_params is None:
            optimizer_params = {}

        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout, bidirectional)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_history = []
        self.best_accuracy = 0.0
        self.best_model = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def fit(self, X, y, sequence_length, batch_size, num_epochs=1, test_size=0.2, random_state=2024, grid_search=False,
            param_grid=None, cv=3):
        """
        训练模型。可以选择是否进行网格搜索。
        """
        try:
            # 将数据转换为DataFrame
            data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]).assign(label=y)

            # 使用TimeSeriesSplit进行时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=cv)

            if grid_search and param_grid is not None:

                param_grid = {
                    'num_layers': param_grid.get('num_layers', [1]),
                    'dropout': param_grid.get('dropout', [0.0]),
                    'bidirectional': param_grid.get('bidirectional', [False]),
                    'optimizer_params': param_grid.get('optimizer_params', [{'lr': 0.01}])
                }

                best_model = None
                best_accuracy = 0.0

                for num_layers in param_grid['num_layers']:
                    for dropout in param_grid['dropout']:
                        for bidirectional in param_grid['bidirectional']:
                            for optimizer_params in param_grid['optimizer_params']:
                                # 打印当前参数组合
                                print(
                                    f"Training with num_layers={num_layers}, dropout={dropout}, bidirectional={bidirectional}, optimizer_params={optimizer_params}")

                                model = LSTMModel(self.input_size, self.hidden_size, self.output_size, num_layers,
                                                  dropout,
                                                  bidirectional)
                                criterion = nn.CrossEntropyLoss()
                                optimizer = optim.Adam(model.parameters(), **optimizer_params)
                                model.to(self.device)

                                fold_accuracies = []

                                for train_index, val_index in tscv.split(data):
                                    train_data = data.iloc[train_index]
                                    val_data = data.iloc[val_index]

                                    train_dataset = SensorDataset(train_data, sequence_length)
                                    val_dataset = SensorDataset(val_data, sequence_length)

                                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                              num_workers=8)
                                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                            num_workers=8)

                                    for epoch in range(num_epochs):
                                        model.train()
                                        running_loss = 0.0
                                        for inputs, labels in train_loader:
                                            inputs, labels = inputs.to(self.device), labels.to(self.device)

                                            optimizer.zero_grad()
                                            outputs = model(inputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer.step()
                                            running_loss += loss.item()

                                        avg_loss = running_loss / len(train_loader)

                                        # 验证模型
                                        model.eval()
                                        y_true, y_pred, y_prob = [], [], []
                                        with torch.no_grad():
                                            for inputs, labels in val_loader:
                                                inputs, labels = inputs.to(self.device), labels.to(self.device)
                                                outputs = model(inputs)
                                                _, predicted = torch.max(outputs, 1)

                                                y_true.extend(labels.cpu().numpy())
                                                y_pred.extend(predicted.cpu().numpy())
                                                y_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())

                                        accuracy = accuracy_score(y_true, y_pred)
                                        print(
                                            f'Epoch [{epoch + 1}/{num_epochs}], Fold [{len(fold_accuracies) + 1}/{cv}], Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

                                    fold_accuracies.append(accuracy)

                                avg_fold_accuracy = np.mean(fold_accuracies)
                                print(f'Average Validation Accuracy for current parameters: {avg_fold_accuracy:.4f}')

                                # 更新最佳模型
                                if avg_fold_accuracy > best_accuracy:
                                    best_accuracy = avg_fold_accuracy
                                    best_model = model.state_dict()
                                    best_params = {
                                        'num_layers': num_layers,
                                        'dropout': dropout,
                                        'bidirectional': bidirectional,
                                        'optimizer_params': optimizer_params
                                    }
                                    # 保存最佳参数
                                    self.best_params = best_params

                if self.best_params:
                    # 使用最佳参数重新初始化模型
                    self.model = LSTMModel(
                        input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        output_size=self.output_size,
                        num_layers=self.best_params['num_layers'],
                        dropout=self.best_params['dropout'],
                        bidirectional=self.best_params['bidirectional']
                    )
                    self.model.to(self.device)

                # 加载最佳模型状态字典
                if best_model:
                    self.model.load_state_dict(best_model)
                    self.best_accuracy = best_accuracy
                    print(f'Best parameters found: {self.best_params}')

                # 使用最佳模型进行最终验证
                self.model.eval()
                y_true, y_pred, y_prob = [], [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs, 1)

                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())
                        y_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            else:
                for train_index, val_index in tscv.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    # 创建数据集和数据加载器
                    train_dataset = SensorDataset(
                        pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])]).assign(
                            label=y_train),
                        sequence_length)
                    val_dataset = SensorDataset(
                        pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])]).assign(
                            label=y_val),
                        sequence_length)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    for epoch in range(num_epochs):
                        self.model.train()
                        running_loss = 0.0
                        for inputs, labels in train_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)

                            self.optimizer.zero_grad()
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            loss.backward()
                            self.optimizer.step()
                            running_loss += loss.item()

                        avg_loss = running_loss / len(train_loader)
                        self.loss_history.append(avg_loss)

                        # 验证模型
                        self.model.eval()
                        y_true, y_pred, y_prob = [], [], []
                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                inputs, labels = inputs.to(self.device), labels.to(self.device)
                                outputs = self.model(inputs)
                                _, predicted = torch.max(outputs, 1)

                                y_true.extend(labels.cpu().numpy())
                                y_pred.extend(predicted.cpu().numpy())
                                y_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())

                        accuracy = accuracy_score(y_true, y_pred)
                        print(
                            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

                        # 更新最佳模型
                        if accuracy > self.best_accuracy:
                            self.best_accuracy = accuracy
                            self.best_model = self.model.state_dict()

                # 绘制损失曲线
                self.plot_loss()

            # 评估模型
            y_t = np.array(y_true)
            y_p = np.array(y_pred)
            self.evaluate(y_t, y_p, y_prob)

        except Exception as e:
            print(f"An error occurred during training: {e}")

    def evaluate(self, y_true, y_pred, y_prob):
        """
        评估模型性能。

        :param y_true: 真实标签。
        :param y_pred: 预测标签。
        :param y_prob: 预测概率。n_split
        """
        # 打印分类报告
        print(classification_report(y_true, y_pred, zero_division=0))

        # 计算并打印每个类别的召回率
        for i in range(len(np.unique(y_true))):
            recall = accuracy_score(y_true[y_true == i], y_pred[y_true == i])
            print(f'Recall for class {i}: {recall:.4f}')

        # 计算并绘制ROC和AUC曲线
        y_true_binarized = label_binarize(y_true, classes=np.arange(len(np.unique(y_true))))
        y_prob_binarized = np.array(y_prob)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_true))):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_prob_binarized[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 绘制所有类别的ROC曲线
        plt.figure(figsize=(10, 8))
        for i in range(len(np.unique(y_true))):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for each class')
        plt.legend(loc="lower right")
        plt.show()

    def plot_loss(self):
        """
        绘制训练过程中损失随epoch变化的曲线。
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.show()

    def predict(self, X, sequence_length, batch_size):
        """
        使用模型进行预测。

        :param X: 特征矩阵。
        :param sequence_length: 序列长度。
        :param batch_size: 批量大小。
        :return: 预测结果。
        """
        self.model.eval()
        dataset = SensorDataset(pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        y_pred = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())

        return y_pred


