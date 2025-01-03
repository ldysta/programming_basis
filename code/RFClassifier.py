from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, cohen_kappa_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt

class RFClassifier:
    def __init__(self, params=None):
        if params is None:
            params = {}

        self.params = params
        self.n_jobs = 8
        try:
            self.model = RandomForestClassifier(oob_score=True, **params)
        except TypeError as e:
            print(f"Error initializing model: {e}")
            self.model = RandomForestClassifier(oob_score=True)  # 使用默认参数初始化模型

        self.scaler = StandardScaler()
        self.best_params_ = None

    def fit(self, X, y, n_splits=5, grid_search=False, param_grid=None, cv=3):
        try:
            # 数据标准化
            X_scaled = self.scaler.fit_transform(X)

            # 使用 TimeSeriesSplit 划分数据集
            tscv = TimeSeriesSplit(n_splits=n_splits)

            for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled)):
                X_train, X_val = X_scaled[train_index], X_scaled[val_index]
                y_train, y_val = y[train_index], y[val_index]

                if grid_search and param_grid is not None:
                    # 网格搜索
                    grid_search_cv = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=self.n_jobs)

                    # 输出当前的参数组合
                    for params in grid_search_cv.param_grid:
                        print(f"Fold {fold + 1} - Evaluating parameters: {params}")

                    grid_search_cv.fit(X_train, y_train)
                    self.model = grid_search_cv.best_estimator_
                    self.best_params_ = grid_search_cv.best_params_
                    print(f"Fold {fold + 1} - Best parameters found:", self.best_params_)
                else:
                    # 训练模型并记录每次迭代的 OOB 评分
                    oob_scores = []
                    for n_estimators in range(1, self.model.n_estimators + 1):
                        self.model.set_params(n_estimators=n_estimators)
                        self.model.fit(X_train, y_train)
                        oob_score = self.model.oob_score_
                        oob_scores.append(oob_score)
                        print(f"Fold {fold + 1} - Iteration {n_estimators}: OOB Score = {oob_score:.4f}")

                    # 绘制 OOB 评分随迭代次数的变化
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(oob_scores) + 1), oob_scores, label=f'Fold {fold + 1} OOB Score', color='blue')
                    plt.title('OOB Score vs. Number of Trees (Iterations)')
                    plt.xlabel('Number of Trees (Iterations)')
                    plt.ylabel('OOB Score')
                    plt.legend()
                    plt.grid()
                    plt.show()

                # 验证模型
                y_pred = self.predict(X_val)
                self.evaluate(y_val, y_pred, X_val, fold=fold + 1)
        except Exception as e:
            print(f"An error occurred during fitting: {e}")

    def predict(self, X):
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None

    def evaluate(self, y_true, y_pred, X_val, fold=None):
        """
        评估模型性能。
        :param y_true: 真实标签。
        :param y_pred: 预测标签。
        :param X_val: 验证特征矩阵。
        :param fold: 当前交叉验证的折数。
        """
        try:
            # 准确率
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Fold {fold} - Validation Accuracy: {accuracy:.4f}")

            # 混淆矩阵
            conf_matrix = confusion_matrix(y_true, y_pred)
            print(f"Fold {fold} - Confusion Matrix:\n", conf_matrix)

            # 分类报告
            report = classification_report(y_true, y_pred, zero_division=0)
            print(f"Fold {fold} - Classification Report:\n", report)

            # ROC 曲线和 AUC
            y_prob = self.model.predict_proba(X_val)
            plt.figure(figsize=(10, 8))
            for i in range(10):  # 对于每个类别
                fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--')  # 对角线
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Fold {fold} - Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()

            # Kappa 分数
            kappa = cohen_kappa_score(y_true, y_pred)
            print(f"Fold {fold} - Cohen's Kappa Score: {kappa:.4f}")

            # F1 分数
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            print(f"Fold {fold} - Macro F1 Score: {f1_macro:.4f}")
            print(f"Fold {fold} - Micro F1 Score: {f1_micro:.4f}")
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
