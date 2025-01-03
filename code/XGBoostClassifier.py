
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import NotFittedError


class XGBoostClassifier:
    def __init__(self, params=None):
        if params is None:
            params = {}

        default_gpu_params = {'tree_method': 'hist', 'objective': 'multi:softprob',
                              'device': 'cuda', 'n_estimators': 100}
        params.update(default_gpu_params)

        # params['eval_metric'] = 'mlogloss'
        self.n_jobs = 6
        self.params = params
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_params_ = None

    @staticmethod
    def encode_labels(y_train, y_val):
        unique_classes = np.unique(np.concatenate((y_train, y_val)))
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_classes)
        y_train_encoded = label_encoder.transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        return y_train_encoded, y_val_encoded, label_encoder

    def train(self, X, y, n_splits=5, grid_search=False, param_grid=None):
        try:
            # 统一类别编码
            unique_classes = np.unique(y)
            self.label_encoder.fit(unique_classes)
            y_encoded = self.label_encoder.transform(y)
            print(set(y_encoded))

            # 数据标准化
            X_scaled = self.scaler.fit_transform(X)

            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=n_splits)

            if grid_search and param_grid is not None:
                # 网格搜索
                grid_search_cv = GridSearchCV(estimator=xgb.XGBClassifier(**self.params),
                                              param_grid=param_grid,
                                              cv=tscv,
                                              scoring='accuracy',
                                              n_jobs=self.n_jobs,
                                              verbose=False)  # 这里通过设置 verbose=1 启用详细输出

                # 开始网格搜索
                print("Starting Grid Search...")
                grid_search_cv.fit(X_scaled, y_encoded)
                self.model = grid_search_cv.best_estimator_
                self.best_params_ = grid_search_cv.best_params_
                print("Best parameters found:", self.best_params_)

                # 输出每次网格搜索的评估结果
                results = grid_search_cv.cv_results_
                for mean_score, params in zip(results["mean_test_score"], results["params"]):
                    print(f"Mean Test Score: {mean_score:.4f} for parameters: {params}")

                # 使用最佳模型做进一步评估
                y_pred = self.model.predict(X_scaled)
                print("Classification Report:")
                print(classification_report(y_encoded, y_pred))

            else:
                # 使用 xgb.XGBClassifier 进行训练
                self.model = xgb.XGBClassifier(**self.params)

                for train_index, test_index in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_val = y_encoded[train_index], y_encoded[test_index]

                    # 训练模型并记录损失
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=True
                    )
                    # 验证模型
                    y_pred = self.model.predict(X_val)
                    y_pred_classes = y_pred  # 这里 y_pred 已经是类别
                    accuracy = accuracy_score(y_val, y_pred_classes)
                    print(f"Validation Accuracy: {accuracy:.4f}")
                    print(classification_report(y_val, y_pred_classes))

        except NotFittedError as nfe:
            print(f"Model not fitted error during prediction: {nfe}")
            return None
        except Exception as e:
            print(f"An error occurred during training: {e}")

    def predict(self, X):
        try:
            X_scaled = self.scaler.transform(X)
            y_pred_encoded = self.model.predict(X_scaled)  # 直接使用 NumPy 数组
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            return y_pred
        except NotFittedError as nfe:
            print(f"Model not fitted error during prediction: {nfe}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
            return None

    def evaluate(self, X_val, y_val):
        """评估模型性能"""
        try:
            # 对验证数据进行缩放
            X_val_scaled = self.scaler.transform(X_val)
            y_val_encoded = self.label_encoder.transform(y_val)

            # 使用 xgb.XGBClassifier 进行预测
            y_pred_encoded = self.model.predict(X_val_scaled)

            # 获取预测的类别索引
            y_pred_classes = y_pred_encoded  # 这里 y_pred 已经是类别

            # 将编码的预测结果转回原始标签
            y_pred = self.label_encoder.inverse_transform(y_pred_classes)

            # 计算准确率
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Validation Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_val, y_pred))

            # 计算每个类别的召回率
            print("Recall for each class:")
            report = classification_report(y_val, y_pred, output_dict=True)
            unique_classes = np.unique(y_val)  # 获取唯一类别
            for i in unique_classes:
                if str(i) in report:  # 确保类别在报告中
                    print(f"Recall for class {i}: {report[str(i)]['recall']:.4f}")

            # 绘制 ROC 曲线和 AUC
            plt.figure(figsize=(10, 8))
            for i in unique_classes:  # 对于每个类别
                fpr, tpr, _ = roc_curve(y_val_encoded, self.model.predict_proba(X_val_scaled)[:, i], pos_label=i)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--')  # 绘制随机猜测的基线
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Multi-class')
            plt.legend(loc='best')
            plt.show()

        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during evaluation: {e}")