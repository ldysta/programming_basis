# 基于HARTH数据集的人类活动识别任务

## 项目介绍

在当今的大数据时代，以智能手机为代表的数字化设备已深入人们的生活中，成为不可或缺的一部分，尤其是在体育运动、健康管理方面，智能手表和手环能够实现运动轨迹记录、GPS独立定位、心率监测、血氧监测、血压血糖监测、睡眠质量监测、情绪监测等众多功能，是人们了解身体状况和运动水平的得力助手。

华为、小米等公司的智能手表和手环时刻在记录用户的各项数据，比如加速计的空间位置随时间变化的运动轨迹，陀螺仪$XYZ$三轴角速度数据以感知设备姿态。智能手表和手环通过蓝牙连接手机，将数据同步到云端，分析运动情况和时间，并依此提供健康建议。人类活动识别任务(Human Activity Recognition)是数字化设备感知人体活动状态的基础性任务，其核心在于通过传感器数据准确判断用户正在进行的活动类型。

本项目基于Pytorch、scikit-learn、xgboost等包共编写了3个深度学习和机器学习算法：LSTM分类、XGBoost分类和随机森林分类，实现识别人类活动的功能。在编写时，本项目特别实现了网格搜索和非网格搜索两种模式，有助于调整超参数。

### 数据来源

本项目的数据集来源于[UCI机器学习仓库-HARTH数据集](https://archive.ics.uci.edu/dataset/779/harth)。该数据集研究招募了22名健康成年人(8名女性)，年龄范围从25到68岁。每位参与者佩戴了两个三轴加速度计(Axivity AX3)，一个放置在大腿前部(膝盖上方约10厘米)，另一个放置在下背部(大约在第三腰椎位置)。参与者在日常工作环境中进行活动，每次记录持续90到120分钟，被要求进行多种活动，包括坐、站、躺、走、跑等，每种活动至少持续2到3分钟。传感器数据经过了时间同步、低通滤波、信号分段等处理，确保了传感器数据能够提供充分准确的信息。

HARTH数据集共有22个csv文件，共6461328个样本，每个csv文件包括时间戳(1列)、两个传感器位置坐标(6列)、运动状态标签(1列)，此外还有某些未处理的无用列。[data_process.py](https://github.com/ldysta/programming_basis/blob/master/code/data_process.py)文件是对数据进行预处理。

## 运行环境

本项目使用Python版本为3.12.4，在一台CPU为Intel(R) i5-10300H CPU @ 2.50GHz，GPU为NVIDIA GTX 1650 Ti的电脑上进行运算，Pytorch版本为2.4.1，CUDA版本为11.8，xgboost库版本2.1.3，scikit-learn库版本1.5.2。

## 复现步骤

harth文件夹存储了S006.csv的数据，harth_all存放了全部的harth数据集，复现时按照code文件夹里[final_project.ipynb](https://github.com/ldysta/programming_basis/blob/master/code/final_project.ipynb)文件操作即可。依次是数据预处理，LSTM，XGBoost，随机森林和EDA部分。code文件夹里还存放了[LSTM分类算法的代码](https://github.com/ldysta/programming_basis/blob/master/code/LSTMClassifier.py)、[XGBoost分类算法的代码](https://github.com/ldysta/programming_basis/blob/master/code/XGBoostClassifier.py)和[随机森林分类算法的代码](https://github.com/ldysta/programming_basis/blob/master/code/RFClassifier.py)
