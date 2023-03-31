import numpy as np
import paddle
import pymysql
import pandas as pd

class Runner(object):
    def __init__(self, model, optimizer, loss_fn, metric):
        # 自定义属性
        self.model = model  # 模型
        self.optimizer = optimizer  # 优化器
        self.loss_fn = loss_fn  # 损失函数
        self.metric = metric  # 评估指标

        # 模型训练
        def train(self, train_dataset, dev_dataset=None, **kwargs):
            pass

        # 模型评价
        def evaluate(self, data_set, **kwargs):
            pass

        # 模型预测
        def predict(self, x, **kwargs):
            pass

        # 模型保存
        def save_model(self, save_path):
            pass

        # 模型加载
        def load_model(self, model_path):
            pass


# 划分数据集
def train_test_split(X, y, train_percent=0.8):
    n = len(X)
    shuffled_indices = paddle.randperm(n)  # 返回一个数值在0到n-1、随机排列的1-D Tensor
    train_set_size = int(n * train_percent)
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:]

    X = X.values
    y = y.values

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    X_min = paddle.min(X_train, axis=0)
    X_max = paddle.max(X_train, axis=0)

    X_train = (X_train - X_min) / (X_max - X_min)
    X_test = (X_test - X_min) / (X_max - X_min)

    return X_train, X_test, y_train, y_test


def dataHandle(data):
    num_features = data.select_dtypes(exclude=['object', 'bool']).columns.tolist()

    for feature in num_features:
        print(feature)
        # 每次进来都能取一列数据
        if feature == 'CHAS':
            continue

        Q1 = data[feature].quantile(q=0.25)  # 下四分位
        Q3 = data[feature].quantile(q=0.75)  # 上四分位

        IQR = Q3 - Q1
        top = Q3 + 1.5 * IQR  # 最大估计值
        bot = Q1 - 1.5 * IQR  # 最小估计值
        values = data[feature].values
        values[values > top] = top  # 临界值取代噪声
        values[values < bot] = bot  # 临界值取代噪声
        data[feature] = values.astype(data[feature].dtypes)
        return data


if __name__ == '__main__':
    data = pd.read_csv("T:/deeplearning/boston_house_prices.csv")
    # 用isna()来判断数据中各元素是否缺失
    x = data.isna().sum()

    ## 处理异常数据，减少噪声
#
    ## 四分位处理异常值
    data = dataHandle(data)
#
    ## 划分数据集 && 归一化
    #X = data.drop(['MEDV'], axis=1)
    #y = data['MEDV']
    #X_train, X_test, y_train, y_test = train_test_split(X, y)  # X_train每一行是个样本，shape[N,D]
#
    #train_dataset = (X_train, y_train)
    ## 测试集构造
    #test_dataset = (X_test, y_test)


