import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


def dataHandle(data):
    num_features = data.select_dtypes(exclude=['object', 'bool']).columns.tolist()
    for feature in num_features:
        if feature == 'total_amount':
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


class LeadDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        # 将pandas转为numpy
        self.x_data = torch.from_numpy(normalize(data.iloc[:, :-1]).values).to(torch.float32)
        self.y_data = torch.from_numpy(normalize(data.iloc[:, [-1]]).values).to(torch.float32)
        # self.x_data = torch.from_numpy(dataHandle(data).values[:, :-1]).to(torch.float32)
        # self.y_data = torch.from_numpy(data.values[:, [-1]]).to(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def normalize(array):
    # 计算均值
    means = np.mean(array, axis=0)
    # 计算标准差
    stds = np.std(array, axis=0)
    return (array - means) / stds


# def normalize(array):
#     # 计算均值
#     min = np.min(array, axis=0)
#     max = np.max(array, axis=0)
#     return (array - min) / (max - min)

# 模型设置，若这里只是Linear(1,1)就说明只是普通的wx+b一条直线，如果多加基层神经元，就可以使曲线变弯曲

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.out = nn.Linear(1, 1)

    def forward(self, x):
        out = self.out(x)
        return out


if __name__ == '__main__':
    dataSet = LeadDataset(pd.read_csv(r'../train/lead_linear.csv'))
    model = Model()
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
    # 若学习率过小，可以看见w在2左右横跳
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    for epoch in range(10000):
        # print("=======", epoch, "=======")
        print("w", model.state_dict()['out.weight'])
        # print("b", model.state_dict()['linear.bias'])
        # y = wx+b
        y_pred = model(dataSet.x_data)
        # print("y_pred", y_pred)
        # MSELoss函数计算方式
        # 1 / 3 * (y_pred - dataSet.y_data).pow(2).sum()
        # w梯度更新的计算方式
        # model.state_dict()['linear.weight'] - ((y_pred - dataSet.y_data) * dataSet.x_data).sum() * 2 / 3 * 0.1

        loss = criterion(y_pred, dataSet.y_data)
        print("loss", loss.item())

        # 查看损失值
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.step(loss)

    # print(model.state_dict()['linear.weight'])
    # print(model.state_dict()['linear.bias'])
    x_test = torch.Tensor([normalize([9, 1437, 0, 206, 753, 185])])
    y_test = model(x_test)
    print(y_test)
