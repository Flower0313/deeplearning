import math
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset


class CrmDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1]).to(torch.float32)
        self.y_data = torch.from_numpy(data[:, [-1]]).to(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()
        self.layer1 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        return x


def make_moons(n_samples=1000, shuffle=True, noise=None):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # 采集第1类数据，特征为(x,y)
    # 使用'paddle.linspace'在0到pi上均匀取n_samples_out个值
    # 使用'paddle.cos'计算上述取值的余弦值作为特征1，使用'paddle.sin'计算上述取值的正弦值作为特征2
    outer_circ_x = torch.cos(torch.linspace(0, math.pi, n_samples_out))
    outer_circ_y = torch.sin(torch.linspace(0, math.pi, n_samples_out))

    inner_circ_x = 1 - torch.cos(torch.linspace(0, math.pi, n_samples_in))
    inner_circ_y = 0.5 - torch.sin(torch.linspace(0, math.pi, n_samples_in))

    # 使用'torch.cat'将两类数据的特征1和特征2分别延维度0拼接在一起，得到全部特征1和特征2
    # 使用'torch.stack'将两类特征延维度1堆叠在一起
    X = torch.stack(
        [torch.cat([outer_circ_x, inner_circ_x]),
         torch.cat([outer_circ_y, inner_circ_y])],
        axis=1
    )

    # 使用'torch.zeros'将第一类数据的标签全部设置为0
    # 使用'torch.ones'将第二类数据的标签全部设置为1
    y = torch.cat(
        [torch.zeros([n_samples_out]), torch.ones([n_samples_in])]
    )

    # 如果shuffle为True，将所有数据打乱
    if shuffle:
        # 使用'torch.randperm'生成一个数值在0到X.shape[0]，随机排列的一维Tensor做索引值，用于打乱数据
        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    # 如果noise不为None，则给特征值加入噪声
    if noise is not None:
        # 使用'paddle.normal'生成符合正态分布的随机Tensor作为噪声，并加到原始特征上
        noise_tensor = torch.randn(X.shape) * noise
        X.add_(noise_tensor)
    return X, y


def decision_boundary(w, b, x1):
    x2 = (- w.numpy()[0][0] * x1 - b.numpy()[0]) / w.numpy()[0][1]
    return x2


if __name__ == '__main__':
    # 采样1000个样本
    n_samples = 1000
    X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.5)

    plt.figure(figsize=(5, 5))
    plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', c=y.tolist())
    plt.xlim(-3, 4)
    plt.ylim(-3, 4)

    model = BinaryClassificationModel()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1000):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, torch.from_numpy(y.numpy().reshape((1000, 1))).to(torch.float32))
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w = model.state_dict()['layer1.weight']
    b = model.state_dict()['layer1.bias']
    x1 = torch.linspace(-2, 3, 1000)
    x2 = decision_boundary(w, b, x1)
    plt.plot(x1.tolist(), x2.tolist(), color="red")
    plt.show()
