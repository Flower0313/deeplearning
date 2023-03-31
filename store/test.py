import math

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pymysql
import pandas as pd
import torch
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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = Model()
# 加载训练好的模型
model.load_state_dict(torch.load('D:/trained_model/store.pt'))
# 将模式设置为预测模式
model.eval()
with torch.no_grad():
    x = []
    for i in range(1, 150):
        x.append([i, i * i])
    x_test = torch.Tensor(x)
    y_test = model(x_test)
    # 期望业绩
    plt.plot(x_test[:, 0], y_test.detach().numpy(), 'red', lw=3)
    plt.show()

    # print((outputs.detach().numpy() / 10 * 6451381) - 1295716)
