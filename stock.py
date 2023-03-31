import csv

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 取消numpy科学计数法
np.set_printoptions(suppress=True)
# 取消tensor科学计数法
torch.set_printoptions(sci_mode=False)


# 数据
class HutongDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :5]).float()
        self.y_data = torch.from_numpy(data[:, [-1]]).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def readDate(path):
    stock = pd.read_csv(path, dtype=np.float64, quoting=csv.QUOTE_NONE)
    dct_data = np.array(stock.loc[:, :])
    return dct_data[:, :-1].astype(np.float)


if __name__ == '__main__':
    # 数据
    stock_data = readDate("./train/stocks.csv")
    my_dataset = HutongDataset(stock_data)
    #
    ## 模型
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 16),
        torch.nn.Sigmoid(),
        torch.nn.Linear(16, 1)
    )

    # 损失
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 训练
    for epoch in range(1000):
        y_pred = model(my_dataset.x_data)
        loss = criterion(y_pred, my_dataset.y_data)
        # 查看损失值
        print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

