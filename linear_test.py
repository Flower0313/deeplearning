import math

import matplotlib.pyplot as plt
import numpy as np
import pymysql
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn


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


def normalize(array, mult):
    # 计算均值
    means = np.mean(array, axis=0)
    # 计算标准差
    stds = np.std(array, axis=0)
    if mult:
        return means, stds, (array - means) / stds
    else:
        return (array - means) / stds


# def normalize(array):
#     # 计算均值
#     min = np.min(array, axis=0)
#     max = np.max(array, axis=0)
#     return (array - min) / (max - min)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(2, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linears(x)
        return out


conn = pymysql.connect(
    host='ht-polardb.rwlb.rds.aliyuncs.com',
    port=3306,
    user='user_bi',
    passwd='H^4zzM8iVc&c',
    charset='utf8'
)
#
sql = '''
SELECT
	count(IF( lead_status = 1, lead_id, NULL ))  as following_lead,
	count(IF( lead_status = 1, lead_id, NULL ))*count(IF( lead_status = 1, lead_id, NULL ))  as following_lead2,
	count(IF( lead_status = 2, lead_id, NULL ))  as orderd_lead
FROM
	dw_bi.rp_dm_lead_analysis where affect_year>='2022'
GROUP BY
	woy order by affect_date
'''

cursor = conn.cursor()
cursor.execute(sql)
content = cursor.fetchall()


class HutongDataset(Dataset):
    def __init__(self, data, lr):
        self.len = data.shape[0]
        self.x_train = torch.from_numpy(
            normalize(dataHandle(data).iloc[:math.floor(self.len * lr), :-1], False).values).to(
            torch.float32)
        self.mean, self.std, self.y_train = normalize(dataHandle(data).iloc[:math.floor(self.len * lr), [-1]], True)
        self.y_train = torch.from_numpy(self.y_train.values).to(torch.float32)
        self.x_test = torch.from_numpy(
            normalize(dataHandle(data).iloc[:math.ceil(self.len * (1 - lr)), :-1], False).values).to(
            torch.float32)
        self.x_test_ori = torch.from_numpy(data.iloc[:math.ceil(self.len * (1 - lr)), :-1].values).to(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    my_dataframe = pd.DataFrame(data=np.array(content, dtype=float))
    dataSet = HutongDataset(my_dataframe, 1)
    model = Model()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    losses = []
    for epoch in range(100):
        y_pred = model(dataSet.x_train)
        loss = criterion(y_pred, dataSet.y_train)
        # print("w", model.state_dict()['out.weight'])
        # print("b", model.state_dict()['out.bias'])
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.cla()  # 清空
    plt.scatter(dataSet.x_train[:, 0], dataSet.y_train, c='blue', s=50, alpha=0.3, label='train')
    plt.xlabel('followed')
    plt.ylabel('ordered')
    with torch.no_grad():
        x = []
        for i in range(2000, 10000):
            x.append([i, i * i, i * i * i, i * i * i * i])
        x_test = torch.Tensor(normalize(x, False))
        y_test = model(x_test)
        # 期望业绩
        plt.plot(x_test[:, 0], y_test.detach().numpy(), 'red', lw=3)

    plt.ioff()
    plt.show()

    plt.plot(list(range(1, len(losses) + 1)), losses, 'black', lw=3)
    plt.xlabel('number')
    plt.ylabel('loss')
    print(losses[-1])
    plt.show()
    # y_test = model(dataSet.x_test)
    # print(dataSet.x_test_ori)
    # print(y_test.detach().numpy() * dataSet.std.mean() + dataSet.mean.mean())
