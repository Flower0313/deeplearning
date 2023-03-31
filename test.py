import math

import matplotlib.pyplot as plt
import numpy as np
import pymysql
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from torchvision import datasets


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


conn = pymysql.connect(
    host='ht-polardb.rwlb.rds.aliyuncs.com',
    port=3306,
    user='user_bi',
    passwd='H^4zzM8iVc&c',
    charset='utf8'
)

sql = '''
SELECT
	count(IF( lead_status = 1, lead_id, NULL ))  as following_lead,
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
        self.x_train = torch.from_numpy(normalize(data.iloc[:math.floor(self.len * lr), :-1], False).values).to(
            torch.float32)
        self.mean, self.std, self.y_train = normalize(data.iloc[:math.floor(self.len * lr), [-1]], True)
        self.y_train = torch.from_numpy(self.y_train.values).to(torch.float32)
        self.x_test = torch.from_numpy(normalize(data.iloc[:math.ceil(self.len * (1 - lr)), :-1], False).values).to(
            torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


my_dataframe = pd.DataFrame(data=np.array(content, dtype=float))
dataSet = HutongDataset(my_dataframe, 1)

plt.cla()  # 清空

colors = ['blue', 'red', 'black', 'green', 'pink', 'orange']

for i in range(dataSet.x_train.shape[1]):
    plt.scatter(dataSet.x_train[:, [i]], dataSet.y_train, c=colors[i], s=50, alpha=0.3, label='train')
    # plt.plot(dataSet.x_train, dataSet.y_train, 'r-', lw=5)
plt.ioff()
plt.show()
