import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pymysql


def load_data():
    conn = pymysql.connect(
        host='rm-bp130tpgwlw1hkbbn8o.mysql.rds.aliyuncs.com',
        port=3306,
        user='user_bi_wr',
        passwd='BykJ_0824!',
        charset='utf8'
    )

    sql = """
    SELECT stu_id,lesson_cnt,status FROM htdw_bi.`rp_student_forecast` where `status`=0 limit 50
union all
SELECT stu_id,lesson_cnt,status FROM htdw_bi.`rp_student_forecast` where `status`=1 limit 50
    """

    cursor = conn.cursor()
    cursor.execute(sql)
    content = cursor.fetchall()
    train_data = []
    for i in content:
        data = str(i[1]).split(",")
        data.insert(10, i[2])
        train_data.append(data)

    x = np.array(train_data, dtype=float)
    return x


# 数据
class StuDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1]).to(torch.float32)
        self.y_data = torch.from_numpy(data[:, [-1]]).to(torch.float32)

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

# 1. 定义神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    # 不用定义backward()，因为会自动根据forward来自动微分


if __name__ == '__main__':
    my_dataset = StuDataset(load_data())
    model = Model()
    # 损失
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1):
        optimizer.zero_grad()
        y_pred = model(my_dataset.x_data)
        loss = criterion(y_pred, my_dataset.y_data)
        # 查看损失值
        # print(epoch, loss.item())
        loss.backward()
        optimizer.step()

    # 保存模型
    # torch.save(model.state_dict(), 'D:/trained_model/student_model.pt')

# x_test = torch.Tensor([[2, 11, 12, 3, 5, 1, 1, 12, 3, 5]])
# y_test = model(x_test)
# print(y_test)
