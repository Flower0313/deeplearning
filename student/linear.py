import numpy as np
import pymysql
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


def load_data():
    conn = pymysql.connect(
        host='rm-bp130tpgwlw1hkbbn8o.mysql.rds.aliyuncs.com',
        port=3306,
        user='user_bi_wr',
        passwd='BykJ_0824!',
        charset='utf8'
    )

    sql = "SELECT stu_id,lesson_cnt,status FROM htdw_bi.`rp_student_forecast`"

    cursor = conn.cursor()
    cursor.execute(sql)
    content = cursor.fetchall()
    train_data = []
    for i in content:
        data = str(i[1]).split(",")
        data.insert(10, i[2])
        data.insert(0, i[0])
        train_data.append(data)

    x = np.array(train_data, dtype=float)
    return x


# 数据
class StuDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.stu = torch.from_numpy(data[:, [0]])
        self.x_data = torch.from_numpy(data[:, 1:-1]).to(torch.float32)
        self.y_data = torch.from_numpy(data[:, [-1]]).to(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    my_dataset = StuDataset(load_data())

    # 定义模型和损失函数
    model = SimpleClassifier()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(1000):
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(my_dataset.x_data)
        loss = criterion(outputs, my_dataset.y_data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'D:/trained_model/student_model.pt')
