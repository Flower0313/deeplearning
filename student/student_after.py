import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pymysql

base_sql = '''
                insert into htdw_bi.rp_forecast_all values("{}","{}","{}","{}","{}")
            '''


def load_data():
    conn = pymysql.connect(
        host='rm-bp130tpgwlw1hkbbn8o.mysql.rds.aliyuncs.com',
        port=3306,
        user='user_bi_wr',
        passwd='BykJ_0824!',
        charset='utf8'
    )

    sql = """
        SELECT
	stu_id,
	lesson_cnt,
STATUS 
FROM
	htdw_bi.`rp_student_forecast` 
WHERE
	`status` = 1 
ORDER BY
	stu_id 
	LIMIT 40000,100
    """

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
    cursor.close()
    conn.close()
    return x


# 数据
class StuDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.stu = torch.from_numpy(data[:, [0]])
        self.x_data = torch.from_numpy(data[:, 1:-1]).to(torch.float32)
        self.y_data = torch.from_numpy(data[:, [-1]]).to(torch.float32)

    def __getitem__(self, index):
        return self.stu[index], self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 1. 定义神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    conn = pymysql.connect(
        host='rm-bp130tpgwlw1hkbbn8o.mysql.rds.aliyuncs.com',
        port=3306,
        user='user_bi_wr',
        passwd='BykJ_0824!',
        charset='utf8'
    )
    cursor = conn.cursor()
    new_data = StuDataset(load_data())

    model = SimpleClassifier()
    # 加载训练好的模型
    model.load_state_dict(torch.load('D:/trained_model/student_model.pt'))
    # 将模式设置为预测模式
    model.eval()
    # 预测模型确保不要计算梯度
    with torch.no_grad():
        # 加载预测数据
        for x, y, z in new_data:
            outputs = model(y)
            print(str(x.item()).split(".")[0], outputs[0])
            sql = base_sql.format(str(x.item()).split(".")[0], -1, outputs.item(), 2, "学员沉睡率")
            cursor.execute(sql)
    # 重新训练模型,设置为训练模式
    # model.train()
    conn.commit()
    cursor.close()
    conn.close()
## criterion = nn.CrossEntropyLoss()
## optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())
#
# print("--- 开始训练新数据 ---")
## 训练新数据
# optimizer.zero_grad()
# outputs = model(new_data.x_data)
# loss = criterion(outputs, new_data.y_data)
# loss.backward()
# optimizer.step()
# print("--- 完成 ---")
#
## 再次保存模型
# torch.save(model.state_dict(), 'D:/trained_model/student_model.pt')
