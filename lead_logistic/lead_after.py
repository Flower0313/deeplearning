import numpy as np
import pymysql
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader, Dataset

conn = pymysql.connect(
    host='rm-bp130tpgwlw1hkbbn8o.mysql.rds.aliyuncs.com',
    port=3306,
    user='user_bi_wr',
    passwd='BykJ_0824!',
    charset='utf8'
)

sql = '''
SELECT
	owner_num,
	following_num,
	boring_num,
	recall_num,
	order_num,
	no_connected_num,
	days,
	subject_id,
	store_id,
	is_sign 
FROM
	htdw_bi.`rp_lead_side` 
	where is_sign=1 limit 10000
'''


class CustomDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1]).to(torch.float32)
        self.y_data = torch.from_numpy(data[:, [-1]]).to(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(9, 5)
        self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


if __name__ == '__main__':
    cursor = conn.cursor()
    cursor.execute(sql)
    content = cursor.fetchall()
    tran_data = np.array(content, dtype=float)
    my_dataset = CustomDataset(tran_data)

    model = CustomModel()
    # 加载训练好的模型
    model.load_state_dict(torch.load('D:/trained_model/lead_model.pt'))
    # 将模式设置为预测模式
    model.eval()

    with torch.no_grad():
        for x, y in my_dataset:
            outputs = model(x)
            print(y, outputs)
        # 加载预测数据
