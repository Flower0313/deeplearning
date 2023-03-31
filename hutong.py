import pymysql
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)

conn = pymysql.connect(
    host='by-rds-yz.mysql.rds.aliyuncs.com',
    port=3306,
    user='user_bi_wr',
    passwd='GIsH-xgNuiYJW3zA',
    charset='utf8'
)

sql = '''
SELECT
	operating_people,# 运营人数
	sales_people,# 销售人数
	operating_cost,# 运营成本
	average_performance,# 人均业绩
	admission_student,# 招生学员
	people_cost,# 人力成本
	site_cost,# 场地成本
	other_cost,# 其他成本
	headquarter_cost,# 总部分摊成本
	c.ss,
	floor( b.area / 10 )+
IF
	( b.area % 10 > 0, 1, 0 ) AS area,# 使用面积
IF
	( store_gross_profits > 0, 1, 0 ) AS result # 结果
	
FROM
	business_analysis.ba_average_profits a
	JOIN ( SELECT id, max( actual_use_area ) AS area FROM htdw_bi.by_store GROUP BY id ) b ON a.store_id = b.id
	JOIN ( SELECT store_id, sum( avg_teacher_finished )/ sum( avg_student_finished ) AS ss FROM business_analysis.`ba_finished_lesson` WHERE store_id IS NOT NULL GROUP BY store_id ) c ON a.store_id = c.store_id 
WHERE
	operating_people IS NOT NULL 
	AND operating_cost > 0 
ORDER BY
	cal_date
	limit 5000
'''
cursor = conn.cursor()
cursor.execute(sql)
content = cursor.fetchall()


# 数据
class HutongDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.from_numpy(data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(11, 32)
        self.linear2 = torch.nn.Linear(32, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.activate = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    # 向前传递
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 数据集
tran_data = np.array(content, dtype=float)
my_dataset = HutongDataset(tran_data)
# tran_loader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True, num_workers=0)

# 模型
# model = Model()
model = torch.nn.Sequential(
    torch.nn.Linear(11, 100),
    torch.nn.ReLU(True),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(True),
    torch.nn.Linear(100, 1)
)

# 损失
criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-2)

# 循环、迭代

for epoch in range(5000):
    y_pred = model(my_dataset.x_data)
    loss = criterion(y_pred, my_dataset.y_data)

    # 查看损失值
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# x_test =

cursor.close()
conn.close()
