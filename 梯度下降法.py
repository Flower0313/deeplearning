from clickhouse_driver import Client
import numpy as np
import matplotlib.pyplot as plt


# 拟合函数(代价函数),另外还要传入数据的x,y
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)

    # 逐点计算平方损失误差,然后求平均数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2

    return total_cost / M


# 学习率
alpha = 0.0001
# 梯度初始值
initial_w = 0
initial_b = 0
num_iter = 10


def grad_desc(points, initial_w, initial_b, alpha, num_iter):
    w = initial_w
    b = initial_b
    # 定义一个list列表保存所有的损失函数值，用来显示下降的过程
    cost_list = []

    for i in range(num_iter):
        # 逐个计算损失函数
        cost_list.append(compute_cost(w, b, points))
        # 迭代,梯度下降
        w, b = step_grad_desc(w, b, alpha, points)
    return [w, b, cost_list]


def step_grad_desc(current_w, current_b, alpha, points):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(points)

    # 对每个点,代入公式求和
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += current_w * x + current_b - y

    # 用公式求当前梯度
    grad_w = 2 / M * sum_grad_w
    grad_b = 2 / M * sum_grad_b

    # 梯度下降，更新当前w和b
    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b
    return updated_w, updated_b


client = Client(host='43.142.117.50', database='spider_base', user='default', password='root',
                port=61616)

res = client.execute(
    "select deal_amount,closing_price from spider_base.stock_detail where code='301266'")
# 转为float类型
data = np.array(res, dtype=np.float64)

deal_amount = data[:, 0]
closing_price = data[:, 1]

print(data)

w, b, cost_list = grad_desc(data, initial_w, initial_b, alpha, num_iter)

plt.plot(cost_list)
plt.show()
