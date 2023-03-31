# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1)  # 设置随机种子
n_hidden = 100  # 隐藏神经元个数
max_iter = 2000  # 最大迭代次数
disp_interval = 200  # plt展示的迭代次数间隔
lr_init = 0.01  # 学习率


# ============================ step 1/5 数据 ============================
def gen_data(num_data=10, x_range=(-1, 1)):
    w = 2
    b = 0.5
    train_x = torch.linspace(x_range[0], x_range[1], num_data).unsqueeze_(1)  # torch.linspace生成1*10的向量，压缩成10*1的向量

    # 模型需要训练的直线：wx+b
    train_y = w * train_x + b + torch.normal(0, 0.5, size=train_x.size())  # 训练集在直线wx+b基础上加入噪声

    test_x = torch.linspace(x_range[0], x_range[1], num_data).unsqueeze_(1)
    test_y = w * test_x + b + torch.normal(0, 0.3, size=test_x.size())  # 测试集在直线wx+b基础上加入噪声

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = gen_data()


# ============================ step 2/5 模型 ============================
class Net(nn.Module):
    def __init__(self, neural_num):
        super(Net, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(1, neural_num),
            nn.ReLU(True),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(True),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(True),
            nn.Linear(neural_num, 1),
        )

    def forward(self, x):
        return self.linears(x)


net_normal = Net(neural_num=n_hidden)
net_weight_decay = Net(neural_num=n_hidden)

# ============================ step 3/5 优化器 ============================
"""不加正则化的优化器 & 加入L2正则化参数λ（weight_decay项）的优化器"""
optim_normal = torch.optim.SGD(net_normal.parameters(), lr=lr_init, momentum=0.9)
optim_wdecay = torch.optim.SGD(net_weight_decay.parameters(), lr=lr_init, momentum=0.9, weight_decay=1e-2)

# ============================ step 4/5 损失函数 ============================
loss_func = torch.nn.MSELoss()
# ============================ step 5/5 迭代训练 ============================
for epoch in range(max_iter):

    # forward
    pred_normal, pred_wdecay = net_normal(train_x), net_weight_decay(train_x)

    loss_normal, loss_wdecay = loss_func(pred_normal, train_y), loss_func(pred_wdecay, train_y)

    # 优化器梯度清零
    optim_normal.zero_grad()
    optim_wdecay.zero_grad()

    # 反向传播
    loss_normal.backward()
    loss_wdecay.backward()

    optim_normal.step()
    optim_wdecay.step()

    if (epoch + 1) % disp_interval == 0:
        test_pred_normal, test_pred_wdecay = net_normal(test_x), net_weight_decay(test_x)

        # 绘图
        # 真实点（训练集 测试集）
        plt.scatter(train_x, train_y, c='blue', s=50, alpha=0.3, label='train')  # s散点面积  alpha散点透明度
        # plt.scatter(test_x, test_y, c='red', s=50, alpha=0.3, label='test')

        # 测试集预测值 拟合的曲线,tensor->numpy格式需加入 .detach().numpy()
        # 这画的不是预测曲线，而是每次y_hat值
        # plt.plot(test_x, test_pred_normal.detach().numpy(), 'r-', lw=3, label='no weight decay')
        # plt.plot(test_x, test_pred_wdecay.detach().numpy(), 'b--', lw=3, label='weight decay')

        # 训练集的loss
        # plt.text(-0.5, -1, 'no weight_decay={:.6f}'.format(loss_normal.item()),
        # fontdict={'size': 15, 'color': 'red'})
        # plt.text(-0.5, -1.25, 'weight_decay={:.6f}'.format(loss_wdecay.item()),
        # fontdict={'size': 15, 'color': 'red'})

with torch.no_grad():
    x = []
    for i in range(-1, 2):
        x.append([i])
    x_test = torch.Tensor(x)
    y_test = net_normal(x_test)
    # 期望业绩
    plt.plot(x_test[:, 0], y_test.detach().numpy(), 'red', lw=3)
    plt.show()
    plt.close()
