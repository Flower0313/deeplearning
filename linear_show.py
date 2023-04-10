import math

import torch.nn
from matplotlib import pyplot as plt


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def draw(w, b):
    x = torch.rand(50)
    y = w * x + b
    plt.plot(x.data.numpy(), y.data.numpy())

    plt.show()


model = LinearModel()

if __name__ == '__main__':
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])
    y_data = torch.Tensor([[2.0], [4.0], [6.0]])

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(50):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(model.linear.weight.item())
        # print("loss:", loss.item())
        draw(model.linear.weight.item(), model.linear.bias.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("w=", model.linear.weight.item())
    print("b=", model.linear.bias.item())

    # plt.text(-0.5, -1, 'train:weight=', model.linear.weight.item())
    # plt.text(x=0.5,  # 文本x轴坐标
    #          y=1.0,  # 文本y轴坐标
    #          s='w=' + str(round(model.linear.weight.item(), 2)),  # 文本内容
    #          fontdict=dict(fontsize=12, color='r',
    #                        family='monospace',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
    #                        weight='bold',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
    #                        ))
    # plt.text(x=0.5,  # 文本x轴坐标
    #          y=0.9,  # 文本y轴坐标
    #          s='b=' + str(round(model.linear.bias.item(), 2)),  # 文本内容
    #          fontdict=dict(fontsize=12, color='r',
    #                        family='monospace',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
    #                        weight='bold',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
    #                        ))
    # draw(model.linear.weight.item(), model.linear.bias.item())

    x_test = torch.Tensor([[4.0]])
    y_test = model(x_test)
    print(y_test)
