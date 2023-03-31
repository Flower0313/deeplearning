import json
import re
import time
from urllib.request import urlopen  # python自带爬虫库
import urllib.request
import requests
from kafka import KafkaProducer
from clickhouse_driver import Client
import numpy as np
import matplotlib.pyplot as plt


# 损失函数(代价函数),另外还要传入数据的x,y
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)

    # 逐点计算平方损失误差,然后求平均数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2

    return total_cost / M


def average(data):
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum / num


# 拟合函数(预测函数)
def fit(points):
    M = len(points)
    x_bar = average(points[:, 0])

    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0

    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - M * (x_bar ** 2))

    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += (y - w * x)

    b = sum_delta / M
    return w, b


client = Client(host='43.142.117.50', database='spider_base', user='default', password='root',
                port=61616)

res = client.execute(
    "select deal_amount,closing_price from spider_base.stock_detail where code='301266'")
data = np.array(res)

deal_amount = data[:, 0]
closing_price = data[:, 1]

w, b = fit(data)
cost = compute_cost(w, b, data)

# 画出拟合曲线
pred_y = w * deal_amount + b
plt.scatter(deal_amount, closing_price)
plt.plot(deal_amount, pred_y, c='r')
plt.show()
