import operator
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from tensorboard.notebook import display
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

# 取消numpy科学计数法
np.set_printoptions(suppress=True)
# 取消tensor科学计数法
torch.set_printoptions(sci_mode=False)


# 再转为tensor之前先改变numpy中数据的类型
# stock_list = torch.tensor(stock_data.astype(float))

def readDate(path):
    stock = pd.read_csv(path, dtype=str, header=0)
    dct_data = np.array(stock.loc[:, :])
    # 获取code
    stock_title = dct_data[:, 0]
    # 获取数值
    stock_data = dct_data[:, 1:-1].astype(float)
    # 获取股票的结果
    stock_result = dct_data[:, -1]
    return stock_title, stock_data, stock_result


def normalize(array):
    # 计算均值
    means = np.mean(array, axis=0)
    # 计算标准差
    stds = np.std(array, axis=0)
    return (array - means) / stds


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加,sum(0)列相加,sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方,计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


# 读取训练数据
stock_tile, stock_data, stock_result = readDate("./train/stock_train.csv")
# 读取测试数据
stock_tile_t, stock_data_t, stock_result_t = readDate("./test/stock_test.csv")
# 归一化
standard_data = normalize(stock_data)
standard_data_t = normalize(stock_data_t)
# 99%训练数据,1%的测试数据
hoRatio = 0.01
total = standard_data.shape[0]
numTestNums = int(total * hoRatio)
rightCount = 0.0

# 第一个参数是第几个股票,第二个参数是已知结果的股票，第三个参数是结果，第四个参数是取前多少
for i in range(standard_data_t.shape[0]):
    classifierResult = classify(standard_data_t[i, :], standard_data[:, :],
                                stock_result[:], 15)

    print("股票code:%s\t预测结果:%s\t真实结果:%s" % (str(stock_tile_t[i]), str(classifierResult), str(stock_result_t[i])))
    if classifierResult == stock_result_t[i]:
        rightCount += 1.0
print("正确率:%f%%" % (rightCount / float(standard_data_t.shape[0]) * 100))
