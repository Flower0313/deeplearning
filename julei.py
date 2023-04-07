import numpy as np


def normalize(array):
    # 计算均值
    means = np.mean(array, axis=0)
    # 计算标准差
    stds = np.std(array, axis=0)
    return (array - means) / stds


# 创建一个5行2列的numpy数组
arr = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]])

# 获取要进行归一化操作的列，并计算该列的最大值和最小值
data_col = arr[:, 0]

# 对该列数据进行归一化操作
normalized_col = normalize(data_col)
# 将归一化后的列替换掉原来的列
arr[:, 0] = normalized_col

# 打印归一化后的结果
