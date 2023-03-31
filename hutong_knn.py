import operator

import pymysql
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
	admission_student,# 招生学员
	people_cost,# 人力成本
	site_cost,# 场地成本
	other_cost,# 其他成本
	headquarter_cost,# 总部分摊成本
	c.ss, # 老师消课/学生消课
	c.st, # 学生数/老师数
	floor( b.area / 10 )+
IF
	( b.area % 10 > 0, 1, 0 ) AS area,# 使用面积
IF
	( store_gross_profits > 0, 1, 0 ) AS result # 结果
	
FROM
	business_analysis.ba_average_profits a
	JOIN ( SELECT id, max( actual_use_area ) AS area FROM htdw_bi.by_store GROUP BY id ) b ON a.store_id = b.id
	JOIN ( SELECT store_id, floor(sum( avg_teacher_finished )/ sum( avg_student_finished )) AS ss,floor(sum(student_num)/sum(teacher_num)) st FROM business_analysis.`ba_finished_lesson` WHERE store_id IS NOT NULL GROUP BY store_id ) c ON a.store_id = c.store_id 
WHERE
	operating_people IS NOT NULL 
	AND operating_cost > 0 
ORDER BY
	cal_date
	limit 3000
'''

sql2 = '''
SELECT
	operating_people,# 运营人数
	sales_people,# 销售人数
	operating_cost,# 运营成本
	admission_student,# 招生学员
	people_cost,# 人力成本
	site_cost,# 场地成本
	other_cost,# 其他成本
	headquarter_cost,# 总部分摊成本
	c.ss,c.st,
	floor( b.area / 10 )+
IF
	( b.area % 10 > 0, 1, 0 ) AS area,# 使用面积
IF
	( store_gross_profits > 0, 1, 0 ) AS result # 结果
	
FROM
	business_analysis.ba_average_profits a
	JOIN ( SELECT id, max( actual_use_area ) AS area FROM htdw_bi.by_store GROUP BY id ) b ON a.store_id = b.id
	JOIN ( SELECT store_id, floor(sum( avg_teacher_finished )/ sum( avg_student_finished )) AS ss,floor(sum(student_num)/sum(teacher_num)) st FROM business_analysis.`ba_finished_lesson` WHERE store_id IS NOT NULL GROUP BY store_id ) c ON a.store_id = c.store_id 
WHERE
	operating_people IS NOT NULL 
	AND operating_cost > 0 
ORDER BY
	cal_date
	limit 3000,300
'''
cursor = conn.cursor()
cursor.execute(sql)
content = cursor.fetchall()
cursor.execute(sql2)
content2 = cursor.fetchall()
cursor.close()


def readDate(data):
    return data[:, :-1], data[:, [-1]]


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
        classCount[str(voteIlabel)] = classCount.get(str(voteIlabel), 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


# 读取训练数据
inputs, labels = readDate(np.array(content, dtype=float))
inputs_t, labels_t = readDate(np.array(content2, dtype=float))
standard = normalize(inputs)
standard_t = normalize(inputs_t)

rightCount = 0.0

for i in range(standard_t.shape[0]):
    classifierResult = classify(standard_t[i, :], standard[:, :],
                                labels[:], 15)

    print("预测结果:%s\t真实结果:%s" % (str(classifierResult), str(labels_t[i])))
    if str(classifierResult) == str(labels_t[i]):
        rightCount += 1.0
print("正确率:%f%%" % (rightCount / float(standard_t.shape[0]) * 100))

conn.close()
