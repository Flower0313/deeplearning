from sklearn.naive_bayes import GaussianNB

# 定义特征和标签
X = [[80, 90], [60, 70], [70, 80], [85, 95], [55, 65]]
y = ['是', '否', '是', '是', '否']

# 训练高斯朴素贝叶斯分类器
clf = GaussianNB()
clf.fit(X, y)

# 进行预测
print(clf.predict([[75, 85]]))
