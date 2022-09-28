from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions as pdr

import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np


iris = datasets.load_iris()
#取第3,4个特征量使用
last_two_features = iris.data[:,[2,3]]
#获取分类标签
teacher_labels = iris.target

#分为训练数据和测试数据，比例为8:2
#控制随机数的参数random_state设置为None，以便每次生成不同的数据
train_features,test_features,train_teacher_labels,test_teacher_labels=train_test_split(last_two_features,teacher_labels,test_size=0.2,random_state=None)

#数据标准化
sc=StandardScaler()
sc.fit(train_features)

#标准化后的特征量训练数据和测试数据
train_features_std = sc.transform(train_features)
test_features_std = sc.transform(test_features)

#生成线性SVM
model = SVC(kernel='linear',random_state=None)

#让模型进行训练
model.fit(train_features_std,train_teacher_labels)

#将训练数据分类为预学习模型时的精度
predict_train = model.predict(train_features_std)

#计算并显示分类精度
accuracy_train = accuracy_score(train_teacher_labels,predict_train)
print('对训练数据的分类精度：%.2f' % accuracy_train)

# 将验证数据分类为预训练模型时的精度
predict_test = model.predict(test_features_std)
accuracy_test = accuracy_score(test_teacher_labels,predict_test)

print('测试数据的分类精度：%.2f' % accuracy_test)

#将特征量数据和教师数据结合起来，用于学习和验证
#vstack:将参数元组的元素数组按竖直方向进行叠加
combined_features_std=np.vstack((train_features_std,test_features_std))
#hstack:将参数元组的元素数组按水平方向进行叠加
combined_teacher_labels=np.hstack((train_teacher_labels,test_teacher_labels))

fig=plt.figure(figsize=(12,8))

#散点图相关设置
scatter_kwargs = {'s':300,'edgecolor':'white','alpha':0.5}
contourf_kwargs = {'alpha':0.2}
scatter_highlight_kwargs = {'s':200,'label':'Test','alpha':0.7}

pdr(combined_features_std,combined_teacher_labels,clf=model,scatter_kwargs=scatter_kwargs,contourf_kwargs=contourf_kwargs,scatter_highlight_kwargs=scatter_highlight_kwargs)
#plt.show()

test_data=np.array([[4.1,5.2]])
print(test_data)
test_result=model.predict(test_data)
print(test_result)