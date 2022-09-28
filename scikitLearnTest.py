import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# 获取鸢尾花的数据，有四个特征量
iris = datasets.load_iris()
#以表格形式查看数据
pd.DataFrame(iris.data,columns=iris.feature_names)
#print(pd.DataFrame(iris.data,columns=iris.feature_names))

#第一个特征量
first_one_feature = iris.data[:,:1]
pd.DataFrame(first_one_feature,columns=iris.feature_names[:1])
#print(pd.DataFrame(first_one_feature,columns=iris.feature_names[:1]))

#提取前两个特征量
first_two_feature = iris.data[:,:2]
#提取最后两个特征量
last_two_feature = iris.data[:,:2]

teacher_labels = iris.target
#print(teacher_labels)

#这是包含全部第一特征量的鸢尾花数据
all_features = iris.data

x_min,x_max = all_features[:,0].min(),all_features[:,0].max()
y_min,y_max = all_features[:,1].min(),all_features[:,1].max()

plt.figure(2,figsize=(12,9))
plt.clf()
#绘制散点图
#s是点的大小，c是表示渐变是哪个方向的；,cmap是渐变色
plt.scatter(all_features[:,0],all_features[:,1],s=300,c=teacher_labels,cmap=plt.cm.Set2,edgecolor='darkgray')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.grid(True)
plt.show()

#三维展示
fig = plt.figure(1,figsize=(12,9))
ax = Axes3D(fig,elev=-140,azim=100)

#降维
reduced_features = PCA(n_components=3).fit_transform(all_features)
#创建散点图
ax.scatter(reduced_features[:,0],reduced_features[:,1],reduced_features[:,2],c=teacher_labels,cmap=plt.cm.Set2,edgecolor='darkgray',s=200)
plt.show()




