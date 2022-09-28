from sklearn.svm import SVC
import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

###对鸢尾花的前两个特征量，通过svm方法进行学习
iris = datasets.load_iris()
first_two_features = iris.data[:,[0,1]]
#print("1####"+str(first_two_features))
teacher_labels = iris.target
#print("2####"+str(teacher_labels))
first_two_features = first_two_features[teacher_labels!=2]
teacher_labels = teacher_labels[teacher_labels!=2]
#print("3####"+str(first_two_features))

model = SVC(C=1.0,kernel='linear')
model.fit(first_two_features,teacher_labels)
print(model.coef_)
print(model.intercept_)

#确定figure对象创建大小
fig,ax=plt.subplots(figsize=(12,9))

#------------------------------
#绘制花的数据
#只提取iris setosa（y=0）的数据
setosa = first_two_features[teacher_labels == 0]
#只提取iris versicolor(y=1)的数据
versicolor = first_two_features[teacher_labels == 1]
#绘制iris setosa数据
plt.scatter(setosa[:,0],setosa[:,1],s=300,c='white',linewidths=0.5,edgecolors='lightgray')
plt.scatter(versicolor[:,0],versicolor[:,1],s=300,c='firebrick',linewidths=0.5,edgecolors='lightgray')
#绘制回归直线
#指定图表的范围
Xi = np.linspace(4,7.25)
#绘制超平面
Y = -model.coef_[0][0]/model.coef_[0][1]*Xi - model.intercept_/model.coef_[0][1]

#在图表上绘制线条
ax.plot(Xi,Y,linestyle='dashed',linewidth=3)

plt.show()


