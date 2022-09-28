import numpy as np
import matplotlib.pyplot as plt

###多项式回归
#读入训练数据
train = np.loadtxt('E:/algorithm/test/data.csv',delimiter=',',skiprows=1)
train_x=train[:,0]
train_y=train[:,1]
#标准化
#训练数据的平均值
mu = train_x.mean()
#训练数据的
# 标准差
sigma = train_x.std()
def standardize(x):
    return (x - mu)/sigma
train_z = standardize(train_x)

theta = np.random.rand(3)

def to_matrix(x):
    #np.ones()函数返回给定形状和数据类型的新数组，其中元素的值设置为1
    #np.vstack()函数是竖直方向堆叠形成一个数组
    return np.vstack([np.ones(x.shape[0]),x,x ** 2]).T

X = to_matrix(train_z)

#预测函数
def f(x):
    #np.dot函数是矩阵乘法运算
    return np.dot(x,theta)

#目标函数
def E(x,y):
    return 0.5 * np.sum((y-f(x))**2)

#学习率
ETA = 1e-3

#误差的差值
diff=1

#重复学习
error = E(X, train_y)
while diff > 1e-2:
    #更新参数
    theta = theta -ETA*np.dot(f(X)-train_y,X)
    #计算与上一次误差的差值
    current_error = E(X,train_y)
    diff = error - current_error
    error = current_error

x=np.linspace(-3,3,100)
plt.plot(train_z,train_y,'o')
plt.plot(x,f(to_matrix(x)))
plt.show()






