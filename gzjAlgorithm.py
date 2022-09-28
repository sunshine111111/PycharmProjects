import numpy as np
import matplotlib.pyplot as plt

##感知机模型
#读入训练数据
train = np.loadtxt('E:/algorithm/test/image.csv',delimiter=',',skiprows=1)

#x[:,m:n]，即取所有数据集的第m到n-1列数据
train_x = train[:,0:2]
#x[:,n]表示在全部数组（维）中取第n个数据
#x[n,:]表示在n个数组（维）中取全部数据
train_y = train[:,2]

