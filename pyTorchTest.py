import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from mlp import MLP
from torch.autograd import *

import torch.optim as optimizer
import torch.nn as nn

##基于PyTorch的MNIST手写数字学习，在“CNN:卷积神经网络”中构建学习模型
data_folder = 'E:/algorithm/data'
BATCH_SIZE = 8

#数据集MNIST,train:True表示作为训练数据载入,False表示作为测试数据载入；download:True表示从互联网上下载数据集，并把数据集放在data_folder目录下；transforms.ToTensor()数据变换
mnist_data = MNIST(data_folder,train=True,download=True,transform=transforms.ToTensor())

#shuffle：洗牌，数据处理后，具有某种共同特征的一类数据需要最终汇聚（aggregate）到一个计算节点上进行计算
data_loader = DataLoader(mnist_data,batch_size=BATCH_SIZE,shuffle=False)

#检索数据，并将检索到的一批数据分配给不同的变量images和labels（因为BATCH_SIZE是8，所有images和labels分别会有八个）
data_iterator = iter(data_loader)
images,labels = data_iterator.next()

location = 4
#转换为numpy矩阵
data = images[location].numpy()
#(1, 28, 28)
#print(data.shape)

#调整数据通道以用于matplotlib的绘制
reshaped_data = data.reshape(28,28)
# plt.imshow(reshaped_data,cmap='inferno',interpolation='bicubic')
# plt.show()
# #TensorFlow用Tensor这种数据结构来表示所有的数据，TensorFlow的computation graph(计算图)中的节点之间只用Tensor传递数据
# #标签： tensor(9)
# print('标签：',labels[location])

#训练数据
train_data_with_labels = MNIST(data_folder,train=True,download=True,transform=transforms.ToTensor())
train_data_loader = DataLoader(train_data_with_labels,batch_size=BATCH_SIZE,shuffle=True)

#测试数据
test_data_with_labels = MNIST(data_folder,train=False,download=True,transform=transforms.ToTensor())
test_data_loader = DataLoader(test_data_with_labels,batch_size=BATCH_SIZE,shuffle=True)

#################学习
model = MLP()

#softMax：交叉熵
lossResult = nn.CrossEntropyLoss()
#SGD
optimizer = optimizer.SGD(model.parameters(),lr=0.01)

#最大学习次数
MAX_EPOCH = 4

for epoch in range(MAX_EPOCH):
    total_loss = 0.0
    for i,data in enumerate(train_data_loader):

        #从数据中检索提取训练数据和教师标签数据
        train_data,teacher_labels = data

        #将输入转换为torch.autograd.Variable
        #这里有其他依赖，需要这样from torch.autograd import *，不能只导入Variable，否则会报错
        train_data,teacher_labels=Variable(train_data),Variable(teacher_labels)

        #删除计算出的梯度信息
        optimizer.zero_grad()

        #为模型提供训练数据来计算预测
        outputs = model(train_data)

        #基于loss和w的微分计算
        loss = lossResult(outputs,teacher_labels)
        loss.backward()

        #更新梯度
        optimizer.step()

        #累计误差
        total_loss += loss.item()

        #以2000个小型批处理为单位显示进度
        if i % 2000 == 1999:
            print('学习进度：[%d,%d] 学习误差（loss）: %.3f' % (epoch + 1,i+1,total_loss/2000))
            total_loss = 0.0

print('学习结束')

##################测试
#总计
total=0
#正确答案
count_when_correct = 0

print(test_data_loader)
for data in test_data_loader:
    # 从测试数据加载器中检索数据，然后将其解包
    test_data,teacher_labels=data
    #将转换测试数据，然后将其传递给模型，使其作出判断
    results = model(Variable(test_data))
    # 获取预测
    _, predicted=torch.max(results.data,1)

    total += teacher_labels.size(0)
    count_when_correct += (predicted == teacher_labels).sum()

print('count_when_correct:%d' % (count_when_correct))
print('total:%d' % (total))

print('正确率： %d / %d = %f' % (count_when_correct,total,int(count_when_correct)/int(total)))

###############用个别数据进行测试
test_iterator = iter(test_data_loader)
#可以增加或者减少次数，以获得不同的测试数据
test_data,teacher_labels=test_iterator.next()
#转换测试数据，然后将其传递给模型，使其作出判断
results = model|(Variable(test_data))
_,predicted_label=torch.max(results.data,1)

location = 1
plt.imshow(test_data[location].numpy().reshape(28,28),cmap='inferno',interpolation='bicubic')
print('标签： ',predicted_label[location])