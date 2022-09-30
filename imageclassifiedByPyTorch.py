import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optimizer
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from cNN import CNN

#基于PyTorch的CIFAR-10图像学习，通过卷积神经网络模型进行学习
if __name__ == '__main__':
    #transforms.Compose函数就是将transforms组合在一起,
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    #训练数据
    train_data_with_teacher_labels = CIFAR10(root='./data',train=True,download=True,transform=transform)
    #训练数据shuffle为true，是因为数据处理完后，需要汇聚到计算节点上进行计算？？(我的猜测)
    train_data_loader = DataLoader(train_data_with_teacher_labels,batch_size=4,shuffle=True,num_workers=2)

    #测试数据
    test_data_with_teacher_labels = CIFAR10(root='./data',train=False,download=True,transform=transform)
    #测试数据shuffle为false，是因为数据用来测试，测试完后不需要汇聚到计算节点上进行计算
    test_data_loader = DataLoader(test_data_with_teacher_labels,batch_size=4,shuffle=False,num_workers=2)

    class_names = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    #显示图像的函数
    def show_image(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()

    #从训练数据中提取一些数据
    # data_iterator = iter(train_data_loader)
    # images,labels = data_iterator.next()
    #显示图像
    #show_image(torchvision.utils.make_grid(images))
    #显示标签
    #print(' '.join('%5s' % class_names[labels[j]] for j in range(4)))

    model = CNN()

    #为优化而设置Optimizer
    criterion = nn.CrossEntropyLoss()
    #优化方法：随机梯度下降法（SGD）
    optimizer = optimizer.SGD(model.parameters(),lr=0.001,momentum=0.9)

    #######训练
    MAX_EPOCH = 3

    for epoch in range(MAX_EPOCH):

        total_loss = 0.0
        for i,data in enumerate(train_data_loader,0):
            #从数据中检索训练数据和标签数据
            train_data,teacher_labels = data

            #删除计算出的梯度信息
            optimizer.zero_grad()

            #计算模型中的预测
            outputs = model(train_data)

            #用loss和w进行微分
            loss = criterion(outputs,teacher_labels)
            loss.backward()

            #更新梯度
            optimizer.step()

            #累计误差
            total_loss += loss.item()

            #以2000个小型批处理为单位显示进度
            if i % 2000 == 1999:
                print('学习进度：[%d,%5d] loss: %.3f' % (epoch + 1,i + 1,total_loss / 2000))
                total_loss = 0.0

    print('学习完成')

    #######测试
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in test_data_loader:
            test_data,teacher_labels = data
            results = model(test_data)
            _,predicted = torch.max(results,1)
            c = (predicted == teacher_labels).squeeze()
            for i in range(4):
                label = teacher_labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print(' %5s 类的正确率是 ： %2d %%' % (class_names[i],100*class_correct[i] / class_total[i]))