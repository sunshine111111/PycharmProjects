import torch.nn as nn
import torch.nn.functional as F

#定义卷积神经网络进行学习
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #nn.Conv1d：一维卷积主要用于文本数据；
        #nn.Conv2d：二维卷积主要用于图像数据，对宽度和高度都进行卷积
        self.conv1 = nn.Conv2d(3,6,5)
        #nn.MaxPool2d：池化，取最大池化窗口覆盖元素中的最大值，提取显著特征和降维
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        #定义输入层、中间层、输出层
        self.layer1 = nn.Linear(16*5*5,120)
        self.layer2 = nn.Linear(120,84)
        self.layer3 = nn.Linear(84,10)

    def forward(self,input_data):
        #F.relu：使用ReLU激活函数
        #self.conv1，F.relu，self.pool依次对应操作：卷积、激活函数、池化
        input_data = self.pool(F.relu(self.conv1(input_data)))
        input_data = self.pool(F.relu(self.conv2(input_data)))
        #这里view函数相当于resize功能,参数不可为空,参数中的-1表示这个位置由其他位置的数字来推断，有点占位的意思
        input_data = input_data.view(-1,16*5*5)
        input_data = F.relu(self.layer1(input_data))
        input_data = F.relu(self.layer2(input_data))
        input_data = self.layer3(input_data)
        return input_data

