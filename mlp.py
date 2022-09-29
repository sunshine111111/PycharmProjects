from torch.autograd import Variable
import torch.nn as nn

#nn.Module是父类，MLP:多层感知机
class MLP(nn.Module):
    #注意init前后是两个‘_’,否则会报错
    def __init__(self):
        super().__init__()
        #输入层
        #nn.Linear(arg1，arg2)：arg1表示在forward中输入Tensor最后一维的通道数；arg2表示在forward中输出Tensor最后一维的通道数
        self.layer1=nn.Linear(28*28,100)
        #中间层
        self.layer2=nn.Linear(100,50)
        #输出层
        self.layer3=nn.Linear(50,10)

    def forward(self,input_data):
        input_data = input_data.view(-1,28*28)
        input_data = self.layer1(input_data)
        input_data = self.layer2(input_data)
        input_data = self.layer3(input_data)
        return input_data
