import torch
from torch import nn
from torchsummary import summary #打印模型结构和参数信息


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.pool1(x)
        x = self.sig(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

if __name__=="__main__":
    # 判断是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    # print(summary(model,(1, 28, 28)))