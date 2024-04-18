import copy
import time

import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST #包含很多数据集
from torchvision import transforms #处理数据，归一化等
import torch.utils.data as Data #数据处理的包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model import LeNet  #导入模型

'''
数据加载

'''
# 处理训练集和验证集
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    # 划分训练集和验证集
    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))]) #划分数据

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=0)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=0)

    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 定义训练所使用到的设备，有GPU用GPU，没有用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备
    model = model.to(device)
    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    '''
    初始化参数
    '''
    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 准确度列表
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    '''
       模型训练
    '''
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print("-"*50)

        #初始化参数
        # 损失函数
        train_loss = 0.0
        # 准确度
        train_correts = 0
        val_loss = 0.0
        val_correts = 0
        # 样本数量
        train_num = 0
        val_num = 0
        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备
            b_x = b_x.to(device)
            # 将标签放入到训练设备
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程， 输入为一个batch，输出为一个batch种对应的预测
            output = model(b_x)
            # 查找每一行种最大值对应的行标
            pre_label = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络参数， 以起到降低loss函数的作用
            optimizer.step()
            # 对损失函数进行累加          第一个维度大小为样本数量
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，准确度+1
            train_correts += torch.sum(pre_label == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)


        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备
            b_x = b_x.to(device)
            # 将标签放入到验证设备
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()

            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            loss = criterion(output,b_y)

            val_loss += loss.item() * b_x.size(0)
            val_correts += torch.sum(pre_label == b_y.data)
            val_num += b_x.size(0)


        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_correts.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correts.double().item() / val_num)

        print("{} train loss:{:.4f} train acc:{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc:{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度
        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())
        # 计算训练耗时
        time_use = time.time() - since
        print("训练耗费的时间{:.0f}m{:.0f}s".format(time_use//60,time_use%60))


    # 选择最优参数
    # 加载最高准确率下的模型参数
    # model.load_state_dict(best_model_wts) x
    torch.save(best_model_wts, './best_model.pth')
    # torch.save(model.state_dict(best_model_wts),'./best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 将模型实例化
    LeNet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
