import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST #包含很多数据集
from torchvision import transforms #处理数据，归一化等
import torch.utils.data as Data #数据处理的包
import numpy as np
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./data',
                          train=True, # 只下载训练集
                          transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                          download=False,)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64, # 64一捆，哈哈哈（Tensor格式的图片+标签）
                               shuffle=True,
                               num_workers=0) # 加载数据的进程

# 获得一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

print("The size of train data",b_x.numpy() .shape)
# squeeze()用于从张量中移除大小为 1 的维度
batch_x = b_x.squeeze().numpy() # 将四维张量移除第一维，并转换成Numpy数组
batch_y = b_y.numpy()  # 张量-》numpy
print("The size of train data",batch_x.shape)


class_label = train_data.classes # 训练集的标签
print(class_label)

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5)) # 创建窗口，大小为（12,5）
for ii in np.arange(len(batch_y)):
    # print(ii)
    plt.subplot(4, 16, ii + 1) #指定了子图的行数、列数和当前子图的索引。
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray) # cmap=plt.cm.viridis 彩色展示
    plt.title(class_label[batch_y[ii]], size=10) # 在当前子图中添加标题（类别标签）
    plt.axis("off") # 关闭子图的坐标轴
    plt.subplots_adjust(wspace=0.05) # 子图之间的水平间距
plt.show()

