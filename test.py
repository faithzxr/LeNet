import torch

# 创建一个具有大小为1的维度的张量
x = torch.randn(1, 3, 1, 5)
print("原始张量的形状:", x.shape)  # 输出：torch.Size([1, 3, 1, 5])

# 使用squeeze()移除大小为1的维度
x_squeezed = x.squeeze()
print("移除大小为1的维度后的形状:", x_squeezed.shape)  # 输出：torch.Size([3, 5])
