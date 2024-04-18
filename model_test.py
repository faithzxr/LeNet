import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet


'''
数据加载

'''
# 处理训练集和验证集
def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    # 划分训练集和验证集

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)



    return test_dataloader


def test_model_process(model, test_dataloader):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 将模型放到训练设备中
    model = model.to(device)
    # 初始化参数
    test_correts = 0.0
    test_num = 0

    # 模型推理，只进行前向传播，不进行梯度计算
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader: #一张一张推理
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度+1
            test_correts += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    test_acc = test_correts.double().item() / test_num
    print("测试准确率为：", test_acc)

if __name__ =="__main__":
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载测试模型
    # test_model_process(model, test_dataloader)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为验证模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output)
            result = pre_lab.item()
            label = b_y.item()

            print("预测值：", classes[result], "------", "真实值", classes[label])
