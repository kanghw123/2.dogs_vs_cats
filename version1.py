# # %% md
#
# ## 猫狗大战
# + 学习目标
#
# 猫狗大战项目要求对一个混合了猫和狗的图片数据集进行二分类，要求使用CNN神经网络模型在训练集上对已分类的猫和狗的图片进行建模。
#
# + 任务介绍
#
# Dogs
# vs.Cats(猫狗大战)
# 来源[Kaggle](https: // www.kaggle.com / c / dogs - vs - cats)上的一个竞赛题，任务为给定一个[数据集](
#     https: // www.kaggle.com / c / dogs - vs - cats / data)，设计一种算法对测试集中的猫狗图片进行判别。
#
# 训练集：
#
# 训练集由标记为cat和dog的猫狗图片组成，各
# `12500`
# 张，总共
# `25000`
# 张，图片为24位jpg格式，即RGB三通道图像， ** 图片尺寸不一 **
# ![ % E5 % 9
# B % BE % E7 % 89 % 87 - 4.
# png](attachment: % E5 % 9B % BE % E7 % 89 % 87-4.png)
# 测试集：
#
# 测试集由
# `12500`
# 张的cat或dog图片组成，未标记，图片也为24位jpg格式，RGB三通道图像， ** 图像尺寸不一 **
# ![ % E5 % 9
# B % BE % E7 % 89 % 87 - 3.
# png](attachment: % E5 % 9B % BE % E7 % 89 % 87-3.png)

# %% md

### Version 1: 自定义网络结构

# 准确率差，仅为理解知识点

# %% md

# 导入必需的库

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import random
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

# %% md

# 一些必要的超参数和事先处理


# %%

# 设置随机数种子方法
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 执行随机数种子方法
seed = 24
seed_everything(seed)

# %%

# 批大小
batch_size = 16
# 裁剪之后的图片尺寸
image_size = (200, 200)

lr = 3e-5
nepoch = 10

# 设置使用CPU 还是 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%

# 讲解device的作用
torch.cuda.is_available()

# %% md

#获得数据

# %%

# 数据所在文件夹
# train_dir = 'D:/workspace/数据/dogs-vs-cats-redux-kernels-edition/train/'
# test_dir = 'D:/workspace/数据/dogs-vs-cats-redux-kernels-edition/test/'
train_dir = '/home/deep/data/day2/train/'
test_dir = '/home/deep/data/day2test/'

# %%

os.listdir(train_dir)

# %%

# 获取所有训练数据和测试数据的文件名
train_list = glob(os.path.join(train_dir, '*'))
test_list = glob(os.path.join(test_dir, '*'))
print(f"Num of train {len(train_list)}")
print(f"Num of test {len(test_list)}")

# %%

# os.listdir('data')
os.listdir('/home/deep/seven_days')  # 打印路径下的文件和目录

# %%

# 讲解glob
# glob是python自带的一个操作文件的相关模块，
# 由于模块功能比较少，所以很容易掌握。
# 用它可以查找符合特定规则的文件路径名。
# 使用该模块查找文件，只需要用到： “*”, “?”, “[]”这三个通配符;
# os.listdir('data/')
glob('/home/deep/seven_days/' + '*.ipynb')

# %%

# 讲解os.path
# os.path 模块主要用于获取文件的属性。
print(os.path.basename('/home/deep/main.py'))  # 返回文件名
print(os.path.dirname('/home/deep/main.py'))  # 返回目录路径
# print(os.path.dirname('D:\\workspace\\数据\\dogs-vs-cats-redux-kernels-edition/test/')) # 返回目录路径
print(os.path.split('/home/deep/main.py'))  # 分割文件名与路径
print(os.path.join('home', 'deep', 'main.py'))
print(os.path.abspath('.'))

# %%

# 取出所有样本的标签，即文件名中的cat或者dog，/../cat.0.jpg
train_labels = [path.split('/')[-1].split('.')[0] for path in train_list]
test_labels = [path.split('/')[-1].split('.')[0] for path in test_list]
print(len(train_labels), len(test_labels))

# %%

train_list[0]

# %%

path = train_list[0]

# %%

path.split('/')[-1].split('.')[0]

# %%

# 讲解str数据类型的split
print(train_list[0])
path = train_list[0]
print(path.split('/'))  # 如果在linux中使用/

# %%

# 讲解列表推导式
# 列表推导式可以利用 range 区间、元组、列表、字典和集合等数据类型，快速生成一个满足指定需求的列表。
# 列表推导式的语法格式如下：
# [表达式 for 迭代变量 in 可迭代对象 [if 条件表达式] ]
alist = [x ** 2 for x in range(10)]
print(alist)

# %%


# %%

# 统计训练集中cat和dog的数量
count = 0
for it in train_labels:
    if it == 'cat':
        count += 1
print(f"num of cat {count}")
print(f"num of dog {len(train_labels) - count}")

# %%

# plot some image
# 随机取几张图片使用画图工具展示
## 随机获取9张图片的索引
random_idx = np.random.randint(1, len(train_list), size=9)
## 设置3行3列的画布，大小为(8,6)
fig, axes = plt.subplots(3, 3, figsize=(8, 6))
# 在画布每个单元格里画一张图片
for idx, ax in enumerate(axes.ravel()):
    # 打开第idx张图片
    img = Image.open(train_list[random_idx[idx]])
    # 打印图片的索引和尺寸
    print(random_idx[idx], img.size)
    # 设置子图的标题
    ax.set_title(train_labels[random_idx[idx]])
    # 坐标轴关闭
    ax.axis("off")
    # 画图
    ax.imshow(img)

# %%

# split data
# 将训练集切分，80%分给训练集，20%当做验证集，按照标签的比例进行切分
list_train, list_valid = train_test_split(train_list, test_size=0.2, stratify=train_labels,
                                          random_state=seed)  # stratify 按照标签比例分配
print(f'Num of train: {len(list_train)}')
print(f'Num of valid: {len(list_valid)}')

# %%

np.random.randint(2, size=(20,))

# %%

# 讲解train_test_split
a = np.arange(20)
b = np.random.randint(2, size=(20,))
print('a', a)
print('b', b)
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=seed)
print('a_train', a_train)
print('a_test', a_test)
print('b_train', b_train)
print('b_test', b_test)

# %% md
#
# 1.
# 定义输入数据

# %%

# Image augmentation
# 图像增强，预处理
train_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),  # 改变图片尺寸
        transforms.RandomResizedCrop(image_size),  # 随机大小，裁剪图片大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转化为tensor
    ]
)

valid_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

# %%

print(train_list[0])
img_path = train_list[0]
img = Image.open(img_path)


# t = transforms.Resize(100)
# t(img)
# t = transforms.RandomResizedCrop(100) # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为指定的大小；（即先随机裁剪，然后对裁剪得到的图像缩放为同一大小）
# t(img)
# t = transforms.RandomHorizontalFlip() # 随机水平翻转
# t(img)
# t = transforms.ToTensor()
# t(img)
# print(type(img))

# %%

# class Dataset
# 自定义数据集
# 数据集一定要继承torch.utils.data.Dataset并提供__getitem__和__len__方法
# 目的是返回一个一个样本数据
# 除了自定义的方式，PyTorch也提供了TensorDataset等封装好的Dataset
class CustomDataset(Dataset):
    # 构造函数，传入一个文件列表和一个预处理方法
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    # 获取数据集长度
    def __len__(self):
        return len(self.file_list)

    # 获取第idx张图片的数据
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        if img_path.split('/')[-1][-3:] == 'jpg':
            img = Image.open(img_path)
            if self.transform is not None:
                img_transform = self.transform(img)
                label = img_path.split('/')[-1].split('.')[0]
                label = 1 if label == 'dog' else 0
        return img_transform, label


# %%

# 训练集，验证集，测试集
train_data = CustomDataset(list_train, transform=train_transforms)
valid_data = CustomDataset(list_valid, transform=valid_transforms)
test_data = CustomDataset(test_list, transform=test_transforms)

# %%

# 训练集加载器，验证集加载器，测试集加载器
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)  # 用PyTorch的DataLoader类封装，实现数据集顺序打乱
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))

# %%

# Dataset和DataLoader的关系
# len(train_data)

# %%

a = torch.randn(10)
l1 = nn.Linear(10, 2)
b = l1(a)
print(a.shape, b.shape)

# %%

# 体会relu
a = torch.randn(10)
b = torch.relu(a)
print(a)
print(b)

# %%

# 体会最大池化max_pool2d
a = torch.randn(size=(1, 4, 4))
b = F.max_pool2d(a, 2)
print(a)
print(b)

# %% md
#
# 2.
# 定义模型


# %%

class Net(nn.Module):  # 新建一个网络类，就是需要搭建的网络，必须继承PyTorch的nn.Module父类
    def __init__(self):  # 构造函数，用于设定网络层
        super(Net, self).__init__()  # 标准语句
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)  # 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)  # 第二个卷积层，输入通道数16，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认

        self.fc1 = nn.Linear(50 * 50 * 16, 128)  # 第一个全连层，线性连接，输入节点数50×50×16，输出节点数128
        self.fc2 = nn.Linear(128, 64)  # 第二个全连层，线性连接，输入节点数128，输出节点数64
        self.fc3 = nn.Linear(64, 2)  # 第三个全连层，线性连接，输入节点数64，输出节点数2

    def forward(self, x):  # 重写父类forward方法，即前向计算，通过该方法获取网络输入数据后的输出值
        x = self.conv1(x)  # 第一次卷积
        x = F.relu(x)  # 第一次卷积结果经过ReLU激活函数处理
        x = F.max_pool2d(x, 2)  # 第一次池化，池化大小2×2，方式Max pooling

        x = self.conv2(x)  # 第二次卷积
        x = F.relu(x)  # 第二次卷积结果经过ReLU激活函数处理
        x = F.max_pool2d(x, 2)  # 第二次池化，池化大小2×2，方式Max pooling

        x = x.view(x.size()[0], -1)  # 由于全连层输入的是一维张量，因此需要对输入的[50×50×16]格式数据排列成[40000×1]形式
        x = F.relu(self.fc1(x))  # 第一次全连，ReLU激活
        x = F.relu(self.fc2(x))  # 第二次全连，ReLU激活
        y = self.fc3(x)  # 第三次激活，ReLU激活

        return y

# 参考于 https://github.com/xbliuHNU/DogsVsCats

# %% md

# ![ % E5 % 9
# B % BE % E7 % 89 % 87.
# png](attachment: % E5 % 9B % BE % E7 % 89 % 87.png)
# ![ % E5 % 9
# B % BE % E7 % 89 % 87 - 2.
# png](attachment: % E5 % 9B % BE % E7 % 89 % 87-2.png)

# %%

# 讲解nn.Conv2d
input = torch.randn(20, 16, 50, 100)  # batch_size 为20,通道数为16，宽为50，高维100
m = nn.Conv2d(16, 33, 3, stride=2)  # 输入通道数16，输出通道数33，卷积核大小3×3，跨度为2
output = m(input)
input.shape, output.shape

# %%

input = torch.randn(1, 1, 3, 3)
m = nn.Conv2d(1, 1, 2)
output = m(input)
print('input', input, )
print('output', output)
m.weight, m.bias

# %%

2.234 * 0.1569 + 2.2565 * -0.2033 + 0.2442 * 0.4643 + 2.3920 * 0.3316 + 0.3275

# %%

# 讲解Linear，relu，max_pool2d

# %% md

# 3.
# 计算输出
#
# 4.
# 计算损失
#
# 5.
# 计算梯度
#
# 6.
# 优化参数


# %%

# 讲解iterator，epoch的概念

# %%

def train(dataloader, device, nepoch):
    model = Net()  # 实例化一个网络
    model = model.to(device)  # 设置使用CPU还是GPU
    model.train()  # 网络设定为训练模式，有两种模式可选，.train()和.eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 实例化一个优化器，即调整网络参数，优化方式为adam方法

    criterion = nn.CrossEntropyLoss()  # 定义loss计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小

    cnt = 0  # 训练图片数量
    total_loss = 0.
    for epoch in range(nepoch):
        # 读取数据集中数据进行训练，因为dataloader的batch_size设置为16，所以每次读取的数据量为16，即img包含了16个图像，label有16个
        for img, label in tqdm.tqdm(dataloader):  # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
            img, label = img.to(device), label.to(device)  # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
            out = model(img)  # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
            loss = criterion(out, label)  # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好
            loss.backward()  # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            optimizer.step()  # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()  # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加

            cnt += 1
            total_loss += loss.item()

            if cnt % 100 == 0:
                print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt * batch_size,
                                                                   total_loss / cnt))  # 打印一个batch size的训练结果

        torch.save(model.state_dict(), 'dogs_vs_cats_cnn_model_epoch{}.pth'.format(epoch))  # 训练所有数据后，保存网络的参数


# %%

train(train_loader, device, nepoch)

# %%

from sklearn import metrics


def test(dataloader, device, model_file):
    # setting model
    model = Net()  # 实例化一个网络
    model.to(device)  # 使用CPU或者GPU
    model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数
    model.eval()  # 设定为评估模式，即计算过程中不要dropout

    prediction = []
    true_labels = []

    for img, label in tqdm.tqdm(dataloader):
        img, label = img.to(device), label.to(device)
        out = model(img)
        prediction += out.argmax(-1).cpu().tolist()
        true_labels += label.cpu().tolist()

    print(metrics.classification_report(true_labels, prediction))


# %%

a = torch.randn(3, 2)
print(a)
print(a.argmax(dim=0))

# %%

test(valid_loader, device, 'dogs_vs_cats_cnn_model_epoch6.pth')