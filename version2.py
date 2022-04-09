### Version 2: 使用ResNet预训练模型
# CVPR 2016 best paper
#
# 论文链接：https://arxiv.org/pdf/1512.03385.pdf
# ![ % E5 % 9
# B % BE % E7 % 89 % 87 - 2.
# png](attachment: % E5 % 9B % BE % E7 % 89 % 87-2.png)

# %%

from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm.notebook import tqdm
from glob import glob
from sklearn import metrics

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

# 设置使用CPU 还是 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据所在文件夹
# train_dir = '/home/deep/data/day2/train/'
# test_dir = '/home/deep/data/day2test/'
#数据所在文件夹
train_dir = 'D:\深度学习-雷课\data\dogs-vs-cats-redux-kernels-edition/train/'
test_dir = 'D:\深度学习-雷课\data\dogs-vs-cats-redux-kernels-edition/test/'

# 获取所有训练数据和测试数据的文件名
train_list = glob(os.path.join(train_dir, '*'))
test_list = glob(os.path.join(test_dir, '*'))
print(f"Num of train {len(train_list)}")
print(f"Num of test {len(test_list)}")

# train_list[0]

# 取出所有样本的标签，即文件名中的cat或者dog
train_labels = [path.split('/')[-1].split('.')[0] for path in train_list]
test_labels = [path.split('/')[-1].split('.')[0] for path in test_list]
print(len(train_labels), len(test_labels))


# train_labels[0]

# 统计训练集中cat和dog的数量
count = 0
for it in train_labels:
    if it == 'cat':
        count += 1
print(f"num of cat {count}")
print(f"num of dog {len(train_labels) - count}")

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

# split data
# 将训练集切分，80%分给训练集，20%当做验证集，按照标签的比例进行切分
list_train, list_valid = train_test_split(train_list, test_size=0.2, stratify=train_labels,
                                          random_state=seed)  # stratify 按照标签比例分配
print(f'Num of train: {len(list_train)}')
print(f'Num of valid: {len(list_valid)}')

# Image augmentation
# 图像增强，预处理
image_size = (224, 224)
train_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),  # 改变图片尺寸为(224,224)
        transforms.RandomResizedCrop(image_size),  # 随机大小，裁剪图片大小为(224,224)。
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

# class Dataset
# 自定义数据集
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
                label = 1 if label == 'dog' else 0  # 1 是狗，0是猫
        return img_transform, label

# 训练集，验证集，测试集
train_data = CustomDataset(list_train, transform=train_transforms)
valid_data = CustomDataset(list_valid, transform=valid_transforms)
test_data = CustomDataset(test_list, transform=test_transforms)

# 批大小
batch_size = 32
# 训练集加载器，验证集加载器，测试集加载器
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))

# model
# 使用预训练模型resnet18，下载模型
model = models.resnet18(pretrained=True)
# 获取全连接层输入大小
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# 设置预训练模型之后的全连接层
model.fc = nn.Linear(num_ftrs, 2)
# 模型加载到device设备上
model = model.to(device)

# 学习率
lr = 3e-5
gamma = 0.2
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler，会动态调整学习率
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

#%%time  #出错：Statement expected, found Py:PERC  %%time 将会给出cell的代码运行一次所花费的时间
# 开始训练
# 数据集迭代多少次
epochs = 20
# 记录最好的验证集准确率
best_val_acc = 0.
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    # 每次循环是一个batch
    for data, label in tqdm(train_loader):
        # 数据和标签，直接从xxx_loader中获取
        data = data.to(device)
        label = label.to(device)
        # 预测输出
        output = model(data)
        # 损失
        loss = criterion(output, label)

        # 梯度清空
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 优化模型参数
        optimizer.step()

        # 计算训练集准确率
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    # 不计算梯度
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        # 同理，计算验证集准确率
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    # 如果当前验证集准确率优于之前最好的，则保存模型和记录当前验证集准确率
    if epoch_val_accuracy > best_val_acc:
        best_val_acc = epoch_val_accuracy
        torch.save(model, 'dogs_vs_cats_resnet_model.pt')
        print('save model.', best_val_acc)

    print(
        f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    # 625/625 [46:33<00:00, 4.47s/it]
    # save model. tensor(0.9477)
    # Epoch : 1 - loss : 0.1468 - acc: 0.9388 - val_loss : 0.1182 - val_acc: 0.9477


# 测试过程
# from sklearn import metrics

model = torch.load('./dogs_vs_cats_resnet_model.pt')
model.to(device)
model.eval()
val_accuracy = 0.
prediction = []
true_labels = []
for data, label in tqdm(valid_loader):
    data = data.to(device)
    label = label.to(device)

    val_output = model(data)
    prediction += val_output.argmax(dim=1).cpu().tolist()
    true_labels += label.cpu().tolist()

    acc = (val_output.argmax(dim=1) == label).float().mean()
    val_accuracy += acc / len(valid_loader)
print(val_accuracy)
print(metrics.classification_report(prediction, true_labels))

model.eval()
valid_loader2 = DataLoader(dataset=valid_data, batch_size=2, shuffle=True)
data, label = next(iter(valid_loader2))
data, label = data.to(device), label.to(device)
pred_label = model(data)
pred_label = pred_label.argmax(dim=1)

fig, axes = plt.subplots(1, 2, figsize=(8, 6))
# 在画布每个单元格里画一张图片
for idx, ax in enumerate(axes.ravel()):
    # 打印图片的prediction和尺寸
    print(label[idx], data[idx].shape)
    # 设置子图的标题
    ax.set_title(pred_label[idx].tolist())
    # 坐标轴关闭
    ax.axis("off")
    # 画图
    ax.imshow(data[idx].permute(1, 2, 0).cpu())



