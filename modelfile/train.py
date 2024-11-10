import os

import cv2
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms, ToTensor

from modelfile import model

write = SummaryWriter("logs")  # 用tensorboard进行可视化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shape = (44, 44)
shape1 = (48, 48)

# 获取信号图像，信噪比为-10~8db，每2db有100张作为测试集，700张为训练集
class DataSetFactory:
    def __init__(self):
        train_images = []
        train_label = []
        test_images = []
        test_label = []
        for i in range(1, 11):
            dirs = ['D:\\python work\\Py work folder\\data\\picture\\2FSK\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\2FSK\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\4FSK\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\4FSK\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\BPSK\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\BPSK\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\DLFM\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\DLFM\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\EQFM\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\EQFM\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\Frank\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\Frank\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\FSKBPSK\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\FSKBPSK\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\LFM\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\LFM\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\LFMBPSK\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\LFMBPSK\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\MLFM\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\MLFM\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\MP\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\MP\\train' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\SFM\\test' + str(i),
                    'D:\\python work\\Py work folder\\data\\picture\\SFM\\train' + str(i)]
            for j, dir in enumerate(dirs):  # 每一个文件对应是不同信号，不同信噪比的训练集或测试集，返回的是索引值和对应文件
                imgList = os.listdir(dir)   # 将读取到的图片文件夹中的图片文件按随机顺序保存到imglist列表中
                # 对读取的图片文件进行排序
                imgList.sort(key=lambda x: int(x.replace("", "").split('.')[0]))  # test为1-100 train为1-700
                for count in range(0, len(imgList)):
                    im_name = imgList[count]    # 排序后可根据顺序索引值在imgList中找到对应的图像的文件名
                    im_path = os.path.join(dir, im_name)    # 合并成文件路径
                    pic = cv2.imread(im_path)  # 读取每一张图片
                    # 对读取到的图片进行处理
                    # 因为直接转灰度图对4FSK和MLFM的处理效果不好，对两者进行高斯滤波处理
                    if im_path.find("4FSK") >= 1 or im_path.find("MLFM") >= 1:
                        pic = cv2.GaussianBlur(pic, (3, 3), 3)      # 用3*3的高斯滤波器卷积核对图像进行平滑处理
                    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)     # 将BGR格式的图像转换成灰度图，方便转换成像素值送入模型进行训练
                    pic = cv2.resize(pic, (48, 48))
                    pic = pic.astype('uint8')       # 将图像的像素值以无符号整形数据类型存储
                    # 因为对应的每个信号下对应着训练集和测试集，所以每两个对应着标签中的同一个标签
                    var = lambda j: int(j / 2) if j % 2 == 0 else int((j + 1) / 2 - 1)
                    # 形成训练集及测试集以及标签
                    if im_path.find("train") >= 0:
                        train_images.append(Image.fromarray(pic))
                        train_label.append(var(j))
                    if im_path.find("test") >= 0:
                        test_images.append(Image.fromarray(pic))
                        test_label.append(var(j))


        print('training size : %d   val size : %d'
              % (len(train_images), len(test_images)))
        # 对训练集和测试集的图像进行预处理
        # Compose是将对图片处理的步骤组合
        train_transform = transforms.Compose([
            transforms.RandomCrop(shape[0]),    # 对图片进行随机位置裁剪
            transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定的PIL图像
            ToTensor(),         # 将PIL图像中的像素值转换成0~1之间的长*高*宽的张量形式
        ])
        val_transform = transforms.Compose([
            transforms.CenterCrop(shape[0]),    # 对图片在中间区域进行裁剪
            ToTensor(),
        ])

        self.training = DataSet(transform=train_transform, images=train_images, wave=train_label)
        self.private = DataSet(transform=val_transform, images=test_images, wave=test_label)


class DataSet(torch.utils.data.Dataset):

    def __init__(self, transform=None, images=None, wave=None):
        self.transform = transform
        self.images = images
        self.wave = wave

    def __getitem__(self, index):
        image = self.images[index]
        wave = self.wave[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, wave

    def __len__(self):
        return len(self.images)


def main():
    batch_size = 128
    lr = 0.007
    epochs = 100
    # 用来调整学习率
    learning_rate_decay_start = 80      # 开始学习率衰减的迭代周期
    learning_rate_decay_every = 5       # 学习率的衰减速度
    learning_rate_decay_rate = 0.9      # 学习率衰减系数

    classes = ["LFM", "2FSK", "4FSK", "BPSK", "DLFM", "EQFM", "Frank", "FSKBPSK", "LFMBPSK", "MLFM", "MP", "SFM"]
    network = model.Model(num_classes=len(classes)).to(device)

    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)       # 权重衰减用于避免模型过拟合
    criterion = nn.CrossEntropyLoss()
    factory = DataSetFactory()
    # num_workers是指进程数
    # 获得训练数据集
    training_loader = DataLoader(factory.training, batch_size=batch_size, shuffle=True, num_workers=1)
    # 获得测试训练集
    validation_loader = DataLoader(factory.private, batch_size=batch_size, shuffle=True, num_workers=1)

    min_validation_loss = 10000     # 损失优化标准
    max_validation_acc = 0          # 精确度优化标准

    for epoch in range(epochs):
        network.train()
        total = 0
        correct = 0
        total_train_loss = 0
        # 学习率衰减机制（指数型）
        # 大于学习率衰减起始时间则对学习率按照学习率衰减机制进行更新
        if epoch > learning_rate_decay_start >= 0:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = lr * decay_factor
            for group in optimizer.param_groups:        # 对梯度下降中的学习率参数进行更新
                group['lr'] = current_lr
        # 否则保持初始学习率
        else:
            current_lr = lr
        print('learning_rate: %s' % str(current_lr))

        for i, (x_train, y_train) in enumerate(training_loader):
            optimizer.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_predicted = network(x_train)
            loss = 0.1 * criterion(y_predicted, y_train)        # 计算交叉熵损失
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_predicted.data, 1)       # 得到模型计算出来的概率最大的索引值，对应标签为预测结果（1代表dim=1）
            total_train_loss += loss.data
            total += y_train.size(0)
            correct += predicted.eq(y_train.data).sum()         # 比较预测标签与真实标签是否相等
        accuracy = 100. * float(correct) / total                # 计算精确度
        print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (i + 1), accuracy))
        if epoch % 10 == 9 or epoch == 0:  # 从0开始每十个轮次可视化一次，使用tensorboard进行可视化
            write.add_scalars('loss', {'Train': total_train_loss / (i + 1), }, epoch)
            write.add_scalars('acc', {'Train': accuracy, }, epoch)

        network.eval()      # 测试集不进行梯度下降反向传播参数更新
        with torch.no_grad():
            total = 0
            correct = 0
            total_validation_loss = 0
            # 计算测试集的预测值、损失以及精确度
            for j, (x_val, y_val) in enumerate(validation_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_predicted = network(x_val)
                val_loss = 0.1 * criterion(y_val_predicted, y_val)
                _, predicted = torch.max(y_val_predicted.data, 1)
                total_validation_loss += val_loss.data
                total += y_val.size(0)
                correct += predicted.eq(y_val.data).sum()

            accuracy = 100. * float(correct) / total  # 计算精确度

            if accuracy > max_validation_acc:   # 满足条件最值更替
                max_validation_acc = accuracy   # 只有比上一轮的精确度高才会被更替
            if total_validation_loss <= min_validation_loss:  # 满足优化条件 参数更替
                if epoch >= 10:
                    print('saving new model')
                    state = network.state_dict()        # 保存状态字典，即将在模型中可以用于反向传播进行学习更新的参数进行保存
                    torch.save(state, 'D:/python work/Py work folder/final/experiment_5/test5/e_5/model_save%s_model_%d_%d.t7'
                               % ("private", epoch + 1, accuracy))      # 后缀是训练的迭代周期数和该轮的精确度
                    min_validation_loss = total_validation_loss

            if epoch % 10 == 9 or epoch == 0:  # 从0开始每十个轮次可视化一次
                write.add_scalars('loss', {'private': total_validation_loss / (j + 1), }, epoch)
                write.add_scalars('acc', {'private': accuracy, }, epoch)

            print('Epoch [%d/%d] %s validation Loss: %.4f, Accuracy: %.4f' % (  # 每轮结果可视化
                epoch + 1, epochs, "private", total_validation_loss / (j + 1), accuracy))

    write.close()
    # 打印全局最优解
    print("Best Private loss: %.4f" % (min_validation_loss / (j + 1)))
    print("Best Private acc: %.4f" % max_validation_acc)


if __name__ == "__main__":
    main()
