import os
import cv2
import torch

from torch.utils.tensorboard import SummaryWriter

write = SummaryWriter("logs")  # 方便tensorboard进行可视化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_images = []
train_label = []
test_images = []
test_label = []
for i in range(1, 2):
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
    for j, dir in enumerate(dirs):  # 每一个dir对应一种调制加一种用途
        k = 0
        imgList = os.listdir(dir)
        # print(imgList)
        imgList.sort(key=lambda x: int(x.replace("", "").split('.')[0]))  # test为1-100 train为1-700
        # print(imgList)
        for count in range(0, 1):
            im_name = imgList[count]
            im_path = os.path.join(dir, im_name)
            face = cv2.imread(im_path)  # 读取每一张图片
            cv2.imshow('imag', face)
            cv2.waitKey(0)      # 无限等待按键事件，按任意键继续
            face = cv2.GaussianBlur(face, (9, 9), 5)
            cv2.imshow('imag', face)
            cv2.waitKey(0)
