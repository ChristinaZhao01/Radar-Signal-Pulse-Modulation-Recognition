import glob
import cv2
import torch
from PIL import Image
from numpy import array
from torch import softmax
from torchvision import transforms
from torchvision.transforms import ToTensor
import model

classes = ["LFM", "2FSK", "4FSK", "BPSK", "DLFM", "EQFM", "Frank", "FSKBPSK", "LFMBPSK", "MLFM", "MP", "SFM"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shape = (44, 44)
num = 101  # 用测试集的图片进行预测
data = []
# 找模型存放地址
address_first = "D:\\python work\\Py work folder\\final\\experiment_5\\test5\\e_5\\"
address_model = "private_model_11_98.t7"
address = address_first + address_model


def predict():
    with torch.no_grad():
        #加载模型参数
        model1 = model.Model(num_classes=len(classes)).to(device)
        model1.load_state_dict(torch.load(address, map_location=device))
        model1 = model1.eval()
        for k in range(0, len(classes)):
            right = 0
            name = classes[k]
            for i in range(1, num):
                imgfile = glob.glob(
                    'D:\\python work\\Py work folder\\data\\picture\\' + name + '\\test1\\' + str(i) + '.png')  # 查找要预测的图片所在路径

                for i in imgfile:
                    imgfile1 = i.replace("\\", "/")  # 最后读的路径要注意 将\\转成/
                    img = cv2.imread(imgfile1)

                    if imgfile1.find("4FSK") >= 1 or imgfile1.find("MLFM") >= 1:
                        img = cv2.GaussianBlur(img, (3, 3), 3)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (48, 48))
                    img = img.astype('uint8')  # numpy
                    img = Image.fromarray(array(img))

                    val_transform = transforms.Compose([transforms.RandomCrop(shape[0]), ToTensor()])
                    img = val_transform(img)
                    img = img.unsqueeze(0).to(device)  # 在第一维增加一个维度，用于图像增强

                    out1 = model1(img)
                    out1 = softmax(out1, dim=1)
                    # print(out1, '\n')
                    predicted, index = torch.max(out1.data, 1)
                    degree = int(index[0])
                    list = ["LFM", "2FSK", "4FSK", "BPSK", "DLFM", "EQFM", "Frank", "FSKBPSK", "LFMBPSK", "MLFM", "MP",
                            "SFM"]
                    if list[degree] == name:
                        right += 1
            acc = right / (num - 1) * 100
            data.append(acc)
            print("the accuracy of " + name + " is:%.3f" % acc)
        acc = sum(data) / 12
        print("the accuracy of total is:%.3f" % acc)
        # print(predicted * 100, list[degree])


if __name__ == '__main__':
    predict()
