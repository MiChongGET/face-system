import pickle
import glob
import numpy as np
import cv2
import os


##################################
##       cifar十分类图片读取    ###
##                             ###
##################################

# 官方提供的解包代码
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 分类，主要用于生成不同的文件夹，以便图片存入对应的文件夹
classification = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']
if __name__ == '__main__':

    # 数据集的路径
    folers = 'D:\\Python\\Work\\face-system\\data_manager\\data\\cifar-10-batches-py'
    train_files = glob.glob(folers + "/data_batch*")
    data = []
    labels = []
    for file in train_files:
        dt = unpickle(file)
        data += list(dt[b"data"])
        labels += list(dt[b"labels"])
    print(np.shape(labels))  # labels形状为（50000,）
    print(np.shape(data))  # data形状为(50000, 3072)
    imgs = np.reshape(data, [-1, 3, 32, 32])  # imgs形状为(50000, 3, 32, 32)
    print(imgs.shape)

    # 将图片写入到对应类别的文件夹
    for i in range(imgs.shape[0]):
        im_data = imgs[i, ...]
        # 转换形状，由（3,32,32）转换为（32,32,3）
        im_data = np.transpose(im_data, [1, 2, 0])
        # 得到cv处理后的
        im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR)
        f = "{}/{}".format("D:\\Python\\Work\\face-system\\data_manager\\data\\images\\train",
                           classification[labels[i]])

        # 判断文件夹是否存在
        if not os.path.exists(f):
            os.mkdir(f)
        # 将图片写入到对应的文件夹
        cv2.imwrite("{}/{}.jpg".format(f, str(i)), im_data)
