##################################
##                              ##
##    cifar10分类图片打包        ##
##                              ##
##################################
import tensorflow as tf
import cv2
import numpy as np

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

import glob

idx = 0
im_labels = []
im_data = []

# 循环10次
for i in classification:
    # 获取图片文件夹
    path = "data\\images\\train\\" + i
    # 获取当前文件夹下面的图片路径集合,例如：['data\\images\\train\\deer\\43327.jpg',...]
    # im_list长度5000
    im_list = glob.glob(path + "/*")
    # 标签集合，例如[0,0,0.....]
    im_label = [idx for i in range(im_list.__len__())]
    idx += 1
    im_data += im_list
    im_labels += im_label

# 设置打包文件存放路径
tfrecord_file = "data\\train.tfrecord"
writer = tf.python_io.TFRecordWriter(tfrecord_file)

# 记录下标
index = [i for i in range(im_data.__len__())]
# 打乱下标，目的是使得图片无序存储
np.random.shuffle(index)

for i in range(im_data.__len__()):
    # 通过opencv读取图片数据
    im_d = im_data[index[i]]
    # 如果使用opencv方式，则在tf.train.Example中使用value=[data.tobytes()]
    data = cv2.imread(im_d)

    # 也可以通过tf服务图片，使用这种方式，下面的data.tobytes()必须换成data,因为此时的data已经是bit类型了
    # 通过tf打包的文件，需要解码解包
    #data = tf.gfile.FastGFile(im_d, "rb").read()

    # 读取图片标签
    im_l = im_labels[index[i]]

    # 封装数据，包括image和label两个标签
    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[data.tobytes()])),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[im_l])),
            }
        )
    )

    # 写入序列化文件
    writer.write(ex.SerializeToString())

# 关闭
writer.close()
