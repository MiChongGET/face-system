import tensorflow as tf
import cv2
import numpy as np
import glob
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

path_list = [
    "/home/kuan/code/mooc_py3_tensorflow/dataset/BioID/*.pgm",
   # "/home/kuan/code/mooc_py3_tensorflow/dataset/BioID"
   # "/home/kuan/code/mooc_py3_tensorflow/dataset/CelebA/Img/img_celeba.7z/img_celeba/*"
]

im_list = []

for path in path_list:
    im_list += glob.glob(path)

tfrecord_file_train = "data/train_2.tfrecord"
tfrecord_file_test = "data/test_2.tfrecord"

im_size = 128
index = [i for i in range(im_list.__len__())]
np.random.shuffle(index)

def write_data(begin, end, tfrecord_file):


    writer = tf.python_io.TFRecordWriter(tfrecord_file)

    for i in range(begin, end):
        print(im_list.__len__(), i)
        #print(index[i])
        img = cv2.imread(im_list[index[i]])
        if img is None:
            continue
        # 取灰度
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 人脸数rects
        rects = detector(img_gray, 0)
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])

            x = []
            y = []
            for idx, point in enumerate(landmarks):
                x.append(point[0, 0])
                y.append(point[0, 1])

            x1 = min(x)
            y1 = min(y)

            x2 = max(x)
            y2 = max(y)


            # x1 = rects[i].left()
            # y1 = rects[i].top()
            # x2 = rects[i].right()
            # y2 = rects[i].bottom()

            y1 = int(y1 - (y2 - y1) * 0.1)
            y2 = int(y2 + (y2 - y1) * 0.05)

            x1 = int(x1 - (x2 - x1) * 0.05)
            x2 = int(x2 + (x2 - x1) * 0.05)


            img = img[y1:y2, x1:x2]
            sp = img.shape

            if sp[0] == 0 or sp[1] == 0:
                continue

            im_point = []
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0] - x1, point[0, 1] - y1)

                im_point.append(float(pos[0] * 1.0 / sp[1]))
                im_point.append(float(pos[1] * 1.0 / sp[0]))

                # cv2.circle(img, (int(im_point[idx * 2] * sp[1]),
                #                  int(im_point[idx * 2 + 1] * sp[0])),
                #            1, (255, 0, 0), 1)

            #print(im_point)
            #cv2.imshow("im", cv2.resize(img, (500, 500)))
            #cv2.waitKey(0)
            data = cv2.resize(img, (im_size, im_size))

            #data = tf.gfile.FastGFile(im_d, "rb").read()
            ex = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        "image":tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[data.tobytes()])),
                        "label": tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=im_point)),
                    }
                )
            )
            writer.write(ex.SerializeToString())

    writer.close()

write_data(0, int(im_list.__len__() * 0.9), tfrecord_file_train)
write_data(int(im_list.__len__() * 0.9), im_list.__len__(), tfrecord_file_test)