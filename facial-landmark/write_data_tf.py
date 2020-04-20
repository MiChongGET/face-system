import tensorflow as tf
import cv2
import numpy as np
import glob
from scipy.io import loadmat

landmark_path_data = "/home/kuan/code/mooc_py3_tensorflow/dataset/300W_LP/landmarks"

landmark_path_folders = glob.glob(landmark_path_data + "/*")

landmark_anno_list = []

for f in landmark_path_folders:
    landmark_anno_list += glob.glob(f + "/*.mat")

print(landmark_anno_list)

writer_train = tf.python_io.TFRecordWriter("train1.tfrecords")
writer_test = tf.python_io.TFRecordWriter("test1.tfrecords")

for idx in range(landmark_anno_list.__len__()):
    landmark_info = landmark_anno_list[idx]
    im_path =  landmark_info.replace("300W_LP/landmarks", "300W_LP").replace("_pts.mat", ".jpg")
    print(im_path)
    im_data = cv2.imread(im_path)
    # cv2.imshow("11", im_data)
    # cv2.waitKey(0)

    landmark = loadmat(landmark_info)['pts_2d']
    # print(landmark)
    # for i in range(68):
    #     cv2.circle(im_data, (int(landmark[i][0]), int(landmark[i][1])),
    #                2, (0, 255, 0), 2)
    #
    # cv2.imshow("11", im_data)
    # cv2.waitKey(0)

    x_max = int(np.max(landmark[0:68, 0]))
    x_min = int(np.min(landmark[0:68, 0]))

    y_max = int(np.max(landmark[0:68, 1]))
    y_min = int(np.min(landmark[0:68, 1]))


    y_min = int(y_min - (y_max - y_min)  * 0.3)
    y_max = int(y_max + (y_max - y_min)  * 0.05)

    x_min = int(x_min - (x_max - x_min)  * 0.05)
    x_max = int(x_max + (x_max - x_min)  * 0.05)

    # cv2.rectangle(im_data, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    # cv2.imshow("11", im_data)
    # cv2.waitKey(0)

    face_data = im_data[y_min:y_max, x_min:x_max]

    sp = face_data.shape

    im_point = []

    for p in range(68):
        im_point.append((landmark[p][0] - x_min) * 1.0 / sp[1])
        im_point.append((landmark[p][1] - y_min) * 1.0 / sp[0])

        cv2.circle(face_data, (int(im_point[p * 2] * sp[1]), int(im_point[p * 2 + 1] * sp[0])),
                   2, (0, 255, 0), 2)

    # cv2.imshow("11", face_data)
    # cv2.waitKey(0)

    face_data = cv2.resize(face_data, (128, 128))

    ex = tf.train.Example(
        features = tf.train.Features(
            feature = {
                "image" : tf.train.Feature(
                    bytes_list = tf.train.BytesList(value=[face_data.tobytes()])
                ),
                "label": tf.train.Feature(
                    float_list=tf.train.FloatList(value=im_point)
                )
            }
        )
    )

    if idx > landmark_anno_list.__len__() * 0.8:
        writer_test.write(ex.SerializeToString())
    else:
        writer_train.write(ex.SerializeToString())

writer_test.close()
writer_train.close()