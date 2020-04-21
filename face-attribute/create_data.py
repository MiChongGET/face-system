import cv2
import tensorflow as tf
import dlib

##################
# 本程序代码将CelebA图片裁剪之后打包成tfrecords文件

anno_file = "H:\\DataSet\\CelebA\\Anno\\list_attr_celeba.txt"
ff = open(anno_file)
anno_info = ff.readlines()
attribute_class = anno_info[1].split(" ")

print(attribute_class.__len__())

# 四种属性Eyeglasses--15，Male--20，Young--31，Smiling--39
writer_train = tf.python_io.TFRecordWriter("train. tfrecords")
writer_test = tf.python_io.TFRecordWriter("test. tfrecords")

detector = dlib.get_frontal_face_detector()

for idx in range(2, anno_info.__len__()):
    info = anno_info[idx]

    attr_val = info.replace("  ", " ").split(" ")

    # print(attr_val.__len__())
    print(attr_val[0])
    print(attr_val[16])
    print(attr_val[21])
    print(attr_val[32])
    print(attr_val[40])

    im_data = cv2.imread("H:\\DataSet\\CelebA\\Img\\img_celeba.7z\\img_celeba\\" + attr_val[0])
    rects = detector(im_data, 0)

    # 如果没有人脸
    if len(rects) == 0:
        continue

    x1 = rects[0].left()
    y1 = rects[0].top()
    x2 = rects[0].right()
    y2 = rects[0].bottom()

    y1 = int(max((y1 - 0.3 * (y2 - y1), 0)))
    # cv2.rectangle(im_data, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # cv2.imshow('1',im_data)
    # cv2.waitKey(0)

    if y2 - y1 < 50 or x2 - x1 < 50 or x1 < 0 or y1 < 0:
        continue

    im_data = im_data[y1:y2, x1:x2]
    im_data = cv2.resize(im_data, (128, 128))

    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[im_data.tobytes()])
                ),
                "Eyeglasses": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[int(attr_val[16])]
                    )
                ),
                "Male": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[int(attr_val[21])]
                    )
                ),
                "Young": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[int(attr_val[32])]
                    )
                ),
                "Smiling": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[int(attr_val[40])]
                    )
                )
            }
        )
    )

    if idx > anno_info.__len__() * 0.95:
        writer_test.write(ex.SerializeToString())
    else:
        writer_train.write(ex.SerializeToString())

writer_test.close()
writer_train.close()
