import tensorflow as tf
import cv2
import glob

import numpy as np

pb_path = "landmark.pb"

sess = tf.Session()

with sess.as_default():
    with tf.gfile.FastGFile(pb_path, "rb") as f:
        graph_def = sess.graph_def
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

#im_list = glob.glob("/home/kuan/code/mooc_py3_tensorflow/flask_server_facerecognition/face_landmark.jpg")
im_list = glob.glob("img\\AFW_134212_1_0.jpg")
landmark = sess.graph.get_tensor_by_name("fully_connected_1/Relu:0")

for im_url in im_list:
    print(im_url)
    im_data = cv2.imread(im_url)
    sp = im_data.shape
    print(sp)
    im_data = im_data[sp[0] * 1//4:,:,:]
    print(im_data.shape)
    im_data = cv2.resize(im_data, (128, 128))

    pred = sess.run(landmark, {"Placeholder:0":
                                   np.expand_dims(im_data, 0)})

    print(pred)
    pred = pred[0]
    for i in range(0, 136, 2):
        # if i == 48*2:
        cv2.circle(im_data, (int(pred[i] * 128), int(pred[i+1] * 128)),
                   2, (0, 255, 0), 2)

    print(pred[44 * 2 + 1] * 128 - pred[46 * 2 + 1] * 128)

    cv2.imshow("11", im_data)
    cv2.waitKey(0)