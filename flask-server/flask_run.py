from flask import Flask, request, make_response, jsonify
from object_detection.utils import ops as utils_ops
import os
import numpy as np
import base64
from flask_cors import *
import dlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from gevent import monkey

monkey.patch_all()
import tensorflow as tf

app = Flask(__name__)
# 解决跨域问题
CORS(app, supports_credentials=True)

PATH_TO_FROZEN_GRAPH = "frozen_inference_graph.pb"
PATH_TO_LABELS = "object_detection\\face_label_map.pbtxt"
IMAGE_SIZE = (256, 256)
detection_sess = tf.Session()

with detection_sess.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, IMAGE_SIZE[0], IMAGE_SIZE[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

#############################
# 人脸特征 face feature
face_feature_sess = tf.Session()
ff_pb_path = "face_recognition_model.pb"
with face_feature_sess.as_default():
    ff_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        ff_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(ff_od_graph_def, name='')
        ff_images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        ff_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        ff_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")


@app.route("/")
def helloworld():
    return '<h1>Hello World!</h1>'


# 图片上传
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    return upload_path


# 人脸检测接口
@app.route("/face_detect", methods=['POST'])
def inference():
    # 获取前端的base64图像字符串（摄像头拍摄的图片）
    img_str = str(request.form['base64Data'])

    # 获取到base64数据
    data = img_str.split("base64,")
    img_data = data[1]

    # 转换base64数据
    img = base64.b64decode(img_data)

    # 转化为cv需要的格式
    ima_data_cv = np.fromstring(img, np.uint8)
    im_data = cv2.imdecode(ima_data_cv, cv2.IMREAD_COLOR)

    cv2.imwrite("take.png", im_data)

    # im_url = request.args.get("url")
    # # 读取图片数据
    # im_data = cv2.imread(im_url)
    # # 重新定义图片大小,将图片转化为256*256大小
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    # 将图像矩阵转化为思维，参数加在第一维上面
    output_dict = detection_sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    # 初始化两个点,防止图片中没有人脸返回结果异常
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # print(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.1:
            bbox = output_dict['detection_boxes'][i]
            cate = output_dict['detection_classes'][i]
            y1 = IMAGE_SIZE[0] * bbox[0]
            x1 = IMAGE_SIZE[1] * bbox[1]
            y2 = IMAGE_SIZE[0] * (bbox[2])
            x2 = IMAGE_SIZE[1] * (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)

    # 当没有检测到人脸的时候
    if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
        return make_response(jsonify({"data": [x1, y1, x2, y2], "code": 201}))

    respose = make_response(jsonify({"data": [x1, y1, x2, y2], "code": 200}))
    return respose


# 使用dlib来检测人脸
@app.route("/face_detect_by_dlib", methods=['POST'])
def face_detect_by_dlib():
    # 获取前端的base64图像字符串（摄像头拍摄的图片）
    img_str = str(request.form['base64Data'])

    # 获取到base64数据
    data = img_str.split("base64,")
    img_data = data[1]

    # 转换base64数据
    img = base64.b64decode(img_data)

    # 转化为cv需要的格式
    ima_data_cv = np.fromstring(img, np.uint8)
    im_data = cv2.imdecode(ima_data_cv, cv2.IMREAD_COLOR)
    # 将摄像机拍摄的图片写入到本地
    cv2.imwrite('tmp/face.png', im_data)

    detector = dlib.get_frontal_face_detector()
    # 由dlib检测出来的人脸
    dets = detector(im_data)

    # 判断没有人脸的时候
    if len(dets) == 0:
        return make_response(jsonify({"data": None, "code": 201}))

    d = dets[0]
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    print('Left %d Top %d Right %d Bottom %d' % (d.left(), d.top(), d.right(), d.bottom()))
    return make_response(jsonify({"data": [x1, y1, x2, y2], "code": 200}))


# 图像数据的标准化，图片预处理过程
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


# 读取图片
def read_image(path):
    ###
    im_data = cv2.imread(path)
    im_data = prewhiten(im_data)
    im_data = cv2.resize(im_data, (160, 160))
    # 1 * h * w * 3
    return im_data


# 得到人脸的128个特征值
@app.route("/face_recognition")
def face_recognition():
    im_data1 = read_image("tmp\\lei.jpg")
    im_data1 = np.expand_dims(im_data1, axis=0)
    emb1 = face_feature_sess.run(ff_embeddings,
                                 feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})
    print(emb1)
    response = make_response(jsonify(str(emb1[0])))

    return response


# 计算欧式距离
@app.route("/face_dis")
def face_dis():
    im_data1 = read_image("tmp\\lei.jpg")

    im_data1 = np.expand_dims(im_data1, axis=0)

    emb1 = face_feature_sess.run(ff_embeddings,
                                 feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})

    im_data1 = read_image("tmp\\face.jpg")

    im_data1 = np.expand_dims(im_data1, axis=0)

    emb2 = face_feature_sess.run(ff_embeddings,
                                 feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})

    dist = np.linalg.norm(emb1 - emb2)

    return str(dist)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=9001, debug=True)
