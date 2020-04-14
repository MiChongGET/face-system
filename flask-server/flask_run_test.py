from flask import Flask, request, make_response, jsonify
from object_detection.utils import ops as utils_ops
import os
import numpy as np
import base64
from flask_cors import *

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


@app.route("/")
def helloworld():
    return '<h1>Hello World!</h1>'


# 图片上传
@app.route("/upload")
def upload():
    return "upload success!"


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
    # 读取图片数据
    # im_data = cv2.imread(ima_data_cv)
    # 重新定义图片大小
    # im_data = cv2.resize(im_data, IMAGE_SIZE)
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
    if (x1 == 0 & x2 == 0 & y1 == 0 & y2 == 0):
        return make_response(jsonify({"data": [x1, y1, x2, y2], "code": 201}))

    respose = make_response(jsonify({"data": [x1, y1, x2, y2], "code": 200}))
    return respose


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=9001, debug=True)
