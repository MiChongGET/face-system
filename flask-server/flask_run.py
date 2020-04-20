from flask import Flask, request, make_response, jsonify
from object_detection.utils import ops as utils_ops
import os
import numpy as np
import base64
from flask_cors import *
import dlib
import glob

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

###################
# 人脸检测准备
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
# 人脸特征准备 face feature
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

#############################
# 人脸关键点检测准备
face_landmark_sess = tf.Session()
ff_pb_path = "landmark.pb"
with face_landmark_sess.as_default():
    ff_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        ff_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(ff_od_graph_def, name='')
        landmark_tensor = tf.get_default_graph().get_tensor_by_name("fully_connected_1/Relu:0")


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
    sp = im_data.shape
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
            # 将坐标点还原到本来的大小
            # y1 = IMAGE_SIZE[0] * bbox[0]
            # x1 = IMAGE_SIZE[1] * bbox[1]
            # y2 = IMAGE_SIZE[0] * (bbox[2])
            # x2 = IMAGE_SIZE[1] * (bbox[3])
            # sp是原图片的大小
            y1 = int(bbox[0] * sp[0])
            x1 = int(bbox[1] * sp[1])
            y2 = int(bbox[2] * sp[0])
            x2 = int(bbox[3] * sp[1])
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


@app.route('/face_register', methods=['POST'])
def face_register():
    # 获取用户姓名
    user_name = str(request.form['name'])
    print("用户姓名：" + user_name)

    ## 获取摄像头捕获的头像
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
    img_path = "face/img/" + user_name + ".png"
    cv2.imwrite(img_path, im_data)

    sp = im_data.shape
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                         np.expand_dims(
                                             im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # 获取人脸框，然后提取人脸
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.1:
            bbox = output_dict['detection_boxes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)
            ## 提取人脸区域
            y1 = int(y1 * sp[0])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            face_data = im_data[y1:y2, x1:x2]
            im_data = prewhiten(face_data)  # 预处理
            im_data = cv2.resize(im_data, (160, 160))
            im_data1 = np.expand_dims(im_data, axis=0)
            ## 人脸特征提取
            emb1 = face_feature_sess.run(ff_embeddings,
                                         feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})

            strr = ",".join(str(i) for i in emb1[0])
            ## 将人脸特征写入到本地文件中
            feature_path = "face/feature/" + user_name + ".txt"
            with open(feature_path, "w") as f:
                f.writelines(strr)
            f.close()
            mess = make_response(jsonify({"data": [x1, y1, x2, y2], "code": 200, "type": True}))
            break
        else:
            mess = make_response(jsonify({"data": None, "code": 200, "type": False}))

    return mess


@app.route('/face_login', methods=['POST', 'GET'])
def face_login():
    # 图片上传
    # 人脸检测
    # 人脸特征提取
    # 加载注册人脸（人脸签到，人脸数很多，加载注册人脸放在face_login,
    # 启动服务加载/采用搜索引擎/ES）
    # 同注册人脸相似性度量
    # 返回度量结果
    # 获取用户姓名

    ## 获取摄像头捕获的头像
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

    sp = im_data.shape
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                         np.expand_dims(
                                             im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # 获取人脸框，然后提取人脸
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.1:
            bbox = output_dict['detection_boxes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)
            ## 提取人脸区域
            y1 = int(y1 * sp[0])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            face_data = im_data[y1:y2, x1:x2]
            im_data = prewhiten(face_data)  # 预处理
            im_data = cv2.resize(im_data, (160, 160))
            im_data1 = np.expand_dims(im_data, axis=0)
            ## 人脸特征提取
            emb1 = face_feature_sess.run(ff_embeddings,
                                         feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})

            # 取出服务端存储的所有人脸特征值，然后挨个比较，求出相应的欧式距离
            features_path = glob.glob("face\\feature" + "/*")

            # 记录用户名称
            user_name = ''
            for features_list in features_path:
                with open(features_list) as f:
                    # 获得各个文件中记录的特征值
                    fea_str = f.readlines()
                    # 获取文件名称
                    file_name = f.name.split("face\\feature\\")[1]
                    # 再次分割，得到用户名称
                    user_name = file_name.split(".txt")[0]

                    f.close()
                    # 计算欧式距离
                    emb2_str = fea_str[0].split(",")
                    emb2 = []
                    for ss in emb2_str:
                        emb2.append(float(ss))
                    emb2 = np.array(emb2)
                    dist = np.linalg.norm(emb1 - emb2)
                    print("dist---->", dist)
                    if dist < 0.5:
                        print("用户：" + user_name + "登录！")
                        return make_response(jsonify({"data": user_name, "code": 200, "type": True}))

    return make_response(jsonify({"data": None, "code": 200, "type": False}))


# 计算欧式距离
@app.route("/face_dis")
def face_dis():
    im_data1 = read_image("tmp\\face2.png")

    im_data1 = np.expand_dims(im_data1, axis=0)

    emb1 = face_feature_sess.run(ff_embeddings,
                                 feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})

    im_data1 = read_image("tmp\\face.png")

    im_data1 = np.expand_dims(im_data1, axis=0)

    emb2 = face_feature_sess.run(ff_embeddings,
                                 feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})

    dist = np.linalg.norm(emb1 - emb2)

    return str(dist)


# 人脸关键点检测
@app.route('/face_landmark', methods=['POST'])
def face_landmark():
    ## 获取摄像头捕获的头像
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
    img_path = "face/img/landmark.png"

    # 记录上传的图片大小
    sp = im_data.shape
    # 重新设置图片大小为256*256
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                         np.expand_dims(
                                             im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    # 获取人脸框，然后提取人脸
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.1:
            bbox = output_dict['detection_boxes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)
            ## 提取人脸区域
            y1 = int((y1 + (y2 - y1) * 0.2) * sp[0])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            face_data = im_data[y1:y2, x1:x2]
            cv2.imwrite(img_path, face_data)

            face_data = cv2.resize(face_data, (128, 128))
            pred = face_landmark_sess.run(landmark_tensor, {"Placeholder:0":
                                                                np.expand_dims(face_data, 0)})

            print(pred[0].shape)
            pred = pred[0]
            res = []
            for i in range(0, 136, 2):
                res.append(str((pred[i] * (x2 - x1) + x1) / sp[1]))
                res.append(str((pred[i + 1] * (y2 - y1) + y1) / sp[0]))

            res = ",".join(res)
            return make_response(jsonify({"data": res, "code": 200, "type": True}))
    return make_response(jsonify({"data": None, "code": 200, "type": False}))


# 人脸关键点检测,添加人脸状态返回
@app.route('/face_landmark_by_dlib', methods=['POST'])
def face_landmark_by_dlib():
    ## 获取摄像头捕获的头像
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

    # 引入dlib
    detector = dlib.get_frontal_face_detector()
    # 人脸检测
    dets = detector(im_data)
    if dets.__len__() == 0:
        make_response(jsonify({"data": None, "code": 200, "type": False}))

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 关键点集合
    landmarks = []
    shape = predictor(im_data, dets[0])  # 寻找人脸的68个标定点

    # 计算人脸热别框边长
    d = dets[0]
    face_width = d.right() - d.left()
    face_higth = d.top() - d.bottom()

    # 分析点的位置关系来作为表情识别的依据
    mouth_width = (shape.part(54).x - shape.part(48).x) / face_width  # 嘴巴咧开程度
    mouth_higth = (shape.part(66).y - shape.part(62).y) / face_width  # 嘴巴张开程度

    # 眉毛直线拟合数据缓冲
    line_brow_x = []
    line_brow_y = []

    # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
    brow_sum = 0  # 高度之和
    frown_sum = 0  # 两边眉毛距离之和
    for j in range(17, 21):
        brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
        frown_sum += shape.part(j + 5).x - shape.part(j).x
        line_brow_x.append(shape.part(j).x)
        line_brow_y.append(shape.part(j).y)

    tempx = np.array(line_brow_x)
    tempy = np.array(line_brow_y)
    # np.ployfit(x,a,n)拟合点集a得到n级多项式，其中x为横轴长度
    z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
    # round(x [,n])返回浮点数x的四舍五入值 round(80.23456, 2)返回80.23
    brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的

    brow_hight = (brow_sum / 10) / face_width  # 眉毛高度占比
    brow_width = (frown_sum / 5) / face_width  # 眉毛距离占比
    # print("眉毛高度与识别框高度之比：",round(brow_arv/self.face_width,3))
    # print("眉毛间距与识别框高度之比：",round(frown_arv/self.face_width,3))

    # 眼睛睁开程度
    eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
               shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
    eye_hight = (eye_sum / 4) / face_width
    # print("眼睛睁开距离与识别框高度之比：",round(eye_open/self.face_width,3))

    # 分情况讨论
    # 张嘴，可能是开心或者惊讶
    face_state = ""
    if round(mouth_higth >= 0.03):
        if eye_hight >= 0.056:
            print("amazing")
            face_state = "amazing"
        else:
            print("happy")
            face_state = "happy"

    # 没有张嘴，可能是正常和生气
    else:
        if brow_k <= -0.3:
            print("angry")
            face_state = "angry"
        else:
            print("nature")
            face_state = "nature"

    # 遍历当前人脸所有的关键点，并且放到list中
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        landmarks.append(pt_pos)

    return make_response(jsonify({"data": landmarks, "face_state": face_state, "code": 200, "type": True}))



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=9001, debug=True)
