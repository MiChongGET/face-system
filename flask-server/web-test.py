import base64
import dlib
import cv2
import numpy as np
from flask import Flask, request, make_response, jsonify
from flask_cors import *
from keras.models import load_model

app = Flask(__name__)
# 解决跨域问题
CORS(app, supports_credentials=True)


# 笔记本摄像头拍照测试
@app.route("/pic", methods=['POST'])
def getPhoto():
    # 获取前端的base64图像字符串（摄像头拍摄的图片）
    img_str = str(request.form['base64Data'])

    # 获取到base64数据
    data = img_str.split("base64,")
    img_data = data[1]

    # 转换base64数据
    img = base64.b64decode(img_data)

    # 转化为cv需要的格式
    ima_data_cv = np.fromstring(img, np.uint8)
    ima_data_cv = cv2.imdecode(ima_data_cv, cv2.IMREAD_COLOR)

    # 将摄像机拍摄的图片写入到本地
    cv2.imwrite('tmp/face.png', ima_data_cv)
    # cv2.imshow('pic', ima_data_cv)
    # cv2.waitKey(0)

    return 'h'


@app.route("/json")
def getjson():
    response = make_response(jsonify({'test': 'good', "code": 403}))
    return jsonify({'test': 'good', "code": 403})


# 使用dlib检测人脸接口
@app.route("/face_detect_by_dlib", methods=['POST'])
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


@app.route('/face_register', methods=['POST'])
def face_register():
    # 获取用户姓名
    user_name = str(request.form['name'])
    print("用户姓名：" + user_name)

    mess = make_response(jsonify({"data": user_name, "code": 200, "type": True}))
    return mess


# 人脸关键点检测
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


# 人脸属性检测
@app.route('/face_attribute', methods=['POST'])
def face_attribute():
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

    face_classifier = cv2.CascadeClassifier(
        "data\\model\\haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(140, 140))

    gender_classifier = load_model(
        "D:\\Python\\Work\\face-system\\flask-server\\data\\model\\simple_CNN.81-0.96.hdf5")
    gender_labels = {0: '女', 1: '男'}
    color = (255, 255, 255)

    if faces.__len__() == 0:
        return make_response(jsonify({"data": None, "code": 200, "type": False}))

    face = faces[0]
    x = face[0]
    y = face[1]
    w = face[2]
    h = face[3]

    face = im_data[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, 0)
    face = face / 255.0
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]
    # img = ChineseText.cv2ImgAddText(img, gender, x + h, y, color, 30)
    print(gender)

    return make_response(
        jsonify({"data": [int(x), int(y), int(w), int(h)], "gender": gender + "", "code": 200, "type": True}))


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=9001, debug=True)
