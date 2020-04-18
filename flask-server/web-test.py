import base64
import dlib
import cv2
import numpy as np
from flask import Flask, request, make_response, jsonify
from flask_cors import *

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


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=9001, debug=True)
