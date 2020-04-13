import base64

import cv2
import numpy as np
from flask import Flask, request

app = Flask(__name__)


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


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=9001, debug=True)
