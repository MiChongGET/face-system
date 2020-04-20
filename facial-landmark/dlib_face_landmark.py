import dlib
import cv2

detector = dlib.get_frontal_face_detector()
img = cv2.imread("img\\AFW_134212_1_0.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 人脸检测
dets = detector(img)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# shape = predictor(img, dets[0])
#
# # 生成 Dlib 的图像窗口
# win = dlib.image_window()
# win.set_image(img)
# # 绘制面部轮廓
# win.add_overlay(shape)
# dlib.hit_enter_to_continue()


dets = detector(img, 1)
#关键点集合
landmarks = []
for face in dets:
    shape = predictor(img, face)  # 寻找人脸的68个标定点
    # 遍历所有点，打印出其坐标，并圈出来
    count = 0
    # 遍历当前人脸所有的关键点
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        count = count + 1
        landmarks.append(pt_pos)

        # if count == 38 or count == 44:
        cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(count), pt_pos, font, 0.4, (255, 255, 255), 1, 1)

    cv2.imshow("image", img)
    cv2.waitKey(0)

print(landmarks)