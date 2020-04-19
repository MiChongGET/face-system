from scipy.io import loadmat
import cv2

m = loadmat("H:\\学习资料\\Python3+TensorFlow打造人脸识别智能小程序\\资料\\数据集\\300W_LP\\landmarks\\AFW\\AFW_134212_1_0_pts.mat")

landmark = m['pts_3d']
im_data = cv2.imread("img\\AFW_134212_1_0.jpg")

for i in range(68):
    cv2.circle(im_data, (int(landmark[i][0]), int(landmark[i][1])), 2, (0, 255, 0), 2)

cv2.imshow('11', im_data)
cv2.waitKey(0)
