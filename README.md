# face-system
### python+TensorFlow实现人脸识别
### 人脸识别API接口的开发
### 日常记录
#### 一、目录详情

```
cifar10                 cifar10图像分类任务
cocoapi-master          目标检测
data_manager            cifar10相关数据读取和打包
dlib                    dlib相关的介绍和一些示例代码
face-net                深度学习模型-FaceNet介绍和源代码
face-web                项目前端代码
facial-landmark         人脸关键点相关的代码
flask-server            项目后端代码
widerface               处理widerface数据集转VOC
```


#### 二、准备

##### 1、开发环境(注意对准版本号)
```
    Anaconda 建议使用这个，开辟新的虚拟环境，适合不同开发环境的调试和切换
    Python 3.7
    TensorFlow 1.13.X
    OpenCv 4.2.X
    numpy  1.16.2
    scipy 1.1.0
    dlib 19.17.99
```
##### 2、FaceNet环境搭建
- 详情请看face-net文件夹下面

##### 3、dlib环境搭建 --- 详细介绍请看dlib文件夹
```
  window下面安装比较繁琐，
  推荐博客https://blog.csdn.net/qq_35756383/article/details/102959409，

  dlib下载：https://github.com/davisking/dlib/releases)
            假如自己动手能力不足的话，就直接安装whl文件吧，丢一个百度云网盘地址：https://pan.baidu.com/s/1MKqW7WH2XP-J8MOLeq3cDA提取码: rfh8
            https://pypi.org/simple/dlib/ 这个网站也提供whl文件下载

  基于dlib的68个特征点检测模型:shape_predictor_68_face_landmarks.dat
  链接: https://pan.baidu.com/s/10ZZNw86SqZL3-0D2XqC6tg 提取码: p2fc
```

##### 4、相关数据集下载：
```
lfw--->链接：https://pan.baidu.com/s/1kH-OcCCAvLVLP1wEQxlvRg
提取码：twmg

64_CASIA-FaceV5
链接：https://pan.baidu.com/s/1vHk_BE6ycoz7ujPSvB9e9A
提取码：20a7

CASIA-WebFace
链接：https://pan.baidu.com/s/17m9Ym45g4km7VLCxedQ7GA
提取码：c951

Celeba
链接：https://pan.baidu.com/s/18RmCCj7uHfvtkmmb8LZoHw
提取码：a6ug
```
