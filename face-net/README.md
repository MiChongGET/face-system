### 一、FaceNet模型

#### (1)、FaceNet源码仓库，本地代码存放在facenet-master文件夹下面
​			https://github.com/davidsandberg/facenet

#### (2)、FaceNet源码解读（网友解读版一），代码存放在understand_facenet文件夹下
​			https://blog.csdn.net/u013044310/article/details/79556099
​			https://github.com/boyliwensheng/understand_facenet（配套源码地址）

#### (3)、FaceNet源码解读（网友解读版二），代码存放在facenet-master文件夹下

​			https://blog.csdn.net/huangshaoyin/article/details/81034551

#### (4)、triplet-reid源码地址

​			https://github.com/VisualComputingInstitute/triplet-reid

#### (5)、FaceNet源码解读(网友解读版三)，（2）是在借鉴此作者的博客，
- 本篇博客也提供了各种人脸数据集的介绍和`预模型`的下载
            https://blog.csdn.net/MrCharles/article/details/80360461

### 二、人脸匹配数据准备
#### 数据集
- LFW  下载地址：http://vis-www.cs.umass.edu/lfw/#views  谷歌网盘（需要梯子）：https://drive.google.com/drive/u/0/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams
- Celeba 下载地址：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- VGGface2
- CASIA-WebFace
- CASIA-faceV5
- 更多人脸数据集请看：https://www.cnblogs.com/ansang/p/8137413.html

#### 数据格式
- 文件夹名/文件夹名_文件名

- 同一个热的图片放在相同文件夹

  

#### Pre-trained models

-  https://blog.csdn.net/MrCharles/article/details/80360461  （一）中的第五点

![](https://ae01.alicdn.com/kf/Hbea52004ac754ea7a7883812f23fade2N.png)

模型下载链接：https://pan.baidu.com/s/1aiSq7wGpdHIe6MUKPnXgrA 密码：4dcn

> 20170512-110547（MS-Celeb-1M数据集训练的模型文件，微软人脸识别数据库，名人榜选择前100万名人，搜索引擎采集每个名人100张人脸图片。预训练模型准确率0.993+-0.004）



#### Inception ResNet v1 模型图

<img src="https://ae01.alicdn.com/kf/Hd053b820ced845f58090b433d513c8f3o.png" style="zoom:80%;" />


### 三、一些问题
#### 1、解决出现`ModuleNotFoundError: No module named 'facenet'`异常
- a.在cmd（需要管理员权限）命令行键入：set PYTHONPATH=...\facenet\src, 例如笔者的是:set PYTHONPATH=D:\Python\Work\face-system\face-net\facenet-master\src
- b.在 计算机-->属性-->高级系统设置-->环境变量 中,新建PYTHONPATH,键入 D:\Python\Work\face-system\face-net\facenet-master\src
- c.如果使用pycharm，请记得重启pycharm

#### 2、重新裁剪LFM图片数据集的大小

> 程序中神经网络使用的是谷歌的“inception resnet v1”网络模型，这个模型的输入时160*160的图像，而我们下载的LFW数据集是250*250限像素的图像，所以需要进行图片的预处理。

- 原本数据集放在raw文件夹下面，新裁剪的图片放在ifw_160文件夹下面

data/lfw/raw ：D:\\Python\\Work\\face-system\\face-net\\facenet-master\\data\\ifw\\raw
data/lfw/lfw_160：D:\\Python\\Work\\face-system\\face-net\\facenet-master\\data\\ifw\\ifw_160

```shell
# 运行脚本，记得将图片文件夹修改为自己的文件夹目录
python src\align\align_dataset_mtcnn.py data/lfw/raw data/lfw/lfw_160 --image_size 160 --margin 32
```

- pycharm中运行记得修改成下面这样

![](https://ae01.alicdn.com/kf/H8bfbd1bb2f2b474681a4a92b42731c65U.png)

#### 3、评估预训练模型的准确率

##### 1）、模型下载

> facenet提供了两个预训练模型，分别是基于CASIA-WebFace和MS-Celeb-1M人脸库训练的，不过需要去谷歌网盘下载，这里给其中一个模型的百度网盘的链接：

https://pan.baidu.com/s/1LLPIitZhXVI_V3ifZ10XNg#list/path=%2F 密码: 12mh
> 模型放在data文件夹下，没有就创建

##### 2）、运行脚本

```shell
# 运行脚本,同样的，目录改为自己的
data\lfw\lfw_160：D:\\Python\\Work\\face-system\\face-net\\facenet-master\\data\\ifw\\ifw_160
src\models\20180408-102900：D:\\Python\\Work\\face-system\\face-net\\facenet-master\\data\\models\\20180408-102900

Python src\validate_on_lfw.py data\lfw\lfw_160 src\models\20180408-102900
```
- 安装网络上面的做法会出现错误，是应为data/pairs.txt读取不到，所以需要在运行脚本上面加上`--lfw_pairs=txt的地址
![](https://ae01.alicdn.com/kf/H9f8ff6240a024da4821d07c405654677G.png)

```shell
运行脚本,同样的，目录改为自己的

data\lfw\lfw_160：D:\\Python\\Work\\face-system\\face-net\\facenet-master\\data\\ifw\\ifw_160
src\models\20180408-102900：D:\\Python\\Work\\face-system\\face-net\\facenet-master\\data\\models\\20180408-102900
data/pairs.txt：D:\\Python\\Work\\face-system\\face-net\\facenet-master\\data\\pairs.txt

Python src\validate_on_lfw.py data\lfw\lfw_160 src\models\20180408-102900 --lfw_pairs=data/pairs.txt
```

##### 3、TensorFlow版本导致报错

```shell
2020-04-17 00:27:11.307949: W tensorflow/core/graph/graph_constructor.cc:1272] Importing a graph with a lower producer version 24 into an existing graph with producer version 27. Shape inference will have run different parts of the graph with different producer versions.
Traceback (most recent call last):
```

- 解决方案

  1.把Tensorflow换为1.7版本的；

  2.在`facenet.py`代码中找到`create_input_pipeline` 再添加一行语句` with tf.name_scope("tempscope"):` 就可以完美解决（貌似Tensorflow 1.10及以上版本才修复这个bug）。

  ![](https://ae01.alicdn.com/kf/H2eb7b8dc5d984ecca4f52703467fb45b0.png)

- 运行结果，可以看出，模型的精度高达99.7%

  <img src="https://ae01.alicdn.com/kf/Hbe49c52d5942488fbed1296a0514254cW.png" style="zoom: 80%;" />

