### 一、FaceNet源码解读（网友解读版），代码存放在understand_facenet文件夹下
https://blog.csdn.net/u013044310/article/details/79556099

https://blog.csdn.net/u013044310/article/details/80481642

https://github.com/boyliwensheng/understand_facenet（配套源码地址）

> facenet提供了两个预训练模型，分别是基于CASIA-WebFace和MS-Celeb-1M人脸库训练的，不过需要去谷歌网盘下载，这里给其中一个模型的百度网盘的链接：

https://pan.baidu.com/s/1LLPIitZhXVI_V3ifZ10XNg#list/path=%2F 密码: 12mh
> 模型放在data文件夹下，没有就创建

### 二、遇到问题
#### 1、GPU内存溢出问题，已经解决
>在`detect_face.py`中加入下面的配置，防止出现GPU内存不足报错，放在代码靠前的位置

```python
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) #此处调整GPU的利用率
config.gpu_options.allow_growth = True
```
####  2、运行人脸比对程序`compare.py`

> 博客中运行方式是错误的，应该按照下面方式运行，应该是data，指的是寻找data文件夹下面的模型，而不是填写模型名称。文件夹路径要写对

![](https://ae01.alicdn.com/kf/H1de3e20962af409c8480a555c5ac12f5J.png)

<img src="https://ae01.alicdn.com/kf/H54230f4742ab47d7bf63c2eb665400a9A.png" style="zoom:80%;" />

> 最后运行两张一模一样的图片，结果如下，两张图片的欧氏距离

![](https://ae01.alicdn.com/kf/Ha23c4862de24470ba9b0de54f7bff20cn.png)

### 三、相关函数

#### 1、主要函数

- align/ ：用于人脸检测与人脸对齐的神经网络
- facenet ：用于人脸映射的神经网络
- util/plot_learning_curves.m:这是用来在训练softmax模型的时候用matlab显示训练过程的程序

#### 2、facenet/contributed/相关函数：

##### 1）、基于mtcnn与facenet的人脸聚类

> 代码：`facenet/contributed/cluster.py`（`facenet/contributed/clustering.py`实现了相似的功能，只是没有`mtcnn`进行检测这一步）

主要功能：

- ① 使用`mtcnn`进行人脸检测并对齐与裁剪

- ② 对裁剪的人脸使用`facenet`进行`embedding`

- ③ 对`embedding`的特征向量使用欧式距离进行聚类




##### 2）、基于mtcnn与facenet的人脸识别（输入单张图片判断这人是谁）

> 代码：`facenet/contributed/predict.py`

主要功能：

- ① 使用`mtcnn`进行人脸检测并对齐与裁剪

- ② 对裁剪的人脸使用`facenet`进行`embedding`

- ③ 执行`predict.py`进行人脸识别（需要训练好的svm模型）




##### 3）、以numpy数组的形式输出人脸聚类和图像标签

代码：`facenet/contributed/export_embeddings.py`

主要功能：

- ① 需要对数据进行对齐与裁剪做为输入数据

- ② 输出`embeddings.npy`；`labels.npy`；`label_strings.npy`
  