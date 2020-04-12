### 处理widerface数据集转VOC

#### 数据集下载地址
http://shuoyang1213.me/WIDERFACE/

#### 数据主要包括三个：
> 前三个是图片数据，最后一个是保存图片路径和图片上面人脸的坐标数据的集合文件（txt文件）
- train 训练集
- test  测试集
- val   验证集
- wider_face_split （Attached the mappings between attribute names and label values.）

#### Passcal VOC介绍
> PV主要包含了三个文件夹：Annotations，ImageSets，JPEGImages
- Annotations       此文件夹主要包含了打包数据之后的xml文件，xml文件中包含了多个信息
- ImageSets\\Main   此文件夹下包含了图片文件名称的集合（TXT文件）
- JPEGImages        此文件夹主要包含了重命名的图片文件