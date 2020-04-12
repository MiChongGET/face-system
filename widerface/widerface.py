import os,cv2,sys,shutil,numpy

# pycharm提示此处报错，提示不存在dom，但是运行是正常的。。。
from xml.dom.minidom import Document

import os
def writexml(filename, saveimg, bboxes, xmlpath):
    doc = Document()

    annotation = doc.createElement('annotation')

    doc.appendChild(annotation)

    folder = doc.createElement('folder')

    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)
    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)
    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)
    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)
    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flickrid)
    owner = doc.createElement('owner')
    annotation.appendChild(owner)
    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('muke'))
    owner.appendChild(flickrid_o)
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('muke'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))

    size.appendChild(width)

    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode('face'))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[0] + bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[1] + bbox[3])))
        bndbox.appendChild(ymax)
    f = open(xmlpath, "w")
    f.write(doc.toprettyxml(indent=''))
    f.close()


rootdir = "D:\\Python\\dataset\\widerface"

# TXT文件中包含图片路径，每张图片中几个人脸，以及人脸所在的位置
gtfile = "D:\\Python\\dataset\\widerface\\data\\wider_face_split\\wider_face_val_bbx_gt.txt"

im_folder = "D:\\Python\\dataset\\widerface\\WIDER_val\\images"

##这里可以是test也可以是val
fwrite = open("D:\\Python\\dataset\\widerface\\ImageSets\\Main\\test.txt", "w")


with open(gtfile, "r") as gt:
    while(True):
        gt_con = gt.readline()[:-1]
        if gt_con is None or gt_con == "":
            break
        im_path = im_folder + "/" + gt_con
        print(im_path)
        im_data = cv2.imread(im_path)
        # 去除图片为空的情况
        if im_data is None:
            continue

        ##需要注意的一点是，图片直接经过resize之后，会存在更多的长宽比例，所以我们直接加pad
        sc = max(im_data.shape)
        im_data_tmp = numpy.zeros([sc, sc, 3], dtype=numpy.uint8)
        off_w = (sc - im_data.shape[1]) // 2
        off_h = (sc - im_data.shape[0]) // 2

        ##对图片进行周围填充，填充为正方形
        im_data_tmp[off_h:im_data.shape[0]+off_h, off_w:im_data.shape[1]+off_w, ...] = im_data
        im_data = im_data_tmp
        #
        # cv2.imshow("1", im_data)
        # cv2.waitKey(0)
        numbox = int(gt.readline())
        #numbox = 0
        bboxes = []
        for i in range(numbox):
            line = gt.readline()
            infos = line.split(" ")
            #x y w h ---人脸坐标
            #去掉最后一个（\n）
            for j in range(infos.__len__() - 1):
                infos[j] = int(infos[j])

            ##注意这里加入了数据清洗
            ##保留resize到640×640 尺寸在8×8以上的人脸
            if infos[2] * 80 < im_data.shape[1] or infos[3] * 80 < im_data.shape[0]:
                continue

            bbox = (infos[0] + off_w, infos[1] + off_h, infos[2], infos[3])
            # 在图片中画矩形框，表明人脸的位置
            cv2.rectangle(im_data, (int(infos[0]) + off_w, int(infos[1]) + off_h),
                          (int(infos[0]) + off_w + int(infos[2]), int(infos[1]) + off_h + int(infos[3])),
                          color=(0, 0, 255), thickness=1)
            # 图片中可能存在多个人脸，所以此处使用list叠加保存数据
            bboxes.append(bbox)

        cv2.imshow("1", im_data)
        cv2.waitKey(0)
        #
        # filename = gt_con.replace("/", "_")
        # # 写入文件名
        # fwrite.write(filename.split(".")[0] + "\n")
        #
        # # 写入文件
        # cv2.imwrite("{}/JPEGImages/{}".format(rootdir, filename), im_data)
        #
        # xmlpath = "{}/Annotations/{}.xml".format(rootdir, filename.split(".")[0])
        # writexml(filename, im_data, bboxes, xmlpath)

fwrite.close()