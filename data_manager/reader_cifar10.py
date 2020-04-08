import tensorflow as tf
import cv2

filelist = ['data\\train.tfrecord']
file_queue = tf.train.string_input_producer(filelist, num_epochs=None, shuffle=True)

reader = tf.TFRecordReader()
_, ex = reader.read(file_queue)
# 通过tf打包的文件，需要解码解包
# 前面使用 data = tf.gfile.FastGFile(im_d, "rb").read()

feature = {
    'image': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
}
batch_size = 2
batch = tf.train.shuffle_batch([ex], batch_size, capacity=batch_size * 10, min_after_dequeue=batch_size * 5)
example = tf.parse_example(batch, features=feature)

image = example['image']
label = example['label']

image = tf.decode_raw(image, tf.uint8)
image = tf.reshape(image, [-1, 32, 32, 3])

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    for i in range(1):
        img, _= sess.run([image, label])
        cv2.imshow("img", img[0, ...])
        cv2.waitKey(0)
