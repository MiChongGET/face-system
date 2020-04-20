import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

##in_dim --->输入特征图通道数
##on_dim --->输出特征图通道数
def senet_blob(net, in_dim, on_dim, stride):
    bk = net

    net = slim.conv2d(net, in_dim // 4, [1, 1])
    ##没有单独定义relu和BN
    net = slim.conv2d(net, in_dim // 4, [3, 3])
    net = slim.conv2d(net, on_dim, [1, 1])
    if stride > 1:
        net = slim.avg_pool2d(net, [stride*2 -1, stride*2 - 1],
                              stride = stride, padding="SAME")

        bk = slim.avg_pool2d(bk, [stride*2 -1, stride*2 - 1],
                              stride = stride, padding="SAME")

    if in_dim != on_dim:
        bk = slim.conv2d(bk, on_dim, [1, 1])

    ##NHWC
    sq = tf.reduce_mean(net, axis=[1, 2])
    ex = slim.fully_connected(sq, on_dim // 16)
    ex = tf.nn.relu(ex)
    ##batchsize*on_dim
    ex = slim.fully_connected(ex, on_dim)
    ex = tf.nn.sigmoid(ex)
    net = net * tf.reshape(ex, [-1, 1, 1, on_dim])
    ##跳连的部分
    net = bk + net
    return net


def CNNNet(input_x, is_training=True, keep_prob=0.8):
    ##resnet + SENet

    bn_param = {
        'is_training': is_training,
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(0.00001),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=bn_param):
        with slim.arg_scope([slim.batch_norm], **bn_param):
            net = slim.conv2d(input_x, 32, [3, 3])
            print(net)
            net = slim.avg_pool2d(net, [3, 3], stride=2, padding="SAME")
            print(net)

            net = slim.conv2d(net, 64, [3, 3])
            net = slim.conv2d(net, 64, [3, 3])
            net = slim.avg_pool2d(net, [3, 3], stride=2, padding="SAME")
            # net = senet_blob(net, 32, 64, 2)
            print(net)

            net = slim.conv2d(net, 64, [3, 3])
            net = slim.conv2d(net, 64, [3, 3])
            net = slim.avg_pool2d(net, [3, 3], stride=2, padding="SAME")
            # net = senet_blob(net, 64, 128, 2)
            print(net)

            net = slim.conv2d(net, 128, [3, 3])
            net = slim.avg_pool2d(net, [3, 3], stride=2, padding="SAME")
            # net = senet_blob(net, 128, 128, 2)
            # net = senet_blob(net, 128, 256, 2)
            print(net)
            net = slim.conv2d(net, 256, [3, 3])
            #net = senet_blob(net, 256, 512, 2)
            print(net)
            net = tf.reduce_mean(net, axis=[1, 2])
            print(net)
            net = slim.fully_connected(net, 1024)
            net = tf.nn.dropout(net, keep_prob=keep_prob)
            print(net)
            net = tf.nn.relu(net)
            net = slim.fully_connected(net, 136)
            print(net)

            return net

def SENet(input_x, is_training=True, keep_prob=0.8):
    ##resnet + SENet

    bn_param = {
        'is_training':is_training,
        'decay': 0.997,
        'epsilon':1e-5,
        'scale':True,
        'updates_collections':tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(0.00001),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params = bn_param):
        with slim.arg_scope([slim.batch_norm], **bn_param):
            net = slim.conv2d(input_x, 32, [3, 3])
            print(net)
            net = slim.avg_pool2d(net, [3, 3], stride=2, padding="SAME")
            print(net)

            net = senet_blob(net, 32, 64, 2)
            print(net)

            net = senet_blob(net, 64, 128, 2)
            print(net)

            net = senet_blob(net, 128, 128, 2)
            print(net)

            net = senet_blob(net, 128, 256, 2)
            print(net)

            net = senet_blob(net, 256, 512, 2)
            print(net)
            net = tf.reduce_mean(net, axis=[1, 2])
            print(net)
            net = slim.fully_connected(net, 1024)
            net = tf.nn.dropout(net, keep_prob=keep_prob)
            print(net)
            net = tf.nn.relu(net)
            net = slim.fully_connected(net, 136)
            print(net)
#read data
def get_one_batch(batch_size, type):
    if type == 0: ##train
        file_list = tf.gfile.Glob("train*.tfrecords")
    else:
        file_list = tf.gfile.Glob("test*.tfrecords")

    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=True
    )

    _, se = reader.read(filename_queue)

    if type == 0:
        batch = tf.train.shuffle_batch([se], batch_size,
                                       capacity=batch_size,
                                       min_after_dequeue=batch_size // 2)
    else:
        batch = tf.train.batch([se], batch_size,
                                       capacity=batch_size)


    features = tf.parse_example(batch, features={
        'image':tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([136], tf.float32)
    })

    batch_im = features['image']
    batch_label = features['label']

    batch_im = tf.decode_raw(batch_im, tf.uint8)
    batch_im = tf.cast(tf.reshape(batch_im,
                                  (batch_size, 128, 128, 3)), tf.float32)
    batch_im = tf.image.resize_images(batch_im, (128, 128))
    return batch_im, batch_label

#net
input_x = tf.placeholder(tf.float32, shape=[1, 128, 128, 3])
print(input_x)
logits = CNNNet(input_x, is_training=False, keep_prob=1.0)
print(logits)

saver = tf.train.Saver(tf.global_variables())


with tf.Session() as session:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=session, coord=coord)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    ckpt = tf.train.get_checkpoint_state("models-3")

    saver.restore(session, ckpt.model_checkpoint_path)
    print(ckpt.model_checkpoint_path)

    output_graph_def = tf.graph_util.convert_variables_to_constants(session,
                                                                    session.graph.as_graph_def(),
                                                                    ['fully_connected_1/Relu'])

    with tf.gfile.FastGFile("landmark-3.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
        f.close()