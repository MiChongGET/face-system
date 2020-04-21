import tensorflow as tf
from tensorflow.contrib.layers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

slim = tf.contrib.slim


def inception_v3(images, drop_out=0.5, is_training=True):
    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        "decay": 0.9997,
        "epsilon": 0.00001,
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_var"]
        }
    }

    weights_regularizer = tf.contrib.layers.l2_regularizer(0.00004)

    with tf.contrib.slim.arg_scope(
            [tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
            weights_regularizer=weights_regularizer,
            trainable=True):
        with tf.contrib.slim.arg_scope(
                [tf.contrib.slim.conv2d],
                weights_regularizer=tf.truncated_normal_initializer(stddev=0.1),
                activation_fn=tf.nn.relu,
                normalizer_params=batch_norm_params):
            nets, endpoints = inception_v3_base(images)
            print(nets)
            print(endpoints)
            net = tf.reduce_mean(nets, axis=[1, 2])
            net = tf.nn.dropout(net, drop_out, name="droplast")
            net = flatten(net, scope="flatten")

    net_eyeglasses = slim.fully_connected(net, 2, activation_fn=None)
    net_young = slim.fully_connected(net, 2, activation_fn=None)
    net_male = slim.fully_connected(net, 2, activation_fn=None)
    net_smiling = slim.fully_connected(net, 2, activation_fn=None)

    return net_eyeglasses, net_young, net_male, net_smiling


input_x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
label_eyeglasses = tf.placeholder(tf.int64, shape=[None, 1])
label_young = tf.placeholder(tf.int64, shape=[None, 1])
label_male = tf.placeholder(tf.int64, shape=[None, 1])
label_smiling = tf.placeholder(tf.int64, shape=[None, 1])

logits_eyeglasses, logits_young, logits_male, logits_smiling = inception_v3(input_x, 0.5, True)
loss_eyeglasses = tf.losses.sparse_softmax_cross_entropy(labels=label_eyeglasses, logits=logits_eyeglasses)
loss_young = tf.losses.sparse_softmax_cross_entropy(labels=label_young, logits=logits_young)
loss_male = tf.losses.sparse_softmax_cross_entropy(labels=label_male, logits=logits_male)
loss_smiling = tf.losses.sparse_softmax_cross_entropy(labels=label_smiling, logits=logits_smiling)

logits_eyeglasses = tf.nn.softmax(logits_eyeglasses)
logits_young = tf.nn.softmax(logits_young)
logits_male = tf.nn.softmax(logits_male)
logits_smiling = tf.nn.softmax(logits_smiling)

loss = loss_eyeglasses + loss_young + loss_male + loss_smiling

reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
l2_loss = tf.add_n(reg_set)

# learn
global_step = tf.Variable(0, trainable=True)
lr = tf.train.exponential_decay(0.0001, global_step, decay_steps=1000, decay_rate=0.98, staircase=False)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    tf.train.AdamOptimizer(lr).minimize(loss + l2_loss, global_step)
