import tensorflow as tf

images = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
labels = [1, 2, 3, 4]

# 此处可以直接获取到tensor的值
[images, labels] = tf.train.slice_input_producer([images, labels],
                              num_epochs=None,
                              shuffle=True)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    tf.train.start_queue_runners(sess=sess)

    for i in range(10):
        print(sess.run([images, labels]))
