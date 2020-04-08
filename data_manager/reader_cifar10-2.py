import tensorflow as tf

filename = ['data\\A.csv', 'data\\B.csv', 'data\\C.csv']

# 此处获取不到tensor值，所以sess.run()无法直接读取file_queue
file_queue = tf.train.string_input_producer(filename,
                                            shuffle=True,
                                            num_epochs=2)
reader = tf.WholeFileReader()
key, value = reader.read(file_queue)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    for i in range(6):
        print(sess.run([key, value]))
