import numpy as np
import tensorflow as tf

EPOCHS = 10000
PRINT_STEP = 1000

# data = np.array([[1, 2, 3, 4, 5], [ 2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
# target = np.array([[6], [7], [8]])

print data.shape[1]
x_ = tf.placeholder(tf.float32, [None, data.shape[1]])
y_ = tf.placeholder(tf.float32, [None, 1])

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=data.shape[1])

outputs, states = tf.nn.rnn(cell, [x_], dtype=tf.float32)
outputs = outputs[-1]

W = tf.Variable(tf.random_normal([data.shape[1], 1]))
b = tf.Variable(tf.random_normal([1]))

y = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.square(y - y_))
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
error = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(EPOCHS):
        sess.run(train_op, feed_dict={x_:data, y_:target})
        if i % PRINT_STEP == 0:
            c = sess.run(cost, feed_dict={x_:data, y_:target})
            print('training cost:', c)

    response = sess.run(y, feed_dict={x_:data})
    print(response)
    # incorrect = sess.run(error,{data: data, target: target})
    incorrect = sess.run(error, feed_dict={x_: data, y_: target})
    #print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    print sess.run(prediction,{data: })
    sess.close()
