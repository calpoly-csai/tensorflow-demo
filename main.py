import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 1])

cost = tf.reduce_mean(tf.pow((y_-y), 2))

train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

# need to make sure all our variables are cleared/init'd
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
steps = 10000


for i in range(steps):
    # create fake data for y = W.x + b, where W = 2, b = 0
    xs = np.array([[i]])
    ys = np.array([[2*i]])

    # Train
    feed = { x: xs, y_ : ys}
    sess.run(train_step, feed_dict=feed)
    print("After %d iterations:" % i)
    print("W: %f" % sess.run(W))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=feed))