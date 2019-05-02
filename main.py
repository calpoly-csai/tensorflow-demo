import numpy as np
import tensorflow as tf

# one feature, house size
# this will hold data to be fed into the model
x = tf.placeholder(tf.float32, [None, 1])

# tf.Variable will hold variables to be trained
# W has one output, house price (first number)
# W has one input, house size (second)
W = tf.Variable(tf.zeros([1,1]))
# B one feature, house size
b = tf.Variable(tf.zeros([1]))

# Tensorflow model - multiply two matricies
# Expands to y = W.x + b
y = tf.matmul(x, W) + b

# actual values 
y_ = tf.placeholder(tf.float32, [None, 1])

# minimize the differences between our model and 
# the actual values
cost = tf.reduce_mean(tf.pow((y_-y), 2))

# tensorflow can do gradient descent for us
train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

# need to make sure all our variables are cleared/init'd
sess = tf.Session()
init = tf.initialize_all_variables()
# Nothing happen in tensorflow until sess.run()
sess.run(init)
steps = 1000


for i in range(steps):
    # create fake data for y = W.x + b, where 
    # W is always 2, b = 0
    xs = np.array([[i]])
    ys = np.array([[2*i]])

    # Train
    feed = { x: xs, y_ : ys}
    sess.run(train_step, feed_dict=feed)
    print("After %d iterations:" % i)
    print("W: %f" % sess.run(W))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=feed))