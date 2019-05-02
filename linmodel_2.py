### lin reg for two variable sytem
### this concept can be generalized for any number of features

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

### Constants ###

datapoint_size = 1000
steps = 10000
actual_theta0 = 7 
actual_theta1 = 2
actual_theta2 = 5
learn_rate = 0.001

### Matrices prevent redundancies ###

### Model for linear regression
theta = tf.Variable(tf.zeros([2,1]))
theta0 = tf.Variable(tf.zeros([1]))
x = tf.placeholder(tf.float32, [None, 2]) # has to be a k x 2 matrix for multiplication

with tf.name_scope("thetax_theta0") as scope:
  product = tf.matmul(x,theta)
  y = product + theta0 					 ## scalar = 1 x n "feature matrix" 
									 ## times n x 1 "coefficient matrix" + scalar

y_ = tf.placeholder(tf.float32, [None, 1]) # actual prices are still a single value

### Define cost function
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_-y))
  cost_sum = tf.summary.scalar("cost", cost)

### Gradient descent minimizes cost
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

### prepare data
xs = []
ys = []
for i in range(datapoint_size):
	x_1 = i%10
	x_2 = np.random.randint(datapoint_size/2)%10
	y = actual_theta1 * x_1 + actual_theta2 * x_2 + actual_theta0
	xs.append([x_1, x_2])
	ys.append(y)

xs = np.array(xs)
ys = np.transpose([ys])

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(steps):
	feed = {x: xs, y_: ys}

	sess.run(train_step, feed_dict=feed)

	print("iteration: %d" % i)
	print("theta: %s" % sess.run(theta))
	print("theta0: %s" % sess.run(theta0))
	print("cost: %f" % sess.run(cost, feed_dict=feed))
	