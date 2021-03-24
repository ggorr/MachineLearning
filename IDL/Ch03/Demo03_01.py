import os

# default 0, filtering INFO 1, filtering WARNING 2, filtering ERROR 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def fig3_6():
	ii = [[[[0], [0], [2], [2]],
		   [[0], [0], [2], [2]],
		   [[0], [0], [2], [2]],
		   [[0], [0], [2], [2]]]]
	I = tf.constant(ii, tf.float32)
	print(I.shape)
	ww = [[[[-1]], [[-1]], [[1]]],
		  [[[-1]], [[-1]], [[1]]],
		  [[[-1]], [[-1]], [[1]]]]
	W = tf.constant(ww, tf.float32)
	print(W.shape)
	C = tf.nn.conv2d(I, W, strides=[1, 1, 1, 1], padding='VALID')
	sess = tf.Session()
	res = sess.run(C)
	print(res.shape)
	print(res)


def fig3_6_1():
	ii = [[[[0, 0, 0], [0, 0, 0], [2, 2, 2], [2, 2, 2]],
		   [[0, 0, 0], [0, 0, 0], [2, 2, 2], [2, 2, 2]],
		   [[0, 0, 0], [0, 0, 0], [2, 2, 2], [2, 2, 2]],
		   [[0, 0, 0], [0, 0, 0], [2, 2, 2], [2, 2, 2]]]]
	I = tf.constant(ii, tf.float32)
	print(I.shape, 'batch size, width, height, input channels(RGB)')
	ww = [[[[-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1]], [[1, 1], [1, 1], [1, 1]]],
		  [[[-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1]], [[1, 1], [1, 1], [1, 1]]],
		  [[[-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1]], [[1, 1], [1, 1], [1, 1]]]]
	W = tf.constant(ww, tf.float32)
	print(W.shape, 'width, height, input channels(RGB), numFlters=output channels')
	C = tf.nn.conv2d(I, W, strides=[1, 1, 1, 1], padding='SAME')
	sess = tf.Session()
	res = sess.run(C)
	print(res.shape, 'batch size, width, height, numFlters=output channels')
	print(res)


def fig3_7():
	old_v = tf.logging.get_verbosity()
	tf.logging.set_verbosity(tf.logging.ERROR)

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	tf.logging.set_verbosity(old_v)

	batchSz = 100

	W = tf.Variable(tf.random_normal([784, 10], stddev=.1))
	b = tf.Variable(tf.random_normal([10], stddev=.1))

	img = tf.placeholder(tf.float32, [batchSz, 784])
	ans = tf.placeholder(tf.float32, [batchSz, 10])

	# ===================================================================================
	image = tf.reshape(img, [batchSz, 28, 28, 1])
	flts = tf.Variable(tf.truncated_normal([4, 4, 1, 4], stddev=0.1))
	convOut = tf.nn.conv2d(image, flts, [1, 2, 2, 1], 'SAME') # try VALID
	convOut = tf.nn.relu(convOut)
	print(convOut.shape, 'batch size, width, height, numFlters=output channels')
	convOut = tf.reshape(convOut, [100, 784])
	prbs = tf.nn.softmax(tf.matmul(convOut, W) + b)
	# prbs = tf.nn.softmax(tf.matmul(img, W) + b)
	# ====================================================================================

	xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))

	train = tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
	numCorrect = tf.equal(tf.argmax(prbs, 1), tf.argmax(ans, 1))
	accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	# ------------------------------------------------------------------------------
	acc = np.zeros(1000)
	for i in range(1000):
		imgs, anss = mnist.train.next_batch(batchSz)
		acc[i], _ = sess.run([accuracy, train], feed_dict={img: imgs, ans: anss})

	sumAcc = 0
	for i in range(1000):
		imgs, anss = mnist.test.next_batch(batchSz)
		sumAcc += sess.run(accuracy, feed_dict={img: imgs, ans: anss})
	print("Test Accuracy: %r" % (sumAcc / 1000))  # > .92
	plt.plot(acc)
	plt.show()

def fig3_8():
	old_v = tf.logging.get_verbosity()
	tf.logging.set_verbosity(tf.logging.ERROR)

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	tf.logging.set_verbosity(old_v)

	batchSz = 100

	# W = tf.Variable(tf.random_normal([784, 10], stddev=.1))
	W = tf.Variable(tf.random_normal([1568, 10], stddev=0.1))
	b = tf.Variable(tf.random_normal([10], stddev=.1))

	img = tf.placeholder(tf.float32, [batchSz, 784])
	ans = tf.placeholder(tf.float32, [batchSz, 10])

	# ===================================================================================
	image = tf.reshape(img, [batchSz, 28, 28, 1])
	flts = tf.Variable(tf.random_normal([4, 4, 1, 16], stddev=0.1))
	convOut = tf.nn.conv2d(image, flts, [1, 2, 2, 1], "SAME")
	convOut = tf.nn.relu(convOut)
	flts2 = tf.Variable(tf.random_normal([2, 2, 16, 32], stddev=0.1))
	convOut2 = tf.nn.conv2d(convOut, flts2, [1, 2, 2, 1], "SAME")
	convOut2 = tf.reshape(convOut2, [100, 1568])
	prbs = tf.nn.softmax(tf.matmul(convOut2, W) + b)
	# ====================================================================================

	xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))

	train = tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
	numCorrect = tf.equal(tf.argmax(prbs, 1), tf.argmax(ans, 1))
	accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	# ------------------------------------------------------------------------------
	acc = np.zeros(1000)
	for i in range(1000):
		imgs, anss = mnist.train.next_batch(batchSz)
		acc[i], _ = sess.run([accuracy, train], feed_dict={img: imgs, ans: anss})

	sumAcc = 0
	for i in range(1000):
		imgs, anss = mnist.test.next_batch(batchSz)
		sumAcc += sess.run(accuracy, feed_dict={img: imgs, ans: anss})
	print("Test Accuracy: %r" % (sumAcc / 1000))  # > .92
	plt.plot(acc)
	plt.show()

def fig3_7_bias():
	old_v = tf.logging.get_verbosity()
	tf.logging.set_verbosity(tf.logging.ERROR)

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	tf.logging.set_verbosity(old_v)

	batchSz = 100

	W = tf.Variable(tf.random_normal([784, 10], stddev=.1))
	b = tf.Variable(tf.random_normal([10], stddev=.1))

	img = tf.placeholder(tf.float32, [batchSz, 784])
	ans = tf.placeholder(tf.float32, [batchSz, 10])

	# ===================================================================================
	image = tf.reshape(img, [batchSz, 28, 28, 1])
	flts = tf.Variable(tf.truncated_normal([4, 4, 1, 4], stddev=0.1))
	bias = tf.Variable(tf.zeros([4]))
	convOut = tf.nn.conv2d(image, flts, [1, 2, 2, 1], 'SAME') + bias
	convOut = tf.nn.relu(convOut)
	print(convOut.shape, 'batch size, width, height, numFlters=output channels')
	convOut = tf.reshape(convOut, [100, 784])
	prbs = tf.nn.softmax(tf.matmul(convOut, W) + b)
	# prbs = tf.nn.softmax(tf.matmul(img, W) + b)
	# ====================================================================================

	xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))

	train = tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
	numCorrect = tf.equal(tf.argmax(prbs, 1), tf.argmax(ans, 1))
	accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	# ------------------------------------------------------------------------------
	acc = np.zeros(1000)
	for i in range(1000):
		imgs, anss = mnist.train.next_batch(batchSz)
		acc[i], _ = sess.run([accuracy, train], feed_dict={img: imgs, ans: anss})

	sumAcc = 0
	for i in range(1000):
		imgs, anss = mnist.test.next_batch(batchSz)
		sumAcc += sess.run(accuracy, feed_dict={img: imgs, ans: anss})
	print("Test Accuracy: %r" % (sumAcc / 1000))  # > .92
	plt.plot(acc)
	plt.show()

# fig3_6()
# fig3_6_1()
# fig3_7()
fig3_8()
# fig3_7_bias()