import os

# default 0, filtering INFO 1, filtering WARNING 2, filtering ERROR 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt


def f0():
	x = tf.constant("Hello World")
	sess = tf.Session()
	print(sess.run(x))


def fig2_1():
	x = tf.constant(2.0)
	z = tf.placeholder(tf.float32)
	sess = tf.Session()
	comp = tf.add(x, z)
	print(sess.run(comp, feed_dict={z: 3.0}))
	print(sess.run(comp, feed_dict={z: 16.0}))
	print(sess.run(x))


# print(sess.run(comp))
def f1():
	bt = tf.random_normal([10], stddev=.1)
	b = tf.Variable(bt)
	W = tf.Variable(tf.random_normal([784, 10], stddev=.1))
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	print(sess.run(b))


def fig2_2():
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

	prbs = tf.nn.softmax(tf.matmul(img, W) + b)
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


def fig2_9():
	old_v = tf.logging.get_verbosity()
	tf.logging.set_verbosity(tf.logging.ERROR)

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	tf.logging.set_verbosity(old_v)

	batchSz = 100

	U = tf.Variable(tf.random_normal([784, 784], stddev=.1))
	bU = tf.Variable(tf.random_normal([784], stddev=.1))
	V = tf.Variable(tf.random_normal([784, 10], stddev=.1))
	bV = tf.Variable(tf.random_normal([10], stddev=.1))

	img = tf.placeholder(tf.float32, [batchSz, 784])
	ans = tf.placeholder(tf.float32, [batchSz, 10])

	L1Output = tf.matmul(img, U) + bU
	L1Output = tf.nn.relu(L1Output)
	prbs = tf.nn.softmax(tf.matmul(L1Output, V) + bV)
	xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))

	train = tf.train.GradientDescentOptimizer(0.05).minimize(xEnt)
	numCorrect = tf.equal(tf.argmax(prbs, 1), tf.argmax(ans, 1))
	accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

	# saveOb = tf.train.Saver()

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	# ------------------------------------------------------------------------------
	acc = np.zeros(1000)
	for i in range(1000):
		imgs, anss = mnist.train.next_batch(batchSz)
		acc[i], _ = sess.run([accuracy, train], feed_dict={img: imgs, ans: anss})
	# saveOb.save(sess, "chpt")

	sumAcc = 0
	for i in range(1000):
		imgs, anss = mnist.test.next_batch(batchSz)
		sumAcc += sess.run(accuracy, feed_dict={img: imgs, ans: anss})
	print("Test Accuracy: %r" % (sumAcc / 1000))  # > .94 for 1000 iterations, > .97 for 10000 iterations
	plt.plot(acc)
	plt.show()


def fig2_10():
	eo = (((1, 2, 3, 4),
		   (1, 1, 1, 1),
		   (1, 1, 1, 1),
		   (-1, 0, -1, 0)),
		  ((1, 2, 3, 4),
		   (1, 1, 1, 1),
		   (1, 1, 1, 1),
		   (-1, 0, -1, 0)))
	encOut = tf.constant(eo, tf.float32)
	print('encOut:', encOut.shape)
	AT = ((.6, .25, .25),
		  (.2, .25, .25),
		  (.1, .25, .25),
		  (.1, .25, .25))
	wAT = tf.constant(AT, tf.float32)
	print('wAT:', wAT.shape)
	encAT = tf.tensordot(encOut, wAT, [[1], [0]])
	sess = tf.Session()
	print(sess.run(encAT))


def f2():
	old_v = tf.logging.get_verbosity()
	tf.logging.set_verbosity(tf.logging.ERROR)

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	tf.logging.set_verbosity(old_v)

	batchSz = 100

	# W = tf.Variable(tf.random_normal([784, 10], stddev=.1))
	# b = tf.Variable(tf.random_normal([10], stddev=.1))

	img = tf.placeholder(tf.float32, [batchSz, 784])
	ans = tf.placeholder(tf.float32, [batchSz, 10])

	# prbs = tf.nn.softmax(tf.matmul(img, W) + b)
	import tensorflow.contrib.layers as layers
	prbs = layers.fully_connected(img, 10, tf.nn.softmax)
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


if __name__ == '__main__':
	# f0()
	# fig2_1()
	# f1()
	# fig2_2()
	# fig2_9()
	# fig2_10()
	f2()
