import os

# default 0, filtering INFO 1, filtering WARNING 2, filtering ERROR 3
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from ptbreader import *
import tensorflow.compat.v1 as tf


def f1():
	trainData, validData, testData, wordId = loadWordIdsFromFiles()
	trainData = np.array(trainData, np.float32)
	# validData = np.array(validData, np.float32)
	testData = np.array(testData, np.float32)
	vocabSz = len(wordId)

	batchSz = 1000
	embedSz = 100
	learnRate = 0.005
	# inpt = tf.placeholder(tf.int32, shape=[batchSz])
	# answr = tf.placeholder(tf.int32, shape=[batchSz])
	inpt = tf.placeholder(tf.int32)
	answr = tf.placeholder(tf.int32)
	E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))
	embed = tf.nn.embedding_lookup(E, inpt)

	W = tf.Variable(tf.random_normal([embedSz, vocabSz], stddev=.1))
	b = tf.Variable(tf.random_normal([vocabSz], stddev=.1))

	logits = tf.matmul(embed, W) + b
	xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answr)
	loss = tf.reduce_sum(xEnt)

	train = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
	prbs = tf.nn.softmax(logits)
	numCorrect = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(prbs, 1), tf.int32), answr), tf.float32))
	# accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	epoch = 0
	while epoch < 10:
		for i in range(trainData.shape[0] // batchSz):
			inp = trainData[i * batchSz:(i + 1) * batchSz]
			ans = trainData[i * batchSz + 1:(i + 1) * batchSz + 1]
			sess.run(train, feed_dict={inpt: inp, answr: ans})
		epoch += 1

	# low memory
	# numCorr, trLoss = sess.run([numCorrect, loss], feed_dict={inpt: trainData[:-1], answr: trainData[1:]})
	# print('train accuracy, train perplexity:', numCorr / (trainData.shape[0] - 1), np.exp(trLoss/trainData.shape[0]))
	numCorr = sess.run(numCorrect, feed_dict={inpt: trainData[:-1], answr: trainData[1:]})
	print('train accuracy:', numCorr / (trainData.shape[0] - 1))
	numCorr, teLoss = sess.run([numCorrect, loss], feed_dict={inpt: testData[:-1], answr: testData[1:]})
	print('test accuracy:', numCorr / (testData.shape[0] - 1), '\ntest perplexity:', np.exp(teLoss/testData.shape[0]))


if __name__ == '__main__':
	f1()
