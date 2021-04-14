import os
import time

# default 0, filtering INFO 1, filtering WARNING 2, filtering ERROR 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow.compat.v1 as tf
import numpy as np

from ptbreader import loadWordIdsFromFiles
from ptbutil import save, drawPerplexity, loadInfo, loadSession, cosSimTable


def startBigram(epochs=10, saveResult=True):
	trainData, validData, testData, wordId = loadWordIdsFromFiles()
	trainData = np.array(trainData, np.float32)
	# validData = np.array(validData, np.float32)
	testData = np.array(testData, np.float32)
	vocabSz = len(wordId)

	batchSz = 100
	embedSz = 100
	learnRate = 0.005
	# inp = tf.placeholder(tf.int32, shape=[batchSz])
	# ans = tf.placeholder(tf.int32, shape=[batchSz])
	inp = tf.placeholder(tf.int32)
	ans = tf.placeholder(tf.int32)
	E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))
	embed = tf.nn.embedding_lookup(E, inp)

	dropRate = tf.placeholder(tf.float32)
	embed = tf.nn.dropout(embed, rate=dropRate)

	W = tf.Variable(tf.random_normal([embedSz, vocabSz], stddev=.1))
	B = tf.Variable(tf.random_normal([vocabSz], stddev=.1))
	logits = tf.matmul(embed, W) + B

	# dropRate = tf.placeholder(tf.float32)
	# logits = tf.nn.dropout(logits, rate=dropRate)

	ents = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ans)
	loss = tf.reduce_sum(ents)
	train = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	startTime = time.time()
	trainPerp = np.zeros(epochs, dtype=np.float32)
	testPerp = np.zeros(epochs, dtype=np.float32)
	epoch = 0
	print('epoch:', end=' ')
	while epoch < epochs:
		for i in range(trainData.shape[0] // batchSz):
			inInp = trainData[i * batchSz:(i + 1) * batchSz]
			inAns = trainData[i * batchSz + 1:(i + 1) * batchSz + 1]

			outLoss, _ = sess.run([loss, train], feed_dict={inp: inInp, ans: inAns, dropRate: 0.5})
			trainPerp[epoch] += outLoss
		testPerp[epoch] = sess.run(loss, feed_dict={inp: testData[:-1], ans: testData[1:], dropRate: 0.0})
		epoch += 1
		print(epoch, end=' ')
	trainPerp = np.exp(trainPerp / (trainData.shape[0] // batchSz * batchSz))
	testPerp = np.exp(testPerp / (testData.shape[0] - 1))
	print(f'\nelapsed: {time.time() - startTime}')
	print('train perplexity:', trainPerp[-1])
	print('test perplexity:', testPerp[-1])

	info = {'style': 'bigram', 'batch size': batchSz, 'embed size': embedSz, 'learning rate': learnRate,
	        'epochs': epochs, 'train perplexity': trainPerp[-1], 'test perplexity': testPerp[-1]}
	if saveResult:
		save(sess, info)
	drawPerplexity(trainPerp, testPerp)


def runMoreBigram(path=None, epochs=10, saveResult=True):
	trainData, validData, testData, wordId = loadWordIdsFromFiles()
	trainData = np.array(trainData, np.float32)
	# validData = np.array(validData, np.float32)
	testData = np.array(testData, np.float32)
	vocabSz = len(wordId)

	info = loadInfo('bigram', path)
	batchSz = info['batch size']
	embedSz = info['embed size']
	learnRate = info['learning rate']

	# inp = tf.placeholder(tf.int32, shape=[info['batch size']])
	# ans = tf.placeholder(tf.int32, shape=[info['embed size']])
	inp = tf.placeholder(tf.int32)
	ans = tf.placeholder(tf.int32)
	E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))
	embed = tf.nn.embedding_lookup(E, inp)

	dropRate = tf.placeholder(tf.float32)
	embed = tf.nn.dropout(embed, rate=dropRate)

	W = tf.Variable(tf.random_normal([embedSz, vocabSz], stddev=.1))
	B = tf.Variable(tf.random_normal([vocabSz], stddev=.1))
	logits = tf.matmul(embed, W) + B

	# dropRate = tf.placeholder(tf.float32)
	# logits = tf.nn.dropout(logits, rate=dropRate)

	ents = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ans)
	loss = tf.reduce_sum(ents)
	train = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

	sess = tf.Session()
	loadSession(sess, 'bigram', path)

	startTime = time.time()
	trainPerp = np.zeros(epochs + 1, dtype=np.float32)
	trainPerp[0] = info['train perplexity']
	testPerp = np.zeros(epochs + 1, dtype=np.float32)
	testPerp[0] = info['test perplexity']
	epoch = 0
	print('epoch:', end=' ')
	while epoch < epochs:
		epoch += 1
		for i in range(trainData.shape[0] // batchSz):
			inInp = trainData[i * batchSz:(i + 1) * batchSz]
			inAns = trainData[i * batchSz + 1:(i + 1) * batchSz + 1]

			# sess.run(train, feed_dict={inp: inp0, ans: ans0})
			outLoss, _ = sess.run([loss, train], feed_dict={inp: inInp, ans: inAns, dropRate: 0.5})
			trainPerp[epoch] += outLoss
		testPerp[epoch] = sess.run(loss, feed_dict={inp: testData[:-1], ans: testData[1:], dropRate: 0.0})
		print(epoch + info['epochs'], end=' ')
	trainPerp[1:] = np.exp(trainPerp[1:] / (trainData.shape[0] // batchSz * batchSz))
	testPerp[1:] = np.exp(testPerp[1:] / (testData.shape[0] - 1))
	print(f'\nelapsed: {time.time() - startTime}')
	print('train perplexity:', trainPerp[-1])
	print('test perplexity:', testPerp[-1])

	info['epochs'] += epochs
	info['train perplexity'] = trainPerp[-1]
	info['test perplexity'] = testPerp[-1]
	if saveResult:
		save(sess, info)
	drawPerplexity(trainPerp, testPerp, info['epochs'] - epochs)


if __name__ == '__main__':
	# startBigram(epochs=1, saveResult=False)
	# startBigram(epochs=1)
	# runMoreBigram(epochs=1)
	cosSimTable(['under', 'above', 'the', 'a', 'recalls', 'says', 'rules', 'laws', 'computer', 'machine'], 'bigram')
