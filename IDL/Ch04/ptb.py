import os

# default 0, filtering INFO 1, filtering WARNING 2, filtering ERROR 3
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np

from ptbreader import *


def oneHot(size, targets) -> np.ndarray:
	return np.eye(size)[targets]


def save(sess: tf.Session, info: dict):
	t = time.localtime()
	path = f'result/{t.tm_year}{t.tm_mon:02d}{t.tm_mday:02d}/{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}'
	if not os.path.exists(path):
		os.makedirs(path)
	with open('result/recent.txt', 'w') as f:
		f.write(path)
	saver = tf.train.Saver()
	saver.save(sess, path + '/ptb')
	with open(f'{path}/ptb-info.txt', 'w') as f:
		f.write(f"batch size {info['batch size']}\n")
		f.write(f"embed size {info['embed size']}\n")
		f.write(f"learning rate {info['learning rate']}\n")
		f.write(f"epochs {info['epochs']}\n")
		f.write(f"train accuracy {info['train accuracy']}\n")
		f.write(f"test accuracy {info['test accuracy']}")


def loadSession(sess, path=None):
	if path is None:
		with open('result/recent.txt', 'r') as f:
			path = f.readline()
	print('loading session ' + path)
	saver = tf.train.Saver()
	saver.restore(sess, path + '/ptb')


def loadInfo(path=None):
	if path is None:
		with open('result/recent.txt', 'r') as f:
			path = f.readline()
	print('loading info ' + path)
	info = {}
	with open(path + '/ptb-info.txt', 'r') as f:
		line = f.readline()
		info['batch size'] = int(line[11:])
		line = f.readline()
		info['embed size'] = int(line[11:])
		line = f.readline()
		info['learning rate'] = float(line[14:])
		line = f.readline()
		info['epochs'] = int(line[7:])
		line = f.readline()
		info['train accuracy'] = float(line[15:])
		line = f.readline()
		info['test accuracy'] = float(line[14:])
	return info


def drawAcc(acc, offset=1):
	plt.plot(acc)
	plt.gcf().canvas.draw()
	ticks = plt.gca().get_xticks()
	labels = [f"{t + offset:1.2f}" for t in ticks]
	plt.gca().set_xticks(ticks)
	plt.gca().set_xticklabels(labels)
	plt.show()

def cosSimTable(words, resultE=None, path=None):
	trainData, validData, testData, wordId = loadWordIdsFromFiles()
	if resultE is None:
		info = loadInfo(path)
		E = tf.Variable(tf.random_normal([len(wordId), info['embed size']], stddev=0.1))

		sess = tf.Session()
		loadSession(sess, path)
		resultE = sess.run(E)
	embeded = np.array([resultE[wordId[w]] for w in words])
	norms = np.linalg.norm(embeded, axis=1, keepdims=True)
	sims = np.tensordot(embeded, embeded, axes=(1,1)) / np.matmul(norms, norms.T)

	fig, ax = plt.subplots(1, 1)
	ax.axis('tight')
	ax.axis('off')
	ax.table(cellText=sims.round(3), colLabels=words, rowLabels=words, loc='center', fontsize=15.0)
	plt.show()

def startBigram(epochs=10, saveResult=True):
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
	b = tf.Variable(tf.random_normal([1, vocabSz], stddev=.1))

	logits = tf.matmul(embed, W) + b

	xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answr)
	loss = tf.reduce_sum(xEnt)

	train = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
	prbs = tf.nn.softmax(logits)
	numCorrect = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(prbs, 1), tf.int32), answr), tf.float32))
	# accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	startTime = time.time()
	trainAcc = np.zeros(epochs)
	epoch = 0
	print('epoch:', end=' ')
	while epoch < epochs:
		for i in range(trainData.shape[0] // batchSz):
			inp = trainData[i * batchSz:(i + 1) * batchSz]
			ans = trainData[i * batchSz + 1:(i + 1) * batchSz + 1]

			# sess.run(train, feed_dict={inpt: inp, answr: ans})
			numCorrBatch, _ = sess.run([numCorrect, train], feed_dict={inpt: inp, answr: ans})
			trainAcc[epoch] += numCorrBatch
		trainAcc[epoch] /= trainData.shape[0] // batchSz * batchSz
		epoch += 1
		print(epoch, end=' ')
	print(f'\nelapsed: {time.time() - startTime}')
	print('train accuracy:', trainAcc[-1])

	numCorr = sess.run(numCorrect, feed_dict={inpt: testData[:-1], answr: testData[1:]})
	testAcc = numCorr / (testData.shape[0] - 1)
	print('test accuracy:', testAcc)

	info = {'batch size': batchSz, 'embed size': embedSz, 'learning rate': learnRate,
			'epochs': epochs, 'train accuracy': trainAcc[-1], 'test accuracy': testAcc}
	if saveResult:
		save(sess, info)
	drawAcc(trainAcc)


def runMoreBigram(path=None, epochs=10, saveResult=True):
	trainData, validData, testData, wordId = loadWordIdsFromFiles()
	trainData = np.array(trainData, np.float32)
	# validData = np.array(validData, np.float32)
	testData = np.array(testData, np.float32)
	vocabSz = len(wordId)

	info = loadInfo(path)
	batchSz = info['batch size']
	embedSz = info['embed size']
	# inpt = tf.placeholder(tf.int32, shape=[info['batch size']])
	# answr = tf.placeholder(tf.int32, shape=[info['embed size']])
	inpt = tf.placeholder(tf.int32)
	answr = tf.placeholder(tf.int32)
	E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))
	embed = tf.nn.embedding_lookup(E, inpt)

	W = tf.Variable(tf.random_normal([embedSz, vocabSz], stddev=.1))
	b = tf.Variable(tf.random_normal([vocabSz], stddev=.1))

	logits = tf.matmul(embed, W) + b

	xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answr)
	loss = tf.reduce_sum(xEnt)

	train = tf.train.GradientDescentOptimizer(info['learning rate']).minimize(loss)
	prbs = tf.nn.softmax(logits)
	numCorrect = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(prbs, 1), tf.int32), answr), tf.float32))
	# accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

	sess = tf.Session()
	loadSession(sess, path)

	startTime = time.time()
	trainAcc = np.zeros(epochs + 1)
	trainAcc[0] = info['train accuracy']
	epoch = 0
	print('epoch:', end=' ')
	while epoch < epochs:
		epoch += 1
		for i in range(trainData.shape[0] // batchSz):
			inp = trainData[i * batchSz:(i + 1) * batchSz]
			ans = trainData[i * batchSz + 1:(i + 1) * batchSz + 1]

			# sess.run(train, feed_dict={inpt: inp, answr: ans})
			numCorrBatch, _ = sess.run([numCorrect, train], feed_dict={inpt: inp, answr: ans})
			trainAcc[epoch] += numCorrBatch
		trainAcc[epoch] /= trainData.shape[0] // batchSz * batchSz
		print(epoch + info['epochs'], end=' ')
	print(f'\nelapsed: {time.time() - startTime}')
	print('train accuracy:', trainAcc[-1])

	numCorr = sess.run(numCorrect, feed_dict={inpt: testData[:-1], answr: testData[1:]})
	testAcc = numCorr / (testData.shape[0] - 1)
	print('test accuracy:', testAcc)

	info['epochs'] += epochs
	info['train accuracy'] = trainAcc[-1]
	info['test accuracy'] = testAcc
	if saveResult:
		save(sess, info)
	drawAcc(trainAcc, info['epochs'] - epochs)


if __name__ == '__main__':
	startBigram(epochs=1, saveResult=False)
	# startBigram(epochs=50)
	# runMoreBigram(epochs=50)
	# print(cosSimTable('above', 'under'))
	# cosSimTable(['under', 'above', 'the', 'a', 'recalls', 'says', 'rules', 'laws', 'computer', 'machine'])

