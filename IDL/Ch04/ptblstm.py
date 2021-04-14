import os
import time

# default 0, filtering INFO 1, filtering WARNING 2, filtering ERROR 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell

from ptbreader import loadWordIdsFromFiles
from ptbutil import save, drawPerplexity, loadInfo, loadSession


def startLstm(epochs=10, saveResult=True):
	trainData, validData, testData, wordId = loadWordIdsFromFiles()
	trainData = np.array(trainData, np.float32)
	# validData = np.array(validData, np.float32)
	testData = np.array(testData, np.float32)
	vocabSz = len(wordId)

	learnRate = 0.001
	embedSz = 128
	rnnSz, batchSz, winSz = 512, 10, 5
	numWin = (trainData.shape[0] - 1) // (batchSz * winSz)
	# each batch has winSz * numWin words
	batchLen = winSz * numWin

	testNumWin = (testData.shape[0] - 1) // (batchSz * winSz)
	testBatchLen = winSz * testNumWin

	inp = tf.placeholder(tf.int32, shape=[batchSz, winSz])
	ans = tf.placeholder(tf.int32, shape=[batchSz * winSz])

	E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))
	embed = tf.nn.embedding_lookup(E, inp)

	rnn = LSTMCell(rnnSz)
	initialState = rnn.zero_state(batchSz, tf.float32)
	output, nextState = tf.nn.dynamic_rnn(rnn, embed, initial_state=initialState)
	output = tf.reshape(output, [batchSz * winSz, rnnSz])

	W = tf.Variable(tf.random_normal([rnnSz, vocabSz], stddev=.1))
	B = tf.Variable(tf.random_normal([vocabSz], stddev=.1))
	logits = tf.matmul(output, W) + B

	ents = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ans)
	loss = tf.reduce_sum(ents)
	train = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

	trainPerp = np.zeros(epochs, dtype=np.float32)
	testPerp = np.zeros(epochs, dtype=np.float32)
	with tf.Session() as sess:
		startTime = time.time()
		sess.run(tf.global_variables_initializer())
		epoch = 0
		print('epoch:', end=' ')
		while epoch < epochs:
			win = 0
			inState = sess.run(initialState)
			testInState = sess.run(initialState)
			# print(inState, testInState)
			winStart, winEnd = 0, winSz
			while win < numWin:
				inInp = np.array([trainData[i * batchLen + winStart:i * batchLen + winEnd] for i in range(batchSz)])
				inAns = np.reshape(np.array([[trainData[i * batchLen + winStart + 1: i * batchLen + winEnd + 1]] for i in range(batchSz)]), batchSz * winSz)
				_, inState, outLoss = sess.run([train, nextState, loss], {inp: inInp, ans: inAns, nextState: inState})
				trainPerp[epoch] += outLoss
				winStart, winEnd = winEnd, winEnd + winSz
				win += 1
				if win < testNumWin:
					inInp = np.array([testData[i * testBatchLen + winStart:i * testBatchLen + winEnd] for i in range(batchSz)])
					inAns = np.reshape(np.array([[testData[i * testBatchLen + winStart + 1: i * testBatchLen + winEnd + 1]] for i in range(batchSz)]), batchSz * winSz)
					testInState, testOutLoss = sess.run([nextState, loss], {inp: inInp, ans: inAns, nextState: testInState})
					testPerp[epoch] += testOutLoss
			epoch += 1
			print(epoch, end=' ')
		trainPerp = np.exp(trainPerp / (trainData.shape[0] // (batchSz * batchLen) * (batchSz * batchLen)))
		testPerp = np.exp(testPerp / (testData.shape[0] // (batchSz * testBatchLen) * (batchSz * testBatchLen)))
		print(f'\nelapsed: {time.time() - startTime}')
		print('train perplexity:', trainPerp[-1])
		print('test perplexity:', testPerp[-1])

		info = {'style': 'lstm', 'batch size': batchSz, 'embed size': embedSz, 'rnn size': rnnSz, 'win size': winSz,
		        'learning rate': learnRate, 'epochs': epochs, 'train perplexity': trainPerp[-1], 'test perplexity': testPerp[-1]}
		if saveResult:
			save(sess, info)
	drawPerplexity(trainPerp, testPerp)


def runMoreLstm(path=None, epochs=10, saveResult=True):
	trainData, validData, testData, wordId = loadWordIdsFromFiles()
	trainData = np.array(trainData, np.float32)
	# validData = np.array(validData, np.float32)
	testData = np.array(testData, np.float32)
	vocabSz = len(wordId)

	info = loadInfo('lstm', path)
	learnRate = info['learning rate']
	batchSz = info['batch size']
	embedSz = info['embed size']
	rnnSz = info['rnn size']
	winSz = info['win size']
	numWin = (trainData.shape[0] - 1) // (batchSz * winSz)
	# each batch has winSz * numWin words
	batchLen = winSz * numWin

	testNumWin = (testData.shape[0] - 1) // (batchSz * winSz)
	testBatchLen = winSz * testNumWin

	inp = tf.placeholder(tf.int32, shape=[batchSz, winSz])
	ans = tf.placeholder(tf.int32, shape=[batchSz * winSz])

	E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))
	embed = tf.nn.embedding_lookup(E, inp)

	rnn = LSTMCell(rnnSz)
	initialState = rnn.zero_state(batchSz, tf.float32)
	output, nextState = tf.nn.dynamic_rnn(rnn, embed, initial_state=initialState)
	output = tf.reshape(output, [batchSz * winSz, rnnSz])

	W = tf.Variable(tf.random_normal([rnnSz, vocabSz], stddev=.1))
	B = tf.Variable(tf.random_normal([vocabSz], stddev=.1))
	logits = tf.matmul(output, W) + B

	ents = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ans)
	loss = tf.reduce_sum(ents)
	train = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

	trainPerp = np.zeros(epochs + 1, dtype=np.float32)
	trainPerp[0] = info['train perplexity']
	testPerp = np.zeros(epochs + 1, dtype=np.float32)
	testPerp[0] = info['test perplexity']
	with tf.Session() as sess:
		loadSession(sess, 'rnn', path)
		startTime = time.time()
		epoch = 0
		print('epoch:', end=' ')
		while epoch < epochs:
			epoch += 1
			win = 0
			inState = sess.run(initialState)
			testInState = sess.run(initialState)
			# print(inState, testInState)
			winStart, winEnd = 0, winSz
			while win < numWin:
				inInp = np.array([trainData[i * batchLen + winStart:i * batchLen + winEnd] for i in range(batchSz)])
				inAns = np.reshape(np.array([[trainData[i * batchLen + winStart + 1: i * batchLen + winEnd + 1]] for i in range(batchSz)]), batchSz * winSz)
				_, inState, outLoss = sess.run([train, nextState, loss], {inp: inInp, ans: inAns, nextState: inState})
				trainPerp[epoch] += outLoss
				winStart, winEnd = winEnd, winEnd + winSz
				win += 1
				if win < testNumWin:
					inInp = np.array([testData[i * testBatchLen + winStart:i * testBatchLen + winEnd] for i in range(batchSz)])
					inAns = np.reshape(np.array([[testData[i * testBatchLen + winStart + 1: i * testBatchLen + winEnd + 1]] for i in range(batchSz)]), batchSz * winSz)
					testInState, testOutLoss = sess.run([nextState, loss], {inp: inInp, ans: inAns, nextState: testInState})
					testPerp[epoch] += testOutLoss
			print(epoch + info['epochs'], end=' ')
		trainPerp[1:] = np.exp(trainPerp[1:] / (trainData.shape[0] // (batchSz * batchLen) * (batchSz * batchLen)))
		testPerp[1:] = np.exp(testPerp[1:] / (testData.shape[0] // (batchSz * testBatchLen) * (batchSz * testBatchLen)))
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
	# startLstm(epochs=1, saveResult=False)
	startLstm(epochs=10)
	# runMoreLstm(epochs=10)
# cosSimTable(['under', 'above', 'the', 'a', 'recalls', 'says', 'rules', 'laws', 'computer', 'machine'], 'rnn')
