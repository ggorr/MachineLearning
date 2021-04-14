import os
import time

# default 0, filtering INFO 1, filtering WARNING 2, filtering ERROR 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np

from ptbreader import loadWordIdsFromFiles


def save(sess: tf.Session, info: dict):
	t = time.localtime()
	path = f"result/{info['style']}/{t.tm_year}{t.tm_mon:02d}{t.tm_mday:02d}/{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}"
	print('saving to', path)
	if not os.path.exists(path):
		os.makedirs(path)
	with open(f"result/{info['style']}/recent.txt", 'w') as f:
		f.write(path)
	saver = tf.train.Saver()
	saver.save(sess, path + '/ptb')
	with open(f'{path}/ptb-info.txt', 'w') as f:
		for k, v in info.items():
			f.write(f"{k}: {v}\n")


def loadSession(sess, style, path=None):
	if path is None:
		with open(f'result/{style}/recent.txt', 'r') as f:
			path = f.readline()
	print('loading session from', path)
	saver = tf.train.Saver()
	saver.restore(sess, path + '/ptb')


def loadInfo(style, path=None):
	if path is None:
		with open(f'result/{style}/recent.txt', 'r') as f:
			path = f.readline()
	print('loading info from', path)
	info = {}
	with open(path + '/ptb-info.txt', 'r') as f:
		while True:
			line = f.readline()
			if line == '':
				break
			kv = line.split(': ')
			sv = kv[1].strip()
			try:
				v = int(sv)
			except ValueError:
				try:
					v = float(sv)
				except ValueError:
					v = sv
			info[kv[0].strip()] = v
	return info


def drawPerplexity(trainPerp=None, testPerp=None, offset=1):
	if trainPerp is not None:
		plt.plot(trainPerp)
	if testPerp is not None:
		plt.plot(testPerp)
	plt.gcf().canvas.draw()
	ticks = plt.gca().get_xticks()
	labels = [f"{t + offset:1.2f}" for t in ticks]
	plt.gca().set_xticks(ticks)
	plt.gca().set_xticklabels(labels)
	plt.show()


def cosSimTable(words, style, outE=None, path=None):
	trainData, validData, testData, wordId = loadWordIdsFromFiles()
	if outE is None:
		info = loadInfo(style, path)
		E = tf.Variable(tf.random_normal([len(wordId), info['embed size']], stddev=0.1))
		with tf.Session() as sess:
			loadSession(sess, style, path)
			outE = sess.run(E)
	embeded = np.array([outE[wordId[w]] for w in words])
	norms = np.linalg.norm(embeded, axis=1, keepdims=True)
	sims = np.tensordot(embeded, embeded, axes=(1, 1)) / np.matmul(norms, norms.T)

	fig, ax = plt.subplots(1, 1)
	ax.axis('tight')
	ax.axis('off')
	ax.table(cellText=sims.round(3), colLabels=words, rowLabels=words, loc='center', fontsize=15.0)
	plt.show()
