# -*- coding: utf-8 -*-
import random
import threading
from datetime import datetime

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

from lib import ImageManager as imgr, pca


def nearestNeighbor_Random():
	N0, N1 = 20, 20
	X0 = np.array([[random.uniform(-10.0, 10.0) for i in range(N0)], \
	               [random.uniform(-10.0, 10.0) for i in range(N0)]])
	X1 = np.array([[random.uniform(-10.0, 10.0) for i in range(N1)], \
	               [random.uniform(-10.0, 10.0) for i in range(N1)]])
	p = np.array([[random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)]]).T
	D0 = np.subtract(X0, p)
	norm0 = map(la.norm, D0.T)
	min0 = min(norm0)
	# minIndex0, minValue0 = min(enumerate(norm0), key=operator.itemgetter(1))
	D1 = np.subtract(X0, p)
	norm1 = map(la.norm, D1.T)
	# minIndex1, minValue1 = min(enumerate(norm1), key=operator.itemgetter(1))
	min1 = min(norm1)
	if min0 < min1:
		print 'class 0'
	else:
		print 'class 1'
	class0, = plt.plot(X0[0, :], X0[1, :], 'o')
	class1, = plt.plot(X1[0, :], X1[1, :], '+')
	test, = plt.plot(p[0], p[1], '*')
	plt.legend([class0, class1, test], ['class 0', 'class 1', 'test'])
	plt.show()


def nearestNeighbor_Digits(digit0, digit1):
	def showColumn(col):
		plt.imshow(col.reshape((height, width)))
		plt.show()

	def readImage(d, n):
		filename = '../data/images/digits/' + str(d) + '/digits' + str(d) + '-' + str(n) + '_reform.png'
		data = mpimg.imread(filename)
		if data.shape[0] % height != 0:
			print 'The height', data.shape[0], 'of', filename, 'is not multiple of', height
			return
		if data.shape[1] % width != 0:
			print 'The width', data.shape[1], 'of', filename, 'is not multiple of', width
			return
		nrow = data.shape[0] / height
		ncol = data.shape[1] / width
		X = np.zeros((D, nrow * ncol))
		N = 0
		for r in range(nrow):
			for c in range(ncol):
				if np.all(data[r * height:(r + 1) * height, c * width:(c + 1) * width] > 0.999):
					break
				X[:, N] = data[r * height:(r + 1) * height, c * width:(c + 1) * width].flatten()
				N += 1
		return X[:, :N]

	def showPcaProjection(X0, X1, legendStr):
		cat = np.concatenate((X0, X1), axis=1)
		Y = pca.decompose(cat)[2]
		print Y.shape
		line1, = plt.plot(Y[0, :X0.shape[1]], Y[1, :X0.shape[1]], '+')
		line2, = plt.plot(Y[0, X0.shape[1]:], Y[1, X0.shape[1]:], '*')
		plt.legend([line1, line2], legendStr)
		plt.show()

	height, width = 56, 40
	D = height * width

	data0 = [readImage(digit0, i) for i in range(1, 10)]
	data1 = [readImage(digit1, i) for i in range(1, 10)]

	# Traingin data
	XT0 = np.concatenate(data0[:8], axis=1).T
	N0 = XT0.shape[0]
	# showColumn(XT0[-1])
	XT1 = np.concatenate(data1[:8], axis=1).T
	N1 = XT1.shape[0]
	# showColumn(XT1[-1])
	print N0, N1

	showPcaProjection(XT0.T, XT1.T, [str(digit0), str(digit1)])

	# ST0 = np.concatenate(data1[8:], axis=1).T
	ST0 = data0[8].T
	# ST1 = np.concatenate(data7[8:], axis=1).T
	ST1 = data1[8].T
	correct0 = 0
	for v in ST0:
		XT_p = np.subtract(XT0, v)
		m0 = min(map(lambda x: np.dot(x, x), XT_p))
		XT_p = np.subtract(XT1, v)
		m1 = min(map(lambda x: np.dot(x, x), XT_p))
		if m0 < m1:
			correct0 += 1
	print str(correct0) + ',', str(ST0.shape[0]), '=', 100.0 * correct0 / ST0.shape[0], '%'
	correct1 = 0
	for v in ST1:
		XT_p = np.subtract(XT0, v)
		m0 = min(map(lambda x: np.dot(x, x), XT_p))
		XT_p = np.subtract(XT1, v)
		m1 = min(map(lambda x: np.dot(x, x), XT_p))
		if m0 > m1:
			correct1 += 1
	print str(correct1) + ',', str(ST1.shape[0]), '=', 100.0 * correct1 / ST1.shape[0], '%'


def nearestNeighbor_Digits_Multithread(digit0, digit1):
	def showColumn(col):
		plt.imshow(col.reshape((height, width)))
		plt.show()

	def readImage(d, n):
		filename = '../data/images/digits/' + str(d) + '/digits' + str(d) + '-' + str(n) + '_reform.png'
		data = mpimg.imread(filename)
		if data.shape[0] % height != 0:
			print 'The height', data.shape[0], 'of', filename, 'is not multiple of', height
			return
		if data.shape[1] % width != 0:
			print 'The width', data.shape[1], 'of', filename, 'is not multiple of', width
			return
		nrow, ncol = data.shape[0] / height, data.shape[1] / width
		X = np.zeros((D, nrow * ncol))
		N = 0
		for r in range(nrow):
			for c in range(ncol):
				if np.all(data[r * height:(r + 1) * height, c * width:(c + 1) * width] > 0.999):
					break
				X[:, N] = data[r * height:(r + 1) * height, c * width:(c + 1) * width].flatten()
				N += 1
		return X[:, :N]

	def showPcaProjection(X0, X1, legendStr):
		cat = np.concatenate((X0, X1), axis=1)
		Y = pca.decompose(cat)[2]
		print Y.shape
		line1, = plt.plot(Y[0, :X0.shape[1]], Y[1, :X0.shape[1]], '+')
		line2, = plt.plot(Y[0, X0.shape[1]:], Y[1, X0.shape[1]:], '*')
		plt.legend([line1, line2], legendStr)
		plt.show()

	height, width = 56, 40
	D = height * width

	data0 = [readImage(digit0, i) for i in range(1, 10)]
	data1 = [readImage(digit1, i) for i in range(1, 10)]

	# Traingin data
	XT0, XT1 = np.concatenate(data0[:8], axis=1).T, np.concatenate(data1[:8], axis=1).T
	N0, N1 = XT0.shape[0], XT1.shape[0]
	# showColumn(XT0[-1])
	# showColumn(XT1[-1])
	print N0, N1
	# showPcaProjection(XT0.T, XT1.T, [str(digit0), str(digit1)])
	# ST0, ST1 = data0[8].T, data1[8].T
	ST0, ST1 = np.concatenate(data0[8:], axis=1).T, np.concatenate(data1[8:], axis=1).T,

	def threadFunc(ST, comp, res):
		correct = 0
		for v in ST:
			XT_p = np.subtract(XT0, v)
			m0 = min(map(lambda x: np.dot(x, x), XT_p))
			XT_p = np.subtract(XT1, v)
			m1 = min(map(lambda x: np.dot(x, x), XT_p))
			if comp(m0, m1):
				correct += 1
		lock.acquire()
		res.append(correct)
		lock.release()

	lock = threading.Lock()
	part0, part1 = ST0.shape[0] // 2, ST1.shape[0] // 2
	res0, res1 = [], []
	comp0, comp1 = lambda x, y: x < y, lambda x, y: x > y
	ths = [threading.Thread(target=threadFunc, args=(ST0[:part0], comp0, res0)), \
	       threading.Thread(target=threadFunc, args=(ST0[part0:], comp0, res0)), \
	       threading.Thread(target=threadFunc, args=(ST1[:part1], comp1, res1)), \
	       threading.Thread(target=threadFunc, args=(ST1[part1:], comp1, res1))]
	for th in ths:
		th.start()
	for th in ths:
		th.join()
	correct0, correct1 = sum(res0), sum(res1)
	print str(correct0) + ',', str(ST0.shape[0]), '->', 100.0 * correct0 / ST0.shape[0], '%'
	print str(correct1) + ',', str(ST1.shape[0]), '->', 100.0 * correct1 / ST1.shape[0], '%'


if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan, linewidth=np.nan)
	# nearestNeighbor_Random()
	start = datetime.now()
	nearestNeighbor_Digits_Multithread(3, 9)
	print datetime.now() - start
