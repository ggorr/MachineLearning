# -*- coding: utf-8 -*-
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

from lib import HintonDiagram as hd, pca


def batchTraining_Random():
	N0, N1 = 100, 100
	X0 = np.array([[random.gauss(0.0, 2.0) for i in range(N0)], \
	               [random.gauss(0.0, 2.0) for i in range(N0)]])
	X1 = np.array([[random.gauss(5.0, 2.0) for i in range(N1)], \
	               [random.gauss(6.0, 2.0) for i in range(N1)]])

	sigma = lambda x: 1.0 / (1.0 + np.exp(-x))
	eta = 0.1
	w = np.array([1.0, 0.0])
	b = 0.0
	for iter in range(10000):
		coef = map(lambda t: -eta * sigma(t), np.dot(w, X0) + b)
		wNew = w + np.matmul(X0, coef)
		bNew = b + np.sum(coef)
		coef = map(lambda t: eta * (1 - sigma(t)), np.dot(w, X1) + b)
		wNew += np.matmul(X1, coef)
		bNew += np.sum(coef)
		# print la.norm(w), la.norm(w-wNew)
		if np.abs(b - bNew) < 0.001:
			print 'iteration:', iter
			break
		w = wNew
		b = bNew

	print 'w =', w, ', b =', b
	# hd.show(w.reshape((2, 1)))

	class0, = plt.plot(X0[0, :], X0[1, :], 'o')
	class1, = plt.plot(X1[0, :], X1[1, :], '+')
	plt.legend([class0, class1], ['class 0', 'class 1'])
	if w[0] <= w[1]:
		rangeX = [min(min(X0[0, :]), min(X1[0, :])), max(max(X0[0, :]), max(X1[0, :]))]
		plt.plot(rangeX, map(lambda t: -(b + w[0] * t) / w[1], rangeX))
		plt.plot(rangeX, map(lambda t: -(b + w[0] * t - np.log(9.0)) / w[1], rangeX), '--')
		plt.plot(rangeX, map(lambda t: -(b + w[0] * t + np.log(9.0)) / w[1], rangeX), '--')
	else:
		rangeY = [min(min(X0[1, :]), min(X1[1, :])), max(max(X0[1, :]), max(X1[1, :]))]
		plt.plot(map(lambda t: -(b + w[1] * t) / w[0], rangeY), rangeY)
		plt.plot(map(lambda t: -(b + w[1] * t - np.log(9.0)) / w[0], rangeY), rangeY, '--')
		plt.plot(map(lambda t: -(b + w[1] * t + np.log(9.0)) / w[0], rangeY), rangeY, '--')
	plt.show()


def batchTraining_Digits(digits0, digits1):
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

	data0 = [readImage(digits0, i) for i in range(1, 10)]
	data1 = [readImage(digits1, i) for i in range(1, 10)]

	# Traingin data
	X0, X1 = np.concatenate(data0[:8], axis=1), np.concatenate(data1[:8], axis=1)
	# showColumn(X0[:, -1])
	# showColumn(X1[:, -1])
	# showPcaProjection(X0, X1, [str(digits0), str(digits1)])

	# Train
	print 'Training ...'
	sigma = lambda x: 1.0 / (1.0 + np.exp(-x))
	eta = 0.1
	w = np.zeros(D)
	w[0] = 1.0
	b = 1.0
	for iter in range(1000):
		coef = map(lambda t: -eta * sigma(t), np.dot(w, X0) + b)
		wNew = w + np.matmul(X0, coef)
		bNew = b + np.sum(coef)
		coef = map(lambda t: eta * (1 - sigma(t)), np.dot(w, X1) + b)
		wNew += np.matmul(X1, coef)
		bNew += np.sum(coef)
		if np.abs(b - bNew) < 0.001:
			break
		w = wNew
		b = bNew

	# print '    w =', w, ', b =', b
	print '    iteration:', iter
	print '   ', la.norm(w - wNew), np.abs(b - bNew)
	hd.buildAxes(w.reshape((height, width)))
	plt.title('Hinton diagram')
	plt.show()

	print 'Confirm for training data'
	correctTrain0 = sum(np.matmul(w, X0) + b < 0)
	print '   ', correctTrain0, '/', X0.shape[1], '->', 100.0 * correctTrain0 / X0.shape[1], '%'

	correctTrain1 = sum(np.matmul(w, X1) + b > 0)
	print '   ', correctTrain1, '/', X1.shape[1], '->', 100.0 * correctTrain1 / X1.shape[1], '%'

	# Test data
	T0, T1 = np.concatenate(data0[8:], axis=1), np.concatenate(data1[8:], axis=1)
	# showColumn(T0[:, -1])
	# showColumn(T1[:, -1])

	# Test
	print 'Testing ...'
	correctTest0 = sum(np.matmul(w, T0) + b < 0)
	print '   ', correctTest0, '/', T0.shape[1], '->', 100.0 * correctTest0 / T0.shape[1], '%'

	correctTest1 = sum(np.matmul(w, T1) + b > 0)
	print '   ', correctTest1, '/', T1.shape[1], '->', 100.0 * correctTest1 / T1.shape[1], '%'


if __name__ == '__main__':
	# logging.captureWarnings(False)
	np.set_printoptions(threshold=np.nan, linewidth=np.nan)
	# batchTraining_Random()
	batchTraining_Digits(1, 7)
