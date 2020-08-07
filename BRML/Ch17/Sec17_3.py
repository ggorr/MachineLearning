# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la


def readData():
	lines = open('../data/finance.txt').read().splitlines()
	names = lines[0].split(',')
	n = len(lines) - 1
	dates = n * [0]
	data = n * [[]]
	for i in range(n):
		strs = lines[i + 1].split(',')
		dates[i], data[i] = int(strs[0]), map(float, strs[1:])
	return names[1:], dates, np.array(data).T


def fitKospi(trainSize=200, regularize=True, normalize=True):
	# 코스피 지수를 다른 지수(다우 존스, 국고채(3년), USD, 금)의 일차 결합으로 approximate한다
	names, dates, X = readData()
	kospi = X[0, :]
	kospiTrain = X[0, :trainSize]
	X = X[1:, :]
	XTrain = X[:, :trainSize]

	if normalize or regularize:
		# 데이터를 균일하게 조정하려면 normalize한다
		# regularizing에도 필요하다.
		meanXTrain = np.array([XTrain.mean(axis=1)]).T
		phi = lambda t: t / meanXTrain
	else:
		phi = lambda t: t

	phiXTrain = phi(XTrain)
	K = np.matmul(phiXTrain.T, phiXTrain)  # 17.3.8
	if regularize:
		regularizingFactor = 0.1
	else:
		regularizingFactor = 0.0

	a = np.matmul(la.inv(K + regularizingFactor * np.identity(K.shape[0])), kospiTrain)  # 17.3.14
	w = np.matmul(phiXTrain, a)  # 17.3.6
	print w  # [2201.59087761   195.91790743 - 352.33608043   224.62355474] if regularizingFactor = 0.1

	lineKospi, = plt.plot(kospi, '--')
	fitData = np.matmul(np.matmul(phi(X).T, phiXTrain), a)  # 17.3.15
	lineFit, = plt.plot(fitData, '-')
	plt.legend([lineKospi, lineFit], ['kospi', 'fit'])
	plt.plot([trainSize, trainSize], [2000, 2800])
	if regularize:
		plt.title('regularized')
	plt.show()


if __name__ == '__main__':
	# fitKospi(200, False, False) # not work
	# fitKospi(100, False, False) # not work
	fitKospi(200, True, True)
