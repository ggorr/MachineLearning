# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

from lib import pca


def getImageList():
	# returns list of list of numpy.arrays
	# 530 x 297 image from http://myselph.de/neuralNet.html
	imageData = mpimg.imread('../data/digits.png')
	imageList = 10 * [0]
	m = 1
	for i in range(10):
		n = 1
		imageList[i] = 20 * [0]
		for j in range(20):
			imageList[i][j] = imageData[m:m + 29, n:n + 26, :]
			if j % 2 == 0:
				n += 26
			else:
				n += 27
			if j == 8:
				n -= 1
		if i == 0 or i == 1 or i == 4 or i == 8:
			m += 29
		else:
			m += 30
		if i == 5:
			m -= 2
		if i == 6:
			m += 3
		if i == 7:
			m -= 1
	return imageList


def buildPlt(imageList):
	m = len(imageList)
	n = len(imageList[0])
	h, w, c = imageList[0][0].shape
	h1, w1 = h + 1, w + 1
	total = np.zeros((m * h1 + 1, n * w1 + 1, c))
	for k in range(c):
		for j in range(n):
			for i in range(m):
				total[i * h1 + 1:(i + 1) * h1, j * w1 + 1:(j + 1) * w1, k] = imageList[i][j][:, :, k]
	ax = plt.gca()
	ax.axis('off')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(total)


def showImageList(imageList):
	if isinstance(imageList[0], list):
		buildPlt(imageList)
	else:
		buildPlt([imageList])
	plt.show()


def saveImageList(imageList, fname):
	if isinstance(imageList[0], list):
		buildPlt(imageList)
	else:
		buildPlt([imageList])
	plt.savefig(fname)


def buildMatrix(row):
	# build matrix from list of images
	# inverse of buildImageList()
	return np.array([el.flatten() for el in row]).T


def buildImageList(X):
	# build list of images from matrix
	# inverse of buildMatrix()
	n = X.shape[1]
	return [X[:, i].reshape((29, 26, 3)) for i in range(n)]


def fitColorRange(X):
	# 0 <= color data <= 1
	d, n = X.shape
	return np.array([[0 if X[i, j] < 0 else 1 if X[i, j] > 1 else X[i, j] for j in range(n)] for i in range(d)])


def rowMean(M):
	return np.array([np.sum(M, axis=1)]).T / M.shape[1]


def pca_projection():
	imageList = getImageList()
	sublist = imageList[3] + imageList[5] + imageList[7]
	X = buildMatrix(sublist)
	Y = pca.compute(X)[2][:2, :]
	digit3, = plt.plot(Y[0, :20], Y[1, :20], 'o')
	digit5, = plt.plot(Y[0, 20:40], Y[1, 20:40], '+')
	digit7, = plt.plot(Y[0, 40:60], Y[1, 40:60], 'd')
	plt.legend([digit3, digit5, digit7], ['3', '5', '7'])
	plt.title('pca projection')
	plt.gca().set_aspect('equal')
	plt.show()


def canonicalVariates(X, K):
	# returns basis of K dimensional subspace
	# try
	# X1 = np.array([[random.uniform(-1, 1) for j in range(3)] for i in range(10)])
	# X2 = np.array([[random.uniform(-1, 1) for j in range(4)] for i in range(10)])
	# W = canonicalVariates([X1, X2], 2)

	C = len(X)  # 클래스 개수
	D = X[0].shape[0]  # data point의 차원
	N = [x.shape[1] for x in X]  # 각 class의 data point 개수
	mean = [rowMean(x) for x in X]  # 각 클래스의 평균
	XFull = np.concatenate(X, axis=1)  # 데이터 포인트 전체
	meanFull = rowMean(XFull)  # data point 전체 평균
	A = reduce(np.add, [N[i] * np.matmul(mean[i] - meanFull, (mean[i] - meanFull).T) for i in range(C)])
	B = reduce(np.add, [N[i] * np.cov(X[i]) for i in range(C)])

	u, s, v = la.svd(XFull, full_matrices=False)
	dim = sum(N) - C
	# 왜 -C 이어야 하는지 모르겠다. 책대로라면 0, 내 생각엔 -1
	Q = u[:, :dim]
	primeA = np.matmul(Q.T, np.matmul(A, Q))  # A' = Q^T A Q
	primeB = np.matmul(Q.T, np.matmul(B, Q))  # B' = Q^T B Q
	tildeB = la.cholesky(primeB)  # 하삼각행렬
	invTildeB = la.inv(tildeB)
	C = np.matmul(invTildeB.T, np.matmul(primeA, invTildeB))
	L, E = la.eigh(C)
	index = sorted(range(dim), key=lambda i: L[i], reverse=True)
	primeW = E[:, index[:K]]
	# column vectors are orthonomal basis of the projection space
	return np.matmul(Q, primeW)


def DemoCanonicalVariates(digits):
	imageList = getImageList()
	markers = ['o', '*', '+', 'x', '4', 's', 'd', '3', '_', '|']
	C = len(digits)
	X = [buildMatrix(imageList[d]) for d in digits]
	W = canonicalVariates(X, 2)

	lines = C * [0]
	for i in range(C):
		Y = np.matmul(W.T, X[i])
		lines[i], = plt.plot(Y[0, :], Y[1, :], markers[i])
	plt.legend(lines, map(str, digits))
	plt.title('Canonical Variates, C = ' + str(C))
	plt.gca().set_aspect('equal')
	plt.show()


# 직접 구현한 것임
def DemoCanonicalVariatesDirect():
	imageList = getImageList()
	X3, X5, X7 = buildMatrix(imageList[3]), buildMatrix(imageList[5]), buildMatrix(imageList[7])
	N3, N5, N7 = X3.shape[1], X5.shape[1], X7.shape[1]  # 여기서는 모두 20
	mean3, mean5, mean7 = rowMean(X3), rowMean(X5), rowMean(X7)
	XFull = np.concatenate((X3, X5, X7), axis=1)  # 데이터 전체: 2262 x 60 행렬
	meanFull = rowMean(XFull)
	A3 = np.matmul(mean3 - meanFull, (mean3 - meanFull).T)
	A5 = np.matmul(mean5 - meanFull, (mean5 - meanFull).T)
	A7 = np.matmul(mean7 - meanFull, (mean7 - meanFull).T)
	A = N3 * A3 + N5 * A5 + N7 * A7
	B = N3 * np.cov(X3) + N5 * np.cov(X5) + N7 * np.cov(X7)

	u, s, v = la.svd(XFull, full_matrices=False)
	# XFull = u diag(s) v, u는 2262 x 60, s는 대각 원소, v는 60 x 60
	dim = N3 + N5 + N7 - 3
	# 왜 -3 이어야 하는지 모르겠다. 책대로라면 0
	# 내 생각대로라면 -1
	Q = u[:, :dim]
	primeA = np.matmul(Q.T, np.matmul(A, Q))  # A' = Q^T A Q
	primeB = np.matmul(Q.T, np.matmul(B, Q))  # B' = Q^T B Q
	tildeB = la.cholesky(primeB)  # 하삼각행렬
	invTildeB = la.inv(tildeB)
	C = np.matmul(invTildeB.T, np.matmul(primeA, invTildeB))
	L, E = la.eigh(C)
	index = sorted(range(dim), key=lambda i: L[i], reverse=True)
	primeW = E[:, index[:2]]
	W = np.matmul(Q, primeW)

	Y = np.matmul(W.T, XFull)
	line3, = plt.plot(Y[0, :N3], Y[1, :N3], 'o')
	line5, = plt.plot(Y[0, N3:N3 + N5], Y[1, N3:N3 + N5], '+')
	line7, = plt.plot(Y[0, N3 + N5:N3 + N5 + N7], Y[1, N3 + N5:N3 + N5 + N7], 'd')
	plt.legend([line3, line5, line7], ['3', '5', '7'])
	plt.title('Canonical Variates, dimension = ' + str(dim))
	plt.gca().set_aspect('equal')
	plt.show()


def DemoCanonicalVariatesDirect_Another():
	imageList = getImageList()
	X3, X5, X7 = buildMatrix(imageList[3]), buildMatrix(imageList[5]), buildMatrix(imageList[7])
	XFull = np.concatenate((X3, X5, X7), axis=1)
	u, s, v = la.svd(XFull, full_matrices=False)

	N3, N5, N7 = X3.shape[1], X5.shape[1], X7.shape[1]
	dim = 57
	MFull = np.matmul(np.diag(s)[:dim, :], v)
	# In BRML 16.3.1, suggest
	# M = np.matmul(np.diag(s), v),
	# but it does not work
	meanFull = rowMean(MFull)
	M3, M5, M7 = MFull[:, :N3], MFull[:, N3:N3 + N5], MFull[:, N3 + N5:]
	mean3, mean5, mean7 = rowMean(M3), rowMean(M5), rowMean(M7)
	A3 = np.matmul(mean3 - meanFull, (mean3 - meanFull).T)
	A5 = np.matmul(mean5 - meanFull, (mean5 - meanFull).T)
	A7 = np.matmul(mean7 - meanFull, (mean7 - meanFull).T)
	S3, S5, S7 = np.cov(M3), np.cov(M5), np.cov(M7)
	A = N3 * A3 + N5 * A5 + N7 * A7
	B = N3 * S3 + N5 * S5 + N7 * S7
	tildeB = la.cholesky(B)
	invTildeB = la.inv(tildeB)
	C = np.matmul(invTildeB.T, np.matmul(A, invTildeB))
	# C is symmetric, so we use la.eigh()
	L, E = la.eigh(C)
	index = sorted(range(dim), key=lambda i: L[i], reverse=True)
	W = E[:, index[:2]]
	Y3, Y5, Y7 = np.matmul(W.T, M3), np.matmul(W.T, M5), np.matmul(W.T, M7)

	line3, = plt.plot(Y3[0, :], Y3[1, :], 'o')
	line5, = plt.plot(Y5[0, :], Y5[1, :], '+')
	line7, = plt.plot(Y7[0, :], Y7[1, :], 'd')
	plt.legend([line3, line5, line7], ['3', '5', '7'])
	plt.title('Canonical Variates, dimension = ' + str(dim))
	plt.gca().set_aspect('equal')
	plt.show()


if __name__ == '__main__':
	pca_projection()
	DemoCanonicalVariates([3, 5, 7])
	DemoCanonicalVariates([3, 5, 7, 2])
