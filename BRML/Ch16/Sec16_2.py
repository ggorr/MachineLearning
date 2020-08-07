# -*- coding: utf-8 -*-

import random

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


def ldaDemo_Random():
	N1, N2 = 10, 10
	x1 = [random.gauss(0.0, 3.0) for i in range(N1)]
	y1 = [random.gauss(0.0, 1.0) for i in range(N1)]
	x2 = [random.gauss(3.0, 3.0) for i in range(N2)]
	y2 = [random.gauss(4.0, 1.0) for i in range(N2)]

	# pca
	X = np.array([x1 + x2, y1 + y2])
	m, B, Y = pca.decompose(X)
	pcaX = m[0] + B[0, 0] * Y[0, :]
	pcaY = m[1] + B[1, 0] * Y[0, :]

	# Fisher's lda
	X1, X2 = np.array([x1, y1]), np.array([x2, y2])
	m1, m2 = rowMean(X1), rowMean(X2)
	S1, S2 = np.cov(X1), np.cov(X2)
	B = (N1 / float(N1 + N2)) * S1 + (N2 / float(N1 + N2)) * S2

	# One may use the Moore-Penrose inverse, la.pinv(), if B is not full rank
	w = np.matmul(la.inv(B), m1 - m2)
	w /= la.norm(w)

	coef = np.matmul(w.T, X)
	# w is 2 x 1 matrix
	# coef is 1 x (N1 + N2) matrix
	projX, projY = w[0][0] * coef[0], w[1][0] * coef[0]

	line1, = plt.plot(x1, y1, 'o')
	line2, = plt.plot(x2, y2, '+')
	line3, = plt.plot(pcaX, pcaY, 'x')
	line4, = plt.plot(projX, projY, '.')
	plt.legend([line1, line2, line3, line4], ['class 1', 'class 2', 'pca projection', 'lsa projection'])
	plt.gca().set_aspect('equal')
	plt.show()


def ldaDemo_Image_PseudoInverse():
	imageList = getImageList()
	X3, X5 = buildMatrix(imageList[3]), buildMatrix(imageList[5])
	m3, m5 = rowMean(X3), rowMean(X5)
	S3, S5 = np.cov(X3), np.cov(X5)
	B = S3 + S5

	w = np.matmul(la.pinv(B), m3 - m5)
	w /= la.norm(w)

	coef3, coef5 = np.matmul(w.T, X3)[0], np.matmul(w.T, X5)[0]
	print coef3

	n = X3.shape[1]
	line3, = plt.plot(coef3, n * [0], 'ro')
	line5, = plt.plot(coef5, n * [0], 'g+')
	plt.legend([line3, line5], ['3', '5'])
	plt.show()


def ldaDemo_Image0():
	def display_pca_projection_dim2(X3, X5):
		X = np.concatenate((X3, X5), axis=1)
		m, B, Y = pca.decompose(X)
		line3, = plt.plot(Y[0][:20], Y[1][:20], 'o')
		line5, = plt.plot(Y[0][20:], Y[1][20:], '+')
		plt.legend([line3, line5], ['3', '5'])
		plt.show()

	imageList = getImageList()
	X3, X5 = buildMatrix(imageList[3]), buildMatrix(imageList[5])
	# display_pca_projection_dim2(X3, X5)

	X = np.concatenate((X3, X5), axis=1)
	# X는 2262 x 40 행렬이다.
	# 일반적으로 X가 D x N 행렬이고 D >= N 이면
	# 행의 평균을 빼고 SVD를 수행한다.
	#X -= rowMean(X)
	u, s, v = la.svd(X, full_matrices=False)

	N3, N5 = X3.shape[1], X5.shape[1]
	# M = np.matmul(np.diag(s), v)
	# s에서 0인 성분을 제거한다.
	M = np.matmul(np.diag(s)[:39, :], v)
	M3, M5 = M[:, :N3], M[:, N3:]
	m3, m5 = rowMean(M3), rowMean(M5)
	S3, S5 = np.cov(M3), np.cov(M5)
	B = (N3 / float(N3 + N5)) * S3 + (N5 / float(N3 + N5)) * S5

	w = np.matmul(la.inv(B), m3 - m5)
	w /= la.norm(w)

	coef3, coef5 = np.matmul(w.T, M3)[0], np.matmul(w.T, M5)[0]

	line3, = plt.plot(coef3, N3 * [0], 'o')
	line5, = plt.plot(coef5, N5 * [0], '+')
	plt.legend([line3, line5], ['3', '5'])
	plt.show()


def ldaDemo_Image():
	def display_pca_projection_dim2(X3, X5):
		X = np.concatenate((X3, X5), axis=1)
		m, B, Y = pca.decompose(X)
		line3, = plt.plot(Y[0][:20], Y[1][:20], 'o')
		line5, = plt.plot(Y[0][20:], Y[1][20:], '+')
		plt.legend([line3, line5], ['3', '5'])
		plt.show()

	imageList = getImageList()
	X3, X5 = buildMatrix(imageList[3]), buildMatrix(imageList[5])
	# display_pca_projection_dim2(X3, X5)

	X = np.concatenate((X3, X5), axis=1)
	# X는 2262 x 40 행렬이다.
	# 일반적으로 X가 D x N 행렬이고 D >= N 이면
	# 행의 평균을 빼고 SVD를 수행한다.
	X -= rowMean(X)
	u, s, v = la.svd(X, full_matrices=False)

	N3, N5 = X3.shape[1], X5.shape[1]
	# M = np.matmul(np.diag(s), v)
	# s에서 0인 성분을 제거한다.
	M = np.matmul(np.diag(s)[:39, :], v)
	M3, M5 = M[:, :N3], M[:, N3:]
	m3, m5 = rowMean(M3), rowMean(M5)
	S3, S5 = np.cov(M3), np.cov(M5)
	B = (N3 / float(N3 + N5)) * S3 + (N5 / float(N3 + N5)) * S5

	w = np.matmul(la.inv(B), m3 - m5)
	w /= la.norm(w)

	coef3, coef5 = np.matmul(w.T, M3)[0], np.matmul(w.T, M5)[0]

	line3, = plt.plot(coef3, N3 * [0], 'o')
	line5, = plt.plot(coef5, N5 * [0], '+')
	plt.legend([line3, line5], ['3', '5'])
	plt.show()


if __name__ == '__main__':
	ldaDemo_Random()
	#ldaDemo_Image()
