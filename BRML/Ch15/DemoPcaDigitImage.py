import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from lib import pca


def splitx(img):
	fig = plt.figure()
	m = 0
	for i in range(10):
		rect = 0, i / 10., 1, 0.1
		ax = fig.add_axes(rect)
		ax.set_xticks([])
		im1 = img[m:m + 29, :, :]
		print m,
		ax.imshow(im1)
		if i == 0 or i == 4:
			m += 29
		else:
			m += 30
		if i == 5:
			m -= 2
		if i == 6:
			m += 3
		if i == 7:
			m -= 1

	plt.show()


def splity(img):
	fig = plt.figure()
	n = 0
	for j in range(20):
		rect = j / 20., 0, 0.05, 1
		ax = fig.add_axes(rect)
		ax.set_xticks([])
		ax.set_yticks([])
		im1 = img[:, n:n + 26, :]
		ax.imshow(im1)
		if j % 2 == 0:
			n += 26
		else:
			n += 27

	plt.show()


def split(img):
	imlist = 10 * [0]
	m = 1
	for i in range(10):
		n = 1
		imlist[i] = 20 * [0]
		for j in range(20):
			imlist[i][j] = img[m:m + 29, n:n + 26, :]
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
	return imlist


def display1_multiaxes(row):
	n = len(row)
	fig = plt.figure()
	step = 1.0 / n
	for i in range(n):
		rect = i * step, 0, step, step
		ax = fig.add_axes(rect)
		# ax.axis('off')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.imshow(row[i])
	plt.show()


def display1(row):
	n = len(row)
	h, w, c = row[0].shape
	total = np.zeros((h + 2, n * (w + 1) + 1, c))
	for k in range(c):
		for j in range(n):
			total[1:h + 1, j * (w + 1) + 1:(j + 1) * (w + 1), k] = row[j][:, :, k]
	ax = plt.gca()
	ax.axis('off')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(total)
	plt.show()


def display2(imlist, fname=None):
	m = len(imlist)
	n = len(imlist[0])
	h, w, c = imlist[0][0].shape
	total = np.zeros((m * (h + 1) + 1, n * (w + 1) + 1, c))
	for k in range(c):
		for j in range(n):
			for i in range(m):
				total[i * (h + 1) + 1:(i + 1) * (h + 1), j * (w + 1) + 1:(j + 1) * (w + 1), k] = imlist[i][j][:, :, k]
	ax = plt.gca()
	ax.axis('off')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(total)
	if fname == None:
		plt.show()
	else:
		plt.savefig(fname)


def sol1():
	def buildMatrix(row):
		X0 = np.array([row[i][:, :, 0].flatten() for i in range(len(row))]).T
		X1 = np.array([row[i][:, :, 1].flatten() for i in range(len(row))]).T
		X2 = np.array([row[i][:, :, 2].flatten() for i in range(len(row))]).T
		return X0, X1, X2

	def buildImage(X0, X1, X2):
		row = 20 * [0]
		for i in range(20):
			row[i] = np.zeros((29, 26, 3))
			for j in range(29):
				row[i][j, :, 0] = X0[26 * j:26 * (j + 1), i]
				row[i][j, :, 1] = X1[26 * j:26 * (j + 1), i]
				row[i][j, :, 2] = X2[26 * j:26 * (j + 1), i]
		return row

	def truncate(X):
		d, n = X.shape
		return np.array([[0 if X[i, j] < 0 else 1 if X[i, j] > 1 else X[i, j] for j in range(n)] for i in range(d)])

	def displayEig(X0, X1, X2):
		L0 = sorted(pca.principalValues(X0), reverse=True)
		L1 = sorted(pca.principalValues(X1), reverse=True)
		L2 = sorted(pca.principalValues(X2), reverse=True)
		ax = plt.gca()
		ax.plot(range(1, 21), L0, 'r^', range(1, 21), L1, 'gx', range(1, 21), L2, 'b.')
		ax.set_xticks(range(21))
		plt.show()

	def displayReduced(row):
		# display1(row)
		X0, X1, X2 = buildMatrix(row)
		displayEig(X0, X1, X2)

		M0, B0, Y0 = pca.compute(X0)
		M1, B1, Y1 = pca.compute(X2)
		M2, B2, Y2 = pca.compute(X2)

		A = 5 * [0]
		A[0] = row
		for i in range(1, 5):
			dim = 17 - 4 * i
			X0_tilde = truncate(np.matmul(B0[:, :dim], Y0[:dim, :]) + np.array(20 * [M0]).T)
			# X0_tilde = truncate(pca.project(X0, dim))
			X1_tilde = truncate(np.matmul(B1[:, :dim], Y1[:dim, :]) + np.array(20 * [M1]).T)
			X2_tilde = truncate(np.matmul(B2[:, :dim], Y2[:dim, :]) + np.array(20 * [M2]).T)
			A[i] = buildImage(X0_tilde, X1_tilde, X2_tilde)
		display2(A)

	img = mpimg.imread('data/digits.png')
	imlist = split(img)
	# display2(imlist)
	for i in range(10):
		displayReduced(imlist[i])


def sol2():
	def buildMatrix(row):
		return np.array([row[i].flatten() for i in range(len(row))]).T

	def buildImage(X):
		n = X.shape[1]
		return [X[:, i].reshape((29, 26, 3)) for i in range(n)]

	def truncate(X):
		d, n = X.shape
		return np.array([[0 if X[i, j] < 0 else 1 if X[i, j] > 1 else X[i, j] for j in range(n)] for i in range(d)])

	def displayEig(X):
		L = sorted(pca.principalValues(X), reverse=True)
		ax = plt.gca()
		ax.plot(range(1, len(L) + 1), L)
		ax.set_xticks(range(len(L) + 1))
		plt.show()

	def displayReduced(row, fname=None):
		# display1(row)
		X = buildMatrix(row)
		displayEig(X)
		M, B, Y = pca.compute(X)

		A = 5 * [0]
		A[0] = row
		for i in range(1, 5):
			dim = 17 - 4 * i
			X_tilde = truncate(np.matmul(B[:, :dim], Y[:dim, :]) + np.array(len(row) * [M]).T)
			A[i] = buildImage(X_tilde)
		display2(A, fname)

	# 530 x 297 image from http://myselph.de/neuralNet.html
	img = mpimg.imread('../data/digits.png')
	imlist = split(img)
	# display2(imlist)
	# displayReduced(imlist[0])

	# for i in range(10):
	# displayReduced(imlist[i])
	# displayReduced(imlist[i], 'demo/output/figure/fig0' + str(i))

	sublist = [imlist[i][j] for i in range(10) for j in range(3)]
	# displayReduced(sublist, 'demo/output/figure/fig30')
	displayReduced(sublist)


# sol1()

sol2()
