# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la


def polynomialFit(x, y, degree):
	# x: 1 x N matrix, N x 1 matrix, or list
	# y: 1 x N matrix, N x 1 matrix, or list
	# returns coefficients of the polynomial fitting x, y
	# try
	# polynomialFit([-1, 0, 1], [1, 0, 1], 2)
	# which returns coefficients of x^2

	if isinstance(x, np.ndarray):
		x = x.flatten()
	if isinstance(y, np.ndarray):
		y = y.flatten()
	M = np.array([[t ** i for t in x] for i in range(degree + 1)])
	return np.matmul(la.inv(np.matmul(M, M.T)), np.matmul(M, y))


def demoPolynomialFit():
	poly = lambda t: 1 - 10 * t + t ** 2 + t ** 3
	x = np.linspace(-3, 3, 30)
	y = map(poly, x) + 4 * np.random.rand(x.shape[0]) - 2
	w = polynomialFit(x, y, 3)

	print w
	fit = lambda t: w[0] + w[1] * t + w[2] * t ** 2 + w[3] * t ** 3

	plt.plot(x, y, 'o')
	line1, = plt.plot(x, map(poly, x))
	line2, = plt.plot(x, map(fit, x), '--')
	plt.legend([line1, line2], ['generator', 'fit'])
	plt.title('polynomial fit')
	# plt.gca().set_aspect('equal')
	plt.show()


def simpleLPM(X, Y):
	# not regularized ordinary least square training
	# X: D x N matrix or list
	# Y: 1 x N matrix or list
	# returns a vector w such that w^T X = y approximately

	if isinstance(X, list):
		X = np.array(X)
	if isinstance(Y, list):
		Y = np.array(Y)
	else:
		Y = Y.flatten()

	return np.matmul(la.inv(np.matmul(X, X.T)), np.matmul(X, Y))


def demoSimpleLPM():
	# use simpleLPM() instead of polynomialFit()
	poly = lambda t: 1 - 10 * t + t ** 2 + t ** 3
	x = np.linspace(-3, 3, 30)
	X = np.array([[t ** i for t in x] for i in range(4)])
	Y = map(poly, x) + 4 * np.random.rand(x.shape[0]) - 2
	w = simpleLPM(X, Y)

	print w
	fit = lambda t: w[0] + w[1] * t + w[2] * t ** 2 + w[3] * t ** 3

	plt.plot(x, Y, 'o')
	line1, = plt.plot(x, map(poly, x))
	line2, = plt.plot(x, map(fit, x), '--')
	plt.legend([line1, line2], ['generator', 'fit'])
	plt.title('simple LPM')
	# plt.gca().set_aspect('equal')
	plt.show()


if __name__ == '__main__':
	demoPolynomialFit()
	demoSimpleLPM()
