# -*- coding: utf-8 -*-
import random

import numpy as np
from matplotlib import pyplot as plt


def linearFit(x, y):
	# x: 1 x N matrix, N x 1 matrix, or list
	# y: 1 x N matrix, N x 1 matrix, or list
	# returns coefficients of the line y = a + b x
	# try
	# linearFit([-1, 0, 1], [1, 0, -1])
	# which returns coefficients of -x

	if isinstance(x, list):
		x = np.array(x)
	else:
		x = x.flatten()
	if isinstance(y, list):
		y = np.array(y)
	else:
		y = y.flatten()
	N = float(x.shape[0])
	Ex = np.sum(x) / N
	Exx = np.dot(x, x) / N
	Ey = np.sum(y) / N
	Exy = np.dot(x, y) / N
	den = Exx - Ex * Ex
	return [(Ey * Exx - Exy * Ex) / den, (Exy - Ex * Exy) / den]


def demoLinearFit():
	func = lambda x: 3 - 2 * x
	x = np.linspace(-3, 3, 30)
	y = map(func, x) + 2 * np.random.rand(x.shape[0]) - 1
	w = linearFit(x, y)
	print w
	fit = lambda x: w[0] + w[1] * x

	plt.plot(x, y, 'o')
	line1, = plt.plot(x, map(func, x))
	line2, = plt.plot(x, map(fit, x), '--')
	plt.legend([line1, line2], ['generator', 'fit'])
	# plt.gca().set_aspect('equal')
	plt.show()


demoLinearFit()
