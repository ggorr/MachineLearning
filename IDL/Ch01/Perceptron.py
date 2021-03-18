from typing import *

import matplotlib.pyplot as plt
import numpy as np


class Line(object):
	def __init__(self, a: float = 1, b: float = 1, c: float = -1):
		self.a, self.b, self.c = a, b, c

	def eval(self, p: np.ndarray = None):
		if p is None:
			p = np.array((0., 0.))
		res = self.a * p[..., 0] + self.b * p[..., 1] + self.c
		return res

	def y(self, x: Union[float, np.ndarray] = 0):
		return -(self.a * np.array(x) + self.c) / self.b

	def reflect(self, p: np.ndarray):
		c11, c12, c22 = self.a * self.a, self.a * self.b, self.b * self.b
		len, lc, ls = c11 + c22, c11 - c22, 2 * c12
		q = np.empty_like(p)
		q[..., 0] = -(lc * p[..., 0] + ls * p[..., 1] + 2 * self.a * self.c) / len
		q[..., 1] = -(ls * p[..., 0] - lc * p[..., 1] + 2 * self.b * self.c) / len
		return q

	def plot(self, xrange: np.ndarray = None, color:str='b'):
		if xrange is None:
			xrange = np.array((0., 1.))
		plt.plot(xrange, self.y(xrange), color)

	def __str__(self):
		return f'[{self.a}, {self.b}, {self.c}]'


def reflect(data: List, line: Line, positive: bool = True):
	result = np.empty_like(data)
	for src, dst in zip(data, result):
		if positive == (line.eval(src) > 0):
			dst[:] = src[:]
		else:
			dst[:] = line.reflect(src)
	return result


def gen_data(line: Line, rand: np.random.RandomState = None, size0: int = 30, zero_one_ratio: float = 1):
	if rand is None:
		rand = np.random.RandomState(seed=0)
	size1 = int(size0 * zero_one_ratio)
	X, Y = np.empty((size0 + size1, 2)), np.empty(size0 + size1)
	X[:size0, :] = reflect(rand.rand(size0, 2), line, True)
	Y[:size0] = 0
	X[size0:, :] = reflect(rand.rand(size1, 2), line, False)
	Y[size0:] = 1
	ind = np.arange(size0 + size1)
	np.random.shuffle(ind)
	return X[ind], Y[ind]


def separate(X: np.ndarray, Y: np.ndarray, iter: int = 50, lr=1):
	line = Line(0, 0, 0)
	n = 0
	while n < iter:
		f = (line.eval(X) > 0).astype(int)
		if all(f == Y):
			break
		for i in range(Y.shape[0]):
			if Y[i] != f[i]:
				line.a += lr * (Y[i] - f[i]) * X[i, 0]
				line.b += lr * (Y[i] - f[i]) * X[i, 1]
				line.c += lr * (Y[i] - f[i])
		n += 1
	return line


if __name__ == '__main__':
	size0 = 30
	seed = 5
	line = Line(-1, 1, -1)
	rand = np.random.RandomState(seed=seed)
	X, Y = gen_data(line=line, rand=rand, size0=size0)
	print(X)
	print(Y)

	plt.plot(X[Y==0, 0], X[Y==0, 1], 'r.')
	plt.plot(X[Y==1, 0], X[Y==1, 1], 'b.')
	sep = separate(X, Y)
	print(sep)
	sep.plot(xrange=[-1,1], color='g')
	# line.plot(xrange=[-1,1], color='m')
	plt.gca().set_ylim([-.5, 2.5])
	plt.show()
