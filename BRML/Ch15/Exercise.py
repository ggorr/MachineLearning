import math
import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import Axes3D


def exer161():
	def circle(n):
		epsilon = 0.1
		x = np.zeros((2, n))
		for i in range(n):
			angle = random.uniform(-math.pi, math.pi)
			mag = random.uniform(1 - epsilon, 1 + epsilon)
			x[:, i] = [mag * math.cos(angle), mag * math.sin(angle)]
		return x

	n = 100
	X = circle(n)

	m = np.sum(X, axis=1) / n
	print "mean: ", m

	X_c = np.zeros((2, n))
	for i in range(n):
		X_c[:, i] = X[:, i] - m

	xxt = np.matmul(X, X.T)
	L, E = la.eigh(xxt)
	print "eigenvalues:", L
	print "eigenvectors:"
	print E

	ax = plt.axes()
	ax.set_aspect('equal')
	ax.grid(True)

	ax.plot(X[0], X[1], '.')
	if L[0] > L[1]:
		ax.plot([m[0], m[0] + E[0, 0]], [m[1], m[1] + E[1, 0]], 'r')
		ax.plot([m[0], m[0] + E[0, 1]], [m[1], m[1] + E[1, 1]], 'b--')
	else:
		ax.plot([m[0], m[0] + E[0, 1]], [m[1], m[1] + E[1, 1]], 'r')
		ax.plot([m[0], m[0] + E[0, 0]], [m[1], m[1] + E[1, 0]], 'b--')

	plt.show()


def exer166():
	data = [1.3, 1.6, 2.8, 4.3, -1.4, 5.8, -0.6, 3.7, 0.7, -0.4, 3.2, 5.8, 3.3, -0.4, 4.3, -0.4, 3.1, 0.9]
	d, n = 3, len(data) / 3
	X = np.array(data).reshape(n, d).T
	print "data: "
	print X

	c = np.sum(X, axis=1) / n
	print "mean: ", c

	X_c = X - np.array(n * [c]).T
	# X_c = np.array([X[:,i]-c for i in range(n)]).T
	# print X_c

	S = np.matmul(X_c, X_c.T) / n
	print "covariance matrix: "
	print S

	L, E = la.eigh(S)
	print "eigenvalues: ", L
	print "eigenvectors: "
	print E

	index = sorted(range(d), key=lambda i: L[i], reverse=True)
	B = E[:, index[:2]]
	print "B: "
	print B

	Y = np.matmul(B.T, X_c)
	print "Y: "
	print Y

	X_tilde = np.matmul(B, Y) + np.array(n * [c]).T
	print "X_tilde: "
	print X_tilde

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_aspect('equal')

	ax.scatter(X[0], X[1], X[2], c='r', marker='o')
	ax.scatter(X_tilde[0], X_tilde[1], X_tilde[2], c='b', marker='^')
	ax.plot(X_tilde[0], X_tilde[1], X_tilde[2], c='b')

	plt.show()


exer161()
exer166()
