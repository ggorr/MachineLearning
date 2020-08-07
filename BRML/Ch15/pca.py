import numpy as np
from numpy import linalg as la


def principalValues(X):
	# eigenvalues of X * X^t or X^t * X
	d, n = X.shape
	mean = np.array([np.sum(X, axis=1)]).T / n
	X_mean = X - mean
	if d <= n:
		S = np.matmul(X_mean, X_mean.T) / n
	else:
		S = np.matmul(X_mean.T, X_mean) / n
	return la.eigh(S)[0]


def project(X, dim):
	d, n = X.shape
	mean = np.array([np.sum(X, axis=1)]).T / n
	X_mean = X - mean
	if d <= n:
		S = np.matmul(X_mean, X_mean.T)
		L, E = la.eigh(S)

		index = sorted(range(d), key=lambda i: L[i], reverse=True)
		B = E[:, index[:dim]]
		Y = np.matmul(B.T, X_mean)

		tildeX = np.matmul(B, Y) + mean
		return tildeX
	else:
		S = np.matmul(X_mean.T, X_mean)
		L, E = la.eigh(S)

		index = sorted(range(n), key=lambda i: L[i], reverse=True)
		B = np.matmul(X, E[:, index[:dim]])
		B = np.matmul(B, np.diag(1 / la.norm(B, axis=0)))
		Y = np.matmul(B.T, X_mean)

		tildeX = np.matmul(B, Y) + mean
		return tildeX


def decompose(X):
	d, n = X.shape
	mean = np.array([np.sum(X, axis=1)]).T / n
	X_mean = X - mean
	if d <= n:
		S = np.matmul(X_mean, X_mean.T)
		L, E = la.eigh(S)

		index = sorted(range(d), key=lambda i: L[i], reverse=True)
		B = E[:, index]
		Y = np.matmul(B.T, X_mean)
		return mean, B, Y
	else:
		S = np.matmul(X_mean.T, X_mean)
		L, E = la.eigh(S)

		index = sorted(range(n), key=lambda i: L[i], reverse=True)
		B = np.matmul(X, E[:, index])
		B = np.matmul(B, np.diag(1 / la.norm(B, axis=0)))
		Y = np.matmul(B.T, X_mean)
		return mean, B, Y


# deprecated, use decompose()
def compute(X):
	d, n = X.shape
	mean = np.sum(X, axis=1) / n
	X_mean = X - np.array(n * [mean]).T
	if d <= n:
		S = np.matmul(X_mean, X_mean.T)
		L, E = la.eigh(S)

		index = sorted(range(d), key=lambda i: L[i], reverse=True)
		B = E[:, index]
		Y = np.matmul(B.T, X_mean)
		return mean, B, Y
	else:
		S = np.matmul(X_mean.T, X_mean)
		L, E = la.eigh(S)

		index = sorted(range(n), key=lambda i: L[i], reverse=True)
		B = np.matmul(X, E[:, index])
		B = np.matmul(B, np.diag(1 / la.norm(B, axis=0)))
		Y = np.matmul(B.T, X_mean)
		return mean, B, Y
