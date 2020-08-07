import numpy as np
import matplotlib.pyplot as plt


def case1():
	"""Find maximum of 3x^2-6x+2y^2+y using gradient descent"""
	f = lambda x: 3 * x[0] * x[0] - 6 * x[0] + 2 * x[1] * x[1] + x[1]
	grad_f = lambda x: np.array([6 * x[0] - 6, 4 * x[1] + 1])

	lr = 0.1  # learning rate
	p = np.random.rand(2)
	value = np.inf
	epoch = 0
	while True:
		prev_value, value = value, f(p)
		if np.abs(value - prev_value) < 0.0000001 or epoch >= 1000:
			break
		epoch += 1
		p -= lr * grad_f(p)

	print(p)


def case2():
	"""Find regression coefficient using gradient descent"""

	def f(X, y, beta):
		diff = y - np.matmul(X, beta)
		return np.matmul(diff.T, diff)[0, 0]

	def grad_f(X, y, beta):
		# 2(X.T X beta - X.T y)
		return 2 * np.matmul(X.T, np.matmul(X, beta) - y)

	X0 = np.array([[0.], [1.], [2.], [3.], [4.]])
	y = np.array([[0.5], [0.4], [0.7], [0.9], [0.7]])
	# append 1
	X = np.concatenate((np.ones((X0.shape[0], 1)), X0), axis=1)
	# coefficient
	beta = np.random.rand(2, 1)
	lr = 0.01  # learning rate
	value = np.inf
	epoch = 0
	while True:
		prev_value, value = value, f(X, y, beta)
		if np.abs(value - prev_value) < 0.0000001 or epoch >= 1000:
			break
		epoch += 1
		beta -= lr * grad_f(X, y, beta)

	print('beta:', beta.T[0])
	print('beta by matrix computaion:', np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y)).T[0])
	print('epochs:', epoch)


	plt.plot(X[:, 1], y[:, 0])
	xlim = np.min(X[:, 1]), np.max(X[:, 1])
	plt.plot(xlim, [beta[0] + beta[1] * xlim[0], beta[0] + beta[1] * xlim[1]])
	plt.show()


case2()
