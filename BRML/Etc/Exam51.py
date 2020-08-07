import matplotlib.pyplot as plt
import numpy as np


def p(v, h, theta):
	return np.exp(-(v - theta * h) ** 2) / np.sqrt(np.pi)


def plotLogLikelihood(v=2.75):
	x = np.linspace(np.floor(v / 2), np.ceil(v), 41)
	y = map(lambda t: np.log(p(v, 1, t) + p(v, 2, t)), x)
	plt.plot(x, y)
	plt.show()


def em(v=2.75, theta=1.8):
	while True:
		a, b = p(v, 1, theta), p(v, 2, theta)

		q2 = b / (a + b)
		newTheta = v * (2 * q2 + (1 - q2)) / (4 * q2 + (1 - q2))
		print q2, newTheta

		# q1 = a / (a + b)
		# newTheta = v * (2 * (1 - q1) + q1) / (4 * (1 - q1) + q1)
		# print q1, newTheta

		if np.abs(newTheta - theta) < 0.0000001:
			return newTheta
		else:
			theta = newTheta


def main():
	print em()
	plotLogLikelihood()


if __name__ == "__main__":
	main()
