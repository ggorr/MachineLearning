import numpy as np
import matplotlib.pyplot as plt

# ####################### data generation #######################
def generateData(n=15):
	n = n // 3
	X = np.empty((3 * n, 2))
	X[:n, :] = np.random.randn(n, 2) / 4
	X[n:2 * n, :] = np.random.randn(n, 2) / 4 + np.array([1., 0.])
	X[2 * n:3 * n, :] = np.random.randn(n, 2) / 4 + np.array([0.5, np.math.sqrt(3) / 2])
	return np.round(X, 1)
#
# n = 15
# X = generateData()
# print(np.array2string(X.T, separator=','))
# plt.plot(X[:n // 3, 0], X[:n // 3, 1], 'r.')
# plt.plot(X[n // 3:2 * n // 3, 0], X[n // 3:2 * n // 3, 1], 'g.')
# plt.plot(X[2 * n // 3:n, 0], X[2 * n // 3:n, 1], 'b.')
# plt.show()
####################################################################
n = 15
# generated data
X = np.array([[0., 0.1, -0.2, -0., 0.4, 1.3, 0.6, 1.2, 0.9, 1.1, 0.7, 0.9, 0.4, 0.7, 0.5],
			  [-0.3, -0., 0.2, 0.2, -0.5, -0.3, 0.1, 0.2, -0.1, 0.1, 0.6, 0.6, 0.5, 1.1, 1.1]]).T
# n = 300
# X = generateData(n)
K = 3
label = np.random.randint(0, K, n)
centroid = np.empty([K, 2])
color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
fixed = False
for k in range(1, 10):
	plt.subplot(330 + k)
	for i in range(K):
		centroid[i, :] = np.mean(X[label == i, :], axis=0)
		plt.plot(centroid[i, 0], centroid[i, 1], color[i] + 'x')
		plt.plot(X[label == i, 0], X[label == i, 1], color[i] + '.')
	plt.title(f'step {k}')
	if fixed:
		break
	fixed = True
	for i in range(X.shape[0]):
		minDist = np.inf
		for j in range(K):
			v = X[i] - centroid[j]
			dist = np.sum(v * v)
			if dist < minDist:
				newLabel = j
				minDist = dist
		if label[i] != newLabel:
			fixed = False
			label[i] = newLabel
plt.show()
