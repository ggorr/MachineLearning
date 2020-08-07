import numpy as np

X = np.array([[0., 0., 1., 1.], [0., 1., 0., 1.]])
target = np.array([[0., 1., 1., 0.]])

# input layer
# number of nodes
p0 = 2
# number of samples
n = 4

# hidden layer 1
# number of nodes
p1 = 2
# bias
b1 = np.random.rand(p1, 1)
# weight
K1 = np.random.rand(p1, p0)
# activation
f1 = lambda x: 1 / (1 + np.exp(-x))

# output layer
# number of nodes
p2 = 1
# bias
b2 = np.random.rand(p2, 1)
# weight
K2 = np.random.rand(p2, p1)
# activation
f2 = lambda x: 1 / (1 + np.exp(-x))

lr = 0.1
epoch = 0
while True:
	# forward propagation
	z1 = b1 + np.matmul(K1, X)
	y1 = f1(z1)
	z2 = b2 + np.matmul(K2, y1)
	y2 = f2(z2)
	# error
	error = .5 * np.sum((target - y2) ** 2)
	if error < 0.0001 or epoch >= 10000:
		break

	# backward propagation
	grad_y2 = y2 - target
	grad_z2 = y2 * (1 - y2) * grad_y2
	grad_b2 = np.sum(grad_z2, axis=1, keepdims=True)
	grad_K2 = np.matmul(grad_z2, y1.T)
	grad_y1 = np.matmul(K2.T, grad_z2)
	grad_z1 = y1 * (1 - y1) * grad_y1
	grad_b1 = np.sum(grad_z1, axis=1, keepdims=True)
	grad_K1 = np.matmul(grad_z1, X.T)

	# update
	b1 -= lr * grad_b1
	K1 -= lr * grad_K1
	b2 -= lr * grad_b2
	K2 -= lr * grad_K2
	epoch += 1

print(y2)
