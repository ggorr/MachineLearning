import numpy as np
import matplotlib.pyplot as plt

sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda y: y * (1 - y)
tanh = np.tanh
dtanh = lambda y: 1 - y * y
relu = lambda x: (x > 0) * x
drelu = lambda y: (y > 0).astype(float)


class Layer:
	def __init__(self, xdim, ydim, f, df):
		"""
		:param xdim: input dimension
		:param ydim: output dimension
		:param f: activation function
		:param df: Jacobian of f
		"""
		self.xdim = xdim
		self.ydim = ydim
		self.f = f
		self.df = df
		# bias
		self.B = np.random.rand(ydim, 1)
		# weight
		self.W = np.random.rand(ydim, xdim)
		# input
		self.x = None
		# output
		self.y = None
		# gradient of bias
		self.gradB = np.empty_like(self.B)
		# gradient of weight
		self.gradW = np.empty_like(self.W)

	def forProp(self, x):
		"""forward propagation"""
		self.x = x
		self.y = self.f(self.B + np.matmul(self.W, x))
		return self.y

	def backProp(self, gradY):
		"""
		:param gradY: gradient of loss w.r.t. y
		:return: gradient of loss w.r.t. x
		"""
		gradZ = self.df(self.y) * gradY
		self.gradB = np.sum(gradZ, axis=1, keepdims=True)
		self.gradW = np.matmul(gradZ, self.x.T)
		return np.matmul(self.W.T, gradZ)

	def update(self, lr):
		self.B -= lr * self.gradB
		self.W -= lr * self.gradW


X = np.array([[0., 0., 1., 1.], [0., 1., 0., 1.]])
target = np.array([[0., 1., 1., 0.]])
# layers = [Layer(2, 5, sigmoid, dsigmoid),
# 		  Layer(5, 1, sigmoid, dsigmoid)]
layers = [Layer(2, 2, sigmoid, dsigmoid),
		  Layer(2, 1, sigmoid, dsigmoid)]

lr = 0.1
epoch = 0
while True:
	# forward propagation
	y = X
	for lay in layers:
		y = lay.forProp(y)

	# error
	error = .5 * np.sum((target - y) ** 2)
	if error < 0.0001 or epoch >= 10000:
		break

	# backward propagation
	grad = y - target
	for lay in layers[::-1]:
		grad = lay.backProp(grad)

	# update
	for lay in layers:
		lay.update(lr)
	epoch += 1

print(y)

X1 = np.linspace(-1, 2, 61)
Y1 = np.linspace(-1, 2, 61)
X1, Y1 = np.meshgrid(X1, Y1)
Z1 = np.empty_like(X1)
for x1, y1, z1 in zip(X1, Y1, Z1):
	xy = np.vstack((x1, y1))
	for lay in layers:
		xy = lay.forProp(xy)
	z1[:] = xy
cs = plt.contour(X1, Y1, Z1)
plt.clabel(cs)
plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
plt.show()
