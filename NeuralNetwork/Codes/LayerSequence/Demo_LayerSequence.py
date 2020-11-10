import os, sys, time

import matplotlib.pyplot as plt
import numpy as np

# import lib.LayerSequence as ls
import LayerSequence as ls


def contour(xr, yr, seq, title):
	delta = 0.01
	x = np.arange(xr[0], xr[1], delta)
	y = np.arange(yr[0], yr[1], delta)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros_like(X)
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			Z[i, j] = seq.pairCombinedOutput(np.array([X[i, j], Y[i, j]]))
	cs = plt.contour(X, Y, Z, 5)
	plt.clabel(cs, inline=1)
	plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
	plt.title(title)


def contour2(pos1, pos2, xr, yr, seq, title):
	delta = 0.01
	x = np.arange(xr[0], xr[1], delta)
	y = np.arange(yr[0], yr[1], delta)
	X, Y = np.meshgrid(x, y)
	Z0 = np.zeros_like(X)
	Z1 = np.zeros_like(X)
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			Z = seq.pairCombinedOutput(np.array([X[i, j], Y[i, j]]))
			Z0[i, j] = Z[0]
			Z1[i, j] = Z[1]
	plt.subplot(pos1)
	cs = plt.contour(X, Y, Z0, 5)
	plt.clabel(cs, inline=1)
	plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
	plt.title(title)
	plt.subplot(pos2)
	cs = plt.contour(X, Y, Z1, 5)
	plt.clabel(cs, inline=1)
	plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])


def plotLoss(seq):
	if seq.testLoss is None:
		plt.plot(seq.trainLoss)
	else:
		l1, = plt.plot(seq.trainLoss, label='train loss')
		l2, = plt.plot(seq.testLoss, label='test loss')
		plt.legend(handles=[l1, l2])
	plt.plot([0, len(seq.trainLoss)], [0, 0])


def xor():
	print('******************** {} ********************'.format(sys._getframe().f_code.co_name))
	trainX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	trainY = np.array([0, 1, 1, 0])
	testX = np.array([[.5, .5]])
	testY = np.array([0])

	def f0():
		seq = ls.Sequence(ls.LearnType.ADAPTIVE)
		# case 1
		seq.addLayers(nodes=(2, 1))
		# case 2
		# seq.addLayer(weight=np.array([[0, 1, 1], [-1, 1, 1]]), activation=ls.relu)
		# seq.addLayer(weight=np.array([[0, 1, -2]]), activation=ls.identity)
		# case 3
		# seq.addLayer(node=2, activation=ls.relu)
		# seq.addLayer(node=1, activation=ls.identity)
		# case 4
		# w1 = np.random.rand(2, 3) * 2 - 1
		# w1[:, 0] = 0.1
		# seq.addLayer(node=2, weight=w1, activation=ls.relu)
		# w2 = np.random.rand(1, 3) * 2 - 1
		# seq.addLayer(node=1, weight=w2, activation=ls.sigmoid)
		# case 5
		# seq.addLayers(weights=[np.random.rand(2, 3) * 2 - 1, np.zeros((1, 3))])
		seq.setTrainData(trainX, trainY)
		# seq.setTestData(testX, testY)
		seq.epochMax = 10000
		seq.lossType = ls.LossType.MULTI_CROSS_ENTROPY
		seq.learningRate = 1
		# seq.regularization = ls.Lasso(0.1)
		# seq.categorical = False
		seq.build()
		seq.saveFullInfo('results/XorInfo1.txt')
		return seq

	def f1():
		seq = ls.Sequence(ls.LearnType.ADAPTIVE)
		# seq.addLayers(nodes=(2, 1))
		seq.addLayers(weights=[np.array([[0.52521228, 0.26351307, -0.27976042],
		                                 [0.01867158, 0.92281241, -0.93475387]]),
		                       np.array([[0.76627397, -0.78167278, 0.28030265]])])
		seq.setTrainData(trainX, trainY)
		# seq.setTestData(testX, testY)
		seq.epochMax = 20000
		seq.lossType = ls.LossType.MULTI_CROSS_ENTROPY
		seq.learningRate = 1
		# seq.regularization = ls.Lasso(0.1)
		# seq.categorical = False
		seq.build()
		seq.saveFullInfo('results/XorInfo1.txt')
		return seq

	def f2():
		seq = ls.Sequence(path='results/XorInfo1.txt', weight='initial')
		seq.setTrainData(trainX, trainY)
		# seq.setTestData(testX, testY)
		seq.build()
		seq.saveFullInfo('results/XorInfo2.txt')
		return seq

	seq = f0()

	print(seq.summary())
	print('final weights:', seq.getFinalWeights())
	print('output:', seq.output())
	print(seq.getActivationFuncs())

	X = np.meshgrid(np.linspace(-0.5, 1.5, 30), np.linspace(-0.5, 1.5, 30))
	X = np.array([X[0].flatten(), X[1].flatten()]).T
	x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	c = np.array(['r', 'g', 'b', 'c'])
	plt.subplot(231)
	plt.title('input')
	plt.plot(X[:, 0], X[:, 1], '.y', alpha=.3, zorder=1)
	plt.scatter(x[:, 0], x[:, 1], s=20, c=c, zorder=2)
	plt.subplot(232)
	plt.title('before activation')
	N = seq.output(X, 1, False)
	plt.plot(N[:, 0], N[:, 1], '.y', alpha=.3, zorder=1)
	n = seq.output(x, 1, False)
	plt.scatter(n[:, 0], n[:, 1], s=20, c=c, alpha=1, zorder=2)
	plt.subplot(233)
	plt.title('after activation')
	O = seq.output(X, 1)
	plt.plot(O[:, 0], O[:, 1], '.y', alpha=.3, zorder=1)
	o = seq.output(x, 1)
	plt.scatter(o[:, 0], o[:, 1], s=20, c=c, alpha=1, zorder=2)
	xlim = np.array([np.min(O[:, 0]), np.max(O[:, 0])])
	ylim = np.array([np.min(O[:, 1]), np.max(O[:, 1])])
	w = seq.getFinalWeights()[-1][0]
	plt.plot(xlim, -(w[0] + w[1] * xlim) / w[2])
	if np.abs(ylim[0] - ylim[1]) > 0.001:
		plt.gca().set_ylim(ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
	plt.subplot(234)
	contour([-0.5, 1.5], [-0.5, 1.5], seq, '')  # 'xor, nodes = {}'.format(seq.getNodes()))
	plt.subplot(235)
	plt.title('loss')
	plotLoss(seq)
	plt.subplot(236)
	plt.title('learning rates')
	plt.plot(seq.appliedRate)
	plt.show()


def xor2():
	print('******************** {} ********************'.format(sys._getframe().f_code.co_name))
	seq = ls.Sequence()
	trainX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	trainY = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
	seq.addLayers(nodes=(2, 2), weights=[np.random.rand(2, 3), np.zeros((2, 3))])
	seq.lossType = ls.LossType.MONO_CROSS_ENTROPY
	seq.setTrainData(trainX, trainY)
	seq.epochMax = 10000
	seq.build()
	print(seq.getFinalWeights())
	print(seq.summary())

	X = np.meshgrid(np.linspace(-.5, 1.5, 30), np.linspace(-.5, 1.5, 30))
	X = np.array([X[0].ravel(), X[1].ravel()]).T
	x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	c = np.array(['r', 'g', 'b', 'c'])
	plt.subplot(231)
	plt.plot(X[:, 0], X[:, 1], '.y', alpha=.3, zorder=1)
	plt.scatter(x[:, 0], x[:, 1], s=20, c=c, zorder=2)
	plt.subplot(232)
	plt.title('before activation')
	N = seq.output(X, 1, False)
	plt.plot(N[:, 0], N[:, 1], '.y', alpha=.3, zorder=1)
	n = seq.output(x, 1, False)
	plt.scatter(n[:, 0], n[:, 1], s=20, c=c, alpha=1, zorder=2)
	plt.subplot(233)
	plt.title('after activation')
	O = seq.output(X, 1)
	plt.plot(O[:, 0], O[:, 1], '.y', alpha=.3, zorder=1)
	o = seq.output(x, 1)
	plt.scatter(o[:, 0], o[:, 1], s=20, c=c, alpha=1, zorder=2)
	xlim = np.array([np.min(O[:, 0]), np.max(O[:, 0])])
	ylim = np.array([np.min(O[:, 1]), np.max(O[:, 1])])
	w = seq.getFinalWeights()[-1]
	plt.plot(xlim, -(w[0, 0] + w[0, 1] * xlim) / w[0, 2])
	plt.plot(xlim, -(w[1, 0] + w[1, 1] * xlim) / w[1, 2])
	plt.gca().set_ylim(ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))

	contour2(234, 235, [-0.5, 1.5], [-0.5, 1.5], seq, 'xor2, nodes = {}'.format(seq.getNodes()))
	plt.subplot(236)
	plotLoss(seq)
	plt.show()


def iris():
	print('******************** {} ********************'.format(sys._getframe().f_code.co_name))
	from sklearn import datasets
	iris = datasets.load_iris()
	print(iris.data.shape)
	trainX = (iris.data - np.mean(iris.data, axis=0)) / np.std(iris.data, axis=0)
	trainY = np.array([iris.target == 0, iris.target == 1, iris.target == 2], int).T
	seq = ls.Sequence(ls.LearnType.ADAPTIVE)
	seq.addLayers((5, 4, 3))
	seq.setTrainData(trainX, trainY)
	seq.epochMax = 10000
	seq.lossType = ls.LossType.MONO_CROSS_ENTROPY
	seq.learningRate = 0.01
	# seq.momentum = 0.1
	seq.regularization = ls.Ridge(0.1)

	start = time.time()
	seq.build()
	print('elapsed time:', time.time() - start)
	output = seq.output()
	print(seq.summary())
	print('final weights:', seq.getFinalWeights())
	plt.subplot(321)
	p1, = plt.plot(iris.target, label='target')
	p2, = plt.plot(np.argmax(output, axis=1), label='output')
	plt.legend(handles=[p1, p2])
	plt.subplot(322)
	plt.title('class 0')
	plt.plot(output[:, 0])
	plt.subplot(324)
	plt.title('class 1')
	plt.plot(output[:, 1])
	plt.subplot(326)
	plt.title('class 2')
	plt.plot(output[:, 2])
	plt.subplot(323)
	plt.title('erros')
	plotLoss(seq)
	plt.subplot(325)
	plt.plot(seq.appliedRate)
	plt.title('applied learning rate')
	plt.show()


if __name__ == '__main__':
	np.set_printoptions(threshold=np.inf, linewidth=np.inf)
	if not os.path.isdir('results'):
		os.makedirs('results')

	# out1()
	# out2()
	# out4()
	# parity(4)
	# addition()

	xor()
	# xor_split()
	# xor2()
	# xor2_split()

	iris()
	# iris_split()
	# wine()
	# wine_split()
	# breastCancer()
	# breastCancer_split()
	# boston()

	start = time.time()
	print(time.time() - start)
