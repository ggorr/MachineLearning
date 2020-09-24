import sys

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
	seq = ls.Sequence(ls.LearnType.ADAPTIVE)

	############# random weights #############################################
	def f0():
		seq.addLayers(nodes=(2, 1))

	##########################################################################
	############# deep learning book##########################################
	def f1():
		seq.addLayer(weight=np.array([[0, 1, 1], [-1, 1, 1]]), activation=ls.relu)
		seq.addLayer(weight=np.array([[0, 1, -2]]), activation=ls.identity)

	##########################################################################
	############# random weights #############################################
	def f2():
		seq.addLayers(weights=[np.array([[0.52521228, 0.26351307, -0.27976042],
		                                 [0.01867158, 0.92281241, -0.93475387]]),
		                       np.array([[0.76627397, -0.78167278, 0.28030265]])])
	##########################################################################

	f0()

	seq.setTrainData(trainX, trainY)
	seq.epochMax = 10000
	seq.lossType = ls.LossType.L2
	seq.learningRate = 1
	seq.build()

	print(seq.summary())
	print(seq.getFinalWeights())
	print(seq.output())

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
	contour([-0.5, 1.5], [-0.5, 1.5], seq, 'xor, nodes = {}'.format(seq.getNodes()))
	plt.subplot(235)
	plt.title('loss')
	plotLoss(seq)
	plt.subplot(236)
	plt.title('learning rates')
	plt.plot(seq.appliedRate)
	plt.show()


if __name__ == '__main__':
	np.set_printoptions(threshold=np.inf, linewidth=np.inf)
	xor()
