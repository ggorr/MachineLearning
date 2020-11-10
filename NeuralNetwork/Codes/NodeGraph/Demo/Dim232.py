import inspect

import numpy as np
import matplotlib.pyplot as plt

from lib32.NodeGraph import *
import lib32.DataTools as dt


def meshgrid_linear(xmin, xmax, xsize, ymin, ymax, ysize):
	X, Y = np.meshgrid(np.linspace(xmin, xmax, xsize), np.linspace(ymin, ymax, ysize))
	return np.array([X.flatten(), Y.flatten()]).T


def draw_line(B, W, c, linetype, xmin, xmax, ymin, ymax):
	if np.abs(W[0, 0]) < np.abs(W[1, 0]):
		if np.abs(W[0, 0]) < 1e-8:
			u = v = -B[0] / W[1, 0]
			a, b = xmin, xmax
		else:
			uv = -(B[0] + np.array([xmin, xmax]) * W[0, 0]) / W[1, 0]
			u, v = map(lambda t: min(ymax, max(ymin, t)), uv)
			a, b = map(lambda t: -(B[0] + W[1, 0] * t) / W[0, 0], (u, v))
	else:
		if np.abs(W[1, 0]) < 1e-8:
			a = b = -B[0] / W[0, 0]
			u, v = ymin, ymax
		else:
			ab = -(B[0] + np.array([ymin, ymax]) * W[1, 0]) / W[0, 0]
			a, b = map(lambda t: min(xmax, max(xmin, t)), ab)
			u, v = map(lambda t: -(B[0] + t * W[0, 0]) / W[1, 0], (a, b))
	plt.plot([a, b], [u, v], c + linetype)


def draw_weight_lines(weight, xmin=-2, xmax=2, ymin=-2, ymax=2, linetype='-'):
	c = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
	for i in range(weight.shape[0]):
		draw_line(weight[i], c[i if i < len(c) else -1], linetype, xmin, xmax, ymin, ymax)


def display(param, net, levels=3):
	plt.subplot(221)
	for trainX, trainY in zip(param.trainX, param.trainY):
		plt.plot(trainX[0], trainX[1], '.r' if trainY > 0 else '.b')
	plt.gca().set_aspect('equal')
	plt.subplot(222)
	x0, x1 = np.min(param.trainX[:, 0]), np.max(param.trainX[:, 0])
	y0, y1 = np.min(param.trainX[:, 1]), np.max(param.trainX[:, 1])
	X, Y = np.meshgrid(np.linspace(x0, x1, 100), np.linspace(y0, y1, 100))
	Z = np.zeros_like(X)
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			Z[i, j] = net.output(np.array([X[i, j], Y[i, j]]))
	cs = plt.contour(X, Y, Z, levels)
	plt.clabel(cs, inline=1)
	# plt.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)), 'r')
	draw_weight_lines(net.finalWeight[0])
	draw_weight_lines(net.initialWeight[0], linetype='--')
	plt.gca().set_aspect('equal')
	plt.subplot(223)
	plt.title('loss')
	plt.plot(net.loss)
	plt.subplot(224)
	plt.title('learning rate')
	plt.plot(net.appliedRate)
	plt.tight_layout()
	plt.show()


def dim2_quarter_plane_sigmoid():
	print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
	grid = 30
	trainX = meshgrid_linear(-1, 1, grid, -1, 1, grid).astype(np.float32)
	n1, n2 = np.array([1, 0]), np.array([0, 1])
	trainTB = np.logical_and(np.sum(trainX * n1, axis=1) >= 0, np.sum(trainX * n2, axis=1) >= 0)
	trainT = trainTB.astype(int).reshape((-1, 1))
	trainX, trainT = dt.addChannel(trainX, trainT)

	# sset(2) ---> den1(2>2) ---> act1(Sigmoid) ---> den2(2>1) ---> act2(Sigmoid) ---> bce(1)
	ng = NodeGraphAdaptive()
	sset = ng.add(StartSet1D(2, name='sset'))
	den1 = Dense1D(2, name='den1')
	ng.add(den1, sset)
	act1 = ng.add(Sigmoid1D('act1'), den1)
	# act1 = ng.add(Relu('act1'), den1)
	den2 = ng.add(Dense1D(1, name='den2'), act1)
	act2 = ng.add(Sigmoid1D('act2'), den2)
	bce = ng.add(ZeroOneBce1D('bce'), act2)
	den1.initTransf = np.array([[[.3, -.5, 30.2], [-3, 30.2, -.5]]], np.float32)
	den2.initTransf = np.array([[[-28.3, 19.0, 19.0]]], np.float32)
	# # relu
	# den1.initW = np.array([[0.04065937, 0.83962917], [6.2536874, 8.23629256], [-8.99669535, 0.01755958]], np.float32)
	# den2.initW = np.array([[-7.14937919, -10.94093724, 8.40722765]], np.float32).T
	ng.lossMax = np.float32(.00001)
	ng.compile()
	ng.epochMax = 1000
	ng.lrInit = np.float32(0.1)
	ng.fit(trainX, trainT)

	print(f'------------ graph info ---------\n{ng.graphInfo()}')
	print(f'------------ summary ------------\n{ng.summary()}')

	quad = trainX[trainTB]
	comp = trainX[~trainTB]
	plt.subplot(221)
	plt.title('trainX')
	plt.plot(quad[:, 0, 0], quad[:, 1, 0], '+k')
	plt.scatter(comp[:, 0, 0], comp[:, 1, 0], s=1, c='k')
	plt.subplot(222)
	plt.title('activation')
	ng.predict(quad)
	plt.plot(act1.prY[:, 0, 0], act1.prY[:, 1, 0], '+k')
	ng.predict(comp)
	plt.scatter(act1.prY[:, 0, 0], act1.prY[:, 1, 0], s=1, c='k')
	draw_line(den2.B[0], den2.W[:, :, 0], 'r', '-', -1, 1, -1, 1)
	plt.subplot(223)
	x0, x1, y0, y1 = -2, 2, -2, 2
	X, Y = np.meshgrid(np.linspace(x0, x1, 30), np.linspace(y0, y1, 30))
	Z = np.zeros_like(X)
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			p = dt.addChannel(np.array([[X[i, j], Y[i, j]]])).astype(np.float32)
			Z[i, j] = ng.predict(p)[0, 0, 0]
	cs = plt.contour(X, Y, Z, [.5])
	plt.clabel(cs, inline=True)
	plt.subplot(224)
	plt.title('loss')
	plt.plot(ng.trLosses)
	plt.tight_layout()
	plt.show()


dim2_quarter_plane_sigmoid()
