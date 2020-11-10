import matplotlib.pyplot as plt
from .NodeGraph import *
from .DataTools import *


def plotLoss(ng: NodeGraph, zero=False, legend=True):
	if zero:
		plt.plot([0, len(ng.trLosses)], [0, 0], label='zeros')
	plt.plot(ng.trLosses, label='train loss')
	if ng.teLosses is not None:
		plt.plot(ng.teLosses, label='test loss')
	if legend:
		plt.legend()


def plotAccuracy(ng: NodeGraph, one=False, legend=True):
	if one:
		plt.plot([0, len(ng.trAccuracy)], [1, 1], label='ones')
	plt.plot(ng.trAccuracy, label='train accuracy')
	if ng.teAccuracy is not None:
		plt.plot(ng.teAccuracy, label='test accuracy')
	if legend:
		plt.legend()


def contour(ng: NodeGraph, xr: List[float], yr: List[float], title: str = None):
	"""
	plot contour for 2-dimensional input
	:param ng: an instance of Sequence
	:param xr: x range
	:param yr: y range
	:param title: title
	:return:
	"""
	delta = 0.01
	x = np.arange(xr[0], xr[1], delta)
	y = np.arange(yr[0], yr[1], delta)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros_like(X)
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			p = np.array([[X[i, j], Y[i, j]]])
			Z[i, j] = ng.predict(addChannel(p))[0, 0, 0]
	cs = plt.contour(X, Y, Z, 5)
	plt.clabel(cs, inline=1)
	plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
	plt.title(title)


def contourDouble(ng: NodeGraph, pos1: int, pos2: int, xr: Union[List[float], Tuple[float]], yr: Union[List[float], Tuple[float]], title: str = None):
	delta = 0.01
	x = np.arange(xr[0], xr[1], delta)
	y = np.arange(yr[0], yr[1], delta)
	X, Y = np.meshgrid(x, y)
	Z0 = np.zeros_like(X)
	Z1 = np.zeros_like(X)
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if ng.startSets[0].ydim == 1:
				Z = ng.predict([np.array([[[X[i, j]]]]), np.array([[[Y[i, j]]]])])
			else:
				p = np.array([[X[i, j], Y[i, j]]])
				Z = ng.predict(addChannel(p))
			Z0[i, j] = Z[0, 0, 0]
			Z1[i, j] = Z[0, 1, 0]
	plt.subplot(pos1)
	cs = plt.contour(X, Y, Z0, 5)
	plt.clabel(cs, inline=1)
	plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
	plt.title(title)
	plt.subplot(pos2)
	cs = plt.contour(X, Y, Z1, 5)
	plt.clabel(cs, inline=1)
	plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
