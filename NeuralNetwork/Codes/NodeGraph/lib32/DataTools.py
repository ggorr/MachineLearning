import os
from typing import Union

import numpy as np


def addChannel(*x: Union[np.ndarray, list, tuple]) -> Union[np.ndarray, tuple]:
	if len(x) == 1:
		return x[0][..., np.newaxis]
	else:
		return tuple(o[..., np.newaxis] for o in x)


def splitTrainTest(x: np.ndarray, t: np.ndarray = None, trainRate=.7, seed: int = None):
	rand = np.random.RandomState(seed)
	if t is None:
		trainSize = int(x.shape[0] * trainRate + .5)
		perm = rand.permutation(x.shape[0])
		return x[perm[:trainSize]], x[perm[trainSize:]]
	else:
		shapeX, shapeT = list(x.shape), list(t.shape)
		shapeX[0] = shapeT[0] = 0
		trainX = np.empty(shapeX, x.dtype)  # array with length 0
		trainT = np.empty(shapeT, t.dtype)
		testX = np.empty_like(trainX)
		testT = np.empty_like(trainT)  # array with length 0
		cls = np.unique(t, axis=0)
		for c in cls:
			ind = list(i for i, t1 in enumerate(t) if np.array_equal(t1, c))
			trX, teX = splitTrainTest(x[ind], None, trainRate, seed)
			trainX = np.concatenate((trainX, trX), 0)
			testX = np.concatenate((testX, teX), 0)
			trT = np.tile(c, [trX.shape[0]] + [1] * c.ndim)
			trainT = np.concatenate((trainT, trT))
			teT = np.tile(c, [teX.shape[0]] + [1] * c.ndim)
			testT = np.concatenate((testT, teT))
		permTrain = rand.permutation(trainT.shape[0])
		permTest = rand.permutation(testT.shape[0])
		return trainX[permTrain], trainT[permTrain], testX[permTest], testT[permTest]


def onehot(target: Union[np.ndarray, tuple, list], order: Union[np.ndarray, tuple, list] = None):
	targets = (target,) if isinstance(target, np.ndarray) else target
	if order is None:
		order = np.unique(targets[0])
	siz = order.shape[0] if isinstance(order, np.ndarray) else len(order)
	vec = np.zeros(siz, int)
	vec[0] = 1
	m = {order[0]: vec.copy()}
	for i in range(1, siz):
		vec[i - 1] = 0
		vec[i] = 1
		m[order[i]] = vec.copy()
	oh = ()
	for t in targets:
		oht = np.empty((t.shape[0], siz), int)
		for i, v in enumerate(t):
			oht[i] = m[v]
		oh += (oht,)
	return oh if isinstance(target, tuple) else list(oh) if isinstance(target, list) else oh[0]


def read_mnist_train():
	path = os.path.abspath(os.path.dirname('../../../Data/MNIST/'))
	with open(path + '/train-labels.idx1-ubyte', 'rb') as f:
		f.seek(8, os.SEEK_SET)
		data = f.read()
	labels = np.frombuffer(data, np.uint8)

	with open(path + '/train-images.idx3-ubyte', 'rb') as f:
		f.seek(4, os.SEEK_SET)
		b = f.read(4)
		count = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
		b = f.read(4)
		width = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
		b = f.read(4)
		height = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
		data = f.read()
	images = np.frombuffer(data, np.uint8).reshape(count, height, width)
	return labels, images


def read_mnist_test():
	path = os.path.abspath(os.path.dirname('../../../Data/MNIST/'))
	with open(path + '/t10k-labels.idx1-ubyte', 'rb') as f:
		f.seek(8, os.SEEK_SET)
		data = f.read()
	labels = np.frombuffer(data, np.uint8)

	with open(path + '/t10k-images.idx3-ubyte', 'rb') as f:
		f.seek(4, os.SEEK_SET)
		b = f.read(4)
		count = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
		b = f.read(4)
		width = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
		b = f.read(4)
		height = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
		data = f.read()
	images = np.frombuffer(data, np.uint8).reshape(count, height, width)
	return labels, images
