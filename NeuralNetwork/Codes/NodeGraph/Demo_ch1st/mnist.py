import time
from cProfile import Profile
from pstats import Stats

import numpy as np
import lib_ch1st.DataTools as dt
from lib_ch1st.NodeGraph import *
import matplotlib.pyplot as plt

train_labels, train_images = dt.read_mnist_train()
test_labels, test_images = dt.read_mnist_test()
ntr, nte = train_images.shape[0], test_images.shape[0]
w, h = train_images.shape[1], train_images.shape[2]


# print(images.shape, labels.shape)
def choose(images, labels, n):
	img = np.empty((10 * n, w, h))
	lab = np.empty(10 * n, int)
	count = np.zeros(10, int)
	i = j = 0
	while not np.all(count == n):
		if count[labels[i]] < n:
			count[labels[i]] += 1
			lab[j] = labels[i]
			img[j] = images[i]
			j += 1
		i += 1
	return img, lab


def f0():
	trX = dt.addChannel(train_images.reshape(ntr , w*h)).astype(np.float)
	trT = dt.addChannel(dt.onehot(train_labels)).astype(np.float)
	teX = dt.addChannel(test_images.reshape(nte, w * h)).astype(np.float)
	teT = dt.addChannel(dt.onehot(test_labels)).astype(np.float)
	# print(trX.shape, trT.shape)
	# return
	# sset(784) ---> den1(784>100) ---> act1() ---> den2(100>10) ---> act2 ---> cce
	ng = NodeGraphAdaptive()
	sset = ng.add(StartSet1D(trX.shape[1]))
	den1 = ng.add(Dense1D(100), sset)
	act1 = ng.add(Sigmoid1D(), den1)
	den2 = ng.add(Dense1D(10), act1)
	act2 = ng.add(Softmax1D(), den2)
	lossFunc = ng.add(Cce1D(), act2)
	ng.compile()
	ng.lrInit = 0.00001
	ng.epochMax = 50
	start = time.time()
	ng.fit(trX, trT, teX, teT, verbose=0)
	print(f'elapsed: {time.time() - start}')

	print(f'=== graph info ===\n{ng.graphInfo("detail")}')
	print(f'=== summary ===\n{ng.summary()}')

	plt.subplot(221)
	plt.plot(ng.trAccuracy, label='train accuracy')
	if teX is not None:
		plt.plot(ng.teAccuracy, label='test accuracy')
	plt.legend()
	plt.subplot(223)
	plt.plot(ng.trLosses, label='train losses')
	if teX is not None:
		plt.plot(ng.teLosses, label='test losses')
	plt.legend()
	plt.subplot(224)
	plt.plot(ng.appLrs, label='learning rates')
	plt.legend()
	plt.tight_layout()
	plt.show()


def f1():
	tr_img, tr_lab = choose(train_images, train_labels, 100)
	te_img, te_lab = choose(test_images, test_labels, 10)
	trX, trT = dt.addChannel(tr_img / 255, dt.onehot(tr_lab).astype(float))
	teX, teT = dt.addChannel(te_img / 255, dt.onehot(te_lab).astype(float))

	# print(trX.shape, trT.shape)
	# print(trX.dtype)
	# return
	################################################
	ng = NodeGraphAdaptiveMinibatch()
	ng.lrMin = 1.0E-6
	ng.lrInit = 0.0001
	################################################
	# ng = NodeGraphBatch()
	################################################
	# sset(28,28) ---> conv((3,3), 32) ---> mp((2,2)) ---> flat ---> den1(100) ---> act1 ---> den2(10) ---> act2 ---> loss
	sset = ng.add(StartSet2D((trX.shape[1], trX.shape[2]), trX.shape[0]))
	conv = ng.add(Conv2D((3, 3), 32), sset)
	act1 = ng.add(Relu2D(), conv)
	mp = ng.add(MaxPool2D((2, 2)), act1)
	flat = ng.add(Flat2D(), mp)
	den1 = ng.add(Dense1D(100), flat)
	act1 = ng.add(Sigmoid1D(), den1)
	den2 = ng.add(Dense1D(10), act1)
	act2 = ng.add(Softmax1D(), den2)
	loss = ng.add(Cce1D(), act2)
	ng.compile()
	ng.epochMax = 100
	start = time.time()
	ng.fit(trX, trT, teX, teT, batchSize=100, verbose=0)
	print(f'elapsed: {time.time() - start}')

	print(f'=== graph info ===\n{ng.graphInfo("detail")}')
	print(f'=== summary ===\n{ng.summary()}')

	plt.subplot(221)
	plt.plot(ng.trAccuracy, label='train accuracy')
	if teX is not None:
		plt.plot(ng.teAccuracy, label='test accuracy')
	plt.legend()
	plt.subplot(222)
	plt.plot(ng.trLosses, label='train losses')
	if teX is not None:
		plt.plot(ng.teLosses, label='test losses')
	plt.legend()
	plt.subplot(223)
	plt.plot(ng.appLrs, label='learning rates')
	plt.legend()
	plt.tight_layout()
	plt.show()


def f2():
	# tr_img, tr_lab = choose(train_images, train_labels, 100)
	# te_img, te_lab = choose(test_images, test_labels, 10)
	# trX, trT = dt.addChannel(tr_img / 255.0, dt.vectorizeTarget(tr_lab).astype(float))
	# teX, teT = dt.addChannel(te_img / 255.0, dt.vectorizeTarget(te_lab).astype(float))
	trX, trT = dt.addChannel(train_images / 255.0, dt.onehot(train_labels).astype(float))
	teX, teT = dt.addChannel(test_images / 255.0, dt.onehot(test_labels).astype(float))

	# print(trX.shape, trT.shape)
	# print(trX.dtype)
	# return
	################################################
	ng = NodeGraphAdaptiveMinibatch()
	ng.lrMin = 1.0E-6
	ng.lrInit = 0.0001
	################################################
	# ng = NodeGraphBatch()
	################################################
	# sset(28,28) ---> conv((3,3), 32) ---> act1 ---> mp((2,2)) ---> flat ---> den1(100) ---> act1 ---> den2(10) ---> act2 ---> loss
	sset = ng.add(StartSet2D((trX.shape[1], trX.shape[2]), trX.shape[0]))
	conv1 = ng.add(Conv2D((3, 3), 32), sset)
	act1 = ng.add(Relu2D(), conv1)
	mp1 = ng.add(MaxPool2D((2, 2)), act1)
	conv2 = ng.add(Conv2D((3, 3), 32), mp1)
	act2 = ng.add(Relu2D(), conv2)
	mp2 = ng.add(MaxPool2D((2, 2)), act2)
	flat = ng.add(Flat2D(), mp2)
	den1 = ng.add(Dense1D(100), flat)
	act1 = ng.add(Sigmoid1D(), den1)
	den2 = ng.add(Dense1D(10), act1)
	act2 = ng.add(Softmax1D(), den2)
	loss = ng.add(Cce1D(), act2)
	ng.compile()
	ng.epochMax = 10
	start = time.time()
	ng.fit(trX, trT, teX, teT, batchSize=10000, verbose=2)
	print(f'elapsed: {time.time() - start}')

	print(f'=== graph info ===\n{ng.graphInfo("detail")}')
	print(f'=== summary ===\n{ng.summary()}')

	plt.subplot(221)
	plt.plot(ng.trAccuracy, label='train accuracy')
	if teX is not None:
		plt.plot(ng.teAccuracy, label='test accuracy')
	plt.legend()
	plt.subplot(222)
	plt.plot(ng.trLosses, label='train losses')
	if teX is not None:
		plt.plot(ng.teLosses, label='test losses')
	plt.legend()
	plt.subplot(223)
	plt.plot(ng.appLrs, label='learning rates')
	plt.legend()
	plt.tight_layout()
	plt.show()


def profile():
	prof = Profile()
	prof.runcall(f1)
	stat = Stats(prof)
	stat.strip_dirs()
	stat.sort_stats('cumulative')
	stat.print_stats()


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# f0()
f1()
# f2()
# profile()
