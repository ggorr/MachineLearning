import time
from cProfile import Profile
from pstats import Stats

import numpy as np
import lib32.DataTools as dt
from lib32.NodeGraph import *
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


def f00():
	trX = dt.addChannel(train_images).astype(np.float32)
	trT = dt.addChannel(dt.onehot(train_labels))
	teX = dt.addChannel(test_images).astype(np.float32)
	teT = dt.addChannel(dt.onehot(test_labels))
	# print(trX.shape, trT.shape)
	# return
	# sset(784) ---> flat ---> den1(784>100) ---> act1() ---> den2(100>10) ---> act2 ---> cce
	ng = NodeGraphAdaptive()
	sset = ng.add(StartSet2D((w, h)))
	flat = ng.add(Flat2D(), sset)
	den1 = ng.add(Dense1D(100), flat)
	# act1 = ng.add(Sigmoid1D(), den1)
	act1 = ng.add(Relu1D(), den1)
	den2 = ng.add(Dense1D(10), act1)
	act2 = ng.add(Softmax1D(), den2)
	lossFunc = ng.add(OneHotCce1D(), act2)
	ng.compile()
	ng.lrInit = np.float32(0.00001)
	ng.lrMin = np.float32(1.0e-8)
	ng.epochMax = 50
	start = time.time()
	ng.fit(trX, trT, teX, teT, verbose=2)
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


def f01():
	trX = dt.addChannel(train_images / 255).astype(np.float32)
	trT = dt.addChannel(dt.onehot(train_labels))
	teX = dt.addChannel(test_images / 255).astype(np.float32)
	teT = dt.addChannel(dt.onehot(test_labels))

	# sset(784) ---> flat ---> den1(784>100) ---> act1() ---> den2(100>10) ---> act2 ---> cce
	ng = NodeGraphAdaptiveMinibatch()
	sset = ng.add(StartSet2D((w,h)))
	flat = ng.add(Flat2D(), sset)
	den1 = ng.add(Dense1D(128), flat)
	act1 = ng.add(Relu1D(), den1)
	den2 = ng.add(Dense1D(10), act1)
	act2 = ng.add(Softmax1D(), den2)
	lossFunc = ng.add(OneHotCce1D(), act2)
	ng.compile()
	ng.lrInit = np.float32(0.0001)
	ng.lrMin = np.float32(1.0e-8)
	ng.epochMax = 50
	start = time.time()
	# ng.fit(trX, trT, verbose=2, batchSize=32)
	# teX = teT = None
	ng.fit(trX, trT, teX, teT, verbose=2, batchSize=6000)
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


def f10():
	tr_img, tr_lab = choose(train_images, train_labels, 100)
	te_img, te_lab = choose(test_images, test_labels, 10)
	trX, trT = dt.addChannel(tr_img.astype(np.float32) / 255, dt.onehot(tr_lab))
	teX, teT = dt.addChannel(te_img.astype(np.float32) / 255, dt.onehot(te_lab))

	# print(trX.shape, trT.shape)
	# print(trX.dtype)
	# return
	################################################
	ng = NodeGraphAdaptive()
	ng.lrMin = np.float32(1.0E-6)
	ng.lrInit = np.float32(0.0001)
	################################################
	# ng = NodeGraphBatch()
	################################################
	# sset(28,28) ---> conv((3,3), 32) ---> mp((2,2)) ---> flat ---> den1(100) ---> act1 ---> den2(10) ---> act2 ---> loss
	sset = ng.add(StartSet2D((trX.shape[1], trX.shape[2]), trX.shape[3]))
	conv = ng.add(Conv2D((3, 3), 32), sset)
	act1 = ng.add(Relu2D(), conv)
	mp = ng.add(MaxPool2D((2, 2)), act1)
	flat = ng.add(Flat2D(), mp)
	den1 = ng.add(Dense1D(100), flat)
	act1 = ng.add(Sigmoid1D(), den1)
	den2 = ng.add(Dense1D(10), act1)
	act2 = ng.add(Softmax1D(), den2)
	loss = ng.add(OneHotCce1D(), act2)
	ng.compile()
	ng.epochMax = 100
	start = time.time()
	ng.fit(trX, trT, teX, teT, batchSize=100, verbose=2)
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


def f11():
	tr_img, tr_lab = choose(train_images, train_labels, 100)
	te_img, te_lab = choose(test_images, test_labels, 50)
	trX, trT = dt.addChannel(tr_img.astype(np.float32) / 255, dt.onehot(tr_lab))
	teX, teT = dt.addChannel(te_img.astype(np.float32) / 255, dt.onehot(te_lab))
	# teX = teT = None
	# print(trX.shape, trT.shape)
	# print(trX.dtype)
	# return
	################################################
	ng = NodeGraphAdaptive()
	ng.lrMin = np.float32(1.0E-8)
	ng.lrInit = np.float32(0.0001)
	################################################
	# ng = NodeGraphBatch()
	################################################
	# sset(28,28) ---> conv1((3,3), 32) ---> act1 ---> mp1((2,2)) ---> conv2((3,3), 32) ---> act2 ---> mp2((2,2))
	# 		---> flat ---> den1(100) ---> act1 ---> den2(10) ---> act2 ---> loss
	sset = ng.add(StartSet2D((trX.shape[1], trX.shape[2]), trX.shape[3]))
	conv1 = ng.add(Conv2D((3, 3), 32), sset)
	act1 = ng.add(Relu2D(), conv1)
	mp1 = ng.add(MaxPool2D((2, 2)), act1)
	conv2 = ng.add(Conv2D((3, 3), 32), mp1)
	act2 = ng.add(Relu2D(), conv2)
	mp2 = ng.add(MaxPool2D((2, 2)), act2)
	flat = ng.add(Flat2D(), mp2)
	den1 = ng.add(Dense1D(100), flat)
	act3 = ng.add(Sigmoid1D(), den1)
	den2 = ng.add(Dense1D(10), act3)
	act4 = ng.add(Softmax1D(), den2)
	loss = ng.add(OneHotCce1D(), act4)
	ng.compile()
	ng.epochMax = 50
	start = time.time()
	# ng.fit(trX, trT, teX, teT, batchSize=None, verbose=0)
	ng.fit(trX, trT)
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


def f20():
	# tr_img, tr_lab = choose(train_images, train_labels, 100)
	# te_img, te_lab = choose(test_images, test_labels, 10)
	# trX, trT = dt.addChannel(tr_img / 255.0, dt.vectorizeTarget(tr_lab).astype(float))
	# teX, teT = dt.addChannel(te_img / 255.0, dt.vectorizeTarget(te_lab).astype(float))
	trX, trT = dt.addChannel(train_images / 255.0, dt.onehot(train_labels))
	teX, teT = dt.addChannel(test_images / 255.0, dt.onehot(test_labels))

	# print(trX.shape, trT.shape)
	# print(trX.dtype)
	# return
	################################################
	ng = NodeGraphAdaptiveMinibatch()
	ng.lrMin = np.float32(1.0E-6)
	ng.lrInit = np.float32(0.0001)
	################################################
	# ng = NodeGraphBatch()
	################################################
	# sset(28,28) ---> conv((3,3), 32) ---> act1 ---> mp((2,2)) ---> flat ---> den1(100) ---> act1 ---> den2(10) ---> act2 ---> loss
	sset = ng.add(StartSet2D((w,h)))
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
	loss = ng.add(OneHotCce1D(), act2)
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
	prof.runcall(f10)
	stat = Stats(prof)
	stat.strip_dirs()
	stat.sort_stats('cumulative')
	stat.print_stats()


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# f00()
# f01()
f10()
# f11()
# f20()
# profile()
