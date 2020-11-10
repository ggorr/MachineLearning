import time

import matplotlib.pyplot as plt

import lib_v1.DataTools as dt
import lib_v1.PlotTools as pt
from lib_v1.NodeGraph import *


def f0():
	from sklearn import datasets
	iris = datasets.load_iris()
	data = (iris.data - np.mean(iris.data, axis=0)) / np.std(iris.data, axis=0)
	trainX, trT, testX, teT = dt.splitTrainTest(data, iris.target)
	# trainT = dt.vectorizeTarget(trT).astype(float)
	# testT = dt.vectorizeTarget(teT).astype(float)
	trainX, testX = data, None
	trainT, testT = dt.vectorizeTarget(iris.target).astype(float), None

	# sset(4) ---> den1(4>3) ---> act1(sigmoid) ---> den2(20>3) ---> act2(softmax) ---> cce
	####### Batch ###############
	# ng = NodeGraphBatch()
	# ng.lr = .01
	# ng.epochMax = 100
	####### Adaptive ###############
	ng = NodeGraphAdaptive()
	ng.lrMin = 1.0e-12
	ng.epochMax = 10000
	####### Common ###############
	sset = ng.add(StartSet(trainX.shape[1]))  # ss0
	den0 = ng.add(Dense(20), sset)  # dn0
	act0 = ng.add(Sigmoid(), den0)  # ac0
	den1 = ng.add(Dense(3), act0)  # dn1
	# act1 = ng.add(Sigmoid(), den1)  # ac1
	# lossFunc = ng.add(Mse(), act1)  # cce
	act1 = ng.add(Softmax(), den1)  # ac1
	lossFunc = ng.add(Cce(), act1)  # cce
	ng.compile()

	start = time.time()
	# ng.fit(trainX, trainT)
	ng.fit(trainX, trainT, testX, testT)
	print('elapsed:', time.time() - start)

	print('=== graph info ===\n' + ng.graphInfo('detail'))
	print('=== summary ===\n' + ng.summary())
	# print(mset.Weight)

	trainY = lossFunc.trY
	plt.subplot(331)
	plt.plot(np.argmax(trainT, axis=1), label='train target')
	plt.plot(np.argmax(trainY, axis=1), label='train output')
	testY = lossFunc.teY
	if testX is not None:
		plt.plot(np.argmax(testT, axis=1), label='test target')
		plt.plot(np.argmax(testY, axis=1), label='test output')
	plt.legend()
	plt.subplot(332)
	plt.title('erros')
	pt.plotLoss(ng)
	plt.subplot(333)
	if isinstance(ng, NodeGraphAdaptive):
		plt.plot(ng.appLrs)
	plt.title('applied learning rate')
	for i in range(3):
		plt.subplot(334 + i)
		plt.plot(trainT[:, i])
		plt.plot(trainY[:, i])
		plt.title(f'train {i}')
		if testX is not None:
			plt.subplot(337 + i)
			plt.plot(testT[:, i])
			plt.plot(testY[:, i])
			plt.title(f'test {i}')
	plt.tight_layout()
	plt.show()


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
f0()
