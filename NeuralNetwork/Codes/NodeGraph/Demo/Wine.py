import time

import matplotlib.pyplot as plt

import lib.DataTools as dt
import lib.PlotTools as pt
from lib.NodeGraph import *


def f0():
	from sklearn import datasets
	wine = datasets.load_wine()
	data = (wine.data - np.mean(wine.data, axis=0)) / np.std(wine.data, axis=0)
	###########################################################################################
	trainX, trT, testX, teT = dt.splitTrainTest(data, wine.target)
	trainT = dt.onehot(trT)
	testT = dt.onehot(teT)
	trainX, trainT, testX, testT = dt.addChannel(trainX, trainT, testX, testT)
	###########################################################################################
	# trainX, testX = data, None
	# trainT, testT = dt.vectorizeTarget(wine.target).astype(float), None
	# trainX, trainT = dt.addChannel(trainX, trainT)
	###########################################################################################

	# sset(13) ---> den0(13>20) ---> act0(sigmoid) ---> mp0(20>4) --> den1(5>3) ---> act1(softmax) ---> cce
	####### Batch ###############
	# ng = NodeGraphBatch()
	####### Adaptive ###############
	ng = NodeGraphAdaptive()
	ng.lrMin = 1.0e-12
	ng.lrInit = 0.1
	####### Common ###############
	sset = ng.add(StartSet1D(trainX.shape[1]))  # ss0
	den0 = ng.add(Dense1D(20), sset)  # dn0
	act0 = ng.add(Sigmoid1D(), den0)  # ac0
	mp0 = ng.add(MaxPool1D(4), act0)
	den1 = ng.add(Dense1D(3), mp0)  # dn1
	act1 = ng.add(Softmax1D(), den1)  # ac1
	lossFunc = ng.add(Cce1D(), act1)  # cce
	ng.compile()

	start = time.time()
	# ng.fit(trainX, trainT)
	ng.fit(trainX, trainT, testX, testT, verbose=2)
	print('elapsed:', time.time() - start)

	print('=== graph info ===\n' + ng.graphInfo('detail'))
	print('=== summary ===\n' + ng.summary())

	trainY = lossFunc.trY
	plt.subplot(331)
	plt.title('accuracy')
	pt.plotAccuracy(ng)
	plt.subplot(332)
	plt.title('erros')
	pt.plotLoss(ng)
	plt.subplot(333)
	if isinstance(ng, NodeGraphAdaptive):
		plt.plot(ng.appLrs)
	plt.title('applied learning rate')
	testY = lossFunc.teY
	for i in range(3):
		plt.subplot(334 + i)
		plt.plot(trainT[0, i, :])
		plt.plot(trainY[0, i, :])
		plt.title(f'train {i}')
		if testX is not None:
			plt.subplot(337 + i)
			plt.plot(testT[0, i, :])
			plt.plot(testY[0, i, :])
			plt.title(f'test {i}')
	plt.tight_layout()
	plt.show()


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
f0()
