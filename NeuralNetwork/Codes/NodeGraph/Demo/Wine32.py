import time

import matplotlib.pyplot as plt

import lib32.DataTools as dt
import lib32.PlotTools as pt
from lib32.NodeGraph import *
from lib32.NodeSet import F1


def f0():
	from sklearn import datasets
	wine = datasets.load_wine()
	data = (wine.data - np.mean(wine.data, axis=0)) / np.std(wine.data, axis=0)
	###########################################################################################
	trX, trT, teX, teT = dt.splitTrainTest(data, wine.target)
	trainX = trX.astype(np.float32)
	trainT = dt.onehot(trT)
	testX = teX.astype(np.float32)
	testT = dt.onehot(teT)
	trainX, trainT, testX, testT = dt.addChannel(trainX, trainT, testX, testT)
	###########################################################################################
	# trainX, testX = data, None
	# trainT, testT = dt.vectorizeTarget(wine.target), None
	# trainX, trainT = dt.addChannel(trainX, trainT)
	###########################################################################################

	# sset(13) ---> den0(13>20) ---> act0(sigmoid) ---> mp0(20>5) --> den1(5>3) ---> act1(softmax) ---> cce
	####### Batch ###############
	# ng = NodeGraphBatch()
	####### Adaptive ###############
	ng = NodeGraphAdaptiveMinibatch()
	ng.lrMin = np.float32(0.001)
	ng.lrInit = np.float32(.1)
	####### Common ###############
	sset = ng.add(StartSet1D(trainX.shape[1]))  # ss0
	den0 = ng.add(Dense1D(20), sset)  # dn0
	act0 = ng.add(Sigmoid1D(), den0)  # ac0
	mp0 = ng.add(MaxPool1D(4), act0)
	den1 = ng.add(Dense1D(3), mp0)  # dn1
	act1 = ng.add(Softmax1D(), den1)  # ac1
	lossFunc = ng.add(OneHotCce1D(), act1)  # cce
	ng.compile()
	ng.epochMax = 1000

	start = time.time()
	# ng.fit(trainX, trainT)
	ng.fit(trainX, trainT, testX, testT, batchSize=118//5)
	print('elapsed:', time.time() - start)

	print('=== graph info ===\n' + ng.graphInfo('detail'))
	print('=== summary ===\n' + ng.summary())

	trainY = ng.predict(trainX)
	plt.subplot(331)
	plt.title('accuracy')
	pt.plotAccuracy(ng)
	plt.subplot(332)
	plt.title('erros')
	pt.plotLoss(ng)
	plt.subplot(333)
	if isinstance(ng, NodeGraphAdaptive) or isinstance(ng, NodeGraphAdaptiveMinibatch):
		plt.plot(ng.appLrs)
	plt.title('applied learning rate')
	testY = lossFunc.teY
	for i in range(3):
		plt.subplot(334 + i)
		plt.plot(trainT[:, i, 0])
		plt.plot(trainY[:, i, 0])
		plt.title(f'train {i}')
		if testX is not None:
			plt.subplot(337 + i)
			plt.plot(testT[:, i, 0])
			plt.plot(testY[:, i, 0])
			plt.title(f'test {i}')
	plt.tight_layout()
	plt.show()


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
f0()
