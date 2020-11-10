import time
import warnings

import matplotlib.pyplot as plt

import lib32.DataTools as dt
import lib32.PlotTools as pt
from lib32.NodeGraph import *

from sklearn import datasets

iris = datasets.load_iris()
data = (iris.data - np.mean(iris.data, axis=0)) / np.std(iris.data, axis=0)
##############################################################################
trX, trT, teX, teT = dt.splitTrainTest(data, iris.target, .7, 1)
trainX, testX = trX.astype(np.float32), teX.astype(np.float32)
trainT, testT = dt.onehot((trT, teT))
trainX, trainT, testX, testT = dt.addChannel(trainX, trainT, testX, testT)
##############################################################################
# trainX, testX = data.astype(np.float32), None
# trainT, testT = dt.onehot(iris.target, np.float32), None
# trainX, trainT = dt.addChannel(trainX, trainT)
##############################################################################
# print(trainX.dtype, trainT.dtype, testX.dtype, testT.dtype)


def f0():
	# sset(4) ---> den1(4>4) ---> act1(sigmoid) ---> den2(4>3) ---> act2(softmax) ---> cce
	####### Batch ###############
	# ng = NodeGraphBatch()
	# ng.lr = np.float32(.01)
	# ng.epochMax = 100
	####### Adaptive ###############
	ng = NodeGraphAdaptive()
	# ng.lrMin = np.float32(1.0e-2)
	ng.epochMax = 1000
	ng.lrInit = 0.01
	####### Common ###############
	sset = ng.add(StartSet1D(trainX.shape[1]))
	den0 = ng.add(Dense1D(4), sset)
	act0 = ng.add(Relu1D(), den0)
	# act0 = ng.add(Sigmoid1D(), den0)
	den1 = ng.add(Dense1D(3), act0)
	# act1 = ng.add(Sigmoid1D(), den1)
	# lossFunc = ng.add(Mse1D(), act1)
	act1 = ng.add(Softmax1D(), den1)
	lossFunc = ng.add(OneHotCce1D(), act1)
	return ng


def f1():
	# sset(4) ---> den0(4>4) ---> act0(sigmoid) ---> flat ---> den2(8>3) ---> act2(softmax) ---> cce
	#		   \								  /
	# 			-> den1(4>4) ---> act1(sigmoid) -
	####### Batch ###############
	# ng = NodeGraphBatch()
	# ng.lr = .01
	# ng.epochMax = 100
	####### Adaptive ###############
	ng = NodeGraphAdaptive()
	# ng.lrMin = np.float32(1.0e-2)
	ng.epochMax = 10000
	ng.lrInit = 0.05
	####### Common ###############
	sset = ng.add(StartSet1D(trainX.shape[1]))
	den0 = ng.add(Dense1D(4), sset)
	act0 = ng.add(Sigmoid1D(), den0)

	den1 = ng.add(Dense1D(4), sset)
	act1 = ng.add(Sigmoid1D(), den1)

	flat = ng.add(Flat1D(), act0, act1)
	den2 = ng.add(Dense1D(3), flat)  # dn1
	# act1 = ng.add(Sigmoid1D(), den1)  # ac1
	# lossFunc = ng.add(Mse1D(), act1)  # cce
	act2 = ng.add(Softmax1D(), den2)  # ac1
	lossFunc = ng.add(OneHotCce1D(), act2)  # cce
	return ng


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# warnings.simplefilter('error')

ng = f0()

ng.compile()

start = time.time()
# ng.fit(trainX, trainT)
ng.fit(trainX, trainT, testX, testT)
print('elapsed:', time.time() - start)

print('=== graph info ===\n' + ng.graphInfo('detail'))
print('=== summary ===\n' + ng.summary())
# print(mset.Weight)

trainY, testY = ng.lossFunc.trY, ng.lossFunc.teY
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
