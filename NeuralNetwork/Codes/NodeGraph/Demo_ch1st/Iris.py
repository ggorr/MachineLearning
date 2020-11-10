import time

import matplotlib.pyplot as plt

import lib_ch1st.DataTools as dt
import lib_ch1st.PlotTools as pt
from lib_ch1st.NodeGraph import *

from sklearn import datasets
iris = datasets.load_iris()
data = (iris.data - np.mean(iris.data, axis=0)) / np.std(iris.data, axis=0)
##############################################################################
# trainX, trT, testX, teT = dt.splitTrainTest(data, iris.target)
# trainT, testT = dt.onehot((trT, teT))
# trainX, trainT, testX, testT = dt.addChannel(trainX, trainT, testX, testT)
##############################################################################
trainX, testX = data, None
trainT, testT = dt.onehot(iris.target), None
trainX, trainT = dt.addChannel(trainX, trainT)
##############################################################################


def f0():
	# sset(4) ---> den1(4>16) ---> act1(sigmoid) ---> den2(16>3) ---> act2(softmax) ---> cce
	####### Batch ###############
	# ng = NodeGraphBatch()
	# # ng.lr = .01
	# ng.epochMax = 1000
	####### Adaptive ###############
	ng = NodeGraphAdaptive()
	ng.lrInit = 0.02
	ng.epochMax = 100
	####### Common ###############
	sset = ng.add(StartSet1D(trainX.shape[1]))
	den0 = ng.add(Dense1D(16), sset)
	act0 = ng.add(Sigmoid1D(), den0)
	den1 = ng.add(Dense1D(3), act0)
	# act1 = ng.add(Sigmoid1D(), den1)
	# lossFunc = ng.add(Mse1D(), act1)
	act1 = ng.add(Softmax1D(), den1)
	lossFunc = ng.add(Cce1D(), act1)
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
	ng.lrMin = 1.0e-2
	ng.lrInit = 0.04
	ng.epochMax = 100
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
	lossFunc = ng.add(Cce1D(), act2)  # cce
	return ng

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
ng = f0()

ng.compile()

start = time.time()
# ng.fit(trainX, trainT)
ng.fit(trainX, trainT, testX, testT, verbose=2)
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

