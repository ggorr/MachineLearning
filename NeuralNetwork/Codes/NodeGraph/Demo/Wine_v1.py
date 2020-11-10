import time

import matplotlib.pyplot as plt

import lib_v1.DataTools as dt
import lib_v1.PlotTools as pt
from lib_v1.NodeGraph import *


def f0():
	from sklearn import datasets
	wine = datasets.load_wine()
	data = (wine.data - np.mean(wine.data, axis=0)) / np.std(wine.data, axis=0)
	trainX, trT, testX, teT = dt.splitTrainTest(data, wine.target)
	trainT = dt.vectorizeTarget(trT).astype(float)
	testT = dt.vectorizeTarget(teT).astype(float)
	# trainX, testX = data, None
	# trainT, testT = dt.vectorizeTarget(wine.target).astype(float), None

	w = np.array([[0., 0., 0.],
				  [-0.28062566, -0.92674063, 0.21886565],
				  [0.62450193, 0.1088061, 0.280161],
				  [0.95312941, 0.52824994, -0.48644398],
				  [-0.85559336, 0.72633092, -0.36064754],
				  [0.52345169, 0.34344775, 0.08545155],
				  [-0.31424921, -0.20702117, -0.35558297],
				  [-0.18214249, 0.57727576, 0.04404145],
				  [-0.24103644, -0.60378356, 0.08339987],
				  [0.16016922, -0.07203351, 0.2427195],
				  [-0.84775982, -0.92997451, 0.12776489],
				  [-0.04465301, -0.23576283, -0.82975385],
				  [-0.49875841, 0.5065488, -0.25812795],
				  [0.03299527, -0.87013903, 0.64624562]])

	# sset(13) ---> den0(13>20) ---> act0(sigmoid) ---> mp0(20>4) --> den1(20>3) ---> act1(softmax) ---> cce
	####### Batch ###############
	# ng = NodeGraphBatch()
	####### Adaptive ###############
	ng = NodeGraphAdaptive()
	ng.lrMin = 1.0e-12
	####### Common ###############
	sset = ng.add(StartSet(trainX.shape[1]))  # ss0
	den0 = ng.add(Dense(20), sset)  # dn0
	act0 = ng.add(Sigmoid(), den0)  # ac0
	mp0 = ng.add(MaxPool1D(4), act0)
	den1 = ng.add(Dense(3), mp0)  # dn1
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
