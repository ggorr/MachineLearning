import inspect
import time

import matplotlib.pyplot as plt

import lib_v1.PlotTools as pt
from lib_v1.NodeGraph import *


def xor1(index=0):
	print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
	trainX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
	trainT = np.array([[0, 1, 1, 0]], float).T
	testX = np.array([[.5, .5]], float)

	def f0():
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		# sset(2) ---> fset1(2>2) ---> act1(Sigmoid) ---> fset2(2>1) ---> act2(Sigmoid) ---> bce(1)
		##### Adaptive ##########
		ng = NodeGraphAdaptive()
		##### Batch #############
		# ng = NodeGraphBatch()
		# ng.lr = 0.5
		# ng.mom = 0.0
		##### Common ###############
		sset = ng.add(StartSet(2, 'sset'))
		fset1 = ng.add(Dense(2, 'fset1'), sset)
		act1 = ng.add(Sigmoid('act1'), fset1)
		fset2 = ng.add(Dense(1, 'fset2'), act1)
		act2 = ng.add(Sigmoid('act2'), fset2)
		lossFunc = ng.add(Bce('bce'), act2)
		ng.compile()

		# ng.reg = Lasso(0.001)
		ng.epochMax = 10000

		start = time.time()
		ng.fit(trainX, trainT, None, None)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		if isinstance(ng, NodeGraphAdaptive):
			print('epoch: ', ng.appLrs.shape[0])
		print('trainY.T:', lossFunc.trY.T)
		print('testY:', lossFunc.teY)
		return ng

	def f1():
		# sset(2) ----> fset11(2>2) --> act11(Sigmoid) ----> fset2(4>1) ---> act2(Sigmoid) ---> bce(1)
		#		   \								    /
		#           --> fset12(2>2) --> act12(Relu) ---
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		ng.lrMin = 1.0e-2
		####### Common ###############
		sset = ng.add(StartSet(2, 'sset'))
		fset11 = ng.add(Dense(2, 'fset11'), sset)
		act11 = ng.add(Sigmoid('act11'), fset11)
		fset12 = ng.add(Dense(2, 'fset12'), sset)
		act12 = ng.add(Relu('act12'), fset12)
		fset2 = ng.add(Dense(1, 'fset2'), act11, act12)
		act2 = ng.add(Sigmoid('act2'), fset2)
		lossFunc = ng.add(Bce('bce'), act2)
		ng.compile()

		ng.epochMax = 10000
		# ng.reg = Ridge(0.1)

		# fset11.initW = np.random.rand(3, 2) * 10 - 5
		# fset11.initW[0, :] = 0.0  # bias
		# fset12.initW = np.random.rand(3, 2) * 10 - 5
		# fset12.initW[0, :] = 0.0  # bias
		# fset2.initW = np.random.rand(5, 1) * 10 - 5
		# fset2.initW[0, :] = 0.0  # bias

		# fset11.initW = np.array([[-4.2838888, 4.0153641],
		# 						 [-8.38238995, - 8.07452261],
		# 						 [8.16167589, 8.36303902]])
		# fset12.initW = np.array([[-0.28976519, - 1.26329357],
		# 						 [-0.14135152, 0.55771471],
		# 						 [-0.74357676, 0.57459211]])
		# fset2.initW = np.array([[8.37800036], [17.80637617], [-17.34347719], [-0.55550275], [-0.61472909]])

		# batch
		# fset11.initW = np.array([[-0.01003381, 0.02639892],
		# 						 [0.73433933, 0.42795873],
		# 						 [0.59814426, -0.80914434]])
		# fset12.initW = np.array([[0.16108624, -0.00472506],
		# 						 [0.21136112, -0.77103574],
		# 						 [0.58521194, 0.77576081]])
		# fset2.initW = np.array([[0.1149501], [-0.07203821], [-0.79247827], [0.2753218], [0.85723796]])

		start = time.time()
		ng.fit(trainX, trainT)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print('trainY.T:', lossFunc.trY.T)
		# print(ng.predict(trainX))

		# print(fset11.Weight)
		# print(fset12.Weight)
		# print(fset2.Weight)

		return ng

	def f2():
		# sset(2) ----> fset11(2>5) ----> act11(Sigmoid) ----> mp(10>2) ---> fset2(2>1) ---> act2(Sigmoid) ---> bce(1)
		#			\									  /
		#			 -> fset12(2>5) ----> act12(Relu) ----
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		ng.lrMin = 1.0e-2
		####### Common ###############
		sset = ng.add(StartSet(2, 'sset'))
		fset11 = ng.add(Dense(5, 'fset11'), sset)
		fset12 = ng.add(Dense(5, 'fset12'), sset)

		act11 = ng.add(Sigmoid('act11'), fset11)
		act12 = ng.add(Relu('act12'), fset12)
		mp = ng.add(MaxPool1D(5, 'mp'), act11, act12)
		fset2 = ng.add(Dense(1, 'fset2'), mp)
		act2 = ng.add(Sigmoid('act2'), fset2)
		# lossFunc = Mse('loss')
		lossFunc = ng.add(Bce('bce'), act2)
		ng.compile()

		ng.epochMax = 10000
		# ng.reg = Ridge(0.1)

		# fset2.initW = np.random.rand(5, 1) * 10 - 5

		start = time.time()
		ng.fit(trainX, trainT)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print('trainY.T:', lossFunc.trY.T)
		# print(ng.predict(trainX))

		# print(fset2.gradX)
		# print(mp.gradY)
		# print(mp.trInd)
		# print(mp.gradX)
		return ng

	ng = eval(f'f{index}()')

	print(f'=== graph info ===\n{ng.graphInfo("detail")}')
	print(f'=== fit summary ===\n{ng.summary()}')

	plt.subplot(221)
	pt.contour(ng, [-0.5, 1.5], [-0.5, 1.5])  # 'xor, nodes = {}'.format(seq.getNodes()))
	plt.subplot(223)
	plt.title('loss')
	pt.plotLoss(ng)
	plt.subplot(224)
	if isinstance(ng, NodeGraphAdaptive):
		plt.title('learning rates')
		plt.plot(ng.appLrs)
	plt.show()


def xor2(index=0):
	print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
	trainX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
	trainT = np.array([[0, 1], [1, 0], [1, 0], [0, 1]], float)
	testX = np.array([[.5, .5]], float)
	testT = np.array([[0, 1]], float)

	def f0():
		# sset(2) ---> fset1(2>2) ---> act1(Sigmoid) ---> fset2(2>2) ---> act2(Softmax) ---> cce(2)
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		####### Common ###############
		sset = ng.add(StartSet(2, 'sset'))
		fset1 = ng.add(Dense(2, 'fset1'), sset)
		act1 = ng.add(Sigmoid('act1'), fset1)
		fset2 = ng.add(Dense(2, 'fset2'), act1)
		act2 = ng.add(Softmax('act2'), fset2)
		lossFunc = ng.add(Cce('cce'), act2)

		ng.epochMax = 10000

		ng.compile()

		start = time.time()
		# ng.fit(trainX, trainT)
		ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print(f'trainY:\n{lossFunc.trY}')
		return ng

	def f1():
		# sset(2) ----> fset11(2->2) --> act11(Sigmoid) --> fset12(2->1) ------> act2(Softmax) ---> cce(2)
		#          \														/
		#           --> fset21(2->2) --> act21(Sigmoid) --> fset22(2->1) --
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		####### Common ###############

		sset = ng.add(StartSet(2, 'sset'))
		fset11 = ng.add(Dense(2, 'fset11'), sset)
		act11 = ng.add(Sigmoid('act11'), fset11)
		fset12 = ng.add(Dense(1, 'fset12'), act11)

		fset21 = ng.add(Dense(2, 'fset21'), sset)
		act21 = ng.add(Sigmoid('act21'), fset21)
		fset22 = ng.add(Dense(1, 'fset22'), act21)

		act2 = ng.add(Softmax('act2'), fset12, fset22)
		# lossFunc = ng.add(MeanSquare('loss'), act2)
		lossFunc = ng.add(Cce('cce'), act2)
		ng.epochMax = 10000
		ng.compile()

		# trainT = np.array([[0.3, .7], [1, 0], [1, 0], [0.3, .7]], float)
		start = time.time()
		# ng.fit(trainX, trainT)
		ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')
		print(f'trainY:\n{lossFunc.trY}')
		return ng

	def f2():
		# sset1(1) ----> fset11(1->2) --> act11(Sigmoid) --> fset12(2->1) ------> act2(Softmax) ---> cce(2)
		#          															 /
		# sset2(1) ----> fset21(1->2) --> act21(Sigmoid) --> fset22(2->1) --
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		# ng.lr = .001
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		ng.lrMin = .001
		####### Common ###############
		sset1 = ng.add(StartSet(1, 'sset1'))
		sset2 = ng.add(StartSet(1, 'sset2'))

		fset11 = ng.add(Dense(2, 'fset11'), sset1)
		act11 = ng.add(Sigmoid('act11'), fset11)
		fset12 = ng.add(Dense(1, 'fset12'), act11)

		fset21 = ng.add(Dense(2, 'fset21'), sset2)
		act21 = ng.add(Sigmoid('act21'), fset21)
		fset22 = ng.add(Dense(1, 'fset22'), act21)

		act2 = ng.add(Softmax('act2'), fset12, fset22)
		# lossFunc = MeanSquare('loss')
		lossFunc = ng.add(Cce('cce'), act2)

		ng.epochMax = 10000
		ng.compile()

		start = time.time()
		ng.fit((trainX[:, :1], trainX[:, 1:]), trainT)
		# ng.fit((trainX[:, :1], trainX[:, 1:]), trainT, (testX[:, :1], testX[:, 1:]), testT)
		print(f'elapsed: {time.time() - start}')
		print(f'trainY:\n{lossFunc.trY}')
		return ng

	def f3():
		# sset(2) ----> fset11(2->4) --> act11(Sigmoid) --> fset12(2->1) -----> act12(Sigmoid) -----> mse(2)
		#			\																			 /
		#            -> fset21(2->12) --> act21(Relu) --> fset22(2->1) -----> act22(Sigmoid) ---/
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		# ng.lr = .01
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		ng.lrInit = .5
		ng.lrMin = .01
		####### Common ###############
		sset = ng.add(StartSet(2, 'sset'))

		fset11 = ng.add(Dense(4, 'fset11'), sset)
		act11 = ng.add(Sigmoid('act11'), fset11)
		fset12 = ng.add(Dense(1, 'fset12'), act11)
		act12 = ng.add(Sigmoid('act12'), fset12)

		fset21 = ng.add(Dense(12, 'fset21'), sset)
		act21 = ng.add(Relu('act21'), fset21)
		fset22 = ng.add(Dense(1, 'fset22'), act21)
		act22 = ng.add(Sigmoid('act22'), fset22)
		lossFunc = ng.add(Mse('mse'), act12, act22)

		ng.epochMax = 10000

		ng.compile()

		start = time.time()
		ng.fit(trainX, trainT)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print(f'trainY:\n{lossFunc.trY}')
		return ng

	def f4():
		# sset(2) ----> fset11(2->4) --> act11(Sigmoid) ---> mp1(4->2) -----> den(4->2) ---> act(Softmax) ---> cce(2)
		#			\									                 /
		#            -> fset21(2->12) --> act21(Relu) --->  mp2(12->2) --
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		# ng.lr = .01
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		ng.lrMin = 1.0E-3
		####### Common ###############
		sset = ng.add(StartSet(2, 'sset'))

		fset11 = ng.add(Dense(4, 'fset11'), sset)
		act11 = ng.add(Sigmoid('act11'), fset11)
		mp1 = ng.add(MaxPool1D(2, 'mp1'), act11)

		fset21 = ng.add(Dense(12, 'fset21'), sset)
		act21 = ng.add(Relu('act21'), fset21)
		mp2 = ng.add(MaxPool1D(6, 'mp2'), act21)

		den = ng.add(Dense(2, 'den'), mp1, mp2)
		act = ng.add(Softmax('sm'), den)
		lossFunc = ng.add(Cce('cce'), act)

		ng.epochMax = 10000

		ng.compile()

		# trainT = np.array([[0.3, .7], [1, 0], [1, 0], [0.3, .7]], float)
		start = time.time()
		ng.fit(trainX, trainT)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print(f'trainY:\n{lossFunc.trY}')
		# print(lossFunc.trBase())
		# print(lossFunc.trLoss())
		return ng

	ng = eval(f'f{index}()')
	print(f'=== graph info ===\n{ng.graphInfo("detail")}')
	print(f'=== summary ===\n{ng.summary()}')

	print(ng.lossFunc.trY)

	pt.contourDouble(ng, 221, 222, [-0.5, 1.5], [-0.5, 1.5])
	plt.subplot(223)
	plt.title('loss')
	pt.plotLoss(ng)
	plt.subplot(224)
	plt.title('learning rates')
	if isinstance(ng, NodeGraphAdaptive):
		plt.plot(ng.appLrs)
	plt.show()


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
xor1(0)
# xor2(0)
