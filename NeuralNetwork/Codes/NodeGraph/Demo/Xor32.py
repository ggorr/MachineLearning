import inspect
import time

import matplotlib.pyplot as plt

import lib32.DataTools as dt
import lib32.PlotTools as pt
from lib32.NodeGraph import *


def xor1(index=0):
	print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
	trainX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
	trainX = dt.addChannel(trainX)
	trainT = np.array([[0], [1], [1], [0]])
	trainT = dt.addChannel(trainT)
	testX = dt.addChannel(np.array([[.5, .5], [1.5, 1.5]], np.float32))
	testT = dt.addChannel(np.array([[0], [1]]))

	# print(trainX.dtype, trainT.dtype)

	def f0():
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		# sset(2) ---> fset1(2>2) ---> act1(Sigmoid) ---> fset2(2>1) ---> act2(Sigmoid) ---> bce(1)
		##### Adaptive ##########
		ng = NodeGraphAdaptive()
		##### Batch #############
		# ng = NodeGraphBatch()
		# ng.lr = np.float32(0.5)
		# ng.mom = 0.0
		##### Common ###############
		sset = ng.add(StartSet1D(2, name='sset'))
		fset1 = ng.add(Dense1D(2, name='fset1'), sset)
		act1 = ng.add(Sigmoid1D('act1'), fset1)
		fset2 = ng.add(Dense1D(1, name='fset2'), act1)
		act2 = ng.add(Sigmoid1D('act2'), fset2)
		lossFunc = ng.add(ZeroOneBce1D('bce'), act2)
		ng.compile()

		# ng.reg = Lasso(0.001)
		ng.epochMax = 10000

		start = time.time()
		ng.fit(trainX, trainT)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		if isinstance(ng, NodeGraphAdaptive):
			print('epoch: ', ng.appLrs.shape[0])
		print('trainY[:,0,0]:', lossFunc.trY[:, 0, 0])
		print('testY:', lossFunc.teY)
		return ng

	def f1():
		# sset(2) ----> fset11(2>2) --> act11(Sigmoid) ----> con1((2,2)>(1,1)) ---> act2(Sigmoid) ---> bce(1)
		#		   \								    /
		#           --> fset12(2>2) --> act12(Relu) ---
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		ng = NodeGraphBatch()
		ng.mom = np.float32(0.01)
		ng.reg = Ridge(0.1)
		####### Adaptive ###############
		# ng = NodeGraphAdaptive()
		# ng.lrMin = np.float32(1.0e-2)
		####### Common ###############
		sset = ng.add(StartSet1D(2, name='sset'))
		fset11 = ng.add(Dense1D(2, name='fset11'), sset)
		act11 = ng.add(Sigmoid1D('act11'), fset11)
		fset12 = ng.add(Dense1D(2, name='fset12'), sset)
		act12 = ng.add(Relu1D('act12'), fset12)
		con1 = ng.add(Conv1D(2, 1, biased=True, name='con1'), act11, act12)
		# fset2 = ng.add(Dense1D(1,2, 'fset2'), act11, act12)
		act2 = ng.add(Sigmoid1D('act2'), con1)
		lossFunc = ng.add(ZeroOneBce1D('bce'), act2)
		ng.compile()

		ng.epochMax = 5000
		# ng.reg = Ridge(0.1)

		start = time.time()
		ng.fit(trainX, trainT)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print('trainY[:,0, 0]:', lossFunc.trY[:, 0, 0])
		# print(ng.predict(trainX))

		# print(fset11.Weight)
		# print(fset12.Weight)
		# print(fset2.Weight)

		return ng

	def f2():
		# sset(2) ----> fset11(2>5) ----> act11(Sigmoid) ----> con1((5,2)>(5,1)) ---> act2(Sigmoid) ---> den2(5>1) ---> act3(Sigmoid) ---> bce(1)
		#			\									  /
		#			 -> fset12(2>5) ----> act12(Relu) ----
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		# ng.mom = np.float32(0.1)
		# ng.reg = Ridge(0.01)
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		ng.lrMin = np.float32(1.0e-2)
		# ng.reg = Ridge(0.01)
		####### Common ###############
		sset = ng.add(StartSet1D(2, name='sset'))
		fset11 = ng.add(Dense1D(5, name='fset11'), sset)
		fset12 = ng.add(Dense1D(5, name='fset12'), sset)
		act11 = ng.add(Sigmoid1D('act11'), fset11)
		act12 = ng.add(Relu1D('act12'), fset12)
		con1 = ng.add(Conv1D(3, 1, biased=True, name='con1'), act11, act12)
		act2 = ng.add(Sigmoid1D('act2'), con1)
		den2 = ng.add(Dense1D(1, name='den1'), act2)
		act3 = ng.add(Sigmoid1D('act3'), den2)
		# lossFunc = Mse('loss')
		lossFunc = ng.add(ZeroOneBce1D('bce'), act3)
		ng.compile()

		ng.epochMax = 10000
		# ng.reg = Ridge(0.1)

		# fset2.initW = np.random.rand(5, 1) * 10 - 5

		start = time.time()
		ng.fit(trainX, trainT)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print('trainY[:, 0, 0]:', lossFunc.trY[:, 0, 0])
		# print(ng.predict(trainX))

		# print(fset2.gradX)
		# print(mp.gradY)
		# print(mp.trInd)
		# print(mp.gradX)
		print('======== B ==========')
		print(con1.B)
		print('======== W ==========')
		print(con1.W)
		return ng

	def f3():
		# sset(2) ----> fset11(2>6) ----> act11(Sigmoid) ----> mp((6,2)>(3,2)) ---> flat((3,2)>6) ---> den2(10>1) ---> act3(Sigmoid) ---> bce(1)
		#			\									  /
		#			 -> fset12(2>6) ----> act12(Relu) ----
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		####### Adaptive ###############
		ng = NodeGraphAdaptiveMinibatch()
		ng.lrMin = np.float32(1.0e-2)
		####### Common ###############
		sset = ng.add(StartSet1D(2, name='sset'))
		fset11 = ng.add(Dense1D(6, name='fset11'), sset)
		fset12 = ng.add(Dense1D(6, name='fset12'), sset)
		act11 = ng.add(Sigmoid1D('act11'), fset11)
		act12 = ng.add(Relu1D('act12'), fset12)
		mp = ng.add(MaxPool1D(2, 'mp'), act11, act12)
		fl1 = ng.add(Flat1D('fl1'), mp)
		den2 = ng.add(Dense1D(1, name='den1'), fl1)
		act3 = ng.add(Sigmoid1D('act3'), den2)
		# lossFunc = Mse('loss')
		lossFunc = ng.add(ZeroOneBce1D('bce'), act3)
		ng.compile()

		ng.epochMax = 1000
		# ng.reg = Ridge(0.1)

		# fset2.initW = np.random.rand(5, 1) * 10 - 5

		start = time.time()
		# ng.fit(trainX, trainT)
		ng.fit(trainX, trainT, testX, testT, batchSize=1)
		print(f'elapsed: {time.time() - start}')

		print(ng.predict(trainX)[:, 0, 0])

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
	plt.subplot(222)
	plt.title('accuracy')
	pt.plotAccuracy(ng)
	plt.subplot(223)
	plt.title('loss')
	pt.plotLoss(ng)
	plt.subplot(224)
	if isinstance(ng, NodeGraphAdaptive):
		plt.title('learning rates')
		plt.plot(ng.appLrs)
	plt.tight_layout()
	plt.show()


def xor2(index=0):
	print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
	trainX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
	trainT = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
	testX = np.array([[.5, .5]], np.float32)
	testT = np.array([[0, 1]])
	trainX, trainT = dt.addChannel(trainX), dt.addChannel(trainT)
	testX, testT = dt.addChannel(testX), dt.addChannel(testT)

	def f0():
		# sset(2) ---> fset1(2>2) ---> act1(Sigmoid) ---> fset2(2>2) ---> act2(Softmax) ---> cce(2)
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		####### Common ###############
		sset = ng.add(StartSet1D(2, name='sset'))
		fset1 = ng.add(Dense1D(2, name='fset1'), sset)
		act1 = ng.add(Sigmoid1D('act1'), fset1)
		fset2 = ng.add(Dense1D(2, name='fset2'), act1)
		act2 = ng.add(Softmax1D('act2'), fset2)
		lossFunc = ng.add(OneHotCce1D('cce'), act2)

		ng.epochMax = 10000

		ng.compile()

		start = time.time()
		# ng.fit(trainX, trainT)
		ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print(f'trainY:\n{lossFunc.trY[:,:,0]}')
		return ng

	def f1():
		# sset(2) ----> fset11(2->2) --> act11(Sigmoid) --> fset12(2->1) ----> con((1,2)>(1,2)) ---> flat ---> act2(Softmax) ---> cce(2)
		#          \													   /
		#           --> fset21(2->2) --> act21(Sigmoid) --> fset22(2->1) -
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		####### Common ###############

		sset = ng.add(StartSet1D(2, name='sset'))
		fset11 = ng.add(Dense1D(2, name='fset11'), sset)
		act11 = ng.add(Sigmoid1D('act11'), fset11)
		fset12 = ng.add(Dense1D(1, name='fset12'), act11)

		fset21 = ng.add(Dense1D(2, name='fset21'), sset)
		act21 = ng.add(Sigmoid1D('act21'), fset21)
		fset22 = ng.add(Dense1D(1, name='fset22'), act21)

		con = ng.add(Conv1D(1, 2, biased=True, name='con'), fset12, fset22)
		flat = ng.add(Flat1D('flt'), con)
		act2 = ng.add(Softmax1D('act2'), flat)
		# lossFunc = ng.add(Mse1D('mse'), act2)
		lossFunc = ng.add(OneHotCce1D('cce'), act2)
		ng.epochMax = 10000
		ng.compile()

		# trainT = np.array([[0.3, .7], [1, 0], [1, 0], [0.3, .7]], float)
		start = time.time()
		# ng.fit(trainX, trainT)
		ng.fit(trainX, trainT, testX, testT, batchSize=2)
		print(f'elapsed: {time.time() - start}')
		print(f'trainY:\n{lossFunc.trY[:,:,0]}')
		return ng

	def f2():
		# sset1(1) ----> fset11(1->2) --> act11(Sigmoid)  ------> con((2, 2)>(1,2)) ---> flt ---> act2(Softmax) ---> cce(2)
		#          											 /
		# sset2(1) ----> fset21(1->2) --> act21(Sigmoid)  --
		print(f'******************** {inspect.currentframe().f_code.co_name} ********************')
		####### Batch ###############
		# ng = NodeGraphBatch()
		# ng.lr = .001
		####### Adaptive ###############
		ng = NodeGraphAdaptive()
		ng.lrMin = .001
		####### Common ###############
		sset1 = ng.add(StartSet1D(1, name='sset1'))
		sset2 = ng.add(StartSet1D(1, name='sset2'))

		fset11 = ng.add(Dense1D(3, name='fset11'), sset1)
		act11 = ng.add(Sigmoid1D('act11'), fset11)

		fset21 = ng.add(Dense1D(3, name='fset21'), sset2)
		act21 = ng.add(Sigmoid1D('act21'), fset21)

		con = ng.add(Conv1D(3, 2, biased=True, padding=0, name='con'), act11, act21)
		flt = ng.add(Flat1D('flt'), con)
		# flt = ng.add(Flat1D('flt'), fset12, fset22)

		act2 = ng.add(Softmax1D('act2'), flt)
		# lossFunc = MeanSquare('loss')
		lossFunc = ng.add(OneHotCce1D('cce'), act2)

		ng.epochMax = 10000
		ng.compile()

		start = time.time()
		ng.fit((trainX[:, :1], trainX[:, 1:]), trainT)
		# ng.fit((trainX[:, :1], trainX[:, 1:]), trainT, (testX[:, :1], testX[:, 1:]), testT)
		print(f'elapsed: {time.time() - start}')
		print(f'trainY:\n{lossFunc.trY[:,:,0]}')
		print(con.B)
		return ng

	def f3():
		# sset(2) ----> fset11(2->4) --> act11(Sigmoid) --> fset12(2->1) -----> act12(Sigmoid) -----> flat ---> mse(2)
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
		sset = ng.add(StartSet1D(2, name='sset'))

		fset11 = ng.add(Dense1D(4, name='fset11'), sset)
		act11 = ng.add(Sigmoid1D('act11'), fset11)
		fset12 = ng.add(Dense1D(1, name='fset12'), act11)
		act12 = ng.add(Sigmoid1D('act12'), fset12)

		fset21 = ng.add(Dense1D(12, name='fset21'), sset)
		act21 = ng.add(Relu1D('act21'), fset21)
		fset22 = ng.add(Dense1D(1, name='fset22'), act21)
		act22 = ng.add(Sigmoid1D('act22'), fset22)
		flat = ng.add(Flat1D('flat'), act12, act22)
		lossFunc = ng.add(Mse1D('mse'), flat)

		ng.epochMax = 10000

		ng.compile()

		start = time.time()
		ng.fit(trainX, trainT)
		# ng.fit(trainX, trainT, testX, testT)
		print(f'elapsed: {time.time() - start}')

		print(f'trainY:\n{lossFunc.trY[:,:,0]}')
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
		ng = NodeGraphAdaptiveMinibatch()
		ng.lrInit = 1.0E-3
		ng.lrMin = 1.0E-3
		####### Common ###############
		sset = ng.add(StartSet1D(2, name='sset'))

		fset11 = ng.add(Dense1D(4, name='fset11'), sset)
		act11 = ng.add(Sigmoid1D(name='act11'), fset11)
		mp1 = ng.add(MaxPool1D(2, name='mp1'), act11)

		fset21 = ng.add(Dense1D(12, name='fset21'), sset)
		act21 = ng.add(Relu1D(name='act21'), fset21)
		mp2 = ng.add(MaxPool1D(6, name='mp2'), act21)

		flat = ng.add(Flat1D(name='flat'), mp1, mp2)
		den = ng.add(Dense1D(2, name='den'), flat)
		act = ng.add(Softmax1D(name='sm'), den)
		lossFunc = ng.add(OneHotCce1D(name='cce'), act)

		ng.epochMax = 5000

		ng.compile()

		start = time.time()
		# ng.fit(trainX, trainT, batchSize=1)
		ng.fit(trainX, trainT, testX, testT, batchSize=4)
		print(f'elapsed: {time.time() - start}')

		print(f'trainY:\n{ng.predict(trainX)[:,:,0]}')
		return ng

	ng = eval(f'f{index}()')
	print(f'=== graph info ===\n{ng.graphInfo("detail")}')
	print(f'=== summary ===\n{ng.summary()}')

	pt.contourDouble(ng, 231, 232, [-0.5, 1.5], [-0.5, 1.5])
	plt.subplot(234)
	plt.title('loss')
	pt.plotLoss(ng)
	plt.subplot(235)
	plt.title('accuracy')
	pt.plotAccuracy(ng)
	plt.subplot(236)
	plt.title('learning rates')
	if isinstance(ng, NodeGraphAdaptive) or isinstance(ng, NodeGraphAdaptiveMinibatch):
		plt.plot(ng.appLrs)
	plt.tight_layout()
	plt.show()


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# xor1(2)
xor2(4)
