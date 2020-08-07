from typing import List, Callable, TextIO

import numpy as np
from scipy.special import expit
from scipy.special import softmax as scipysoftmax

EPSILON = 1.115E-16


def identity(x: np.ndarray, out: np.ndarray):
	np.copyto(out, x)


def didentity_y(_: np.ndarray, out: np.ndarray):
	out.fill(1.0)


def relu(x, out):
	''' do not use as relu(x, x) '''
	# out[:] = (x > 0) * x
	# np.multiply(0.5, np.add(np.fabs(x, out), x, out), out)
	np.multiply(np.greater(x, 0, out), x, out)


def drelu(x, out):
	np.greater(x, 0, out)


def drelu_y(y, out):
	np.greater(y, 0, out)


def sigmoid(x, out):
	# out[:] = 1 / (1 + np.exp(-x))
	# np.reciprocal(np.add(np.exp(np.negative(x, out), out), 1, out), out)
	expit(x, out)


def dsigmoid(x, out):
	# y = 1 / (1 + np.exp(-x))
	# out[:] = y * (1 - y)
	sigmoid(x, out)
	out *= 1 - out


def dsigmoid_y(y, out):
	# out[:] = (1 - y) * y
	np.multiply(np.subtract(1, y, out), y, out)


def softplus(x, out):
	# out[:] = np.log(1 + np.exp(x))
	np.log1p(np.exp(x, out), out)


def dsoftplus(x, out):
	# out[:] = 1 / (1 + np.exp(-x))
	np.reciprocal(np.add(np.exp(np.negative(x, out), out), 1, out), out)


def dsoftplus_y(y, out):
	# out[:] = 1 - np.exp(-y)
	np.subtract(1, np.exp(np.negative(y, out), out), out)


tanh = np.tanh


def dtanh(x, out):
	# out[:] = 1 - np.tanh(x) ** 2
	np.subtract(1, np.square(np.tanh(x, out), out), out)


def dtanh_y(y, out):
	# out[:] = 1 - y * y
	np.subtract(1, np.square(y, out), out)


def softmax(x, out):
	# expx = np.exp(x)
	# out[:] = expx / np.sum(expx, axis=0)
	# np.exp(x, out)
	# out /= np.sum(out, axis=0)
	out[:] = scipysoftmax(x, axis=0)


def dsoftmax(x):
	pass


def dsoftmax_y(y):
	pass


_df_y = {relu: drelu_y, sigmoid: dsigmoid_y, tanh: dtanh_y, softplus: dsoftplus_y, softmax: dsoftmax_y, identity: didentity_y, None: None}


def crossEntropy_multi(y: np.ndarray, output: np.ndarray, entropy: float, t1: np.ndarray, t2: np.ndarray):
	# t = y * np.log(output + EPSILON) + (1 - y) * np.log((1 + EPSILON) - output)
	# return -np.sum(t) / y.shape[1] + entropy
	np.multiply(np.subtract(1, y, t1), np.log(np.subtract(1 + EPSILON, output, t2), t2), t1)
	np.multiply(y, np.log(np.add(output, EPSILON, t2), t2), t2)
	return -np.sum(np.add(t1, t2, t1)) / y.shape[1] + entropy


def crossEntropy_mono(y: np.ndarray, output: np.ndarray, entropy: float, t1: np.ndarray, t2: np.ndarray):
	# return -np.sum(y * np.log(output + EPSILON)) / y.shape[1] + entropy
	return -np.sum(np.multiply(y, np.log(np.add(output, EPSILON, t1), t1), t1)) / y.shape[1] + entropy


def l2(y: np.ndarray, output: np.ndarray, entropy: float, t1: np.ndarray, t2: np.ndarray):
	# return np.sum((y - output) ** 2) / (2 * y.shape[1])
	return np.sum(np.square(np.subtract(y, output, t1), t1)) / (y.shape[1] << 1)


def entropy_multi(y: np.ndarray, t1: np.ndarray, t2: np.ndarray):
	#  return np.sum(y * np.log(y + EPSILON) + (1 - y) * np.log((1 + EPSILON) - y)) / y.shape[1]
	np.multiply(np.subtract(1, y, t1), np.log(np.subtract(1 + EPSILON, y, t2), t2), t1)
	np.multiply(np.log(np.add(y, EPSILON, t2), t2), y, t2)
	return np.sum(np.add(t1, t2, t1)) / y.shape[1]


def entropy_mono(y: np.ndarray, t1: np.ndarray):
	# return np.sum(y * np.log(y + EPSILON)) / y.shape[1]
	return np.sum(np.multiply(np.log(np.add(y, EPSILON, t1), t1), y, t1)) / y.shape[1]


# internal
def _combinePair(src: np.ndarray, pairY: list):
	return np.array([np.sum(src[p, :], axis=0) if isinstance(p, (list, tuple)) else src[p, :] for p in pairY])


# external
def combinePair(src: np.ndarray, pairY: list):
	if pairY is None or len(pairY) == 0:
		return np.copy(src)
	if src.ndim == 1:
		if len(pairY) == 1 and (isinstance(pairY[0], int) or len(pairY[0]) == 1):
			# input is one dimensional vectors
			return np.copy(src)
		else:
			# src is a vector
			src = src.reshape(src.shape[0], 1)
	else:
		src = src.T
	dst = _combinePair(src, pairY)
	return dst.flatten() if dst.shape[0] == 1 or dst.shape[1] == 1 else dst.T


def readWeight(path: str = None, file: TextIO = None):
	f = open(path, 'r') if file is None else file
	weightList, w = [], []
	while True:
		line = f.readline()
		if len(line) <= 2 or line[2] != '[':
			break
		w.append([float(v) for v in line[3:line.find(']')].split()])
		if line.find(']]') > 0:
			weightList.append(np.array(w))
			w = []
	if file is None:
		f.close()
	return weightList


def saveWeight(weightList: List, path: str = None, file: TextIO = None, precision: int = 16):
	f = open(path, 'w') if file is None else file
	for i, wi in enumerate(weightList):
		f.write('[' if i == 0 else ' ')
		f.write(np.array2string(wi, precision=precision, max_line_width=np.inf, threshold=np.inf).replace('\n', '\n '))
		f.write(']' if i == len(weightList) - 1 else '\n')
	if file is None:
		f.close()


class LearnType:
	BATCH = 3
	ADAPTIVE = 4
	BATCH_SINGLE = 13
	ADAPTIVE_SINGLE = 14


class LossType:
	MULTI_CROSS_ENTROPY = 0  # MUST BE: The last activation function is the logistic sigmoid
	MONO_CROSS_ENTROPY = 1  # MUST BE: The last activation function is the soft max
	L2 = 2


class Activation:
	IDENTITY = identity
	TANH = tanh
	SIGMOID = sigmoid
	RELU = relu
	SOFTMAX = softmax
	SOFTPLUS = softplus


class WeightType:
	GENERAL = 0
	FILTER = 1
	FIXED = 2


class Ridge:
	def __init__(self, rate=0.1):
		self.rate = rate


class Lasso:
	def __init__(self, rate=0.1):
		self.rate = rate


def getField(cls, value):
	for key in cls.__dict__.keys():
		if cls.__dict__[key] == value:
			return key


class Layer(object):
	def __init__(self, node: int = None, weight: np.ndarray = None, weightType: int = WeightType.GENERAL, biased: bool = True, activation=Activation.SIGMOID):
		self.initialWeight: np.ndarray = weight
		self.xdim: int = None
		self.ydim: int = node
		self.weightType: int = weightType
		self.biased: bool = biased
		self.f = activation
		# self.df = None
		self.df_y = _df_y[activation]
		# self.row: int = None # for filter
		# self.col: int = None # for filter
		self.trainX: np.ndarray = None
		self.testX: np.ndarray = None

		self.B: np.ndarray = None
		self.W: np.ndarray = None
		self.BPrev: np.ndarray = None
		self.WPrev: np.ndarray = None
		self.trainN: np.ndarray = None
		self.testN: np.ndarray = None
		self.Delta: np.ndarray = None
		self.WDelta: np.ndarray = None
		self.gradB: np.ndarray = None
		self.gradW: np.ndarray = None
		self.gradBPrev: np.ndarray = None
		self.gradWPrev: np.ndarray = None
		self.gradBLike: np.ndarray = None
		self.gradWLike: np.ndarray = None

		self.gradient = None
		self.gradientAtLast = None
		# self.update = None
		self.regularizationRate: float = None

	def initialize(self, learnType: int, regularization, xdim: int, trainSize: int, testSize: int):
		if self.initialWeight is None:
			assert xdim is not None, 'cannot determine xdim'
			assert self.ydim is not None, 'cannot determine ydim'
			self.initialWeight = np.random.rand(self.ydim, xdim + 1) * 2 - 1
			if not self.biased:
				self.initialWeight[:, 0] = 0
		else:
			if xdim is not None:
				assert xdim + 1 == self.initialWeight.shape[1], 'mismatch between xdim and weight'
			if self.ydim is not None:
				assert self.ydim == self.initialWeight.shape[0], 'mismatch between ydim and weight'
			self.ydim = self.initialWeight.shape[0]
		self.xdim = xdim
		self.B = self.initialWeight[:, :1].astype(float)
		self.W = self.initialWeight[:, 1:].astype(float)
		self.trainX = np.empty((self.xdim, trainSize))
		self.trainN = np.empty((self.ydim, trainSize))
		self.Delta = np.empty_like(self.trainN)
		self.WDelta = np.empty((self.xdim, trainSize))
		self.gradB = np.zeros_like(self.B)
		self.gradW = np.zeros_like(self.W)
		self.gradBPrev = np.zeros_like(self.B)
		self.gradWPrev = np.zeros_like(self.W)
		self.gradBLike = np.empty_like(self.B)
		self.gradWLike = np.empty_like(self.W)
		if testSize > 0:
			self.testX = np.empty((self.xdim, testSize))
			self.testN = np.empty((self.ydim, testSize))
		if learnType == LearnType.ADAPTIVE:
			# self.update = self.updateAdaptive
			self.BPrev = np.empty_like(self.B)
			self.WPrev = np.empty_like(self.W)
		# else: #if learnType == LearnType.BATCH:
		# self.update = self.updateBatch
		if regularization is None:
			self.gradient = self.gradientPlain
			self.gradientAtLast = self.gradientPlainAtLast
		elif isinstance(regularization, Ridge):
			self.gradient = self.gradientRidge
			self.gradientAtLast = self.gradientRidgeAtLast
			self.regularizationRate = regularization.rate
		else:  # if isinstance(regularization, Lasso):
			self.gradient = self.gradientLasso
			self.gradientAtLast = self.gradientLassoAtLast
			self.regularizationRate = regularization.rate

	def getFinalWeight(self) -> np.ndarray:
		return np.concatenate((self.B, self.W), axis=1)

	def evaluateTrain(self, y: np.ndarray):
		self.f(np.add(self.B, np.matmul(self.W, self.trainX, self.trainN), self.trainN), y)

	def evaluateTest(self, y: np.ndarray):
		self.f(np.add(self.B, np.matmul(self.W, self.testX, self.testN), self.testN), y)

	def evaluate(self, x: np.ndarray, y: np.ndarray = None, activation: bool = True) -> np.ndarray:
		if y is None:
			y = np.empty((self.ydim, x.shape[1]))
		if activation:
			N = np.empty((self.ydim, x.shape[1]))
			self.f(np.add(self.B, np.matmul(self.W, x, N), N), y)
		else:
			np.add(self.B, np.matmul(self.W, x, y), y)
		return y

	def gradientPlain(self, y: np.ndarray, WDelta: np.ndarray):
		self.df_y(y, self.Delta)
		self.Delta *= WDelta
		np.matmul(self.W.T, self.Delta, self.WDelta)
		if self.weightType == WeightType.GENERAL:
			if self.biased:
				np.sum(self.Delta, axis=1, out=self.gradB, keepdims=True)
			np.matmul(self.Delta, self.trainX.T, self.gradW)

	def gradientPlainAtLast(self, y: np.ndarray, trainY: np.ndarray, lossType: int):
		np.subtract(trainY, y, self.Delta)
		if lossType == LossType.L2:
			DeltaLike = np.empty_like(self.Delta)
			self.df_y(y, DeltaLike)
			self.Delta *= DeltaLike
		# elif self.lossType == LossFunc.MULTI_CROSS_ENTROPY or self.lossType == LossFunc.MONO_CROSS_ENTROPY:
		# 	# MUST BE: The last activation function is the logistic sigmoid if loss function is MULT_CROSS_ENTROPY
		# 	# MUST BE: The last activation function is the softmax if loss function is SINGLE_CROSS_ENTROPY
		# 	pass
		np.matmul(self.W.T, self.Delta, self.WDelta)
		if self.weightType == WeightType.GENERAL:
			if self.biased:
				np.sum(self.Delta, axis=1, out=self.gradB, keepdims=True)
			np.matmul(self.Delta, self.trainX.T, self.gradW)

	def gradientRidge(self, y: np.ndarray, WDelta: np.ndarray):
		self.gradientPlain(y, WDelta)
		# self.gradB -= np.multiply(self.regularizationRate, self.B, self.gradBLike)
		self.gradW -= np.multiply(self.regularizationRate, self.W, self.gradWLike)

	def gradientRidgeAtLast(self, y: np.ndarray, trainY: np.ndarray, lossType: int):
		self.gradientPlainAtLast(y, trainY, lossType)
		# self.gradB -= np.multiply(self.regularizationRate, self.B, self.gradBLike)
		self.gradW -= np.multiply(self.regularizationRate, self.W, self.gradWLike)

	def gradientLasso(self, y: np.ndarray, WDelta: np.ndarray):
		self.gradientPlain(y, WDelta)
		# self.gradB -= np.multiply(self.regularizationRate, np.sign(self.B, self.gradBLike), self.gradBLike)
		self.gradW -= np.multiply(self.regularizationRate, np.sign(self.W, self.gradWLike), self.gradWLike)

	def gradientLassoAtLast(self, y: np.ndarray, trainY: np.ndarray, lossType: int):
		self.gradientPlainAtLast(y, trainY, lossType)
		# self.gradB -= np.multiply(self.regularizationRate, np.sign(self.B, self.gradBLike), self.gradBLike)
		self.gradW -= np.multiply(self.regularizationRate, np.sign(self.W, self.gradWLike), self.gradWLike)

	def updateBatch(self, learningRate: float, momentum: float):
		if self.weightType != WeightType.FIXED:
			# self.gradBPrev = self.momentum * self.gradBPrev + self.learningRate * self.gradB
			# self.gradWPrev = self.momentum * self.gradWPrev + self.learningRate * self.gradW
			self.gradBPrev *= momentum
			self.gradWPrev *= momentum
			self.gradBPrev += np.multiply(learningRate, self.gradB, self.gradBLike)
			self.gradWPrev += np.multiply(learningRate, self.gradW, self.gradWLike)
			self.B += self.gradBPrev
			self.W += self.gradWPrev

	def updateAdaptive(self, learningRate: float):
		# self.B = self.BPrev + learningRate * self.gradBPrev
		# self.W = self.WPrev + learningRate * self.gradWPrev
		np.add(self.BPrev, np.multiply(learningRate, self.gradBPrev, self.B), self.B)
		np.add(self.WPrev, np.multiply(learningRate, self.gradWPrev, self.W), self.W)

	def sendCurrentToPrevious(self):
		self.B, self.BPrev = self.BPrev, self.B
		self.W, self.WPrev = self.WPrev, self.W
		self.gradB, self.gradBPrev = self.gradBPrev, self.gradB
		self.gradW, self.gradWPrev = self.gradWPrev, self.gradW

	def gradientDot(self) -> float:
		return np.sum(self.gradB * self.gradBPrev) + np.sum(self.gradW * self.gradWPrev)


class Sequence(object):
	def __init__(self, learnType: int = LearnType.ADAPTIVE, path: str = None, file: TextIO = None, weight: str = 'initial'):
		self.learnType: int = learnType
		self.lossType: int = LossType.L2
		self.lossMax: float = 0.001
		self.learningRate: float = 0.1
		self.momentum: float = 0.1
		self.regularization = None
		self.categorical: bool = True  # problem is categorical or fitting
		self.pairY: list = None
		self.epochMax: int = 1000
		self.layers: list[Layer] = None
		self.trainX: np.ndarray = None
		self.trainY: np.ndarray = None
		self.testX: np.ndarray = None
		self.testY: np.ndarray = None
		self.trainLoss: np.ndarray = None
		self.testLoss: np.ndarray = None
		self.appliedRate: np.ndarray = None
		self.depth: int = 0
		self.trainOutput: np.ndarray = None
		self.testOutput: np.ndarray = None
		if path is not None or file is not None:
			self.readFullInfo(path=path, file=file, weight=weight)

	def initialize(self):
		assert self.trainX is not None, 'trainX is None'
		assert self.trainY is not None, 'trainY is None'
		assert self.trainX.shape[1] == self.trainY.shape[1], 'mismatch data size: trainX.shape[1]={0}, trainY.shape[1]={1}'.format(self.trainX.shape[1], self.trainY.shape[1])
		self.trainOutput = np.empty_like(self.trainY)
		self.testOutput = np.empty_like(self.testY)
		xdim = self.trainX.shape[0]
		for layer in self.layers:
			layer.initialize(self.learnType, self.regularization, xdim, self.trainX.shape[1], 0 if self.testX is None else self.testX.shape[1])
			xdim = layer.ydim
		assert self.trainY.shape[0] == self.layers[-1].ydim, 'mismatch output dim: trainY.shape[0]={0}, output.shape[0]={1}'.format(self.trainY.shape[1], self.layers[-1].ydim)
		if self.pairY is None:
			self.pairY = list(range(self.trainY.shape[0]))
		else:
			temp = []
			for el in self.pairY:
				temp += el if isinstance(el, (list, tuple)) else [el]
			assert temp == list(range(self.trainY.shape[0])), '"pairY" is a rearrangement of %s.' % str(list(range(self.trainY.shape[0])))
		if self.lossType == LossType.MONO_CROSS_ENTROPY and self.trainY.shape[0] == 1:
			self.lossType = LossType.MULTI_CROSS_ENTROPY
		if self.lossType == LossType.MONO_CROSS_ENTROPY:
			self.layers[-1].f = softmax
		elif self.lossType == LossType.MULTI_CROSS_ENTROPY:
			self.layers[-1].f = sigmoid

	def _addLayer(self, layer: Layer):
		if self.layers is None:
			self.layers = [layer]
		else:
			self.layers.append(layer)
		self.depth += 1

	def addLayer(self, node: int = None, weight: np.ndarray = None, weightType: int = WeightType.GENERAL, biased: bool = True, activation=sigmoid):
		self._addLayer(Layer(node, weight, weightType, biased, activation))

	def addLayers(self, nodes=None, weights=None):
		if nodes is None:
			assert weights is not None, 'either nodes or weights is not None'
			for weight in weights:
				self.addLayer(weight=weight)
		elif isinstance(nodes, (tuple, list)):
			if weights is None:
				for node in nodes:
					self.addLayer(node=node)
			else:
				assert len(nodes) == len(weights), 'mismatch between len(nodes)={0} and len(weights)={1}'.format(len(nodes), len(weights))
				for node, weight in zip(nodes, weights):
					self.addLayer(node=node, weight=weight)
		else:  # if isinstance(nodes, int):
			if weights is None:
				self.addLayer(node=nodes)
			else:
				if weights is (tuple, list):
					assert len(weights) == 1, 'mismatch between nodes and weights'
					self.addLayer(node=nodes, weight=weights[0])
				else:
					assert isinstance(weights, np.ndarray), 'weights is either np.ndarray or list/tuple of np.ndarray'
					self.addLayer(node=nodes, weight=weights)

	def setTrainData(self, x: np.ndarray, y: np.ndarray):
		self.trainX = x.reshape(1, x.shape[0]).astype(float) if x.ndim == 1 else x.T.astype(float)
		self.trainY = y.reshape(1, y.shape[0]).astype(float) if y.ndim == 1 else y.T.astype(float)

	# use carefully. deviced for reinforcement learning. see setTrainData
	def setTrainDataDirectly(self, x: np.ndarray, y: np.ndarray):
		self.trainX = x
		self.trainY = y

	def setTestData(self, x: np.ndarray, y: np.ndarray):
		self.testX = x.reshape(1, x.shape[0]).astype(float) if x.ndim == 1 else x.T.astype(float)
		self.testY = y.reshape(1, y.shape[0]).astype(float) if y.ndim == 1 else y.T.astype(float)

	# use carefully. deviced for reinforcement learning. see setTrainData
	def setTestDataDirectly(self, x: np.ndarray, y: np.ndarray):
		self.testX = x
		self.testY = y

	def getNodes(self) -> List[int]:
		return [layer.ydim for layer in self.layers]

	def getActivationFuncs(self) -> List[Callable]:
		return [layer.f for layer in self.layers]

	def setInitialWeights(self, weights):
		assert len(weights) == len(self.layers), 'len(weights)={0} does not coincide len(layers)={1}'.format(len(weights), len(self.layers))
		for weight, layer in zip(weights, self.layers):
			layer.initialWeight = weight

	def getInitialWeights(self) -> List[np.ndarray]:
		return [layer.initialWeight for layer in self.layers]

	def getFinalWeights(self) -> List[np.ndarray]:
		return [layer.getFinalWeight() for layer in self.layers]

	def getFinalTrainLoss(self) -> float:
		return self.trainLoss[-1]

	# weight is either 'initial' or 'final'
	def readFullInfo(self, path: str = None, file: TextIO = None, weight: str = 'initial'):
		f = open(path, 'r') if file is None else file
		self.readBasicInfo(file=f)
		s = 'initial weights' if weight[0] == 'i' else 'final weights'
		line = f.readline().strip()
		# empty line is not permitted
		while len(line) > 1 and not line.startswith(s):
			line = f.readline().strip()
		if len(line) > 1:
			self.setInitialWeights(readWeight(file=f))
		if file is None:
			f.close()

	def readBasicInfo(self, path: str = None, file: TextIO = None):
		import ast
		f = open(path, 'r') if file is None else file
		while True:
			line = f.readline()
			if line.startswith('learn type'):
				break
		self.learnType = LearnType.__dict__[line[line.find(':') + 2:].strip()]
		line = f.readline()
		self.lossType = LossType.__dict__[line[line.find(':') + 2:].strip()]
		line = f.readline()
		self.lossMax = float(line[line.find(':') + 1:])
		line = f.readline()
		self.learningRate = float(line[line.find(':') + 1:])
		line = f.readline()
		self.momentum = float(line[line.find(':') + 1:])
		line = f.readline()
		ss = line[line.find(':') + 2:].split()
		self.regularization = None if len(ss) == 1 else eval('%s(%s)' % (ss[0], ss[1]))
		line = f.readline()
		self.categorical = ast.literal_eval(line[line.find(':') + 2:])
		line = f.readline()
		while '  ' in line:
			line = line.replace('  ', ' ')
		self.pairY = ast.literal_eval(line[line.find(':') + 2:].replace(' ', ','))
		line = f.readline()
		self.epochMax = int(line[line.find(':') + 1:])

		line = f.readline()
		nodes = [int(c) for c in line[line.find('[') + 1:line.find(']')].split()]
		line = f.readline()
		biased = [c == 'True' for c in line[line.find('[') + 1:line.find(']')].split()]
		line = f.readline()
		weightType = np.array([WeightType.__dict__[c] for c in line[line.find('[') + 1:line.find(']')].split()])
		line = f.readline()
		activations = [None if c == 'None' else eval(c) for c in line[line.find('[') + 1:line.find(']')].split()]
		for i in range(len(nodes)):
			self.addLayer(node=nodes[i], weightType=weightType[i], biased=biased[i], activation=activations[i])
		if file is None:
			f.close()

	def saveFullInfo(self, path: str = None, file: TextIO = None):
		f = open(path, 'w') if file is None else file
		from datetime import datetime
		f.write(str(datetime.now()))
		f.write('\n')
		f.write(self.basicInfo())
		f.write('\ninitial weights:\n')
		saveWeight(self.getInitialWeights(), file=f)
		f.write('\nfinal weights:\n')
		saveWeight(self.getFinalWeights(), file=f)
		f.write('\n')
		f.write(self.summary())
		f.write('\ntrain losses:\n')
		f.write(str(self.trainLoss))
		f.write('\n')
		if self.testX is not None:
			f.write('test losses:\n')
			f.write(str(self.testLoss))
			f.write('\n')
		if file is None:
			f.close()

	def basicInfo(self) -> str:
		def getFloatStr(value):
			return np.format_float_scientific(value) if value < 1e-4 else np.format_float_positional(value)

		c = 'learn type: %s\n' % getField(LearnType, self.learnType)
		c += 'loss type: %s\n' % getField(LossType, self.lossType)
		c += 'lossMax: %s\n' % getFloatStr(self.lossMax)
		c += 'learning rate: %s\n' % getFloatStr(self.learningRate)
		c += 'momentum: %s\n' % getFloatStr(self.momentum)
		if self.regularization is None:
			c += 'regularization: None\n'
		else:
			c += 'regularization: %s %s\n' % (type(self.regularization).__name__, getFloatStr(self.regularization.rate))
		c += 'categorical: %s\n' % str(self.categorical)
		c += 'pair y: %s\n' % str(self.pairY).replace(',', '')
		c += 'epoch max: %d\n' % self.epochMax
		ns = 'nodes: ['
		bs = 'biased: ['
		wts = 'weight types: ['
		afs = 'activations: ['
		for layer in self.layers:
			ns += str(layer.ydim) + ' '
			bs += str(layer.biased) + ' '
			wts += getField(WeightType, layer.weightType) + ' '
			afs += layer.f.__name__ + ' '
		c += '{0}]\n{1}]\n{2}]\n{3}]'.format(ns[:-1], bs[:-1], wts[:-1], afs[:-1])
		return c

	def summary(self) -> str:
		def categoricalMultiDimY(output: np.ndarray, y: np.ndarray):
			label = np.argmax(y, axis=0)
			decision = np.argmax(output, axis=0)
			correct = np.sum(label == decision)
			s = '    correct outputs: {0} / {1} = {2:1.2f}%\n'.format(correct, label.shape[0], correct / label.shape[0] * 100)
			maxValues = np.max(output, axis=0)
			s += '    max under 0.5: %d\n' % np.sum(maxValues < .5)
			minInd = np.argmin(maxValues)
			s += '    min max: index {0}, label {1}, decision {2}, value {3},\n             output {4}'.format(minInd, label[minInd], decision[minInd], maxValues[minInd], output[:, minInd])
			return s

		def categoricalSingleDimY(output: np.ndarray, y: np.ndarray):
			output = output.flatten()
			label = (y.flatten() > .5).astype(np.int)
			decision = (output > .5).astype(np.int)
			correct = np.sum(label == decision)
			s = '    correct outputs: {0} / {1} = {2:1.2f}%\n'.format(correct, label.shape[0], correct / label.shape[0] * 100)
			diff = np.abs(output - .5)
			s += '    values in the interval (0.25, 0.75): %d\n' % np.sum(diff < .25)
			minInd = np.argmin(diff)
			s += '    nearest to 0.5: index {0}, label {1}, decision {2}, value {3}'.format(minInd, label[minInd], decision[minInd], output[minInd])
			return s

		def fitting(output: np.ndarray, y: np.ndarray):
			if self.lossType == LossType.MULTI_CROSS_ENTROPY:
				lossVector = -np.sum(y * np.log(EPSILON + output) + (1 - y) * np.log(1 + EPSILON - output), axis=0) + entropy_multi(y, np.empty_like(y), np.empty_like(y))
			elif self.lossType == LossType.MONO_CROSS_ENTROPY:
				lossVector = -np.sum(y * np.log(EPSILON + output), axis=0) + entropy_mono(y, np.empty_like(y))
			else:
				lossVector = .5 * np.sum((y - output) ** 2, axis=0)
			maxLossInd = np.argmax(lossVector)
			return '    max loss: index {0}, loss {1},\n              y {2},\n              output {3}'.format(maxLossInd, lossVector[maxLossInd], y[:, maxLossInd], output[:, maxLossInd])

		epochStr = 'epoch: %d\n' % (self.trainLoss.shape[0] - 1)
		trainStr = 'train summary\n'
		trainStr += '    final loss: %f\n' % self.trainLoss[-1]
		if self.categorical:
			y = _combinePair(self.trainY, self.pairY)
			trainOutput = _combinePair(self.trainOutput, self.pairY)
			trainStr += categoricalSingleDimY(trainOutput, y) if y.shape[0] == 1 else categoricalMultiDimY(trainOutput, y)
		else:
			trainStr += fitting(self.trainOutput, self.trainY)
		if self.testX is None:
			testStr = ''
		else:
			testStr = '\ntest summary\n'
			testStr += '    final loss: %f\n' % self.testLoss[-1]
			if self.categorical:
				y = _combinePair(self.testY, self.pairY)
				testOutput = _combinePair(self.testOutput, self.pairY)
				testStr += categoricalSingleDimY(testOutput, y) if y.shape[0] == 1 else categoricalMultiDimY(testOutput, y)
			else:
				testStr += fitting(self.testOutput, self.testY)
		return epochStr + trainStr + testStr

	# internal
	def evaluateTrain(self):
		for i in range(self.depth - 1):
			self.layers[i].evaluateTrain(self.layers[i + 1].trainX)
		self.layers[-1].evaluateTrain(self.trainOutput)

	# internal
	def evaluateTest(self):
		for i in range(self.depth - 1):
			self.layers[i].evaluateTest(self.layers[i + 1].testX)
		self.layers[-1].evaluateTest(self.testOutput)

	# internal or reinforcement learning
	def evaluate(self, x: np.ndarray, depth: int = None, lastActivation: bool = True) -> np.ndarray:
		if depth is None:
			depth = self.depth
		for i in range(depth - 1):
			x = self.layers[i].evaluate(x)
		return self.layers[depth - 1].evaluate(x, None, lastActivation)

	# external
	def output(self, x: np.ndarray = None, depth: int = None, lastActivation: bool = True) -> np.ndarray:
		if x is None:
			x = self.trainX.T
		if x.ndim == 1:
			if self.layers[0].xdim == 1:
				# x is a list of one dimensional vector
				x = x.reshape(1, x.shape[0]).astype(float)
			else:
				# x is a vector
				x = x.reshape(x.shape[0], 1).astype(float)
		else:
			x = x.T.astype(float)
		y = self.evaluate(x, depth, lastActivation)
		return y.flatten() if y.shape[0] == 1 or y.shape[1] == 1 else y.T

	# external. apply _combinePair directly for internal use
	def pairCombinedTrainY(self) -> np.ndarray:
		dst = _combinePair(self.trainY, self.pairY)
		return dst.flatten() if dst.shape[0] == 1 or dst.shape[1] == 1 else dst.T

	# external. apply _combinePair directly for internal use
	def pairCombinedOutput(self, x: np.ndarray = None) -> np.ndarray:
		return combinePair(self.output(x), self.pairY)

	# external. apply _combinePair directly for internal use
	def pairCombinedTestY(self) -> np.ndarray:
		dst = _combinePair(self.testY, self.pairY)
		return dst.flatten() if dst.shape[0] == 1 or dst.shape[1] == 1 else dst.T

	# external. apply _combinePair directly for internal use
	def pairCombinedTestOutput(self) -> np.ndarray:
		return combinePair(self.output(self.testX.T), self.pairY)

	def _updateBatch(self):
		lastLayer = self.layers[-1]
		lastLayer.gradientAtLast(self.trainOutput, self.trainY, self.lossType)
		for i in range(self.depth - 2, -1, -1):
			self.layers[i].gradient(lastLayer.trainX, lastLayer.WDelta)
			lastLayer = self.layers[i]
		for layer in self.layers:
			layer.updateBatch(self.learningRate, self.momentum)

	RATE_DOWN = 0.25  # .5
	# RATE_DOWN = 0.9  # .5
	RATE_UP = 1.1  # 2

	def _updateAdaptive(self, rate: float) -> float:
		for layer in self.layers:
			layer.sendCurrentToPrevious()
		rate *= Sequence.RATE_UP
		if rate > self.learningRate:
			rate = self.learningRate
		while True:
			for i in range(self.depth - 1):
				self.layers[i].updateAdaptive(rate)
				self.layers[i].evaluateTrain(self.layers[i + 1].trainX)
			layer = self.layers[-1]
			layer.updateAdaptive(rate)
			layer.evaluateTrain(self.trainOutput)
			layer.gradientAtLast(self.trainOutput, self.trainY, self.lossType)
			dot = layer.gradientDot()
			for i in range(self.depth - 2, -1, -1):
				self.layers[i].gradient(layer.trainX, layer.WDelta)
				layer = self.layers[i]
				dot += layer.gradientDot()
			if dot >= 0:
				break
			rate *= Sequence.RATE_DOWN
		return rate

	def build(self):
		self.initialize()
		self.buildNext()

	def buildNext(self):
		if self.learnType == LearnType.ADAPTIVE:
			self._buildAdaptive()
		else:  # if self.learnType == LearnType.BATCH:
			self._buildBatch()

	def _buildBatch(self):
		lossFunc = l2 if self.lossType == LossType.L2 else crossEntropy_mono if self.lossType == LossType.MONO_CROSS_ENTROPY else crossEntropy_multi
		self.layers[0].trainX = self.trainX
		trainLoss = np.empty(self.epochMax + 1)
		trainYLike1 = np.empty_like(self.trainY)
		trainYLike2 = np.empty_like(self.trainY)
		trainEntropy = None if self.lossType == LossType.L2 else entropy_mono(self.trainY, trainYLike1) if self.lossType == LossType.MONO_CROSS_ENTROPY else entropy_multi(self.trainY, trainYLike1, trainYLike2)
		if self.testX is not None:
			self.layers[0].testX = self.testX
			testLoss = np.empty(self.epochMax + 1)
			testYLike1 = np.empty_like(self.testY)
			testYLike2 = np.empty_like(self.testY)
			testEntropy = None if self.lossType == LossType.L2 else entropy_mono(self.testY, testYLike1) if self.lossType == LossType.MONO_CROSS_ENTROPY else entropy_multi(self.testY, testYLike1, testYLike2)
		epoch = 0
		while True:
			self.evaluateTrain()
			trainLoss[epoch] = lossFunc(self.trainY, self.trainOutput, trainEntropy, trainYLike1, trainYLike2)
			if self.testX is not None:
				self.evaluateTest()
				testLoss[epoch] = lossFunc(self.testY, self.testOutput, testEntropy, testYLike1, testYLike2)
			if trainLoss[epoch] < self.lossMax or epoch == self.epochMax:
				break
			self._updateBatch()
			epoch += 1
		self.trainLoss = trainLoss[:epoch + 1]
		if self.testX is not None:
			self.testLoss = testLoss[:epoch + 1]
		self.appliedRate = [self.learningRate]

	def _buildAdaptive(self):
		lossFunc = l2 if self.lossType == LossType.L2 else crossEntropy_mono if self.lossType == LossType.MONO_CROSS_ENTROPY else crossEntropy_multi
		self.layers[0].trainX = self.trainX
		appliedRate = np.empty(self.epochMax + 1)
		trainLoss = np.empty(self.epochMax + 1)
		trainYLike1 = np.empty_like(self.trainY)
		trainYLike2 = np.empty_like(self.trainY)
		trainEntropy = None if self.lossType == LossType.L2 else entropy_mono(self.trainY, trainYLike1) if self.lossType == LossType.MONO_CROSS_ENTROPY else entropy_multi(self.trainY, trainYLike1, trainYLike2)
		if self.testX is not None:
			self.layers[0].testX = self.testX
			testLoss = np.empty(self.epochMax + 1)
			testYLike1 = np.empty_like(self.testY)
			testYLike2 = np.empty_like(self.testY)
			testEntropy = None if self.lossType == LossType.L2 else entropy_mono(self.testY, testYLike1) if self.lossType == LossType.MONO_CROSS_ENTROPY else entropy_multi(self.testY, testYLike1, testYLike2)
		epoch = 0
		appliedRate[0] = self.learningRate
		# self.evaluateTrain()
		for i in range(self.depth - 1):
			self.layers[i].evaluateTrain(self.layers[i + 1].trainX)
		layer = self.layers[-1]
		layer.evaluateTrain(self.trainOutput)
		layer.gradientAtLast(self.trainOutput, self.trainY, self.lossType)
		for i in range(self.depth - 2, -1, -1):
			self.layers[i].gradient(layer.trainX, layer.WDelta)
			layer = self.layers[i]
		trainLoss[epoch] = lossFunc(self.trainY, self.trainOutput, trainEntropy, trainYLike1, trainYLike2)
		if self.testX is not None:
			self.evaluateTest()
			testLoss[epoch] = lossFunc(self.testY, self.testOutput, testEntropy, testYLike1, testYLike2)
		while trainLoss[epoch] >= self.lossMax and epoch < self.epochMax:
			appliedRate[epoch + 1] = self._updateAdaptive(appliedRate[epoch])
			epoch += 1
			trainLoss[epoch] = lossFunc(self.trainY, self.trainOutput, trainEntropy, trainYLike1, trainYLike2)
			if self.testX is not None:
				self.evaluateTest()
				testLoss[epoch] = lossFunc(self.testY, self.testOutput, testEntropy, testYLike1, testYLike2)
		self.appliedRate = appliedRate[1:epoch + 1]
		self.trainLoss = trainLoss[:epoch + 1]
		if self.testX is not None:
			self.testLoss = testLoss[:epoch + 1]
