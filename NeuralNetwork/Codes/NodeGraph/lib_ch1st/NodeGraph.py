from typing import Tuple

from .Activation import *
from .Dense import *
from .Flatten import *
from .Loss import *
from .Convolution import *
from .Pooling import *
from .Regularizer import *
from .Util import *


class NodeGraph(metaclass=ABCMeta):
	def __init__(self):
		self.startSets: Tuple = ()
		self.fitSets: Tuple = ()
		self.acts: Tuple = ()
		self.etc: Tuple = ()
		self.lossFunc: Union[LossFunc1D, None] = None
		self.midSets: Tuple = ()  # midSets = fitSets or acts or pools

		self.lossMax: float = 0.0001
		self.epochMax: int = 1000
		self.reg: Regularizer = RegNone()  # regularizer

		self.trX: Union[np.ndarray, Tuple, List, None] = None
		self.trT: Union[np.ndarray, None] = None
		self.teX: Union[np.ndarray, Tuple, List, None] = None
		self.teT: Union[np.ndarray, None] = None

		self.trAccuracy: Union[np.ndarray, None] = None
		self.teAccuracy: Union[np.ndarray, None] = None
		self.trLosses: Union[np.ndarray, None] = None
		self.teLosses: Union[np.ndarray, None] = None

	def add(self, nodeSet: NodeSet, *prevSets: NodeSet):
		if isinstance(nodeSet, StartSet1D) or isinstance(nodeSet, StartSet2D):
			self.startSets += (nodeSet,)
		elif isinstance(nodeSet, FitSet1D) or isinstance(nodeSet, FitSet2D):
			self.midSets += (nodeSet,)
			self.fitSets += (nodeSet,)
			nodeSet.addPrevSets(*prevSets)
		elif isinstance(nodeSet, Activation1D) or isinstance(nodeSet, Activation2D):
			self.midSets += (nodeSet,)
			self.acts += (nodeSet,)
			nodeSet.addPrevSets(*prevSets)
		elif isinstance(nodeSet, MaxPool1D) or isinstance(nodeSet, Flat1D) or isinstance(nodeSet, MaxPool2D) or isinstance(nodeSet, Flat2D):
			self.midSets += (nodeSet,)
			self.etc += (nodeSet,)
			nodeSet.addPrevSets(*prevSets)
		elif isinstance(nodeSet, LossFunc1D):
			self.lossFunc = nodeSet
			nodeSet.addPrevSets(*prevSets)
		else:
			raise Exception
		return nodeSet

	def compile(self):
		compiled = False
		while not compiled:
			compiled = True
			for mset in self.midSets:
				if not mset.compile():
					compiled = False
			if not self.lossFunc.compile():
				compiled = False

	def prePropBoth(self, trSiz, teSiz):
		for mset in self.midSets:
			mset.prePropBoth(trSiz, teSiz)
		self.lossFunc.prePropBoth(trSiz, teSiz)

	def prePropTr(self, trSiz):
		for mset in self.midSets:
			mset.prePropTr(trSiz)
		self.lossFunc.prePropTr(trSiz)

	def prePropTe(self, teSiz):
		for mset in self.midSets:
			mset.prePropTe(teSiz)
		self.lossFunc.prePropTe(teSiz)

	def prePropPr(self, prSiz):
		for mset in self.midSets:
			mset.prePropPr(prSiz)
		self.lossFunc.prePropPr(prSiz)

	def pushBoth(self, trXFrag: Union[np.ndarray, Tuple, List], teXFrag: Union[np.ndarray, Tuple, List]):
		for mset in self.midSets:
			mset.resetForPush()
		if isinstance(trXFrag, np.ndarray):
			for sset in self.startSets:
				sset.pushBoth(trXFrag, teXFrag)
		else:
			for sset, trx, tex in zip(self.startSets, trXFrag, teXFrag):
				sset.pushBoth(trx, tex)

	def pushTr(self, trXFrag: Union[np.ndarray, Tuple, List]):
		for mset in self.midSets:
			mset.resetForPush()
		if isinstance(trXFrag, np.ndarray):
			for sset in self.startSets:
				sset.pushTr(trXFrag)
		else:
			for sset, trx in zip(self.startSets, trXFrag):
				sset.pushTr(trx)

	def pushTe(self, teXFrag: Union[np.ndarray, Tuple, List]):
		for mset in self.midSets:
			mset.resetForPush()
		if isinstance(teXFrag, np.ndarray):
			for sset in self.startSets:
				sset.pushTe(teXFrag)
		else:
			for sset, trx in zip(self.startSets, teXFrag):
				sset.pushTe(trx)

	def pushPr(self, prX: Union[np.ndarray, Tuple, List]):
		for mset in self.midSets:
			mset.resetForPush()
		if isinstance(prX, np.ndarray):
			for sset in self.startSets:
				sset.pushPr(prX)
		else:
			for sset, prx in zip(self.startSets, prX):
				sset.pushPr(prx)

	def pullGrad(self):
		for mset in self.midSets:
			mset.resetForPull()
		self.lossFunc.pullGrad()

	def fit(self, trX: Union[np.ndarray, Tuple, List], trT: np.ndarray,
			teX: Union[np.ndarray, Tuple, List] = None, teT: np.ndarray = None,
			**kwargs):
		self.trX, self.trT, self.teX, self.teT = trX, trT, teX, teT

		# set attributes:learnType, lossMax, and so on if any
		for k, v in kwargs.items():
			if v is not None:
				self.__setattr__(k, v)

	def predict(self, prX: Union[np.ndarray, tuple, list], clear: bool = False) -> np.ndarray:
		"""
		:param prX: either array, tuple of arrays or list of arrays
		:param clear: clear data for predict
		:return: prediction
		"""
		prSiz = prX.shape[-1] if isinstance(prX, np.ndarray) else prX[0].shape[-1]
		self.prePropPr(prSiz)
		self.pushPr(prX)
		if clear:
			self.clearPredict()
		return self.lossFunc.prY

	def clearPredict(self):
		for sset in self.startSets:
			sset.prX = None
		for mset in self.midSets:
			mset.prX = None
			mset.prY = None
		self.lossFunc.prY = None

	def graphInfo(self, form: str = 'short') -> str:
		"""
		:param form: one of 'long', 'save', 'short'
		:return: string of graph information
		"""

		c = f'loss maximum: {getFloatStr(self.lossMax)}\n'
		c += f'regularizer: {str(self.reg)}\n'
		c += f'epoch maximum: {self.epochMax}\n'
		if form == 'save':
			s = ''
			for sset in self.startSets:
				s += f'{sset.saveStr()}, '
			c += f'start sets: {s[:-2]}\n'
			s = ''
			for mset in self.midSets:
				s += f'{mset.saveStr()}, '
			c += f'mid sets: {s[:-2]}\n'
			c += f'loss function: {self.lossFunc.saveStr()}'
		else:
			s = ''
			for sset in self.startSets:
				s += f'{sset.shortStr() if form == "short" else sset}, '
			c += f'start sets: {s[:-2]}\n'
			s = ''
			for fset in self.fitSets:
				s += f'{fset.shortStr() if form == "short" else fset}, '
			c += f'fit sets: {s[:-2]}\n'
			s = ''
			for act in self.acts:
				s += f'{act.shortStr() if form == "short" else act}, '
			c += f'ativations: {s[:-2]}\n'
			s = ''
			for pol in self.etc:
				s += f'{pol.shortStr() if form == "short" else pol}, '
			c += f'etc: {s[:-2]}\n'
			c += f'lossfunction: {self.lossFunc.shortStr() if form == "short" else self.lossFunc}'
		return c

	def summary(self, categorical=True) -> str:
		def catNDY(y: np.ndarray, t: np.ndarray):
			ry = y.reshape((-1, y.shape[2]))
			rt = t.reshape((-1, t.shape[2]))
			label, decision = np.argmax(rt, axis=0), np.argmax(ry, axis=0)
			correct = np.sum(label == decision)
			s = f'    correct outputs: {correct} / {label.shape[0]} = {correct / label.shape[0] * 100.0:1.2f}%\n'
			maxVal = np.max(ry, axis=0)
			s += f'    max under 0.5: {np.sum(maxVal < .5)}\n'
			minInd = np.argmin(maxVal)
			s += f'    min max: index {minInd}, label {label[minInd]}, decision {decision[minInd]}, value {maxVal[minInd]},\n'
			s += f'             output {y[:, :, minInd]}'
			return s

		def cat1DY(y: np.ndarray, t: np.ndarray):
			y = y.flatten()
			label, decision = (t.flatten() > .5).astype(np.int), (y > .5).astype(np.int)
			correct = np.sum(label == decision)
			s = f'    correct outputs: {correct} / {label.shape[0]} = {correct / label.shape[0] * 100.0:1.2f}%\n'
			diff = np.abs(y - .5)
			s += f'    values in the interval (0.25, 0.75): {np.sum(diff < .25)}\n'
			minInd = np.argmin(diff)
			s += f'    nearest to 0.5: index {minInd}, label {label[minInd]}, decision {decision[minInd]}, value {y[minInd]}'
			return s

		def fitting(y, t, lossVec):
			maxInd = np.argmax(lossVec)
			return f'    max loss: index {maxInd}, loss {lossVec[maxInd]},\n{"":14}y {t[maxInd, :]},\n{"":14}output {y[maxInd, :]}'

		epochStr = f'epochs: {self.trLosses.shape[0] - 1}\n'
		trStr = 'train result\n'
		trStr += f'    last accuracy(average if minibatch): {self.trAccuracy[-1]}\n'
		trStr += f'    last loss(average if minibatch): {self.trLosses[-1]}\n'
		trY = self.predict(self.trX)
		self.lossFunc.trT = self.trT
		if categorical:
			if self.trT.shape[0] == 1 and self.trT.shape[1] == 1:
				trStr += cat1DY(trY, self.trT)
			else:
				trStr += catNDY(trY, self.trT)
		else:
			trStr += fitting(trY, self.trT, self.lossFunc.lossVec(self.trT, trY) - self.lossFunc.baseVec(self.trT))
		if self.teT is None:
			teStr = ''
		else:
			teStr = '\ntest result\n'
			trStr += f'    last accuracy: {self.teAccuracy[-1]}\n'
			teStr += f'    last loss: {self.teLosses[-1]}\n'
			if categorical:
				if self.trT.shape[1] == 1:
					teStr += cat1DY(self.lossFunc.teY, self.teT)
				else:
					teStr += catNDY(self.lossFunc.teY, self.teT)
			else:
				teStr += fitting(self.lossFunc.teY, self.teT, self.lossFunc.lossVec(self.teT, self.lossFunc.teY) - self.lossFunc.baseVec(self.teT))
		return epochStr + trStr + teStr


class NodeGraphAdaptive(NodeGraph):
	def __init__(self):
		super().__init__()

		self._lr: Union[float, None] = None  # learning rate
		self.lrInit: Union[float, None] = None
		self.lrMax: float = 10.0
		self.lrMin: float = 1.0E-12
		self.lrUp = 1.1
		self.lrDown = .9
		self.appLrs: Union[np.ndarray, None] = None  # applied learning rates

	def __update(self):
		for fset in self.fitSets:
			fset.sendCurrToPrev()
		self._lr *= self.lrUp
		if self._lr > self.lrMax:
			self._lr = self.lrMax
		while True:
			for fset in self.fitSets:
				fset.updateAdaptive(self._lr)
			self.pushTr(self.trX)
			self.pullGrad()
			dot = 0.0
			for fset in self.fitSets:
				dot += fset.gradDot()
			if dot >= 0.0 or self._lr <= self.lrMin:
				break
			self._lr *= self.lrDown
		return dot

	def __fitTrain(self, verbose: int):
		# applied learning rates
		appLrs = np.empty(self.epochMax + 1)
		trAccuracy = np.empty(self.epochMax + 1)
		trLosses = np.empty(self.epochMax + 1)

		trLossBase = self.lossFunc.base(self.trT)
		trSiz = self.trX.shape[-1] if isinstance(self.trX, np.ndarray) else self.trX[0].shape[-1]
		self.prePropTr(trSiz)

		epoch = 0
		while True:
			self.__update()  # executes pushTr()
			appLrs[epoch] = self._lr
			trAccuracy[epoch] = self.lossFunc.trAccuracy()
			trLosses[epoch] = (self.lossFunc.trLoss() - trLossBase) / trSiz
			if verbose > 0:
				print(f'epoch: {epoch}/{self.epochMax}, train accuracy: {trAccuracy[epoch]}')
				if verbose > 1:
					print(f'\t\t\tlearning rate: {appLrs[epoch]}, train loss: {trLosses[epoch]}')
			if trLosses[epoch] < self.lossMax or epoch == self.epochMax:
				break
			epoch += 1
		self.appLrs = appLrs[:epoch + 1]
		self.trAccuracy = trAccuracy[:epoch + 1]
		self.trLosses = trLosses[:epoch + 1]

	def __fitWithTest(self, verbose: int):
		# applied learning rates
		appLrs = np.empty(self.epochMax + 1)
		trAccuracy = np.empty(self.epochMax + 1)
		teAccuracy = np.empty(self.epochMax + 1)
		trLosses = np.empty(self.epochMax + 1)
		teLosses = np.empty(self.epochMax + 1)

		trLossBase, teLossBase = self.lossFunc.base(self.trT), self.lossFunc.base(self.teT)
		trSiz, teSiz = (self.trX.shape[-1], self.teX.shape[-1]) if isinstance(self.trX, np.ndarray) else (self.trX[0].shape[-1], self.teX[0].shape[-1])
		self.prePropBoth(trSiz, teSiz)

		epoch = 0
		while True:
			self.__update()  # executes pushTrain()
			appLrs[epoch] = self._lr
			trAccuracy[epoch] = self.lossFunc.trAccuracy()
			trLosses[epoch] = (self.lossFunc.trLoss() - trLossBase) / trSiz
			self.pushTe(self.teX)
			teAccuracy[epoch] = self.lossFunc.teAccuracy()
			teLosses[epoch] = (self.lossFunc.teLoss() - teLossBase) / teSiz
			if verbose > 0:
				print(f'epoch: {epoch}/{self.epochMax}, train accuracy: {trAccuracy[epoch]}, test accuracy: {teAccuracy[epoch]}')
				if verbose > 1:
					print(f'\t\t\tlearning rate: {appLrs[epoch]}, train loss: {trLosses[epoch]}, test loss: {teLosses[epoch]}')
			if trLosses[epoch] < self.lossMax or epoch == self.epochMax:
				break
			epoch += 1
		self.appLrs = appLrs[:epoch + 1]
		self.trAccuracy = trAccuracy[:epoch + 1]
		self.teAccuracy = teAccuracy[:epoch + 1]
		self.trLosses = trLosses[:epoch + 1]
		self.teLosses = teLosses[:epoch + 1]

	def fit(self, trX: Union[np.ndarray, Tuple, List], trT: np.ndarray,
			teX: Union[np.ndarray, Tuple, List] = None, teT: np.ndarray = None,
			verbose: int = 0, **kwargs):
		super().fit(trX, trT, teX, teT, **kwargs)

		if self.lrInit is None:
			self._lr = self.lrMax
		else:
			self._lr = self.lrInit
		for fset in self.fitSets:
			fset.preFit(regularizer=self.reg, momentum=None)

		self.lossFunc.trT = trT
		self.lossFunc.teT = teT
		if teT is None:
			# train only
			self.__fitTrain(verbose)
		else:
			# train and test
			self.__fitWithTest(verbose)

	def graphInfo(self, form: str = 'short') -> str:
		c = f'{self.__class__.__name__}\n'
		c += f'learning rate init: {self.lrInit if self.lrInit is None else getFloatStr(self.lrInit)}\n'
		c += f'learning rate: {getFloatStr(self._lr)}\n'
		c += f'learning rate min, max: [{getFloatStr(self.lrMin)}, {getFloatStr(self.lrMax)}]\n'
		c += f'learning rate up, down: [{getFloatStr(self.lrUp)}, {getFloatStr(self.lrDown)}]\n'
		return c + super().graphInfo(form)


class NodeGraphAdaptiveMinibatch(NodeGraph):
	def __init__(self):
		super().__init__()

		self.batchSize: int = UNKNOWN

		self._lr: Union[float, None] = None  # learning rate
		self.lrInit: Union[float, None] = None
		self.lrMax: float = 10.0
		self.lrMin: float = 1.0E-12
		self.lrUp = 1.1
		self.lrDown = .9
		self.appLrs: Union[np.ndarray, None] = None  # applied learning rates

	def __update(self, currTrX):
		for fset in self.fitSets:
			fset.sendCurrToPrev()
		self._lr *= self.lrUp
		if self._lr > self.lrMax:
			self._lr = self.lrMax
		while True:
			for fset in self.fitSets:
				fset.updateAdaptive(self._lr)
			self.pushTr(currTrX)
			self.pullGrad()
			dot = 0.0
			for fset in self.fitSets:
				dot += fset.gradDot()
			if dot >= 0.0 or self._lr <= self.lrMin:
				break
			self._lr *= self.lrDown
		return dot

	def __fitTrain(self, verbose: int):
		# applied learning rates
		appLrs = np.zeros(self.epochMax + 1)
		trAccuracy = np.zeros(self.epochMax + 1)
		trLosses = np.zeros(self.epochMax + 1)

		trLossBase = self.lossFunc.base(self.trT)
		trSiz = self.trX.shape[-1] if isinstance(self.trX, np.ndarray) else self.trX[0].shape[-1]
		batchCount = trSiz // self.batchSize
		# number of data joining each train step
		trSizNet = batchCount * self.batchSize
		self.prePropTr(self.batchSize)

		epoch = 0
		while True:
			perm = np.random.permutation(trSiz)
			start = 0
			trAccurateCount = 0
			for _ in range(batchCount):
				end = start + self.batchSize
				self.lossFunc.trT = self.trT[:, :, perm[start:end]]
				if self.trX.ndim == 3:
					self.__update(self.trX[:, :, perm[start:end]])
				else:
					self.__update(self.trX[:, :, :, perm[start:end]])
				appLrs[epoch] += self._lr
				trAccurateCount += self.lossFunc.trAccurateCount()
				trLosses[epoch] += self.lossFunc.trLoss()
				start = end
			appLrs[epoch] /= batchCount
			trAccuracy[epoch] = trAccurateCount / trSizNet
			trLosses[epoch] = (trLosses[epoch] - trLossBase) / trSizNet
			if verbose > 0:
				print(f'epoch: {epoch}/{self.epochMax}, train accuracy: {trAccuracy[epoch]}')
				if verbose > 1:
					print(f'\t\t\tlearning rate: {appLrs[epoch]}, train loss: {trLosses[epoch]}')
			if trLosses[epoch] < self.lossMax or epoch == self.epochMax:
				break
			epoch += 1
		self.appLrs = appLrs[:epoch + 1]
		self.trAccuracy = trAccuracy[:epoch + 1]
		self.trLosses = trLosses[:epoch + 1]

	def __fitWithTest(self, verbose: int):
		# applied learning rates
		appLrs = np.zeros(self.epochMax + 1)
		trAccuracy = np.zeros(self.epochMax + 1)
		teAccuracy = np.zeros(self.epochMax + 1)
		trLosses = np.zeros(self.epochMax + 1)
		teLosses = np.zeros(self.epochMax + 1)

		trLossBase, teLossBase = self.lossFunc.base(self.trT), self.lossFunc.base(self.teT)
		trSiz, teSiz = (self.trX.shape[-1], self.teX.shape[-1]) if isinstance(self.trX, np.ndarray) else (self.trX[0].shape[-1], self.teX[0].shape[-1])
		batchCount = trSiz // self.batchSize
		# number of data joining each train step
		trSizNet = batchCount * self.batchSize
		self.prePropBoth(self.batchSize, teSiz)

		epoch = 0
		while True:
			perm = np.random.permutation(trSiz)
			start = 0
			trAccurateCount = 0
			for i in range(batchCount):
				end = start + self.batchSize
				self.lossFunc.trT = self.trT[:, :, perm[start:end]]
				if self.trX.ndim == 3:
					self.__update(self.trX[:, :, perm[start:end]])
				else:
					self.__update(self.trX[:, :, :, perm[start:end]])
				appLrs[epoch] += self._lr
				trAccurateCount += self.lossFunc.trAccurateCount()
				trLosses[epoch] += self.lossFunc.trLoss()
				start = end
			appLrs[epoch] /= batchCount
			trAccuracy[epoch] = trAccurateCount / trSizNet
			trLosses[epoch] = (trLosses[epoch] - trLossBase) / trSizNet
			self.pushTe(self.teX)
			teAccuracy[epoch] = self.lossFunc.teAccuracy()
			teLosses[epoch] = (self.lossFunc.teLoss() - teLossBase) / teSiz
			if verbose > 0:
				print(f'epoch: {epoch}/{self.epochMax}, train accuracy: {trAccuracy[epoch]}, test accuracy: {teAccuracy[epoch]}')
				if verbose > 1:
					print(f'\t\t\tlearning rate: {appLrs[epoch]}, train loss: {trLosses[epoch]}, test loss: {teLosses[epoch]}')
			if trLosses[epoch] < self.lossMax or epoch == self.epochMax:
				break
			epoch += 1
		self.appLrs = appLrs[:epoch + 1]
		self.trAccuracy = trAccuracy[:epoch + 1]
		self.teAccuracy = teAccuracy[:epoch + 1]
		self.trLosses = trLosses[:epoch + 1]
		self.teLosses = teLosses[:epoch + 1]

	def fit(self, trX: Union[np.ndarray, Tuple, List], trT: np.ndarray,
			teX: Union[np.ndarray, Tuple, List] = None, teT: np.ndarray = None,
			batchSize: int = None, verbose: int = 0, **kwargs):
		super().fit(trX, trT, teX, teT, **kwargs)

		if self.lrInit is None:
			self._lr = self.lrMax
		else:
			self._lr = self.lrInit
		for fset in self.fitSets:
			fset.preFit(regularizer=self.reg, momentum=None)

		if batchSize is not None:
			self.batchSize = batchSize
		elif self.batchSize == UNKNOWN:
			self.batchSize = trT.shape[-1]

		self.lossFunc.teT = self.teT
		if teT is None:
			# train only
			self.__fitTrain(verbose)
		else:
			# train and test
			self.__fitWithTest(verbose)

	def graphInfo(self, form: str = 'short') -> str:
		c = f'{self.__class__.__name__}\n'
		c += f'learning rate init: {self.lrInit if self.lrInit is None else getFloatStr(self.lrInit)}\n'
		c += f'learning rate: {getFloatStr(self._lr)}\n'
		c += f'learning rate min, max: [{getFloatStr(self.lrMin)}, {getFloatStr(self.lrMax)}]\n'
		c += f'learning rate up, down: [{getFloatStr(self.lrUp)}, {getFloatStr(self.lrDown)}]\n'
		return c + super().graphInfo(form)


class NodeGraphBatch(NodeGraph):
	def __init__(self):
		super().__init__()
		self.lr: float = 0.01  # learning rate
		self.mom: float = 0.0  # momentum

	def __update(self):
		self.pullGrad()
		for fset in self.fitSets:
			fset.updateBatch(self.lr)

	def __fitTrain(self, verbose: int):
		trAccuracy = np.empty(self.epochMax + 1)
		trLosses = np.empty(self.epochMax + 1)

		trLossBase = self.lossFunc.base(self.trT)
		trSiz = self.trX.shape[-1] if isinstance(self.trX, np.ndarray) else self.trX[0].shape[-1]
		self.prePropTr(trSiz)

		epoch = 0
		while True:
			self.pushTr(self.trX)
			trAccuracy[epoch] = self.lossFunc.trAccuracy()
			trLosses[epoch] = (self.lossFunc.trLoss() - trLossBase) / trSiz
			if verbose > 0:
				print(f'epoch: {epoch}/{self.epochMax}, train accuracy: {trAccuracy[epoch]}')
				if verbose > 1:
					print(f'\t\t\ttrain loss: {trLosses[epoch]}')
			if trLosses[epoch] < self.lossMax or epoch == self.epochMax:
				break
			self.__update()
			epoch += 1
		self.trAccuracy = trAccuracy[:epoch + 1]
		self.trLosses = trLosses[:epoch + 1]

	def __fitWithTest(self, verbose: int):
		trAccuracy = np.empty(self.epochMax + 1)
		teAccuracy = np.empty(self.epochMax + 1)
		trLosses = np.empty(self.epochMax + 1)
		teLosses = np.empty(self.epochMax + 1)  # test

		trLossBase, teLossBase = self.lossFunc.base(self.trT), self.lossFunc.base(self.teT)
		trSiz, teSiz = (self.trX.shape[-1], self.teX.shape[-1]) if isinstance(self.trX, np.ndarray) else (self.trX[0].shape[-1], self.teX[0].shape[-1])
		self.prePropBoth(trSiz, teSiz)

		epoch = 0
		while True:
			self.pushBoth(self.trX, self.teX)
			trAccuracy[epoch] = self.lossFunc.trAccuracy()
			teAccuracy[epoch] = self.lossFunc.teAccuracy()
			trLosses[epoch] = (self.lossFunc.trLoss() - trLossBase) / trSiz
			teLosses[epoch] = (self.lossFunc.teLoss() - teLossBase) / teSiz
			if verbose > 0:
				print(f'epoch: {epoch}/{self.epochMax}, train accuracy: {trAccuracy[epoch]}, test accuracy: {teAccuracy[epoch]}')
				if verbose > 1:
					print(f'\t\t\ttrain loss: {trLosses[epoch]}, test loss: {teLosses[epoch]}')
			if trLosses[epoch] < self.lossMax or epoch == self.epochMax:
				break
			self.__update()
			epoch += 1
		self.trAccuracy = trAccuracy[:epoch + 1]
		self.teAccuracy = teAccuracy[:epoch + 1]
		self.trLosses = trLosses[:epoch + 1]
		self.teLosses = teLosses[:epoch + 1]  # test

	def fit(self, trX: Union[np.ndarray, Tuple, List], trT: np.ndarray,
			teX: Union[np.ndarray, Tuple, List] = None, teT: np.ndarray = None,
			verbose: int = 0, **kwargs):
		super().fit(trX, trT, teX, teT, **kwargs)

		for fset in self.fitSets:
			fset.preFit(regularizer=self.reg, momentum=self.mom)

		self.lossFunc.trT = self.trT
		self.lossFunc.teT = self.teT
		if teT is None:
			# train only
			self.__fitTrain( verbose)
		else:
			# train and test
			self.__fitWithTest(verbose)

	def graphInfo(self, form: str = 'short') -> str:
		c = f'{self.__class__.__name__}\n'
		c += f'learning rate: {getFloatStr(self.lr)}\n'
		c += f'momentum: {getFloatStr(self.mom)}\n'
		return c + super().graphInfo(form)
