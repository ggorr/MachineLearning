from typing import List

import numpy as np

from .MidSet import *
from .NodeSet import *
from .StartSet import *

# E0 = np.nextafter(1, 2) - 1
# LOSS_E1 = np.nextafter(1, 2)
LOSS_E0 = np.nextafter(0, 1)


# binary entropy
def be(t: np.ndarray) -> float:
	return -np.sum(t * np.log(t + LOSS_E0) + (1 - t) * np.log((1 - t) + LOSS_E0))


def zeroOneBe(t: np.ndarray) -> float:
	return 0.0


# binary cross entropy
def bce(t: np.ndarray, y: np.ndarray) -> float:
	return -np.sum(t * np.log(y + LOSS_E0) + (1 - t) * np.log((1 - y) + LOSS_E0))


# binary cross entropy
def zeroOneBce(t: np.ndarray, y: np.ndarray) -> float:
	return -np.sum(np.log(y[t == 1])) - np.sum(np.log(1 - y[t == 0]))


# categorical entropy
def ce(t: np.ndarray) -> float:
	return -np.sum(t * np.log(t + LOSS_E0))


# categorical entropy
def oneHotCe(t: np.ndarray) -> float:
	return 0.0


# categorical cross entropy
def cce(t: np.ndarray, y: np.ndarray) -> float:
	return -np.sum(t * np.log(y + LOSS_E0))


def oneHotCce(t: np.ndarray, y: np.ndarray) -> float:
	return -np.sum(np.log(y[t.nonzero()]))


class LossFunc1D(NodeSet, metaclass=ABCMeta):
	def __init__(self, name: str = None):
		super().__init__(name)
		self.ydim: Union[int, tuple] = UNKNOWN
		self.ychs = UNKNOWN
		# train output
		self.trY: Union[np.ndarray, None] = None
		# train target
		self.trT: Union[np.ndarray, None] = None

		# test output
		self.teY: Union[np.ndarray, None] = None
		# test target
		self.teT: Union[np.ndarray, None] = None

		# prediction output
		self.prY: Union[np.ndarray, None] = None

		self.gradY: Union[np.ndarray, None] = None
		self.prevSets: tuple = ()

		# indices where prevSets start
		#	trY[:, startYs[i]:endYs[i]] = prevSets[i].trY
		self.startYs: Union[np.ndarray, None] = None
		# indices where prevSets end
		#	trY[:, startYs[i]:endYs[i]] = prevSets[i].trY
		self.endYs: Union[np.ndarray, None] = None

	@abstractmethod
	def base(self, t: np.ndarray) -> float:
		pass

	@abstractmethod
	def baseVec(self, t: np.ndarray) -> Union[float, np.ndarray]:
		pass

	@abstractmethod
	def lossVec(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
		pass

	@abstractmethod
	def trLoss(self) -> float:
		pass

	def trLossMean(self) -> float:
		return self.trLoss() / self.trT.shape[-1]

	def trAccurateCount(self):
		if self.trT.shape[1] == 1:
			return np.sum((self.trY > 0.5) == (self.trT > 0.5))
		else:
			return np.sum(self.trY.argmax(1) == self.trT.argmax(1))

	def trAccuracy(self) -> np.float:
		if self.trT.shape[1] == 1:
			return np.sum((self.trY > 0.5) == (self.trT > 0.5)) / self.trT.shape[-1]
		else:
			return np.sum(self.trY.argmax(1) == self.trT.argmax(1)) / self.trT.shape[-1]

	@abstractmethod
	def teLoss(self) -> float:
		pass

	def teLossMean(self) -> float:
		return self.teLoss() / self.teT.shape[-1]

	def teAccurateCount(self):
		if self.teT.shape[1] == 1:
			return np.sum((self.teY > 0.5) == (self.teT > 0.5))
		else:
			return np.sum(self.teY.argmax(1) == self.teT.argmax(1))

	def teAccuracy(self) -> np.float:
		if self.teT.shape[1] == 1:
			return np.sum((self.teY > 0.5) == (self.teT > 0.5)) / self.teT.shape[-1]
		else:
			return np.sum(self.teY.argmax(1) == self.teT.argmax(1)) / self.teT.shape[-1]

	@abstractmethod
	def evalGrad(self):
		pass

	def addPrevSets(self, *prevSets: Union[StartSet1D, MidSet1D]):
		for pset in prevSets:
			pset.nextInd = np.append(pset.nextInd, len(self.prevSets))
			self.prevSets += (pset,)
			pset.nextSets += (self,)

	def compile(self):
		for pset in self.prevSets:
			if pset.ydim == UNKNOWN:
				return False
		self.ydim = self.prevSets[0].ydim
		self.startYs = np.empty(len(self.prevSets), int)
		self.endYs = np.empty(len(self.prevSets), int)
		self.ychs = 0
		for i, pset in enumerate(self.prevSets):
			self.startYs[i] = self.ychs
			self.ychs += pset.ychs
			self.endYs[i] = self.ychs
		return True

	def prePropBoth(self, trSiz: int, teSiz: int):
		if isinstance(self.ydim, int):
			self.trY = np.empty((self.ychs, self.ydim, trSiz))
			self.teY = np.empty((self.ychs, self.ydim, teSiz))
		else:
			self.trY = np.empty((self.ychs, self.ydim[0], self.ydim[1], trSiz))
			self.teY = np.empty((self.ychs, self.ydim[0], self.ydim[1], teSiz))
		self.gradY = np.empty_like(self.trY)

	def prePropTr(self, trSiz: int):
		if isinstance(self.ydim, int):
			self.trY = np.empty((self.ychs, self.ydim, trSiz))
		else:
			self.trY = np.empty((self.ychs, self.ydim[0], self.ydim[1], trSiz))
		self.gradY = np.empty_like(self.trY)

	def prePropTe(self, teSiz: int):
		if isinstance(self.ydim, int):
			self.teY = np.empty((self.ychs, self.ydim, teSiz))
		else:
			self.teY = np.empty((self.ychs, self.ydim[0], self.ydim[1], teSiz))

	def prePropPr(self, prSiz: int):
		if isinstance(self.ydim, int):
			self.prY = np.empty((self.ychs, self.ydim, prSiz))
		else:
			self.prY = np.empty((self.ychs, self.ydim[0], self.ydim[1], prSiz))

	def pushTr(self, index: int, trYFrag: np.ndarray):
		"""forward propagation
		"""
		self.trY[self.startYs[index]:self.endYs[index]] = trYFrag

	def pushTe(self, index: int, teYFrag: np.ndarray):
		self.teY[self.startYs[index]:self.endYs[index]] = teYFrag

	def pushPr(self, index: int, prYFrag: np.ndarray):
		self.prY[self.startYs[index]:self.endYs[index]] = prYFrag

	def pushBoth(self, index: int, trYFrag: np.ndarray, teYFrag: np.ndarray):
		"""forward propagation
		"""
		self.trY[self.startYs[index]:self.endYs[index]] = trYFrag
		self.teY[self.startYs[index]:self.endYs[index]] = teYFrag

	def pullGrad(self):
		self.evalGrad()
		for i, pset in enumerate(self.prevSets):
			pset.pullGrad(self.gradY[self.startYs[i]:self.endYs[i]])

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.ydim}, {self.ychs})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, {self.ydim}, {self.ychs};'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'


class Mse1D(LossFunc1D):
	"""Mean squared error"""

	def __init__(self, name: str = None):
		super().__init__(name)

	def base(self, t: np.ndarray) -> float:
		return 0.0

	def baseVec(self, t: np.ndarray) -> float:
		return 0.0

	def lossVec(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
		v = y - t
		return .5 * np.sum(v * v, axis=(0, 1))

	def trLoss(self):
		v = self.trY - self.trT
		return .5 * np.sum(v * v)

	def teLoss(self):
		v = self.teY - self.teT
		return .5 * np.sum(v * v)

	def evalGrad(self):
		# self.gradY = self.trY - self.trT
		np.subtract(self.trY, self.trT, self.gradY)


class Bce1D(LossFunc1D):
	"""Binary cross entropy"""

	def __init__(self, name: str = None):
		super().__init__(name)

	def base(self, t: np.ndarray) -> float:
		return -np.sum(t * np.log(t + LOSS_E0) + (1 - t) * np.log((1 - t) + LOSS_E0))

	def baseVec(self, t: np.ndarray) -> np.ndarray:
		return -np.sum(t * np.log(t + LOSS_E0) + (1 - t) * np.log((1 - t) + LOSS_E0), (0, 1))

	def lossVec(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
		return -np.sum(t * np.log(y + LOSS_E0) + (1 - t) * np.log((1 - y) + LOSS_E0), (0, 1))

	def trLoss(self) -> float:
		return -np.sum(self.trT * np.log(self.trY + LOSS_E0) + (1 - self.trT) * np.log((1 - self.trY) + LOSS_E0))

	def teLoss(self) -> float:
		return -np.sum(self.teT * np.log(self.teY + LOSS_E0) + (1 - self.teT) * np.log((1 - self.teY) + LOSS_E0))

	def evalGrad(self):
		# (y - t) / ((y + EPSILON) * ((1 + EPSILON) - y))
		# self.gradY = (self.trY - self.trT) / (self.trY * (1 - self.trY))
		np.divide(self.trY - self.trT, self.trY * (1 - self.trY) + LOSS_E0, self.gradY)


class ZeroOneBce1D(LossFunc1D):
	"""Binary cross entropy"""

	def __init__(self, name: str = None):
		super().__init__(name)

	def base(self, t: np.ndarray) -> float:
		return 0.0

	def baseVec(self, t: np.ndarray) -> np.ndarray:
		return np.zeros(t.shape[2])

	def lossVec(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
		vec = np.empty(t.shape)
		tf = np.equal(t, 1)
		vec[tf] = -np.log(y[tf])
		vec[~tf] = -np.log(1 - y[~tf])
		return vec.reshape(-1)

	def trLoss(self):
		return -np.sum(np.log(self.trY[self.trT == 1])) - np.sum(np.log(1 - self.trY[self.trT == 0]))

	def teLoss(self):
		return -np.sum(np.log(self.teY[self.teT == 1])) - np.sum(np.log(1 - self.teY[self.teT == 0]))

	def evalGrad(self):
		# (y - t) / ((y + EPSILON) * ((1 + EPSILON) - y))
		# self.gradY = (self.trY - self.trT) / (self.trY * (1 - self.trY))
		# self.gradY = (self.trY - self.trT) / ((self.trY + E0) * (E1 - self.trY))
		np.divide(self.trY - self.trT, self.trY * (1 - self.trY) + LOSS_E0, self.gradY)


class Cce1D(LossFunc1D):
	"""Categorical cross entropy"""

	def __init__(self, name: str = None):
		super().__init__(name)

	def base(self, t: np.ndarray) -> float:
		return -np.sum(t * np.log(t + LOSS_E0))

	def baseVec(self, t: np.ndarray) -> np.ndarray:
		return -np.sum(t * np.log(t + LOSS_E0), (0, 1))

	def lossVec(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
		return -np.sum(t * np.log(y + LOSS_E0), (0, 1))

	def trLoss(self):
		# log_loss is too slow
		# from sklearn.metrics import log_loss
		# return log_loss(self.trT, self.trY) * self.trT.shape[0]
		return -np.sum(self.trT * np.log(self.trY + LOSS_E0))

	def teLoss(self):
		return -np.sum(self.teT * np.log(self.teY + LOSS_E0))

	def evalGrad(self):
		# t / (E0 + y)
		# self.gradY = -self.trT / (E0 + self.trY)
		np.divide(-self.trT, LOSS_E0 + self.trY, self.gradY)


class OneHotCce1D(LossFunc1D):
	"""Categorical cross entropy for one hot target"""

	def __init__(self, name: str = None):
		super().__init__(name)

	def base(self, t: np.ndarray) -> float:
		return 0.0

	def baseVec(self, t: np.ndarray) -> np.ndarray:
		return np.zeros(t.shape[2], float)

	def lossVec(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
		return -np.log(y[t.nonzero()])

	def trLoss(self):
		return -np.sum(np.log(self.trY[self.trT.nonzero()]))

	def teLoss(self):
		return -np.sum(np.log(self.teY[self.teT.nonzero()]))

	def evalGrad(self):
		self.gradY.fill(0.0)
		ind = self.trT.nonzero()
		self.gradY[ind] = -1 / self.trY[ind]
