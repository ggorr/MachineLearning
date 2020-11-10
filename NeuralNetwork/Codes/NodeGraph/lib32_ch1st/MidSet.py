from typing import Tuple

from .StartSet import *


class MidSet1D(NodeSet, metaclass=ABCMeta):
	"""base class of all middle sets with channels"""

	def __init__(self, ydim: int, ychs: int, name: str):
		""" ydim: output dimension
				UNKNOWN if self is an Activation
				fixed dimension otherwise"""
		super().__init__(name)
		self.ydim: int = ydim
		self.ychs: int = ychs
		self.xdim: int = UNKNOWN
		self.xchs: int = UNKNOWN

		# train inputs
		#	trX[:, startX[i]:endX[i]] = prevSet[i].trY
		self.trX: Union[np.ndarray, None] = None

		# train output
		#	trY == nextSet[i].trX[:, nextSet[i].startX[j]:nextSet[i].endX[j]]
		self.trY: Union[np.ndarray, None] = None

		# test inputs
		#	teX[:, startX[i]:endX[i]] = prevSet[i].teY
		self.teX: Union[np.ndarray, None] = None

		# test output
		#	teY == nextSet[i].teX[:, nextSet[i].startX[j]:nextSet[i].endX[j]]
		self.teY: Union[np.ndarray, None] = None

		# inputs for prediction
		self.prX: Union[np.ndarray, None] = None
		# outputs for prediction
		self.prY: Union[np.ndarray, None] = None

		# gradient of loss w.r.t. trX
		#	gradX = f^* gradY where f is either a weight matrix or an activation
		self.gradX: Union[np.ndarray, None] = None

		# gradient of loss w.r.t. trY
		self.gradY: Union[np.ndarray, None] = None

		# previous MidSets
		#	trX[:, startX[i]:endX[i]] = prevSet[i].trY
		self.prevSets: tuple = ()

		# indices where prevSets start
		#	trX[:, startX[i]:endX[i]] = prevSet[i].trY
		self.startXs: Union[np.ndarray, None] = None

		# indices where prevSets end
		#	trX[:, startX[i]:endX[i]] = prevSet[i].trY
		self.endXs: Union[np.ndarray, None] = None

		# the number of trX fragment's that are not forwardly propagated
		self.xNeeds: int = 0

		# next NodeSets: MidSets and/or LossFunc
		#	self.nextSet[i].prevSet[nextInd[i]] == self
		self.nextSets: tuple = ()

		# index of self in each next MidSets
		self.nextInd: np.ndarray = np.array([], int)

		# the number of gradY fragment's that are not backward propagated
		self.gradNeeds: int = 0

	def addPrevSets(self, *prevSets):
		for pset in prevSets:
			pset.nextInd = np.append(pset.nextInd, len(self.prevSets))
			pset.nextSets += (self,)
			self.prevSets += (pset,)

	def compile(self) -> bool:
		for pset in self.prevSets:
			if pset.ydim == UNKNOWN:
				# If pset is an Activation then pset.ydim may not be determined yet
				# In the 2nd or 3rd ... execution, this problem is fixed
				return False
		self.xdim = self.prevSets[0].ydim
		self.startXs = np.empty(len(self.prevSets), int)
		self.endXs = np.empty(len(self.prevSets), int)
		self.xchs = 0
		for i, pset in enumerate(self.prevSets):
			self.startXs[i] = self.xchs
			self.xchs += pset.ychs
			self.endXs[i] = self.xchs
		return True

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.trX = np.empty((self.xchs, self.xdim, trSiz), np.float32)
		self.trY = np.empty((self.ychs, self.ydim, trSiz), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)
		self.teX = np.empty((self.xchs, self.xdim, teSiz), np.float32)
		self.teY = np.empty((self.ychs, self.ydim, teSiz), np.float32)

	def prePropTr(self, trSiz: int):
		self.trX = np.empty((self.xchs, self.xdim, trSiz), np.float32)
		self.trY = np.empty((self.ychs, self.ydim, trSiz), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)

	def prePropTe(self, teSiz: int):
		self.teX = np.empty((self.xchs, self.xdim, teSiz), np.float32)
		self.teY = np.empty((self.ychs, self.ydim, teSiz), np.float32)

	def prePropPr(self, prSiz: int):
		self.prX = np.empty((self.xchs, self.xdim, prSiz), np.float32)
		self.prY = np.empty((self.ychs, self.ydim, prSiz), np.float32)

	def resetForPush(self):
		self.xNeeds = len(self.prevSets)

	def pushBoth(self, index: int, trXFrag: np.ndarray, teXFrag: np.ndarray):
		"""forward propagation
		"""
		self.trX[self.startXs[index]:self.endXs[index]] = trXFrag
		self.teX[self.startXs[index]:self.endXs[index]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushBothX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushBoth(nind, self.trY, self.teY)

	def pushTr(self, index: int, trXFrag: np.ndarray):
		self.trX[self.startXs[index]:self.endXs[index]] = trXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTr(nind, self.trY)

	def pushTe(self, index: int, teXFrag: np.ndarray):
		self.teX[self.startXs[index]:self.endXs[index]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTeX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTe(nind, self.teY)

	def pushPr(self, index: int, prXFrag: np.ndarray):
		self.prX[self.startXs[index]:self.endXs[index]] = prXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushPrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushPr(nind, self.prY)

	@abstractmethod
	def pushBothX(self):
		"""forward propagation
			obtain self.trY and self.teY from self.trX and self.teX resp.
		"""
		pass

	@abstractmethod
	def pushTrX(self):
		"""forward propagation
			obtain self.trY from self.trX
		"""
		pass

	@abstractmethod
	def pushTeX(self):
		"""forward propagation
			obtain self.teY from self.teX resp.
		"""
		pass

	@abstractmethod
	def pushPrX(self):
		"""forward propagation
			obtain self.prY from self.prX
		"""
		pass

	def resetForPull(self):
		self.gradNeeds = len(self.nextSets)
		self.gradY.fill(F0)

	def pullGrad(self, gradYFrag: np.ndarray):
		"""backward propagation
		"""
		self.gradY += gradYFrag
		self.gradNeeds -= 1
		if self.gradNeeds == 0:
			self.pullGradY()
			for i, pset in enumerate(self.prevSets):
				pset.pullGrad(self.gradX[self.startXs[i]:self.endXs[i]])

	@abstractmethod
	def pullGradY(self):
		"""backward propagation
			obtain self.gradX from self.gradY
			obtain gradients of bias and/or weight
		"""
		pass


class MidSet2D(NodeSet, metaclass=ABCMeta):
	"""base class of all middle sets with channels"""

	def __init__(self, ydim: Union[int, Tuple, List], ychs: int, name: str):
		""" ydim: output dimension
				UNKNOWN if self is an Activation
				fixed dimension otherwise"""
		super().__init__(name)
		self.ydim: Union[int, tuple] = ydim if isinstance(ydim, int) else tuple(ydim)
		self.ychs: int = ychs
		self.xdim: Union[int, tuple] = UNKNOWN
		self.xchs: int = UNKNOWN

		# train inputs
		#	trX[:, startX[i]:endX[i]] = prevSet[i].trY
		self.trX: Union[np.ndarray, None] = None

		# train output
		#	trY == nextSet[i].trX[:, nextSet[i].startX[j]:nextSet[i].endX[j]]
		self.trY: Union[np.ndarray, None] = None

		# test inputs
		#	teX[:, startX[i]:endX[i]] = prevSet[i].teY
		self.teX: Union[np.ndarray, None] = None

		# test output
		#	teY == nextSet[i].teX[:, nextSet[i].startX[j]:nextSet[i].endX[j]]
		self.teY: Union[np.ndarray, None] = None

		# inputs for prediction
		self.prX: Union[np.ndarray, None] = None
		# outputs for prediction
		self.prY: Union[np.ndarray, None] = None

		# gradient of loss w.r.t. trX
		#	gradX = f^* gradY where f is either a weight matrix or an activation
		self.gradX: Union[np.ndarray, None] = None

		# gradient of loss w.r.t. trY
		self.gradY: Union[np.ndarray, None] = None

		# previous MidSets
		#	trX[:, startX[i]:endX[i]] = prevSet[i].trY
		self.prevSets: tuple = ()

		# indices where prevSets start
		#	trX[:, startX[i]:endX[i]] = prevSet[i].trY
		self.startXs: Union[np.ndarray, None] = None

		# indices where prevSets end
		#	trX[:, startX[i]:endX[i]] = prevSet[i].trY
		self.endXs: Union[np.ndarray, None] = None

		# the number of trX fragment's that are not forwardly propagated
		self.xNeeds: int = 0

		# next NodeSets: MidSets and/or LossFunc
		#	self.nextSet[i].prevSet[nextInd[i]] == self
		self.nextSets: tuple = ()

		# index of self in each next MidSets
		self.nextInd: np.ndarray = np.array([], int)

		# the number of gradY fragment's that are not backward propagated
		self.gradNeeds: int = 0

	def addPrevSets(self, *prevSets):
		for pset in prevSets:
			pset.nextInd = np.append(pset.nextInd, len(self.prevSets))
			pset.nextSets += (self,)
			self.prevSets += (pset,)

	def compile(self) -> bool:
		for pset in self.prevSets:
			if pset.ydim == UNKNOWN:
				# If pset is an Activation then pset.ydim may not be determined yet
				# In the 2nd or 3rd ... execution, this problem is fixed
				return False
		self.xdim = self.prevSets[0].ydim
		self.startXs = np.empty(len(self.prevSets), int)
		self.endXs = np.empty(len(self.prevSets), int)
		self.xchs = 0
		for i, pset in enumerate(self.prevSets):
			self.startXs[i] = self.xchs
			self.xchs += pset.ychs
			self.endXs[i] = self.xchs
		return True

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.trX = np.empty((self.xchs, self.xdim, trSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.xchs, self.xdim[0], self.xdim[1], trSiz), np.float32)
		self.trY = np.empty((self.ychs, self.ydim, trSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.ychs, self.ydim[0], self.ydim[1], trSiz), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)
		self.teX = np.empty((self.xchs, self.xdim, teSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.xchs, self.xdim[0], self.xdim[1], teSiz), np.float32)
		self.teY = np.empty((self.ychs, self.ydim, teSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.ychs, self.ydim[0], self.ydim[1], teSiz), np.float32)

	def prePropTr(self, trSiz: int):
		self.trX = np.empty((self.xchs, self.xdim, trSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.xchs, self.xdim[0], self.xdim[1], trSiz), np.float32)
		self.trY = np.empty((self.ychs, self.ydim, trSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.ychs, self.ydim[0], self.ydim[1], trSiz), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)

	def prePropTe(self, teSiz: int):
		self.teX = np.empty((self.xchs, self.xdim, teSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.xchs, self.xdim[0], self.xdim[1], teSiz), np.float32)
		self.teY = np.empty((self.ychs, self.ydim, teSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.ychs, self.ydim[0], self.ydim[1], teSiz), np.float32)

	def prePropPr(self, prSiz: int):
		self.prX = np.empty((self.xchs, self.xdim, prSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.xchs, self.xdim[0], self.xdim[1], prSiz), np.float32)
		self.prY = np.empty((self.ychs, self.ydim, prSiz), np.float32) if isinstance(self.xdim, int) else \
			np.empty((self.ychs, self.ydim[0], self.ydim[1], prSiz), np.float32)

	def resetForPush(self):
		self.xNeeds = len(self.prevSets)

	def pushBoth(self, index: int, trXFrag: np.ndarray, teXFrag: np.ndarray):
		"""forward propagation
		"""
		self.trX[self.startXs[index]:self.endXs[index]] = trXFrag
		self.teX[self.startXs[index]:self.endXs[index]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushBothX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushBoth(nind, self.trY, self.teY)

	def pushTr(self, index: int, trXFrag: np.ndarray):
		self.trX[self.startXs[index]:self.endXs[index]] = trXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTr(nind, self.trY)

	def pushTe(self, index: int, teXFrag: np.ndarray):
		self.teX[self.startXs[index]:self.endXs[index]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTeX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTe(nind, self.teY)

	def pushPr(self, index: int, prXFrag: np.ndarray):
		self.prX[self.startXs[index]:self.endXs[index]] = prXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushPrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushPr(nind, self.prY)

	@abstractmethod
	def pushBothX(self):
		"""forward propagation
			obtain self.trY and self.teY from self.trX and self.teX resp.
		"""
		pass

	@abstractmethod
	def pushTrX(self):
		"""forward propagation
			obtain self.trY from self.trX
		"""
		pass

	@abstractmethod
	def pushTeX(self):
		"""forward propagation
			obtain self.teY from self.teX resp.
		"""
		pass

	@abstractmethod
	def pushPrX(self):
		"""forward propagation
			obtain self.prY from self.prX
		"""
		pass

	def resetForPull(self):
		self.gradNeeds = len(self.nextSets)
		self.gradY.fill(F0)

	def pullGrad(self, gradYFrag: np.ndarray):
		"""backward propagation
		"""
		self.gradY += gradYFrag
		self.gradNeeds -= 1
		if self.gradNeeds == 0:
			self.pullGradY()
			for i, pset in enumerate(self.prevSets):
				pset.pullGrad(self.gradX[self.startXs[i]:self.endXs[i]])

	@abstractmethod
	def pullGradY(self):
		"""backward propagation
			obtain self.gradX from self.gradY
			obtain gradients of bias and/or weight
		"""
		pass
