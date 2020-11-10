from typing import List

import numpy as np

from .NodeSet import *


class StartSet1D(NodeSet):
	""" start set with channels
		channel is the last(= 3rd) coordinate """

	def __init__(self, ydim: int, ychs: int = 1, name: str = None):
		super().__init__(name)
		self.ydim: int = ydim
		self.ychs: int = ychs
		# next ChMidSet1D's
		#	self.nextSets[i].prevSet[nextIndex[i]] == self
		self.nextSets: tuple = ()

		# index of self in each next MidSets
		self.nextInd: np.ndarray = np.array([], int)

	def pushBoth(self, trXFrag: np.ndarray, teXFrag: np.ndarray):
		for nset, nind in zip(self.nextSets, self.nextInd):
			nset.pushBoth(nind, trXFrag, teXFrag)

	def pushTr(self, trXFrag: np.ndarray):
		for nset, nind in zip(self.nextSets, self.nextInd):
			nset.pushTr(nind, trXFrag)

	def pushTe(self, teXFrag: np.ndarray):
		for nset, nind in zip(self.nextSets, self.nextInd):
			nset.pushTe(nind, teXFrag)

	def pushPr(self, prX: np.ndarray):
		for nset, nind in zip(self.nextSets, self.nextInd):
			nset.pushPr(nind, prX)

	def pullGrad(self, gradYFrag: np.ndarray):
		pass

	def __str__(self) -> str:
		return f'{self.__class__.__name__}({self.Id}, {self.name}, {self.ydim}, {self.ychs})'

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.ydim}, {self.ychs})'

	def saveStr(self) -> str:
		return f'{self.__class__.__name__}({self.Id}, {self.ydim}, {self.ychs}, {self.name})'


class StartSet2D(NodeSet):
	""" start set with channels
		channel is the last(= 3rd) coordinate """

	def __init__(self, ydim: Union[tuple, list], ychs: int = 1, name: str = None):
		super().__init__(name)
		self.ydim: tuple = tuple(ydim)
		self.ychs: int = ychs
		# next ChMidSet1D's
		#	self.nextSets[i].prevSet[nextIndex[i]] == self
		self.nextSets: tuple = ()

		# index of self in each next MidSets
		self.nextInd: np.ndarray = np.array([], int)

	def pushBoth(self, trXFrag: np.ndarray, teXFrag: np.ndarray):
		for nset, nind in zip(self.nextSets, self.nextInd):
			nset.pushBoth(nind, trXFrag, teXFrag)

	def pushTr(self, trXFrag: np.ndarray):
		for nset, nind in zip(self.nextSets, self.nextInd):
			nset.pushTr(nind, trXFrag)

	def pushTe(self, teXFrag: np.ndarray):
		for nset, nind in zip(self.nextSets, self.nextInd):
			nset.pushTe(nind, teXFrag)

	def pushPr(self, prX: np.ndarray):
		for nset, nind in zip(self.nextSets, self.nextInd):
			nset.pushPr(nind, prX)

	def pullGrad(self, gradYFrag: np.ndarray):
		pass

	def __str__(self) -> str:
		return f'{self.__class__.__name__}({self.Id}, {self.name}, ({self.ydim[0]},{self.ydim[1]}), {self.ychs})'

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, ({self.ydim[0]},{self.ydim[1]}), {self.ychs})'

	def saveStr(self) -> str:
		return f'{self.__class__.__name__}({self.Id}, ({self.ydim[0]},{self.ydim[1]}), {self.ychs}, {self.name})'
