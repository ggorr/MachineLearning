from .MidSet import *


class Flat1D(MidSet1D):
	def __init__(self, name: str = None):
		super().__init__(UNKNOWN, UNKNOWN, name)

	def compile(self) -> bool:
		if super(Flat1D, self).compile():
			self.ychs = 1
			self.ydim = self.xchs * self.xdim
			return True
		else:
			return False

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.empty((trSiz, self.xdim, self.xchs), np.float32)
		self.gradY = np.empty((trSiz, self.ydim, 1), np.float32)
		self.teX = np.empty((teSiz, self.xdim, self.xchs), np.float32)

	def prePropTr(self, trSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.empty((trSiz, self.xdim, self.xchs), np.float32)
		self.gradY = np.empty((trSiz, self.ydim, 1), np.float32)

	def prePropTe(self, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.teX = np.empty((teSiz, self.xdim, self.xchs), np.float32)

	def prePropPr(self, prSiz: int):
		self.xNeeds = len(self.prevSets)
		self.prX = np.empty((prSiz, self.xdim, self.xchs), np.float32)

	def pushBothX(self):
		self.trY = self.trX.reshape((self.trX.shape[0], -1, 1))
		self.teY = self.teX.reshape((self.teX.shape[0], -1, 1))

	def pushTrX(self):
		self.trY = self.trX.reshape((self.trX.shape[0], -1, 1))

	def pushTeX(self):
		self.teY = self.teX.reshape((self.teX.shape[0], -1, 1))

	def pushPrX(self):
		self.prY = self.prX.reshape((self.prX.shape[0], -1, 1))

	def pullGradY(self):
		self.gradX = self.gradY.reshape(self.trX.shape)

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.ydim})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, ({self.xdim},{self.xchs})>({self.ydim},{self.ychs});'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'


class Flat2D(MidSet2D):
	def __init__(self, name: str = None):
		super().__init__(UNKNOWN, UNKNOWN, name)

	def compile(self) -> bool:
		if super().compile():
			self.ychs: int = 1
			self.ydim: int = self.xchs * self.xdim[0] * self.xdim[1]
			return True
		else:
			return False

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.empty((trSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)
		self.gradY = np.empty((trSiz, self.ydim, 1), np.float32)
		self.teX = np.empty((teSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)

	def prePropTr(self, trSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.empty((trSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)
		self.gradY = np.empty((trSiz, self.ydim, 1), np.float32)

	def prePropTe(self, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.teX = np.empty((teSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)

	def prePropPr(self, prSiz: int):
		self.xNeeds = len(self.prevSets)
		self.prX = np.empty((prSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)

	def pushBothX(self):
		self.trY = self.trX.reshape((self.trX.shape[0], -1, 1))
		self.teY = self.teX.reshape((self.teX.shape[0], -1, 1))

	def pushTrX(self):
		self.trY = self.trX.reshape((self.trX.shape[0], -1, 1))

	def pushTeX(self):
		self.teY = self.teX.reshape((self.teX.shape[0], -1, 1))

	def pushPrX(self):
		self.prY = self.prX.reshape((self.prX.shape[0], -1, 1))

	def pullGradY(self):
		self.gradX = self.gradY.reshape(self.trX.shape)

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.ydim})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, ({self.xdim[0]},{self.xdim[1]},{self.xchs})>({self.ydim},{self.ychs});'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'
