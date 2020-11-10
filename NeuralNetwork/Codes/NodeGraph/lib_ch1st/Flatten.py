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
		self.trX = np.empty((self.xchs, self.xdim, trSiz))
		self.gradY = np.empty((1, self.ydim, trSiz))
		self.teX = np.empty((self.xchs, self.xdim, teSiz))

	def prePropTr(self, trSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.empty((self.xchs, self.xdim, trSiz))
		self.gradY = np.empty((1, self.ydim, trSiz))

	def prePropTe(self, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.teX = np.empty((self.xchs, self.xdim, teSiz))

	def prePropPr(self, prSiz: int):
		self.xNeeds = len(self.prevSets)
		self.prX = np.empty((self.xchs, self.xdim, prSiz))

	def pushBothX(self):
		self.trY = self.trX.reshape((1, -1, self.trX.shape[2]))
		self.teY = self.teX.reshape((1, -1, self.teX.shape[2]))

	def pushTrX(self):
		self.trY = self.trX.reshape((1, -1, self.trX.shape[2]))

	def pushTeX(self):
		self.teY = self.teX.reshape((1, -1, self.teX.shape[2]))

	def pushPrX(self):
		self.prY = self.prX.reshape((1, -1, self.prX.shape[2]))

	def pullGradY(self):
		self.gradX = self.gradY.reshape((self.xchs, self.xdim, -1))

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
		self.trX = np.empty((self.xchs, self.xdim[0], self.xdim[1], trSiz))
		self.gradY = np.empty((1, self.ydim, trSiz))
		self.teX = np.empty((self.xchs, self.xdim[0], self.xdim[1], teSiz))

	def prePropTr(self, trSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.empty((self.xchs, self.xdim[0], self.xdim[1], trSiz))
		self.gradY = np.empty((1, self.ydim, trSiz))

	def prePropTe(self, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.teX = np.empty((self.xchs, self.xdim[0], self.xdim[1], teSiz))

	def prePropPr(self, prSiz: int):
		self.xNeeds = len(self.prevSets)
		self.prX = np.empty((self.xchs, self.xdim[0], self.xdim[1], prSiz))

	def pushBothX(self):
		self.trY = self.trX.reshape((1, -1, self.trX.shape[3]))
		self.teY = self.teX.reshape((1, -1, self.teX.shape[3]))

	def pushTrX(self):
		self.trY = self.trX.reshape((1, -1, self.trX.shape[3]))

	def pushTeX(self):
		self.teY = self.teX.reshape((1, -1, self.teX.shape[3]))

	def pushPrX(self):
		self.prY = self.prX.reshape((1, -1, self.prX.shape[3]))

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
