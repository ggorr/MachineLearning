from .MidSet import *


class MaxPool1D(MidSet1D):
	def __init__(self, step: int, name: str = None):
		super().__init__(UNKNOWN, UNKNOWN, name)
		self.step: int = step
		# index of maximum
		self.ind: Union[np.ndarray, None] = None

	def compile(self) -> bool:
		if super(MaxPool1D, self).compile():
			self.ydim = (self.xdim - 1) // self.step + 1
			self.ychs = self.xchs
			return True
		else:
			return False

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.trX = np.empty((trSiz, self.xdim, self.xchs), np.float32)
		self.trY = np.empty((trSiz, self.ydim, self.ychs), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)
		self.teX = np.empty((teSiz, self.xdim, self.xchs), np.float32)
		self.teY = np.empty((teSiz, self.ydim, self.ychs), np.float32)
		self.ind = np.empty_like(self.trY, int)

	def prePropTr(self, trSiz: int):
		self.trX = np.empty((trSiz, self.xdim, self.xchs), np.float32)
		self.trY = np.empty((trSiz, self.ydim, self.ychs), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)
		self.ind = np.empty_like(self.trY, int)

	def pushBothX(self):
		start = 0
		for i in range(self.ydim):
			end = start + self.step
			self.ind[:, i] = np.argmax(self.trX[:, start:end], 1) + start
			self.teY[:, i] = np.max(self.teX[:, start:end], 1)
			start = end
		self.trY[:] = np.take_along_axis(self.trX, self.ind, 1)

	def pushTrX(self):
		start = 0
		for i in range(self.ydim):
			end = start + self.step
			self.ind[:, i] = np.argmax(self.trX[:, start:end], 1) + start
			start = end
		self.trY[:] = np.take_along_axis(self.trX, self.ind, 1)

	def pushTeX(self):
		start = 0
		for i in range(self.ydim):
			end = start + self.step
			self.teY[:, i] = np.max(self.teX[:, start:end], 1)
			start = end

	def pushPrX(self):
		start = 0
		for i in range(self.ydim):
			end = start + self.step
			self.prY[:, i] = np.max(self.prX[:, start:end], 1)
			start = end

	def pullGradY(self):
		self.gradX.fill(0.0)
		np.put_along_axis(self.gradX, self.ind, self.gradY, 1)

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.ydim})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, ({self.xdim},{self.xchs})>({self.ydim},{self.ychs});'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, {self.step};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'


class MaxPool2D(MidSet2D):
	def __init__(self, step: Union[tuple, list], name: str = None):
		super().__init__(UNKNOWN, UNKNOWN, name)
		self.step: tuple = tuple(step)
		# index of maximum
		self.ind0: Union[np.ndarray, None] = None
		self.ind1: Union[np.ndarray, None] = None
		self.trMed: Union[np.ndarray, None] = None
		self.teMed: Union[np.ndarray, None] = None
		self.prMed: Union[np.ndarray, None] = None

	def compile(self) -> bool:
		if super().compile():
			self.ydim = ((self.xdim[0] - 1) // self.step[0] + 1, (self.xdim[1] - 1) // self.step[1] + 1)
			self.ychs = self.xchs
			return True
		else:
			return False

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.trX = np.empty((trSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)
		# self.trY = np.empty((trSiz, self.ydim[0], self.ydim[1], self.ychs), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty((trSiz, self.ydim[0], self.ydim[1], self.ychs), np.float32)
		self.teX = np.empty((teSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)
		self.teY = np.empty((teSiz, self.ydim[0], self.ydim[1], self.ychs), np.float32)

		# self.trMed = np.empty((trSiz, self.ydim[0], self.xdim[1], self.xchs), np.float32)
		self.teMed = np.empty((teSiz, self.ydim[0], self.xdim[1], self.xchs), np.float32)
		self.ind0 = np.empty((trSiz, self.ydim[0], self.xdim[1], self.xchs), int)
		self.ind1 = np.empty((trSiz, self.ydim[0], self.ydim[1], self.ychs), int)

	def prePropTr(self, trSiz: int):
		self.trX = np.empty((trSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)
		# self.trY = np.empty((trSiz, self.ydim[0], self.ydim[1], self.ychs), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty((trSiz, self.ydim[0], self.ydim[1], self.ychs), np.float32)

		# self.trMed = np.empty((trSiz, self.ydim[0], self.xdim[1], self.xchs), np.float32)
		self.ind0 = np.empty((trSiz, self.ydim[0], self.xdim[1], self.xchs), int)
		self.ind1 = np.empty((trSiz, self.ydim[0], self.ydim[1], self.ychs), int)

	def prePropTe(self, teSiz: int):
		self.teX = np.empty((teSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)
		self.teY = np.empty((teSiz, self.ydim[0], self.ydim[1], self.ychs), np.float32)
		self.teMed = np.empty((teSiz, self.ydim[0], self.xdim[1], self.xchs), np.float32)

	def prePropPr(self, prSiz: int):
		self.prX = np.empty((prSiz, self.xdim[0], self.xdim[1], self.xchs), np.float32)
		self.prY = np.empty((prSiz, self.ydim[0], self.ydim[1], self.ychs), np.float32)
		self.prMed = np.empty((prSiz, self.ydim[0], self.xdim[1], self.xchs), np.float32)

	def pushBothX(self):
		start = 0
		for i in range(self.ydim[0]):
			end = start + self.step[0]
			self.trX[:, start:end].argmax(1, self.ind0[:, i])
			self.ind0[:, i] += start
			self.teX[:, start:end].max(1, self.teMed[:, i])
			start = end

		self.trMed = np.take_along_axis(self.trX, self.ind0, 1)
		start = 0
		for i in range(self.ydim[1]):
			end = start + self.step[1]
			self.trMed[:, :, start:end].argmax(2, self.ind1[:, :, i])
			self.ind1[:, :, i] += start
			self.teMed[:, :, start:end].max(2, self.teY[:, :, i])
			start = end
		self.trY = np.take_along_axis(self.trMed, self.ind1, 2)

	def pushTrX(self):
		start = 0
		for i in range(self.ydim[0]):
			end = start + self.step[0]
			self.trX[:, start:end].argmax(1, self.ind0[:, i])
			self.ind0[:, i] += start
			start = end

		self.trMed = np.take_along_axis(self.trX, self.ind0, 1)
		start = 0
		for i in range(self.ydim[1]):
			end = start + self.step[1]
			self.trMed[:, :, start:end].argmax(2, self.ind1[:, :, i])
			self.ind1[:, :, i] += start
			start = end
		self.trY = np.take_along_axis(self.trMed, self.ind1, 2)

	def pushTeX(self):
		start = 0
		for i in range(self.ydim[0]):
			end = start + self.step[0]
			self.teX[:, start:end].max(1, self.teMed[:, i])
			start = end
		start = 0
		for i in range(self.ydim[1]):
			end = start + self.step[1]
			self.teMed[:, :, start:end].max(2, self.teY[:, :, i])
			start = end

	def pushPrX(self):
		start = 0
		for i in range(self.ydim[0]):
			end = start + self.step[0]
			self.prX[:, start:end].max(1, self.prMed[:, i])
			start = end
		start = 0
		for i in range(self.ydim[1]):
			end = start + self.step[1]
			self.prMed[:, :, start:end].max(2, self.prY[:, :, i])
			start = end

	def pullGradY(self):
		self.gradX.fill(F0)
		self.trMed.fill(F0)

		np.put_along_axis(self.trMed, self.ind1, self.gradY, 2)
		np.put_along_axis(self.gradX, self.ind0, self.trMed, 1)

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.ydim})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, ({self.xdim},{self.xchs})>({self.ydim},{self.ychs});'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, {self.step};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'
