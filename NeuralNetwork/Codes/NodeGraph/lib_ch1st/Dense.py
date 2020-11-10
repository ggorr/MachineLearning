from .FitSet import *
from .Regularizer import RegNone, Regularizer


class Dense1D(FitSet1D):
	def __init__(self, ydim: int, ychs: int = 1, biased: bool = True, name: str = None):
		super().__init__(ydim, ychs, name)

		# initial affine transformation
		#	B = initTransf[0]
		#	W = initTransf[1:]
		self.initB: Union[np.ndarray, None] = None
		self.initW: Union[np.ndarray, None] = None

		# whether the NodeSet is biased#
		self.biased: bool = biased

		# bias vector
		self.B: Union[np.ndarray, None] = None
		self.BPrev: Union[np.ndarray, None] = None

		# weight matrix
		self.W: Union[np.ndarray, None] = None
		self.WPrev: Union[np.ndarray, None] = None

		# gradient of bias
		self.gradB: Union[np.ndarray, None] = None
		self.gradBPrev: Union[np.ndarray, None] = None

		# gradient of weight
		self.gradW: Union[np.ndarray, None] = None
		self.gradWPrev: Union[np.ndarray, None] = None

		# regularizer
		self.reg: Regularizer = RegNone()

		# momentum
		self.mom: float = 0.0

	def transf(self) -> tuple:
		return self.B, self.W

	def preFit(self, **kwargs):
		if self.initB is None:
			self.initB = np.zeros((self.xchs, self.ydim, 1))
		if self.initW is None:
			self.initW = (np.random.rand(self.xchs, self.ydim, self.xdim) * 2 - 1) / (self.xdim * self.ydim * self.ychs)
		self.B = self.initB.copy() if self.biased else np.zeros((self.xchs, self.ydim, 1))
		self.W = self.initW.copy()
		self.BPrev = np.zeros_like(self.B)
		self.WPrev = np.zeros_like(self.W)
		self.gradB = np.zeros_like(self.B)
		self.gradW = np.zeros_like(self.W)
		self.gradBPrev = np.zeros_like(self.B)
		self.gradWPrev = np.zeros_like(self.W)

		self.reg = kwargs['regularizer']
		self.mom = kwargs['momentum']

	def pushTrX(self):
		# self.trY = self.B + np.matmul(self.trX, self.W)
		np.add(self.B, np.matmul(self.W, self.trX, self.trY), self.trY)

	def pushTeX(self):
		np.add(self.B, np.matmul(self.W, self.teX, self.teY), self.teY)

	def pushPrX(self):
		# self.trY = self.B + np.matmul(self.prX, self.W)
		np.add(self.B, np.matmul(self.W, self.prX, self.prY), self.prY)

	def pushBothX(self):
		np.add(self.B, np.matmul(self.W, self.trX, self.trY), self.trY)
		np.add(self.B, np.matmul(self.W, self.teX, self.teY), self.teY)

	def pullGradY(self):
		np.matmul(self.W.transpose((0, 2, 1)), self.gradY, out=self.gradX)
		if self.biased:
			np.sum(self.gradY, axis=2, keepdims=True, out=self.gradB)
		np.matmul(self.gradY, self.trX.transpose((0, 2, 1)), out=self.gradW)
		self.gradW += self.reg.grad(self.W)

	def updateBatch(self, lr: float):
		"""
		:param lr: learning rate
		"""
		# self.gradBPrev = self.momentum * self.gradBPrev + lr * self.gradB
		# self.gradWPrev = self.momentum * self.gradWPrev + lr * self.gradW
		self.gradBPrev *= self.mom
		self.gradWPrev *= self.mom
		self.gradBPrev += lr * self.gradB
		self.gradWPrev += lr * self.gradW
		self.B -= self.gradBPrev
		self.W -= self.gradWPrev

	def updateAdaptive(self, lr: float):
		"""
		:param lr: learning rate
		"""
		# self.B = self.BPrev - lr * self.gradBPrev
		# self.W = self.WPrev - lr * self.gradWPrev
		np.subtract(self.BPrev, np.multiply(lr, self.gradBPrev, self.B), self.B)
		np.subtract(self.WPrev, np.multiply(lr, self.gradWPrev, self.W), self.W)

	def sendCurrToPrev(self):
		self.B, self.BPrev = self.BPrev, self.B
		self.W, self.WPrev = self.WPrev, self.W
		self.gradB, self.gradBPrev = self.gradBPrev, self.gradB
		self.gradW, self.gradWPrev = self.gradWPrev, self.gradW

	def gradDot(self) -> float:
		# return np.tensordot(self.gradB, self.gradBPrev, 3) + np.tensordot(self.gradW, self.gradWPrev, 3)
		return np.sum(self.gradB * self.gradBPrev) + np.sum(self.gradW * self.gradWPrev)

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.ydim}, {self.ychs})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, ({self.xdim},{self.xchs})>({self.ydim},{self.ychs}), {self.biased};'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.ydim}, {self.ychs}, {self.biased}, {self.name};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'
