from threading import Thread

from .FitSet import *
from .Regularizer import *


class Conv1D(FitSet1D):
	def __init__(self, filSiz: int, nFilter: int, biased: bool = True, padding: int = None, stride: int = 1, name: str = None):
		""" nFilter = ychs: the number of output channels
		    if padding is None, then padding = (kernelSize - 1) // 2 """
		super().__init__(UNKNOWN, nFilter, name)
		self.filSiz: int = filSiz
		self.pad: int = (filSiz - 1) // 2 if padding is None else padding
		self.stride: int = stride
		self.biased: bool = biased
		self.xdimExt: int = UNKNOWN
		self.initB: Union[np.ndarray, None] = None
		self.initW: Union[np.ndarray, None] = None
		self.B: Union[np.ndarray, None] = None
		self.W: Union[np.ndarray, None] = None
		self.BPrev: Union[np.ndarray, None] = None
		self.WPrev: Union[np.ndarray, None] = None
		self.gradB: Union[np.ndarray, None] = None
		self.gradW: Union[np.ndarray, None] = None
		self.gradBPrev: Union[np.ndarray, None] = None
		self.gradWPrev: Union[np.ndarray, None] = None
		# regularizer
		self.reg: Regularizer = RegNone()
		# momentum
		self.mom: float = 0.0

	def compile(self) -> bool:
		if super(Conv1D, self).compile():
			self.xdimExt = self.xdim + 2 * self.pad
			self.ydim = (self.xdimExt - self.filSiz) // self.stride + 1
			return True
		else:
			return False

	def preFit(self, **kwargs):
		self.reg = kwargs['regularizer']
		self.mom = kwargs['momentum']
		if self.initB is None:
			self.initB = np.zeros(self.ychs)
		if self.initW is None:
			self.initW = np.random.rand(self.filSiz, self.xchs, self.ychs)
		self.B = self.initB.copy() if self.biased else np.zeros(self.ychs)
		self.W = self.initW.copy()
		self.BPrev = np.zeros_like(self.B)
		self.WPrev = np.zeros_like(self.W)
		self.gradB = np.zeros_like(self.B)
		self.gradW = np.zeros_like(self.W)
		self.gradBPrev = np.zeros_like(self.B)
		self.gradWPrev = np.zeros_like(self.W)

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.zeros((trSiz, self.xdimExt, self.xchs))
		self.trY = np.empty((trSiz, self.ydim, self.ychs))
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)
		self.teX = np.zeros((teSiz, self.xdimExt, self.xchs))
		self.teY = np.empty((teSiz, self.ydim, self.ychs))

	def prePropTr(self, trSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.zeros((trSiz, self.xdimExt, self.xchs))
		self.trY = np.empty((trSiz, self.ydim, self.ychs))
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)

	def prePropTe(self, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.teX = np.zeros((teSiz, self.xdimExt, self.xchs))
		self.teY = np.empty((teSiz, self.ydim, self.ychs))

	def prePropPr(self, prSiz: int):
		self.xNeeds = len(self.prevSets)
		self.prX = np.zeros((prSiz, self.xdimExt, self.xchs))
		self.prY = np.empty((prSiz, self.ydim, self.ychs))

	def pushBoth(self, index: int, trXFrag: np.ndarray, teXFrag: np.ndarray):
		"""forward propagation
		"""
		self.trX[:, self.pad:self.pad + self.xdim, self.startXs[index]:self.endXs[index]] = trXFrag
		self.teX[:, self.pad:self.pad + self.xdim, self.startXs[index]:self.endXs[index]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushBothX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushBoth(nind, self.trY, self.teY)

	def pushTr(self, index: int, trXFrag: np.ndarray):
		self.trX[:, self.pad:self.pad + self.xdim, self.startXs[index]:self.endXs[index]] = trXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTr(nind, self.trY)

	def pushTe(self, index: int, teXFrag: np.ndarray):
		self.teX[:, self.pad:self.pad + self.xdim, self.startXs[index]:self.endXs[index]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTeX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTe(nind, self.teY)

	def pushPr(self, index: int, prXFrag: np.ndarray):
		self.prX[:, self.pad:self.pad + self.xdim, self.startXs[index]:self.endXs[index]] = prXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushPrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushPr(nind, self.prY)

	def __pushXPartial(self, x: np.ndarray, y: np.ndarray, start: int, end: int):
		s = 0
		for j in range(self.ydim):
			y[start:end, j] = np.tensordot(x[start:end, s:s + self.filSiz], self.W)
			s += self.stride
		y[start:end] += self.B

	def __pushX(self, x: np.ndarray, y: np.ndarray):
		if y.shape[0] < 1024:  # number of data
			self.__pushXPartial(x, y, 0, y.shape[0])
			return
		if y.shape[0] < 4096:
			nth = 16
		else:
			# import multiprocessing
			# nth = 2 * multiprocessing.cpu_count()
			nth = 32  # number of threads
		siz = (y.shape[0] - 1) // nth + 1
		ths = []
		start = 0
		for i in range(nth):
			th = Thread(target=self.__pushXPartial, args=(x, y, start, start + siz))
			ths.append(th)
			th.start()
			start += siz
		for th in ths:
			th.join()

	def pushTrX(self):
		"""forward propagation
			obtain self.trY from self.trX
			inherited from MidSet
		"""
		########################################################################################
		# #### definition
		# for i in range(self.trY.shape[0]):
		# 	s = 0
		# 	for j in range(self.ydim):
		# 		for k in range(self.ychs):
		# 			self.trY[i, j, k] = self.B[0, 0, k]
		# 			for l in range(self.filSiz):
		# 				for m in range(self.xchs):
		# 					self.trY[i, j, k] += self.trX[i, s + l, m] * self.W[l, m, k]
		# 		s += self.stride
		#########################################################################################
		# t = self.ydim * self.stride
		# self.trY[:] = self.B
		# for k in range(self.filSiz):
		# 	self.trY += np.tensordot(self.trX[:, k:k + t:self.stride], self.W[k], 1)
		#########################################################################################
		# s = 0
		# for j in range(self.ydim):
		# 	self.trY[:, j] = np.tensordot(self.trX[:, s:s + self.filSiz], self.W)
		# 	s += self.stride
		# self.trY += self.B
		#########################################################################################
		self.__pushX(self.trX, self.trY)

	def pushBothX(self):
		"""forward propagation
			obtain self.trY and self.teY from self.trX and self.teX resp.
			inherited from MidSet
		"""
		if self.trY.shape[0] < 1024:  # number of data
			s = 0
			for j in range(self.ydim):
				self.trY[:, j] = np.tensordot(self.trX[:, s:s + self.filSiz], self.W)
				self.teY[:, j] = np.tensordot(self.teX[:, s:s + self.filSiz], self.W)
				s += self.stride
			self.trY += self.B
			self.teY += self.B
		else:
			self.__pushX(self.trX, self.trY)
			self.__pushX(self.teX, self.teY)

	def pushTeX(self):
		"""forward propagation
			obtain self.teY from self.teX resp.
			inherited from MidSet
		"""
		self.__pushX(self.teX, self.teY)

	def pushPrX(self):
		"""forward propagation
			obtain self.prY from self.prX
			inherited from MidSet
		"""
		self.__pushX(self.prX, self.prY)

	def resetForPull(self):
		self.gradNeeds = len(self.nextSets)
		# self.gradX.fill(0.0)
		# self.gradW.fill(0.0)
		self.gradY.fill(0.0)

	def pullGrad(self, gradYFrag: np.ndarray):
		"""backward propagation
		"""
		self.gradY += gradYFrag
		self.gradNeeds -= 1
		if self.gradNeeds == 0:
			self.pullGradY()
			for i, pset in enumerate(self.prevSets):
				pset.pullGrad(self.gradX[:, self.pad:self.xdim + self.pad, self.startXs[i]:self.endXs[i]])

	def __pullGradYPartial(self, w: np.ndarray, start: int, end: int):
		s = 0
		for k in range(self.ydim):
			t = s + self.filSiz
			w += np.tensordot(self.trX[start:end, s:t], self.gradY[start:end, k], axes=(0, 0))
			self.gradX[start:end, s:t] += np.tensordot(self.gradY[start:end, k], self.W, axes=(1, 2))
			s += self.stride

	def pullGradY(self):
		############################### gradW ###################################################################
		########## definition
		# t = self.ydim * self.stride
		# for i in range(self.filSiz):
		# 	for j in range(self.xchs):
		# 		for k in range(self.ychs):
		# 			self.gradW[i, j, k] = (self.trX[:, i:i + t:self.stride, j] * self.gradY[:, :, k]).sum()
		#####################################################################################################
		# s = 0
		# xt = self.trX.T.copy()
		# for k in range(self.ydim):
		# 	# either
		# 	# self.gradW += np.tensordot(self.trX[:, s:s+self.filSiz], self.gradY[:, k], axes=(0,0))
		# 	# or
		# 	for j in range(self.xchs):
		# 		# self.gradW[:, j] += np.matmul(self.trX[:, s:s+self.filSiz, j].T, self.gradY[:, k, :])
		# 		self.gradW[:, j] += np.matmul(xt[j, s:s + self.filSiz], self.gradY[:, k, :])
		# 	# end either
		# 	s += self.stride
		################################################################################################

		################################ gradX #########################################################
		# #### slow
		# t = self.ydim * self.stride
		# for i in range(self.filSiz):
		# 	for j in range(self.xchs):
		# 		for k in range(self.ychs):
		# 			self.gradX[:, i:i+t:self.stride, j] += self.gradY[:, :, k] * self.W[i, j, k]
		################################################################################################
		# s = 0
		# for k in range(self.ydim):
		# 	for j in range(self.xchs):
		# 		self.gradX[:, s:s + self.filSiz, j] += np.matmul(self.gradY[:, k], self.W[:, j].T, )
		# 	s += self.stride
		################################################################################################
		# s = 0
		# for k in range(self.ydim):
		# 	self.gradX[:, s:s + self.filSiz] += np.tensordot(self.gradY[:, k], self.W, axes=(1, 2))
		# 	s += self.stride
		################################################################################################
		############ non thread version
		# if self.biased:
		# 	np.sum(self.gradY, (0, 1), out=self.gradB)
		# s = 0
		# # yt = self.gradY.transpose((1, 0, 2)).copy()
		# for k in range(self.ydim):
		# 	self.gradW += np.tensordot(self.trX[:, s:s + self.filSiz], self.gradY[:, k], axes=(0, 0))
		# 	self.gradX[:, s:s + self.filSiz] += np.tensordot(self.gradY[:, k], self.W, axes=(1, 2))
		# 	# self.gradW += np.tensordot(self.trX[:, s:s + self.filSiz], yt[k], axes=(0, 0))
		# 	# self.gradX[:, s:s + self.filSiz] += np.tensordot(yt[k], self.W, axes=(1, 2))
		# 	s += self.stride
		################################################################################################
		if self.biased:
			self.gradY.sum((0,1), out=self.gradB)
		self.gradX.fill(0.0)
		if self.trY.shape[0] < 1024:  # size of data
			self.gradW.fill(0.0)
			self.__pullGradYPartial(self.gradW, 0, self.trY.shape[0])
			self.gradW += self.reg.grad(self.W)
			return
		if self.trY.shape[0] < 4096:
			nth = 16  # number of threads
		else:
			# import multiprocessing
			# nth = multiprocessing.cpu_count()
			nth = 32  # number of threads
		siz = (self.trY.shape[0] - 1) // nth + 1
		ws = np.zeros((nth,) + self.W.shape)
		ths = []
		start = 0
		for w in ws:
			th = Thread(target=self.__pullGradYPartial, args=(w, start, start + siz))
			ths.append(th)
			th.start()
			start += siz
		for th in ths:
			th.join()
		ws.sum(0, out=self.gradW)
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
		np.subtract(self.BPrev, np.multiply(lr, self.gradBPrev, self.B), self.B)
		np.subtract(self.WPrev, np.multiply(lr, self.gradWPrev, self.W), self.W)

	def sendCurrToPrev(self):
		self.B, self.BPrev = self.BPrev, self.B
		self.gradB, self.gradBPrev = self.gradBPrev, self.gradB
		self.W, self.WPrev = self.WPrev, self.W
		self.gradW, self.gradWPrev = self.gradWPrev, self.gradW

	# noinspection PyTypeChecker
	def gradDot(self) -> float:
		return np.sum(self.gradB * self.gradBPrev) + np.sum(self.gradW * self.gradWPrev)

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.filSiz})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, {self.filSiz}, ({self.xdim}, {self.xchs})>({self.ydim},{self.ychs}),' \
			f' {self.biased}, {self.pad}, {self.stride};'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, {self.filSiz}, {self.ychs}, {self.biased}, {self.pad}, {self.stride};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'


class Conv2D(FitSet2D):
	def __init__(self, filSiz: Union[tuple, list], nFilter: int, biased: bool = True, padding: Union[int, tuple, list] = None,
				 stride: Union[int, tuple, list] = 1, name: str = None):
		""" nFilter = ychs: the number of output channels
		    if padding is None, then padding = ((filSiz[0] - 1) // 2, (filSiz[1] - 1) // 2) """
		super().__init__(UNKNOWN, nFilter, name)
		self.filSiz: tuple = tuple(filSiz)
		self.pad: tuple = ((filSiz[0] - 1) // 2, (filSiz[1] - 1) // 2) if padding is None else \
			(padding, padding) if isinstance(padding, int) else tuple(padding)
		self.stride: tuple = (stride, stride) if isinstance(stride, int) else tuple(stride)
		self.biased: bool = biased
		self.xdimExt: tuple = UNKNOWN
		self.initB: Union[np.ndarray, None] = None
		self.initW: Union[np.ndarray, None] = None
		self.B: Union[np.ndarray, None] = None
		self.W: Union[np.ndarray, None] = None
		self.BPrev: Union[np.ndarray, None] = None
		self.WPrev: Union[np.ndarray, None] = None
		self.gradB: Union[np.ndarray, None] = None
		self.gradW: Union[np.ndarray, None] = None
		self.gradBPrev: Union[np.ndarray, None] = None
		self.gradWPrev: Union[np.ndarray, None] = None
		# regularizer
		self.reg: Regularizer = RegNone()
		# momentum
		self.mom: float = 0.0

	def compile(self) -> bool:
		if super().compile():
			self.xdimExt = (self.xdim[0] + 2 * self.pad[0], self.xdim[1] + 2 * self.pad[1])
			self.ydim = ((self.xdimExt[0] - self.filSiz[0]) // self.stride[0] + 1, (self.xdimExt[1] - self.filSiz[1]) // self.stride[1] + 1)
			return True
		else:
			return False

	def preFit(self, **kwargs):
		self.reg = kwargs['regularizer']
		self.mom = kwargs['momentum']
		if self.initB is None:
			self.initB = np.zeros(self.ychs)
		if self.initW is None:
			self.initW = np.random.rand(self.filSiz[0], self.filSiz[1], self.xchs, self.ychs)
		self.B = self.initB.copy() if self.biased else np.zeros(self.ychs)
		self.W = self.initW.copy()
		self.BPrev = np.zeros_like(self.B)
		self.WPrev = np.zeros_like(self.W)
		self.gradB = np.zeros_like(self.B)
		self.gradW = np.zeros_like(self.W)
		self.gradBPrev = np.zeros_like(self.B)
		self.gradWPrev = np.zeros_like(self.W)

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.zeros((trSiz, self.xdimExt[0], self.xdimExt[1], self.xchs))
		self.trY = np.empty((trSiz, self.ydim[0], self.ydim[1], self.ychs))
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)
		self.teX = np.zeros((teSiz, self.xdimExt[0], self.xdimExt[1], self.xchs))
		self.teY = np.empty((teSiz, self.ydim[0], self.ydim[1], self.ychs))

	def prePropTr(self, trSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.zeros((trSiz, self.xdimExt[0], self.xdimExt[1], self.xchs))
		self.trY = np.empty((trSiz, self.ydim[0], self.ydim[1], self.ychs))
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)

	def prePropTe(self, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.teX = np.zeros((teSiz, self.xdimExt[0], self.xdimExt[1], self.xchs))
		self.teY = np.empty((teSiz, self.ydim[0], self.ydim[1], self.ychs))

	def prePropPr(self, prSiz: int):
		self.xNeeds = len(self.prevSets)
		self.prX = np.zeros((prSiz, self.xdimExt[0], self.xdimExt[1], self.xchs))
		self.prY = np.empty((prSiz, self.ydim[0], self.ydim[1], self.ychs))

	def pushBoth(self, index: int, trXFrag: np.ndarray, teXFrag: np.ndarray):
		"""forward propagation
		"""
		self.trX[:, self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1], self.startXs[index]:self.endXs[index]] = trXFrag
		self.teX[:, self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1], self.startXs[index]:self.endXs[index]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushBothX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushBoth(nind, self.trY, self.teY)

	def pushTr(self, index: int, trXFrag: np.ndarray):
		self.trX[:, self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1], self.startXs[index]:self.endXs[index]] = trXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTr(nind, self.trY)

	def pushTe(self, index: int, teXFrag: np.ndarray):
		self.teX[:, self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1], self.startXs[index]:self.endXs[index]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTeX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTe(nind, self.teY)

	def pushPr(self, index: int, prXFrag: np.ndarray):
		self.prX[:, self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1], self.startXs[index]:self.endXs[index]] = prXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushPrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushPr(nind, self.prY)

	def __pushXPartial(self, x: np.ndarray, y: np.ndarray, start: int, end: int):
		s0 = 0
		for i in range(self.ydim[0]):
			t0 = s0 + self.filSiz[0]
			s1 = 0
			for j in range(self.ydim[1]):
				y[start:end, i, j] = np.tensordot(x[start:end, s0:t0, s1:s1 + self.filSiz[1]], self.W, axes=3)
				s1 += self.stride[1]
			s0 += self.stride[0]
		self.trY[start:end] += self.B

	def __pushX(self, x: np.ndarray, y: np.ndarray):
		if y.shape[0] < 64:  # number of data
			self.__pushXPartial(x, y, 0, y.shape[0])
			return
		if y.shape[0] < 256:
			nth = 2  # the number of threads
		elif y.shape[0] < 512:
			nth = 4
		elif y.shape[0] < 1024:
			nth = 7
		else:
			nth = 10
		siz = (y.shape[0] - 1) // nth + 1
		ths = []
		start = 0
		for i in range(nth):
			th = Thread(target=self.__pushXPartial, args=(x, y, start, start + siz))
			ths.append(th)
			th.start()
			start += siz
		for th in ths:
			th.join()

	def pushTrX(self):
		"""forward propagation
			obtain self.trY from self.trX
			inherited from MidSet
		"""
		##############################################################################################
		# for i in range(self.trY.shape[0]):
		# 	for l in range(self.ychs):
		# 		self.trY[i, :, :, l] = self.B[l]
		# 		s0 = 0
		# 		for j in range(self.ydim[0]):
		# 			s1 = 0
		# 			for k in range(self.ydim[1]):
		# 				for m in range(self.filSiz[0]):
		# 					for n in range(self.filSiz[1]):
		# 						for o in range(self.xchs):
		# 							self.trY[i, j, k, l] += self.trX[i, s0+m, s1+n,o] * self.W[m, n, o, l]
		# 				s1 += self.stride[1]
		# 			s0 += self.stride[0]
		##############################################################################################
		########### single thread
		# s0 = 0
		# for j in range(self.ydim[0]):
		# 	s1 = 0
		# 	for k in range(self.ydim[1]):
		# 		self.trY[:, j, k] = np.tensordot(self.trX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]], self.W, axes=3)
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		# self.trY += self.B
		##############################################################################################
		self.__pushX(self.trX, self.trY)

	def pushBothX(self):
		"""forward propagation
			obtain self.trY and self.teY from self.trX and self.teX resp.
			inherited from MidSet
		"""
		if self.trY.shape[0] < 64:
			s0 = 0
			for i in range(self.ydim[0]):
				t0 = s0 + self.filSiz[0]
				s1 = 0
				for j in range(self.ydim[1]):
					self.trY[:, i, j] = np.tensordot(self.trX[:, s0:t0, s1:s1 + self.filSiz[1]], self.W, axes=3)
					self.teY[:, i, j] = np.tensordot(self.teX[:, s0:t0, s1:s1 + self.filSiz[1]], self.W, axes=3)
					s1 += self.stride[1]
				s0 += self.stride[0]
			self.trY += self.B
			self.teY += self.B
		else:
			self.__pushX(self.trX, self.trY)
			self.__pushX(self.teX, self.teY)

	def pushTeX(self):
		"""forward propagation
			obtain self.teY from self.teX resp.
			inherited from MidSet
		"""
		self.__pushX(self.teX, self.teY)

	def pushPrX(self):
		"""forward propagation
			obtain self.prY from self.prX
			inherited from MidSet
		"""
		self.__pushX(self.prX, self.prY)

	def resetForPull(self):
		self.gradNeeds = len(self.nextSets)
		# self.gradW.fill(0.0)
		# self.gradX.fill(0.0)
		self.gradY.fill(0.0)

	def pullGrad(self, gradYFrag: np.ndarray):
		"""backward propagation
		"""
		self.gradY += gradYFrag
		self.gradNeeds -= 1
		if self.gradNeeds == 0:
			self.pullGradY()
			for i, pset in enumerate(self.prevSets):
				pset.pullGrad(self.gradX[:, self.pad[0]:self.xdim[0] + self.pad[0], self.pad[1]:self.xdim[1] + self.pad[1], self.startXs[i]:self.endXs[i]])

	def __pullGradYPartial(self, w: np.ndarray, start: int, end: int):
		# xt = self.trX.transpose((1,2,3,0))
		s0 = 0
		for i in range(self.ydim[0]):
			t0 = s0 + self.filSiz[0]
			s1 = 0
			for j in range(self.ydim[1]):
				t1 = s1 + self.filSiz[1]
				# either
				self.gradX[start:end, s0:t0, s1:t1] += np.matmul(self.gradY[start:end, i, j], self.W.transpose((0, 1, 3, 2))).transpose((2, 0, 1, 3))
				# or
				# self.gradX[start:end, s0:t0, s1:t1] += np.tensordot(self.gradY[start:end, i, j], self.W.transpose((3, 0, 1, 2)), axes=1)
				# end either
				# either
				# w += np.tensordot(self.trX[start:end, s0:t0, s1:t1], self.gradY[start:end, i, j], axes=(0,0))
				# or
				w += np.matmul(self.trX[start:end, s0:t0, s1:t1].transpose((1, 2, 3, 0)), self.gradY[start:end, i, j])
				# w += np.matmul(xt[s0:t0, s1:t1, :, start:end], self.gradY[start:end, i, j])
				# end either
				s1 += self.stride[1]
			s0 += self.stride[0]

	def pullGradY(self):
		############## gradX ##############################################################################################################
		########## basic
		# self.gradX.fill(0.0)
		# s0 = 0
		# for j in range(self.ydim[0]):
		# 	s1 = 0
		# 	for k in range(self.ydim[1]):
		# 		for i in range(self.xchs):
		# 			for l in range(self.ychs):
		# 				for m in range(self.trX.shape[0]):
		# 					self.gradX[m, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1], i] += self.W[:, :, i, l] * self.gradY[m, j, k, l]
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		###################################################################################################################################
		# self.gradX.fill(0.0)
		# t0 = self.ydim[0] * self.stride[0]
		# t1 = self.ydim[1] * self.stride[1]
		# for i in range(self.filSiz[0]):
		# 	for j in range(self.filSiz[1]):
		# 		self.gradX[:, i:i + t0:self.stride[0], j:j + t1:self.stride[1]] += np.matmul(self.gradY, self.W[i, j].T)
		###################################################################################################################################
		# self.gradX.fill(0.0)
		# s0 = 0
		# for i in range(self.ydim[0]):
		# 	s1 = 0
		# 	for j in range(self.ydim[1]):
		# 		# either
		# 		self.gradX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]] += np.matmul(self.gradY[:, i, j], self.W.transpose((0, 1, 3, 2))).transpose((2, 0, 1, 3))
		# 		# or
		# 		# self.gradX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]] += np.tensordot(self.gradY[:, i, j], self.W.transpose((3, 0, 1, 2)), axes=1)
		# 		# end either
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		###################################################################################################################################

		################# gradW ###########################################################################################################
		########## basic
		# self.gradW.fill(0.0)
		# t0 = self.ydim[0] * self.stride[0]
		# t1 = self.ydim[1] * self.stride[1]
		# for i in range(self.xchs):
		# 	for j in range(self.ychs):
		# 		for k in range(self.filSiz[0]):
		# 			for l in range(self.filSiz[1]):
		# 				self.gradW[k, l, i, j] = np.sum(self.trX[:, k:k + t0:self.stride[0], l:l + t1:self.stride[1], i] * self.gradY[:, :, :, j])
		###################################################################################################################################
		# self.gradW.fill(0.0)
		# s0 = 0
		# for i in range(self.ydim[0]):
		# 	s1 = 0
		# 	for j in range(self.ydim[1]):
		# 		# one of
		# 		self.gradW += np.tensordot(self.trX[:, s0:s0+self.filSiz[0], s1:s1+self.filSiz[1]], self.gradY[:, i, j], axes=(0,0))
		# 		# or
		# 		# self.gradW += np.matmul(self.trX[:, s0:s0+self.filSiz[0], s1:s1+self.filSiz[1]].transpose((1,2,3,0)), self.gradY[:, i, j])
		# 		# end one of
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		###################################################################################################################################
		################## single thread
		# self.gradW.fill(0.0)
		# self.gradX.fill(0.0)
		# if self.biased:
		# 	self.gradY.sum((0, 1, 2), out=self.gradB)
		# # xt = self.trX.transpose((1,2,3,0))
		# s0 = 0
		# for i in range(self.ydim[0]):
		# 	t0 = s0 + self.filSiz[0]
		# 	s1 = 0
		# 	for j in range(self.ydim[1]):
		# 		t1 = s1 + self.filSiz[1]
		# 		# either
		# 		self.gradX[:, s0:t0, s1:t1] += np.matmul(self.gradY[:, i, j], self.W.transpose((0, 1, 3, 2))).transpose((2, 0, 1, 3))
		# 		# or
		# 		# self.gradX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]] += np.tensordot(self.gradY[:, i, j], self.W.transpose((3, 0, 1, 2)), axes=1)
		# 		# end either
		# 		# either
		# 		# self.gradW += np.tensordot(self.trX[:, s0:s0+self.filSiz[0], s1:s1+self.filSiz[1]], self.gradY[:, i, j], axes=(0,0))
		# 		# or
		# 		self.gradW += np.matmul(self.trX[:, s0:t0, s1:t1].transpose((1, 2, 3, 0)), self.gradY[:, i, j])
		# 		# self.gradW += np.matmul(xt[s0:s0+self.filSiz[0], s1:s1+self.filSiz[1]], self.gradY[:, i, j])
		# 		# end either
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		# self.gradW += self.reg.grad(self.W)
		###################################################################################################################################
		if self.biased:
			self.gradY.sum((0, 1, 2), out=self.gradB)
		self.gradX.fill(0.0)
		if self.trY.shape[0] < 64:  # number of data
			self.gradW.fill(0.0)
			self.__pullGradYPartial(self.gradW, 0, self.trY.shape[0])
			self.gradW += self.reg.grad(self.W)
			return
		elif self.trY.shape[0] < 256:
			nth = 4  # the number of threads
		elif self.trY.shape[0] < 512:
			nth = 6
		elif self.trY.shape[0] < 1024:
			nth = 8
		elif self.trY.shape[0] < 2048:
			nth = 14
		else:
			nth = 16
		siz = (self.trY.shape[0] - 1) // nth + 1
		ws = np.zeros((nth,) + self.W.shape)
		ths = []
		start = 0
		for w in ws:
			th = Thread(target=self.__pullGradYPartial, args=(w, start, start + siz))
			ths.append(th)
			th.start()
			start += siz
		for th in ths:
			th.join()
		ws.sum(0, out=self.gradW)
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
		np.subtract(self.BPrev, np.multiply(lr, self.gradBPrev, self.B), self.B)
		np.subtract(self.WPrev, np.multiply(lr, self.gradWPrev, self.W), self.W)

	def sendCurrToPrev(self):
		self.B, self.BPrev = self.BPrev, self.B
		self.W, self.WPrev = self.WPrev, self.W
		self.gradB, self.gradBPrev = self.gradBPrev, self.gradB
		self.gradW, self.gradWPrev = self.gradWPrev, self.gradW

	# noinspection PyTypeChecker
	def gradDot(self) -> float:
		return np.sum(self.gradB * self.gradBPrev) + np.sum(self.gradW * self.gradWPrev)

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, {self.filSiz})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, {self.filSiz}, ({self.xdim}, {self.xchs})>({self.ydim},{self.ychs}),' \
			f' {self.biased}, {self.pad}, {self.stride};'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, {self.filSiz}, {self.ychs}, {self.biased}, {self.pad}, {self.stride};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'
