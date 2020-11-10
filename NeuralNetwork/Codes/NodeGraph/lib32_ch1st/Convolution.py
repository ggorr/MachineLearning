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
		self.mom: float = F0

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
			self.initB = np.zeros((self.ychs, 1, 1), np.float32)
		elif self.initB.ndim == 1:
			self.initB = self.initB[:, np.newaxis, np.newaxis]
		if self.initW is None:
			self.initW = np.random.random((self.ychs, self.xchs, self.filSiz)).astype(np.float32)
		self.B = self.initB.astype(np.float32) if self.biased else np.zeros((self.ychs, 1, 1), np.float32)
		self.W = self.initW.astype(np.float32)
		self.BPrev = np.zeros_like(self.B)
		self.WPrev = np.zeros_like(self.W)
		self.gradB = np.zeros_like(self.B)
		self.gradW = np.zeros_like(self.W)
		self.gradBPrev = np.zeros_like(self.B)
		self.gradWPrev = np.zeros_like(self.W)

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.zeros((self.xchs, self.xdimExt, trSiz), np.float32)
		self.trY = np.empty((self.ychs, self.ydim, trSiz), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)
		self.teX = np.zeros((self.xchs, self.xdimExt, teSiz), np.float32)
		self.teY = np.empty((self.ychs, self.ydim, teSiz), np.float32)

	def prePropTr(self, trSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.zeros((self.xchs, self.xdimExt, trSiz), np.float32)
		self.trY = np.empty((self.ychs, self.ydim, trSiz), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)

	def prePropTe(self, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.teX = np.zeros((self.xchs, self.xdimExt, teSiz), np.float32)
		self.teY = np.empty((self.ychs, self.ydim, teSiz), np.float32)

	def prePropPr(self, prSiz: int):
		self.xNeeds = len(self.prevSets)
		self.prX = np.zeros((self.xchs, self.xdimExt, prSiz), np.float32)
		self.prY = np.empty((self.ychs, self.ydim, prSiz), np.float32)

	def pushBoth(self, index: int, trXFrag: np.ndarray, teXFrag: np.ndarray):
		"""forward propagation
		"""
		self.trX[self.startXs[index]:self.endXs[index], self.pad:self.pad + self.xdim] = trXFrag
		self.teX[self.startXs[index]:self.endXs[index], self.pad:self.pad + self.xdim] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushBothX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushBoth(nind, self.trY, self.teY)

	def pushTr(self, index: int, trXFrag: np.ndarray):
		self.trX[self.startXs[index]:self.endXs[index], self.pad:self.pad + self.xdim] = trXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTr(nind, self.trY)

	def pushTe(self, index: int, teXFrag: np.ndarray):
		self.teX[self.startXs[index]:self.endXs[index], self.pad:self.pad + self.xdim] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTeX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTe(nind, self.teY)

	def pushPr(self, index: int, prXFrag: np.ndarray):
		self.prX[self.startXs[index]:self.endXs[index], self.pad:self.pad + self.xdim] = prXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushPrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushPr(nind, self.prY)

	def __pushPartial(self, x: np.ndarray, y: np.ndarray, start: int, end: int):
		s = 0
		for k in range(self.ydim):
			y[:, k, start:end] = np.tensordot(self.W, x[:, s:s + self.filSiz, start:end])
			s += self.stride
		y[:, :, start:end] += self.B

	def __pushX(self, x, y):
		if y.shape[2] < 1024:  # number of data
			self.__pushPartial(x, y, 0, y.shape[2])
		else:
			# import multiprocessing
			# nth = 2 * multiprocessing.cpu_count()
			nth = 16  # number of threads
			siz = (y.shape[2] - 1) // nth + 1
			ths = []
			start = 0
			for i in range(nth):
				th = Thread(target=self.__pushPartial, args=(x, y, start, start + siz))
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
		#### definition
		# for i in range(self.ychs):
		# 	s = 0
		# 	for j in range(self.ydim):
		# 		for k in range(self.trY.shape[2]):
		# 			self.trY[i, j, k] = 0.0
		# 			for l in range(self.xchs):
		# 				self.trY[i, j, k] += self.B[i, l, 0]
		# 				for m in range(self.filSiz):
		# 					self.trY[i, j, k] += self.W[i, l, m] * self.trX[l, s+m, k]
		# 		s += self.stride
		#########################################################################################
		#### slow version
		# t = self.ydim * self.stride
		# for i in range(self.ychs):
		# 	self.trY[i, :, :] = np.sum(self.B[i, :, 0])
		# 	for j in range(self.xchs):
		# 		for k in range(self.filSiz):
		# 			self.trY[i] += self.W[i, j, k] * self.trX[j, k:k + t:self.stride]
		#########################################################################################
		# t = self.ydim * self.stride
		# self.trY[:] = self.B.sum(1, keepdims=True)
		# for k in range(self.filSiz):
		# 	self.trY += np.tensordot(self.W[:, :, k], self.trX[:, k:k + t:self.stride], axes=1)
		#########################################################################################
		# s = 0
		# for k in range(self.ydim):
		# 	self.trY[:, k] = np.matmul(self.W.transpose((1,0,2)), self.trX[:, s:s + self.filSiz]).sum(0)
		# 	s += self.stride
		# self.trY += self.B.sum(1, keepdims=True)
		#########################################################################################
		s = 0
		for k in range(self.ydim):
			self.trY[:, k] = np.tensordot(self.W, self.trX[:, s:s + self.filSiz])
			s += self.stride
		self.trY += self.B
		#########################################################################################
		self.__pushX(self.trX, self.trY)

	def pushBothX(self):
		"""forward propagation
			obtain self.trY and self.teY from self.trX and self.teX resp.
			inherited from MidSet
		"""
		if self.trY.shape[2] < 1024:
			s = 0
			for k in range(self.ydim):
				self.trY[:, k] = np.tensordot(self.W, self.trX[:, s:s + self.filSiz])
				self.teY[:, k] = np.tensordot(self.W, self.teX[:, s:s + self.filSiz])
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
		# self.gradW.fill(F0)
		# self.gradX.fill(F0)
		self.gradY.fill(F0)

	def pullGrad(self, gradYFrag: np.ndarray):
		"""backward propagation
		"""
		self.gradY += gradYFrag
		self.gradNeeds -= 1
		if self.gradNeeds == 0:
			self.pullGradY()
			for i, pset in enumerate(self.prevSets):
				pset.pullGrad(self.gradX[self.startXs[i]:self.endXs[i], self.pad:self.xdim + self.pad])

	def __pullGradYPartial(self, w, start, end):
		s = 0
		for k in range(self.ydim):
			t = s + self.filSiz
			self.gradX[:, s:t, start:end] += np.matmul(self.W.transpose((1, 2, 0)), self.gradY[:, k, start:end])
			w += np.tensordot(self.gradY[:, k, start:end], self.trX[:, s:t, start:end], axes=(1, 2))
			# for j in range(self.xchs):
			# 	self.gradX[j, s:t, start:end] += np.matmul(self.W[:, j].T, self.gradY[:, k, start:end])
			# 	w[:,j] += np.matmul(self.gradY[:, k, start:end], self.trX[j, s:t, start:end].T)
			s += self.stride

	def pullGradY(self):
		##################################################################################################
		# t = self.ydim * self.stride
		# for j in range(self.xchs):
		# 	for i in range(self.ychs):
		# 		for k in range(self.filSiz):
		# 			self.gradW[i, j, k] = (self.trX[j, k:k+t:self.stride, :] * self.gradY[i, :, :]).sum()
		#####################################################################################################
		#### fast
		# s = 0
		# for k in range(self.ydim):
		# 	# either
		# 	# for j in range(self.xchs):
		# 	# 	self.gradW[:, j] += np.matmul(self.gradY[:, k, :], self.trX[j, s:s+self.filSiz, :].T)
		# 	# or
		# 	self.gradW += np.tensordot(self.gradY[:, k], self.trX[:, s:s + self.filSiz], axes=(1, 2))
		# 	# end either
		# 	s += self.stride
		################################################################################################

		################################ gradX #########################################################
		# #### slow version
		# t = self.ydim * self.stride
		# for k in range(self.filSiz):
		# 	for j in range(self.xchs):
		# 		for i in range(self.ychs):
		# 			self.gradX[j, k:k+t:self.stride, :] += self.W[i, j, k] * self.gradY[i, :, :]
		################################################################################################
		#### similar to below
		# s = 0
		# for k in range(self.ydim):
		# 	for j in range(self.xchs):
		# 		self.gradX[j, s:s + self.filSiz] += np.matmul(self.W[:, j].T, self.gradY[:, k])
		# 	s += self.stride
		################################################################################################
		#### fast
		# s = 0
		# for k in range(self.ydim):
		# 	self.gradX[:, s:s + self.filSiz, :] += np.matmul(self.W.transpose((1, 2, 0)), self.gradY[:, k, :])
		# 	s += self.stride
		################################################################################################
		# s = 0
		# wt = self.W.transpose((1, 2, 0))
		# for k in range(self.ydim):
		# 	self.gradX[:, s:s + self.filSiz] += np.matmul(wt, self.gradY[:, k])
		# 	s += self.stride
		################################################################################################
		########## single thread version
		# if self.biased:
		# 	np.sum(self.gradY, (1, 2), out=self.gradB, keepdims=True)
		# xt = self.trX.transpose((0, 2, 1))
		# s = 0
		# for k in range(self.ydim):
		# 	# self.gradX[:, s:s + self.filSiz] += np.matmul(wt, self.gradY[:, k])
		# 	# self.gradW += np.tensordot(self.gradY[:, k], self.trX[:, s:s + self.filSiz], axes=(1, 2))
		# 	for j in range(self.xchs):
		# 		self.gradX[j, s:s + self.filSiz] += np.matmul(self.W[:, j].T, self.gradY[:, k])
		# 		self.gradW[:, j] += np.matmul(self.gradY[:, k], xt[j, :, s:s + self.filSiz])
		# 	s += self.stride
		# self.gradW += self.reg.grad(self.W)
		################################################################################################
		if self.biased:
			np.sum(self.gradY, (1, 2), out=self.gradB, keepdims=True)
		self.gradX.fill(F0)
		if self.trY.shape[2] < 1024:  # size of data
			self.gradW.fill(F0)
			self.__pullGradYPartial(self.gradW, 0, self.trY.shape[2])
		else:
			# import multiprocessing
			# nth = 2 * multiprocessing.cpu_count()
			nth = 16  # number of threads
			siz = (self.trY.shape[2] - 1) // nth + 1
			ws = np.zeros((nth,) + self.W.shape, np.float32)
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
		self.mom: float = F0

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
			self.initB = np.zeros((self.ychs, 1, 1, 1), np.float32)
		elif self.initB.ndim == 1:
			self.initB = self.initB[:, np.newaxis, np.newaxis, np.newaxis]
		if self.initW is None:
			self.initW = np.random.random((self.ychs, self.xchs, self.filSiz[0], self.filSiz[1])).astype(np.float32)
		self.B = self.initB.astype(np.float32) if self.biased else np.zeros((self.ychs, 1, 1, 1), np.float32)
		self.W = self.initW.astype(np.float32)
		self.BPrev = np.zeros_like(self.B)
		self.WPrev = np.zeros_like(self.W)
		self.gradB = np.zeros_like(self.B)
		self.gradW = np.zeros_like(self.W)
		self.gradBPrev = np.zeros_like(self.B)
		self.gradWPrev = np.zeros_like(self.W)

	def prePropBoth(self, trSiz: int, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.zeros((self.xchs, self.xdimExt[0], self.xdimExt[1], trSiz), np.float32)
		self.trY = np.empty((self.ychs, self.ydim[0], self.ydim[1], trSiz), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)
		self.teX = np.zeros((self.xchs, self.xdimExt[0], self.xdimExt[1], teSiz), np.float32)
		self.teY = np.empty((self.ychs, self.ydim[0], self.ydim[1], teSiz), np.float32)

	def prePropTr(self, trSiz: int):
		self.xNeeds = len(self.prevSets)
		self.trX = np.zeros((self.xchs, self.xdimExt[0], self.xdimExt[1], trSiz), np.float32)
		self.trY = np.empty((self.ychs, self.ydim[0], self.ydim[1], trSiz), np.float32)
		self.gradX = np.empty_like(self.trX)
		self.gradY = np.empty_like(self.trY)

	def prePropTe(self, teSiz: int):
		self.xNeeds = len(self.prevSets)
		self.teX = np.zeros((self.xchs, self.xdimExt[0], self.xdimExt[1], teSiz), np.float32)
		self.teY = np.empty((self.ychs, self.ydim[0], self.ydim[1], teSiz), np.float32)

	def prePropPr(self, prSiz: int):
		self.xNeeds = len(self.prevSets)
		self.prX = np.zeros((self.xchs, self.xdimExt[0], self.xdimExt[1], prSiz), np.float32)
		self.prY = np.empty((self.ychs, self.ydim[0], self.ydim[1], prSiz), np.float32)

	def pushBoth(self, index: int, trXFrag: np.ndarray, teXFrag: np.ndarray):
		"""forward propagation
		"""
		self.trX[self.startXs[index]:self.endXs[index], self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1]] = trXFrag
		self.teX[self.startXs[index]:self.endXs[index], self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushBothX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushBoth(nind, self.trY, self.teY)

	def pushTr(self, index: int, trXFrag: np.ndarray):
		self.trX[self.startXs[index]:self.endXs[index], self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1]] = trXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTr(nind, self.trY)

	def pushTe(self, index: int, teXFrag: np.ndarray):
		self.teX[self.startXs[index]:self.endXs[index], self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1]] = teXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushTeX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushTe(nind, self.teY)

	def pushPr(self, index: int, prXFrag: np.ndarray):
		self.prX[self.startXs[index]:self.endXs[index], self.pad[0]:self.pad[0] + self.xdim[0], self.pad[1]:self.pad[1] + self.xdim[1]] = prXFrag
		self.xNeeds -= 1
		if self.xNeeds == 0:
			self.pushPrX()
			for nset, nind in zip(self.nextSets, self.nextInd):
				nset.pushPr(nind, self.prY)

	def __pushXPartial(self, x: np.ndarray, y: np.ndarray, start: int, end: int):
		s0 = 0
		for i in range(self.ydim[0]):
			s1 = 0
			t0 = s0 + self.filSiz[0]
			for j in range(self.ydim[1]):
				y[:, i, j, start:end] = np.tensordot(self.W, x[:, s0:t0, s1:s1 + self.filSiz[1], start:end], axes=3)
				s1 += self.stride[1]
			s0 += self.stride[0]
		y[:, :, :, start:end] += self.B

	def __pushX(self, x: np.ndarray, y: np.ndarray):
		if y.shape[3] < 32:  # number of data
			self.__pushXPartial(x, y, 0, y.shape[3])
			return
		if y.shape[3] < 256:
			nth = 2  # the number of threads
		elif y.shape[3] < 512:
			nth = 4
		elif y.shape[3] < 1024:
			nth = 7
		else:
			nth = 10
		siz = (y.shape[3] - 1) // nth + 1
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
		# self.trY[:] = self.B
		# s0 = 0
		# for j in range(self.ydim[0]):
		# 	s1 = 0
		# 	for k in range(self.ydim[1]):
		# 		self.trY[:, j, k] += np.sum(self.W[:, :, :, :, np.newaxis] * self.trX[np.newaxis, :, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]], (1, 2, 3))
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		##############################################################################################
		# s0 = 0
		# for k in range(self.ydim[0]):
		# 	s1 = 0
		# 	for l in range(self.ydim[1]):
		# 		self.trY[:, k , l] = np.matmul(self.W[:, :, :, np.newaxis], self.trX[:, s0:s0+self.filSiz[0], s1:s1+self.filSiz[1]]).sum((1, 2, 3))
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		# self.trY += self.B
		##############################################################################################
		########### single thread
		# s0 = 0
		# for k in range(self.ydim[0]):
		# 	s1 = 0
		# 	for l in range(self.ydim[1]):
		# 		self.trY[:, k, l] = np.tensordot(self.W, self.trX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]], axes=3)
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
		if self.trY.shape[3] < 32:
			s0 = 0
			for i in range(self.ydim[0]):
				t0 = s0 + self.filSiz[0]
				s1 = 0
				for j in range(self.ydim[1]):
					self.trY[:, i, j] = np.tensordot(self.W, self.trX[:, s0:t0, s1:s1 + self.filSiz[1]], axes=3)
					self.teY[:, i, j] = np.tensordot(self.W, self.teX[:, s0:t0, s1:s1 + self.filSiz[1]], axes=3)
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
		# self.gradW.fill(F0)
		# self.gradX.fill(F0)
		self.gradY.fill(F0)

	def __pullGradYPartial(self, w: np.ndarray, start: int, end: int):
		# xt = self.trX.transpose((0, 1, 3, 2))
		# yt = self.gradY.transpose((3, 1, 2, 0))
		s0 = 0
		for i in range(self.ydim[0]):
			t0 = s0 + self.filSiz[0]
			s1 = 0
			for j in range(self.ydim[1]):
				self.gradX[:, s0:t0, s1:s1 + self.filSiz[1], start:end] += np.matmul(self.W.transpose((1, 2, 3, 0)), self.gradY[:, i, j, start:end])
				w += np.tensordot(self.gradY[:, i, j, start:end], self.trX[:, s0:t0, s1:s1 + self.filSiz[1], start:end], axes=(1, 3))
				# w += np.dot(self.gradY[:, i, j, start:end], xt[:, s0:t0, start:end, s1:s1 + self.filSiz[1]])
				# w += np.dot(self.trX[:, s0:t0, s1:s1 + self.filSiz[1], start:end], yt[start:end, i, j, :]).transpose((3, 0, 1, 2))
				s1 += self.stride[1]
			s0 += self.stride[0]

	def pullGrad(self, gradYFrag: np.ndarray):
		"""backward propagation
		"""
		self.gradY += gradYFrag
		self.gradNeeds -= 1
		if self.gradNeeds == 0:
			self.pullGradY()
			for i, pset in enumerate(self.prevSets):
				pset.pullGrad(self.gradX[self.startXs[i]:self.endXs[i], self.pad[0]:self.xdim[0] + self.pad[0], self.pad[1]:self.xdim[1] + self.pad[1]])

	def pullGradY(self):
		############## gradX ##############################################################################################################
		########## basic
		# s0 = 0
		# for j in range(self.ydim[0]):
		# 	s1 = 0
		# 	for k in range(self.ydim[1]):
		# 		for i in range(self.xchs):
		# 			for l in range(self.ychs):
		# 				self.gradX[i, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]] += self.W[l, i, :, :, np.newaxis] * self.gradY[l, j, k]
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		###################################################################################################################################
		# wt = self.W.transpose((1, 2, 3, 0))
		# s0 = 0
		# for i in range(self.ydim[0]):
		# 	s1 = 0
		# 	for j in range(self.ydim[1]):
		# 		# either
		# 		self.gradX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]] += np.matmul(wt, self.gradY[:, i, j])
		# 		# or
		# 		# self.gradX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]] += np.tensordot(self.W, self.gradY[:, i, j], axes=(0, 0))
		# 		# end either
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		###################################################################################################################################

		################# gradW ###########################################################################################################
		########## basic
		# t0 = self.ydim[0] * self.stride[0]
		# t1 = self.ydim[1] * self.stride[1]
		# for i in range(self.xchs):
		# 	for j in range(self.ychs):
		# 		for k in range(self.filSiz[0]):
		# 			for l in range(self.filSiz[1]):
		# 				self.gradW[j, i, k, l] = np.sum(self.trX[i, k:k+t0:self.stride[0], l:l+t1:self.stride[1]] * self.gradY[j])
		###################################################################################################################################
		# xt = self.trX.transpose((0, 1, 3, 2))
		# yt = self.gradY.transpose((3, 1, 2, 0))
		# s0 = 0
		# for i in range(self.ydim[0]):
		# 	s1 = 0
		# 	for j in range(self.ydim[1]):
		# 		# one of
		# 		# for k in range(self.xchs):
		# 		# 	self.gradW[:, k] += np.dot(self.gradY[:, i, j, :], xt[k, s0:s0 + self.filSiz[0], :, s1:s1 + self.filSiz[1]])
		# 		# or
		# 		# self.gradW += np.tensordot(self.gradY[:, i, j], self.trX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]], axes=(1, 3))
		# 		# or
		# 		# self.gradW += np.dot(self.gradY[:, i, j], xt[:, s0:s0 + self.filSiz[0], :, s1:s1 + self.filSiz[1]])
		# 		# or
		# 		# self.gradW += np.dot(self.trX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]], yt[:, i, j]).transpose((3, 0, 1, 2))
		# 		# end one of
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		###################################################################################################################################
		############## single thread
		# if self.biased:
		# 	self.gradY.sum((1, 2, 3), out=self.gradB, keepdims=True)
		# wt = self.W.transpose((1, 2, 3, 0))
		# # xt = self.trX.transpose((0, 1, 3, 2))
		# # yt = self.gradY.transpose((3, 1, 2, 0))
		# s0 = 0
		# for i in range(self.ydim[0]):
		# 	s1 = 0
		# 	for j in range(self.ydim[1]):
		# 		self.gradX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]] += np.matmul(wt, self.gradY[:, i, j])
		# 		self.gradW += np.tensordot(self.gradY[:, i, j], self.trX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]], axes=(1, 3))
		# 		# self.gradW += np.dot(self.gradY[:, i, j], xt[:, s0:s0 + self.filSiz[0], :, s1:s1 + self.filSiz[1]])
		# 		# self.gradW += np.dot(self.trX[:, s0:s0 + self.filSiz[0], s1:s1 + self.filSiz[1]], yt[:, i, j]).transpose((3, 0, 1, 2))
		# 		s1 += self.stride[1]
		# 	s0 += self.stride[0]
		# self.gradW += self.reg.grad(self.W)
		###################################################################################################################################

		if self.biased:
			self.gradY.sum((1, 2, 3), out=self.gradB, keepdims=True)
		self.gradX.fill(F0)
		if self.trY.shape[3] < 11:  # number of data
			self.gradW.fill(F0)
			self.__pullGradYPartial(self.gradW, 0, self.trY.shape[3])
			self.gradW += self.reg.grad(self.W)
			return
		elif self.trY.shape[3] < 21:
			nth = 2  # the number of threads
		elif self.trY.shape[3] < 64:
			nth = 3
		elif self.trY.shape[3] < 256:
			nth = 4
		elif self.trY.shape[3] < 512:
			nth = 6
		elif self.trY.shape[3] < 1024:
			nth = 8
		elif self.trY.shape[3] < 1536:
			nth = 10
		else:
			nth = 16
		siz = (self.trY.shape[3] - 1) // nth + 1
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

	def updateBatch(self, lr: np.float32):
		"""
		:param lr: learning rate
		"""
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
