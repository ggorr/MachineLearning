from scipy.special import expit, softmax

from .MidSet import *


class Activation1D(MidSet1D, metaclass=ABCMeta):
	"""Activation
	"""

	def __init__(self, name: str = None):
		super().__init__(UNKNOWN, UNKNOWN, name)

	def compile(self) -> bool:
		if super().compile():
			self.ydim = self.xdim
			self.ychs = self.xchs
			return True
		else:
			return False

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


class Identity1D(Activation1D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushBothX(self):
		self.trY[:] = self.trX
		self.teY[:] = self.teX

	def pushTrX(self):
		self.trY[:] = self.trX

	def pushTeX(self):
		self.teY[:] = self.teX

	def pushPrX(self):
		self.prY[:] = self.prX

	def pullGradY(self):
		self.gradX[:] = self.gradY


class Relu1D(Activation1D):
	EPSILON = 1.0e-300
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushBothX(self):
		np.multiply(self.trX > Relu1D.EPSILON, self.trX, self.trY)
		np.multiply(self.teX > Relu1D.EPSILON, self.teX, self.teY)

	def pushTrX(self):
		"""trY = (trX > 0) * trX
		"""
		np.multiply(self.trX > Relu1D.EPSILON, self.trX, self.trY)

	def pushTeX(self):
		np.multiply(self.teX > Relu1D.EPSILON, self.teX, self.teY)

	def pushPrX(self):
		np.multiply(self.prX > Relu1D.EPSILON, self.prX, self.prY)

	def pullGradY(self):
		"""
		gradX = (trX > 0) * gradY
		"""
		np.multiply(self.trX > Relu1D.EPSILON, self.gradY, self.gradX)


class Sigmoid1D(Activation1D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushTrX(self):
		expit(self.trX, self.trY)

	def pushBothX(self):
		expit(self.trX, self.trY)
		expit(self.teX, self.teY)

	def pushTeX(self):
		expit(self.teX, self.teY)

	def pushPrX(self):
		expit(self.prX, self.prY)

	def pullGradY(self):
		"""
		gradX = df * gradY, df = (1 - y) * y
		"""
		np.multiply((1 - self.trY) * self.trY, self.gradY, self.gradX)


class Softplus1D(Activation1D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushTrX(self):
		np.logaddexp(0, self.trX, self.trY)

	def pushTeX(self):
		np.logaddexp(0, self.teX, self.teY)

	def pushPrX(self):
		np.logaddexp(0, self.prX, self.prY)

	def pushBothX(self):
		np.logaddexp(0, self.trX, self.trY)
		np.logaddexp(0, self.teX, self.teY)

	def pullGradY(self):
		"""
		gradX = df * gradY,  df = 1 - np.exp(-y)
		"""
		np.multiply(1 - np.exp(-self.trY), self.gradY, self.gradX)


class Tanh1D(Activation1D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushTrX(self):
		np.tanh(self.trX, self.trY)

	def pushTeX(self):
		np.tanh(self.teX, self.teY)

	def pushPrX(self):
		np.tanh(self.prX, self.prY)

	def pushBothX(self):
		np.tanh(self.trX, self.trY)
		np.tanh(self.teX, self.teY)

	def pullGradY(self):
		"""
		gradX = df * gradY,  df = 1 - y * y
		"""
		np.multiply(1 - self.trY * self.trY, self.gradY, self.gradX)


class Softmax1D(Activation1D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushTrX(self):
		self.trY[:] = softmax(self.trX, axis=1)

	def pushBothX(self):
		self.trY[:] = softmax(self.trX, axis=1)
		self.teY[:] = softmax(self.teX, axis=1)

	def pushTeX(self):
		self.teY[:] = softmax(self.teX, axis=1)

	def pushPrX(self):
		self.prY[:] = softmax(self.prX, axis=1)

	def pullGradY(self):
		"""
		suppose dimY = 2 and #data = n
		dsoftmax = [y_1-y_1^2  -y_1*y_2 ] = dsoftmax^T
				   [-y_2*y_1   y_2-y_2^2]
		gradY =	[dL/dy_11  ...  dL/dy_1n]
				[dL/dy_21  ...  dL/dy_2n]
		Let Y_i = [ y_1i ], gradY_i = [ dL/dy_1i ] and dsoftmax_i = [ y_1i-y_1i^2  -y_1i*y_2i  ]
		          [ y_2i ]			  [ dL/dy_2i ]					[ -y_2i*y_1i   y_2i-y_2i^2 ]
		gradX_i = dsoftmax_i gradY_i
				= [ (y_1i-y_1i^2) * dL/dy_1i - y_1i*y_2i * dL/dy_2i  ]
				  [ -y_2i*y_1i * dL/dy_1i + (y_2i-y_2i^2) * dL/dy_2i ]
				= [ y_1i*(dL/dy_1i*(1-y_1i) - dL/dy_2i*y_2i)  ]
				  [ y_2i*(-dL/dy_1i*y_1i + dL/dy_2i*(1-y_2i)) ]
				= Y_i * [ dL/dy_1i*(1-y_1i) - dL/dy_2i*y_2i  ]
						[ -dL/dy_1i*y_1i + dL/dy_2i*(1-y_2i) ]
				= Y_i * [ dL/dy_1i - dL/dy_1i*y_1i - dL/dy_2i*y_2i ]
						[ dL/dy_2i - dL/dy_1i*y_1i - dL/dy_2i*y_2i ]
				= Y_i * (gradY_i - [ dL/dy_1i*y_1i + dL/dy_2i*y_2i ])
								   [ dL/dy_1i*y_1i + dL/dy_2i*y_2i ]
				= Y_i * (gradY_i - sum(gradY_i * Y_i) * [ 1 ])
														[ 1 ]
		gradX = dsoftmax gradY
			  = Y * (gradY - [ column_sum(Y*gradY) ])
		 					 [ column_sum(Y*gradY) ]
		using numpy
			gradX = dsoftmax gradY = Y * (gradY - np.sum(Y * gradY, axis=0, keepdims=True))
		or
			gradX = dsoftmax gradY = Y * (gradY - np.sum(Y * gradY, axis=0))
		"""
		tempSum = np.sum(self.trY * self.gradY, axis=1, keepdims=True)
		np.multiply(self.trY, self.gradY - tempSum, self.gradX)


class Activation2D(MidSet2D, metaclass=ABCMeta):
	"""Activation
	"""

	def __init__(self, name: str = None):
		super().__init__(UNKNOWN, UNKNOWN, name)

	def compile(self) -> bool:
		if super().compile():
			self.ydim:tuple = self.xdim
			self.ychs:int = self.xchs
			return True
		else:
			return False

	def shortStr(self) -> str:
		return f'{self.__class__.__name__}({self.name}, ({self.ydim[0]},{self.ydim[1]}), {self.ychs})'

	def __str__(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name}, ({self.ydim[0]},{self.ydim[1]}), {self.ychs};'
		for pset in self.prevSets:
			s += f' {pset.name},'
		return s[:-1] + ')'

	def saveStr(self) -> str:
		s = f'{self.__class__.__name__}({self.Id}, {self.name};'
		for pset in self.prevSets:
			s += f' {pset.Id},'
		return s[:-1] + ')'


class Identity2D(Activation2D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushBothX(self):
		self.trY[:] = self.trX
		self.teY[:] = self.teX

	def pushTrX(self):
		self.trY[:] = self.trX

	def pushTeX(self):
		self.teY[:] = self.teX

	def pushPrX(self):
		self.prY[:] = self.prX

	def pullGradY(self):
		self.gradX[:] = self.gradY


class Relu2D(Activation2D):
	EPSILON = 1.0e-300
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushBothX(self):
		np.multiply(self.trX > Relu2D.EPSILON, self.trX, self.trY)
		np.multiply(self.teX > Relu2D.EPSILON, self.teX, self.teY)

	def pushTrX(self):
		"""trY = (trX > 0) * trX
		"""
		np.multiply(self.trX > Relu2D.EPSILON, self.trX, self.trY)

	def pushTeX(self):
		np.multiply(self.teX > Relu2D.EPSILON, self.teX, self.teY)

	def pushPrX(self):
		np.multiply(self.prX > Relu2D.EPSILON, self.prX, self.prY)

	def pullGradY(self):
		"""
		gradX = (trX > 0) * gradY
		"""
		np.multiply(self.trX > Relu2D.EPSILON, self.gradY, self.gradX)


class Sigmoid2D(Activation2D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushTrX(self):
		expit(self.trX, self.trY)

	def pushBothX(self):
		expit(self.trX, self.trY)
		expit(self.teX, self.teY)

	def pushTeX(self):
		expit(self.teX, self.teY)

	def pushPrX(self):
		expit(self.prX, self.prY)

	def pullGradY(self):
		"""
		gradX = df * gradY, df = (1 - y) * y
		"""
		np.multiply((1 - self.trY) * self.trY, self.gradY, self.gradX)


class Softplus2D(Activation2D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushTrX(self):
		np.logaddexp(0, self.trX, self.trY)

	def pushTeX(self):
		np.logaddexp(0, self.teX, self.teY)

	def pushPrX(self):
		np.logaddexp(0, self.prX, self.prY)

	def pushBothX(self):
		np.logaddexp(0, self.trX, self.trY)
		np.logaddexp(0, self.teX, self.teY)

	def pullGradY(self):
		"""
		gradX = df * gradY,  df = 1 - np.exp(-y)
		"""
		np.multiply(1 - np.exp(-self.trY), self.gradY, self.gradX)


class Tanh2D(Activation2D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushTrX(self):
		np.tanh(self.trX, self.trY)

	def pushTeX(self):
		np.tanh(self.teX, self.teY)

	def pushPrX(self):
		np.tanh(self.prX, self.prY)

	def pushBothX(self):
		np.tanh(self.trX, self.trY)
		np.tanh(self.teX, self.teY)

	def pullGradY(self):
		"""
		gradX = df * gradY,  df = 1 - y * y
		"""
		np.multiply(1 - self.trY * self.trY, self.gradY, self.gradX)


class Softmax2D(Activation2D):
	def __init__(self, name: str = None):
		super().__init__(name)

	def pushTrX(self):
		self.trY[:] = softmax(self.trX, axis=(1,2))

	def pushBothX(self):
		self.trY[:] = softmax(self.trX, axis=(1,2))
		self.teY[:] = softmax(self.teX, axis=(1,2))

	def pushTeX(self):
		self.teY[:] = softmax(self.teX, axis=(1,2))

	def pushPrX(self):
		self.prY[:] = softmax(self.prX, axis=(1,2))

	def pullGradY(self):
		"""
		suppose dimY = 2 and #data = n
		dsoftmax = [y_1-y_1^2  -y_1*y_2 ] = dsoftmax^T
				   [-y_2*y_1   y_2-y_2^2]
		gradY =	[dL/dy_11  ...  dL/dy_1n]
				[dL/dy_21  ...  dL/dy_2n]
		Let Y_i = [ y_1i ], gradY_i = [ dL/dy_1i ] and dsoftmax_i = [ y_1i-y_1i^2  -y_1i*y_2i  ]
		          [ y_2i ]			  [ dL/dy_2i ]					[ -y_2i*y_1i   y_2i-y_2i^2 ]
		gradX_i = dsoftmax_i gradY_i
				= [ (y_1i-y_1i^2) * dL/dy_1i - y_1i*y_2i * dL/dy_2i  ]
				  [ -y_2i*y_1i * dL/dy_1i + (y_2i-y_2i^2) * dL/dy_2i ]
				= [ y_1i*(dL/dy_1i*(1-y_1i) - dL/dy_2i*y_2i)  ]
				  [ y_2i*(-dL/dy_1i*y_1i + dL/dy_2i*(1-y_2i)) ]
				= Y_i * [ dL/dy_1i*(1-y_1i) - dL/dy_2i*y_2i  ]
						[ -dL/dy_1i*y_1i + dL/dy_2i*(1-y_2i) ]
				= Y_i * [ dL/dy_1i - dL/dy_1i*y_1i - dL/dy_2i*y_2i ]
						[ dL/dy_2i - dL/dy_1i*y_1i - dL/dy_2i*y_2i ]
				= Y_i * (gradY_i - [ dL/dy_1i*y_1i + dL/dy_2i*y_2i ])
								   [ dL/dy_1i*y_1i + dL/dy_2i*y_2i ]
				= Y_i * (gradY_i - sum(gradY_i * Y_i) * [ 1 ])
														[ 1 ]
		gradX = dsoftmax gradY
			  = Y * (gradY - [ column_sum(Y*gradY) ])
		 					 [ column_sum(Y*gradY) ]
		using numpy
			gradX = dsoftmax gradY = Y * (gradY - np.sum(Y * gradY, axis=0, keepdims=True))
		or
			gradX = dsoftmax gradY = Y * (gradY - np.sum(Y * gradY, axis=0))
		"""
		tempSum = np.sum(self.trY * self.gradY, axis=(1,2), keepdims=True)
		np.multiply(self.trY, self.gradY - tempSum, self.gradX)

