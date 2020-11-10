from .MidSet import *


class FitSet1D(MidSet1D, metaclass=ABCMeta):
	def __init__(self, ydim: int, ychs: int, name: str):
		super().__init__(ydim, ychs, name)

	@abstractmethod
	def preFit(self, **kwargs):
		pass

	@abstractmethod
	def updateBatch(self, learningRate: np.float32):
		pass

	@abstractmethod
	def updateAdaptive(self, learningRate: np.float32):
		pass

	@abstractmethod
	def sendCurrToPrev(self):
		pass

	@abstractmethod
	def gradDot(self) -> np.float32:
		pass


class FitSet2D(MidSet2D, metaclass=ABCMeta):
	def __init__(self, ydim: Union[Tuple, List], ychs: int, name: str):
		super().__init__(ydim, ychs, name)

	@abstractmethod
	def preFit(self, **kwargs):
		pass

	@abstractmethod
	def updateBatch(self, learningRate: np.float32):
		pass

	@abstractmethod
	def updateAdaptive(self, learningRate: np.float32):
		pass

	@abstractmethod
	def sendCurrToPrev(self):
		pass

	@abstractmethod
	def gradDot(self) -> np.float32:
		pass
