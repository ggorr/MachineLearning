from abc import abstractmethod, ABCMeta
from typing import Union

import numpy as np

from lib32.NodeSet import F0


class Regularizer(metaclass=ABCMeta):
	def __init__(self):
		pass

	#
	# @abstractmethod
	# def loss(self, fitSets: Iterable[FitSet]) -> Union[float, np.ndarray]:
	# 	pass

	@abstractmethod
	def grad(self, K: np.ndarray) -> Union[float, np.ndarray]:
		pass


class RegNone(Regularizer):
	def __init__(self):
		super().__init__()

	#
	# def loss(self, fitSets: Iterable[FitSet]) -> float:
	# 	return 0.0

	def grad(self, K: np.ndarray) -> float:
		return F0

	def __str__(self):
		return 'RegNone'


class Ridge(Regularizer):
	def __init__(self, rate=np.float32(0.01)):
		super().__init__()
		self.rate = rate

	# def loss(self, fitSets: Iterable[FitSet]) -> float:
	# 	value = 0.0
	# 	for fset in fitSets:
	# 		value += 0.5 * self.rate * np.sum(fset.Kernel * fset.Kernel)
	# 	return value

	def grad(self, K: np.ndarray) -> np.ndarray:
		return self.rate * K

	def __str__(self):
		return f'Ridge({self.rate})'


class Lasso(Regularizer):
	def __init__(self, rate=np.float32(0.01)):
		super().__init__()
		self.rate = rate

	# def loss(self, fitSets: Iterable[FitSet]) -> float:
	# 	value = 0.0
	# 	for fset in fitSets:
	# 		value += self.rate * np.sum(np.abs(fset.Kernel))
	# 	return value

	def grad(self, K: np.ndarray) -> np.ndarray:
		return self.rate * np.sign(K)

	def __str__(self):
		return f'Lasso({self.rate})'
