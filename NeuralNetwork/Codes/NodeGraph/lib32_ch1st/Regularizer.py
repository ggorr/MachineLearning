from abc import abstractmethod, ABCMeta
from typing import Union

import numpy as np

from .NodeSet import F0


class Regularizer(metaclass=ABCMeta):
	def __init__(self):
		pass

	#
	# @abstractmethod
	# def loss(self, fitSets: Iterable[FitSet]) -> Union[np.float32, np.ndarray]:
	# 	pass

	@abstractmethod
	def grad(self, K: np.ndarray) -> Union[np.float32, np.ndarray]:
		pass


class RegNone(Regularizer):
	def __init__(self):
		super().__init__()

	#
	# def loss(self, fitSets: Iterable[FitSet]) -> np.float32:
	# 	return F0

	def grad(self, K: np.ndarray) -> np.float32:
		return F0

	def __str__(self):
		return 'RegNone'


class Ridge(Regularizer):
	def __init__(self, rate: Union[np.float, np.float32] = 0.01):
		super().__init__()
		self.rate: np.float32 = np.float32(rate)

	# def loss(self, fitSets: Iterable[FitSet]) -> np.float32:
	# 	value = F0
	# 	for fset in fitSets:
	# 		value += FH * self.rate * np.sum(fset.Kernel * fset.Kernel)
	# 	return value

	def grad(self, K: np.ndarray) -> np.ndarray:
		return self.rate * K

	def __str__(self):
		return f'Ridge({self.rate})'


class Lasso(Regularizer):
	def __init__(self, rate: Union[np.float, np.float32] = 0.01):
		super().__init__()
		self.rate: np.float32 = np.float32(rate)

	# def loss(self, fitSets: Iterable[FitSet]) -> np.float32:
	# 	value = F0
	# 	for fset in fitSets:
	# 		value += self.rate * np.sum(np.abs(fset.Kernel))
	# 	return value

	def grad(self, K: np.ndarray) -> np.ndarray:
		return self.rate * np.sign(K)

	def __str__(self):
		return f'Lasso({self.rate})'
