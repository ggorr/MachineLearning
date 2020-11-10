from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np

UNKNOWN = -1
FH = np.float32(.5)
F0 = np.float32(0.0)
F1 = np.float32(1.0)
F2 = np.float32(2.0)


class NodeSet(metaclass=ABCMeta):
	"""base class of all node sets"""

	# prefixes of the name of a NodeSet object
	__prefix = {'StartSet1D': 'ss1_', 'Dense1D': 'dn1_', 'Activation1D': 'ac1_',
				'Conv1D': 'cv1_', 'MaxPool1D': 'pl1_', 'Flat1D': 'fl1_',
				'StartSet2D': 'ss2_', 'Activation2D': 'ac2_',
				'Conv2D': 'cv2_', 'MaxPool2D': 'pl2_', 'Flat2D': 'fl2_',
				'Mse1D': 'ms1', 'Bce1D': 'bc1', 'ZeroOneBce1D': 'bc1', 'Cce1D': 'cc1', 'OneHotCce1D': 'cc1'}
	__index = {'ss1_': 0, 'dn1_': 0, 'ac1_': 0, 'pl1_': 0, 'cv1_': 0, 'sc1_': 0, 'fl1_': 0,
			   'ss2_': 0, 'dn2_': 0, 'ac2_': 0, 'pl2_': 0, 'cv2_': 0, 'sc2_': 0, 'fl2_': 0}
	__nextId = 0

	@staticmethod
	def __naming(name: Union[str, None], nodeSet: object) -> str:
		"""auto naming of nodeSet"""
		cls = nodeSet.__class__
		while cls.__name__ not in NodeSet.__prefix.keys():
			cls = cls.__base__
		prefix = NodeSet.__prefix[cls.__name__]
		if name is None:
			# automatic naming
			if prefix in NodeSet.__index.keys():
				# nodeSet is not a loss function
				name = f'{prefix}{NodeSet.__index[prefix]}'
				NodeSet.__index[prefix] += 1
			else:
				# Loss function is unique, and hence it has no index
				name = NodeSet.__prefix[cls.__name__]
		else:
			if name.startswith(prefix) and prefix in NodeSet.__index.keys():
				try:
					NodeSet.__index[prefix] = max(NodeSet.__index[prefix], int(name[len(prefix):])) + 1
				except ValueError or TypeError:
					pass
		return name

	def __init__(self, name: str = None):
		self.name: str = NodeSet.__naming(name, self)
		# id is a unique number for each NodeSet object
		self.__id: int = NodeSet.__nextId
		NodeSet.__nextId += 1

	@property
	def Id(self) -> int:
		return self.__id

	@abstractmethod
	def shortStr(self) -> str:
		pass

	@abstractmethod
	def saveStr(self) -> str:
		pass
