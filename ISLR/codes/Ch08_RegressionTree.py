from typing import Union

import numpy as np
import matplotlib.pyplot as plt


class Node:
	def __init__(self, X: np.ndarray, y: np.ndarray):
		# inputs
		self.X: Union[np.ndarray, None] = X
		# response
		self.y: Union[np.ndarray, None] = y
		# residual sum of square in this region
		# noinspection PyTypeChecker
		self.rss: float = np.sum((y - np.mean(y)) ** 2)
		# predictor to be splitted
		self.splitIndex: Union[int, None] = None
		# predictor value at which region is splitted
		self.splitValue: Union[float, None] = None
		# residual sum of square when this region is splitted
		self.splitRss: float = np.inf
		self.leftNode: Union[Node, None] = None
		self.rightNode: Union[Node, None] = None
		# whether self is a leaf
		self.isLeaf: bool = True
		# the added order
		# if self is a leaf then order is None
		self.order: Union[int, None] = None
		self.span()

	def addToTree(self, order: int):
		"""adds self to tree"""
		if self.X.shape[0] > 1:
			left = self.X[:, self.splitIndex] < self.splitValue
			# Left and right nodes are created.
			# They are candidates of next split.
			self.leftNode = Node(self.X[left], self.y[left])
			self.rightNode = Node(self.X[~left], self.y[~left])
		self.X = None
		self.y = None
		# leaves are added
		self.isLeaf = False
		self.order = order

	def span(self):
		"""split self into left and right node"""
		if self.X.shape[0] == 1:
			return
		for i in range(self.X.shape[1]):
			comp = np.sort(np.unique(self.X[:, i]))
			splitValueI = None
			splitRssI = np.inf
			for j in range(comp.shape[0] - 1):
				splitValueJ = (comp[j] + comp[j + 1]) / 2
				isLeft = self.X[:, i] < splitValueJ
				leftDiff = self.y[isLeft] - np.mean(self.y[isLeft])
				rightDiff = self.y[~isLeft] - np.mean(self.y[~isLeft])
				splitRssJ = np.sum(leftDiff * leftDiff) + np.sum(rightDiff * rightDiff)
				if splitRssJ < splitRssI:
					splitValueI = splitValueJ
					splitRssI = splitRssJ
			if splitRssI < self.splitRss:
				self.splitIndex = i
				self.splitValue = splitValueI
				self.splitRss = splitRssI

	def getSubnodeWithMaxRssDiff(self, node, maxRssDiff):
		"""find subnode with rss - splitRss < maxRssDiff"""
		if self.isLeaf:
			rssDiff = self.rss - self.splitRss
			if rssDiff > maxRssDiff:
				node, maxRssDiff = self, rssDiff
		else:
			node, maxRssDiff = self.leftNode.getSubnodeWithMaxRssDiff(node, maxRssDiff)
			node, maxRssDiff = self.rightNode.getSubnodeWithMaxRssDiff(node, maxRssDiff)
		return node, maxRssDiff

	def print(self, depth=0):
		if not self.leftNode.isLeaf:
			self.leftNode.print(depth + 1)
		print(f'{depth * "    "}X_{self.splitIndex + 1}({self.splitValue}) added at {self.order}')
		if not self.rightNode.isLeaf:
			self.rightNode.print(depth + 1)

	def getSubnodeCount(self):
		if self.isLeaf:
			return 0
		else:
			return self.leftNode.getSubnodeCount() + self.rightNode.getSubnodeCount() + 2

	def getHeight(self):
		if self.isLeaf:
			return 0
		else:
			return max(self.leftNode.getHeight(), self.rightNode.getHeight()) + 1

	def plot(self, parentX: float = None, parentY: float = None, lr: str = 'root', regionNo=0):
		"""lr is one of 'root', 'left' or 'right'"""
		if lr == 'root':
			x = 0
			parentY = 0
		elif self.isLeaf:
			x = parentX - 1 if lr == 'left' else parentX + 1
			plt.plot([parentX, x, x], [parentY, parentY, parentY - 1], 'k')
			regionNo += 1
			plt.text(x, parentY - 1, f'$R_{regionNo}$', ha='center', va='top')
			return regionNo
		else:
			if lr == 'left':
				x = parentX - self.rightNode.getSubnodeCount() - 2
			else:  # if lr == 'right'
				x = parentX + self.leftNode.getSubnodeCount() + 2
			plt.plot([parentX, x, x], [parentY, parentY, parentY - 1], 'k')
		plt.text(x, parentY - 1, f'${self.order}$     \n$X_{self.splitIndex + 1}({self.splitValue:1.3})$', ha='center', va='center')
		regionNo = self.leftNode.plot(x, parentY - 1, 'left', regionNo)
		regionNo = self.rightNode.plot(x, parentY - 1, 'right', regionNo)
		return regionNo


class RegressionTree:
	def __init__(self, X: np.ndarray, y: np.ndarray):
		self.X = X
		self.y = y
		self.root: Node = Node(X, y)
		self.nodeCount: int = 0

	def span(self, n=1):
		for i in range(n):
			node, _ = self.root.getSubnodeWithMaxRssDiff(None, -np.inf)
			node.addToTree(self.nodeCount)
			self.nodeCount += 1

	def printTree(self):
		if self.root.isLeaf:
			return
		self.root.print()

	def plotTree(self):
		if self.root.isLeaf:
			return
		self.root.plot()
		plt.show()

if __name__=='__main__':
	X = np.array([[1.1, 1.1], [1.0, 1.3], [1.0, 1.0], [1.15, 1.1], [1.5, 1.5], [1.6, 1.3], [1.4, 1.4], [1.4, 1.0], [1.5, 0.8], [1.6, 0.9]])
	y = np.array([4.0, 4.2, 4.0, 4.1, 2.7, 1.8, 2.3, 3.1, 3.9, 3.5])
	rt = RegressionTree(X, y)
	rt.span(2)
	rt.printTree()
	rt.plotTree()
