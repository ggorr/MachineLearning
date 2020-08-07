import time
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt


class Node:
	def __init__(self, index: int = -1, height: int = 0, left=None, right=None):
		# height of self
		self.height: int = height
		# index of data
		# -1 if self is not terminal
		self.index: int = index
		# number of leaves contained in this node, self
		self.leafCount: int = 1 if index >= 0 else left.leafCount + right.leafCount
		# left branch
		# None if self is terminal
		self.left: Union[Node, None] = left
		# right branch
		# None if self is terminal
		self.right: Union[Node, None] = right

	def getIndices(self):
		"""returns indices of leaves contained in self"""
		return [self.index] if self.height == 0 else self.left.getIndices() + self.right.getIndices()

	def dist(self, node, distTable: np.ndarray):
		"""average distance"""
		return np.sum(distTable[self.getIndices()][:, node.getIndices()]) / (self.leafCount * node.leafCount)

	def getClusters(self, bound):
		if self.height <= bound:
			return [self.getIndices()]
		else:
			return self.left.getClusters(bound) + self.right.getClusters(bound)

	def getSubnodeCount(self):
		if self.height == 0:
			return 0
		else:
			return self.left.getSubnodeCount() + self.right.getSubnodeCount() + 2

	def plot(self, parentX: float = None, parentY: float = None, lr: str = 'root'):
		"""lr is one of 'root', 'left' or 'right'"""
		if lr == 'root':
			x = 0
		elif self.height == 0:
			x = parentX - 1 if lr == 'left' else parentX + 1
			plt.plot([parentX, x, x], [parentY, parentY, self.height], 'k')
			plt.text(x, self.height, f'${self.index}$', ha='center', va='top')
			return
		else:
			if lr == 'left':
				x = parentX - self.right.getSubnodeCount() - 2
			else:  # if lr == 'right'
				x = parentX + self.left.getSubnodeCount() + 2
			plt.plot([parentX, x, x], [parentY, parentY, self.height], 'k')
		plt.text(x, self.height, f'${self.height}$', ha='center', va='top')
		self.left.plot(x, self.height, 'left')
		self.right.plot(x, self.height, 'right')


class Cluster:
	def __init__(self, X: np.ndarray):
		self.X: np.ndarray = X
		self.distTable: np.ndarray = np.zeros((X.shape[0], X.shape[0]))
		self.root: Node = self.buildTree()

	@property
	def Height(self):
		return self.root.height

	def buildTree(self):
		"""builds tree and returns root node"""
		n = self.X.shape[0]
		diff = X - X.reshape(n, 1, -1)
		self.distTable = np.sqrt(np.sum(diff * diff, 2))

		nodes = np.array([Node(i) for i in range(n + 1)])
		distSum = np.empty((n + 1, n + 1))
		distSum[:n, :n] = self.distTable
		distMean = distSum.copy()
		for i in range(n + 1):
			distMean[i, i] = np.inf
		leafCount = np.ones(n + 1)
		height = 0
		while n > 1:
			height += 1
			ind = np.divmod(np.argmin(distMean[:n, :n]), n)
			ind0, ind1 = np.min(ind), np.max(ind)
			nodes[n] = Node(height=height, left=nodes[ind0], right=nodes[ind1])
			leafCount[n] = nodes[n].leafCount

			distSum[:n, n] = distSum[n, :n] = distSum[:n, ind0] + distSum[:n, ind1]
			distMean[:n, n] = distMean[n, :n] = distSum[n, :n] / (leafCount[:n] * leafCount[n])
			n1, n2 = n + 1, n - 1
			distSum[ind0:ind1 - 1, :n1] = distSum[ind0 + 1:ind1, :n1]
			distSum[ind1 - 1:n2, :n1] = distSum[ind1 + 1:n1, :n1]
			distSum[:n2, ind0:ind1 - 1] = distSum[:n2, ind0 + 1:ind1]
			distSum[:n2, ind1 - 1:n2] = distSum[:n2, ind1 + 1:n1]
			distMean[ind0:ind1 - 1, :n1] = distMean[ind0 + 1:ind1, :n1]
			distMean[ind1 - 1:n2, :n1] = distMean[ind1 + 1:n1, :n1]
			distMean[:n2, ind0:ind1 - 1] = distMean[:n2, ind0 + 1:ind1]
			distMean[:n2, ind1 - 1:n2] = distMean[:n2, ind1 + 1:n1]
			nodes[ind0:ind1 - 1] = nodes[ind0 + 1:ind1]
			nodes[ind1 - 1:n2] = nodes[ind1 + 1:n1]
			leafCount[ind0:ind1 - 1] = leafCount[ind0 + 1:ind1]
			leafCount[ind1 - 1:n2] = leafCount[ind1 + 1:n1]
			n -= 1
			# if n == 2:
			# 	print(distMean[:2, :2])
		return nodes[0]

	def buildTree2(self):
		"""builds tree and returns root node"""
		n = self.X.shape[0]
		diff = X - X.reshape(n, 1, -1)
		self.distTable = np.sqrt(np.sum(diff * diff, 2))

		nodes = np.array([Node(i) for i in range(n + 1)])
		distSum = np.empty((n + 1, n + 1))
		distSum[:n, :n] = self.distTable
		distMean = distSum.copy()
		for i in range(n + 1):
			distMean[i, i] = np.inf
		leaves = np.ones(n + 1)
		tf = np.array((n + 1) * [False])
		ll = np.array(range(n + 1))
		height = 0
		while n > 1:
			height += 1
			ind = np.divmod(np.argmin(distMean[:n, :n]), n)
			ind0, ind1 = np.min(ind), np.max(ind)
			nodes[n] = Node(height=height, left=nodes[ind0], right=nodes[ind1])
			leaves[n] = nodes[n].leafCount

			distSum[:n, n] = distSum[n, :n] = distSum[:n, ind0] + distSum[:n, ind1]
			distMean[:n, n] = distMean[n, :n] = distSum[n, :n] / leaves[:n] * leaves[n]
			tf.fill(False)
			tf[ind0 + 1:ind1] = True
			tf[ind1 + 1: n + 1] = True
			l = ll[tf]
			distSum[ind0:n - 1, :n + 1] = distSum[l, :n + 1]
			distSum[:n - 1, ind0:n - 1] = distSum[:n - 1, l]
			distMean[ind0:n - 1, :n + 1] = distMean[l, :n + 1]
			distMean[:n - 1, ind0:n - 1] = distMean[:n - 1, l]
			nodes[ind0:n - 1] = nodes[l]
			leaves[ind0:n - 1] = leaves[l]
			n -= 1
		return nodes[0]

	def buildTree1(self):
		"""builds tree and returns root node"""
		n = self.X.shape[0]
		diff = X - X.reshape(n, 1, -1)
		self.distTable = np.sqrt(np.sum(diff * diff, 2))
		nodes = np.array([Node(i) for i in range(n)])
		distSum = self.distTable.copy()
		distMean = distSum.copy()
		for i in range(n):
			distMean[i, i] = np.inf
		height = 0
		while n > 1:
			height += 1
			ind = np.divmod(np.argmin(distMean), n)
			nodes = np.append(nodes, Node(height=height, left=nodes[ind[0]], right=nodes[ind[1]]))

			sumBack = distSum
			distSum = np.zeros((n + 1, n + 1))
			distSum[:n, :n] = sumBack
			distSum[:, -1] = distSum[-1, :] = distSum[:, ind[0]] + distSum[:, ind[1]]
			meanBack = distMean
			distMean = np.empty_like(distSum)
			distMean[:n, :n] = meanBack
			distMean[:n, -1] = distMean[-1, :n] = distSum[-1, :n] / np.array([nodes[i].leafCount * nodes[-1].leafCount for i in range(n)])
			distMean[-1, -1] = np.inf
			tf = np.array((n + 1) * [True])
			tf[ind[0]] = tf[ind[1]] = False
			distSum = distSum[tf][:, tf]
			distMean = distMean[tf][:, tf]
			nodes = nodes[tf]
			n -= 1
		return nodes[0]

	def buildTree0(self):
		"""builds tree and returns root node"""
		n = self.X.shape[0]
		for i in range(n):
			for j in range(i + 1, n):
				self.distTable[i, j] = np.sum((self.X[i] - self.X[j]) ** 2)
				self.distTable[j, i] = self.distTable[i, j]

		nodes = [Node(i) for i in range(n)]
		distSum = self.distTable.copy()
		distMean = distSum.copy()
		for i in range(n):
			distMean[i, i] = np.inf
		height = 0
		while n > 1:
			height += 1
			ind = np.unravel_index(np.argmin(distMean, axis=None), distMean.shape)
			ind0, ind1 = np.min(ind), np.max(ind)
			nodes.append(Node(height=height, left=nodes[ind0], right=nodes[ind1]))
			sumBack = distSum
			# distSum = np.concatenate((distSum, np.zeros((n, 1))), axis=1)
			# distSum = np.concatenate((distSum, np.zeros((1, n + 1))), axis=0)
			distSum = np.zeros((n + 1, n + 1))
			distSum[:n, :n] = sumBack
			distSum[:, -1] = distSum[-1, :] = distSum[:, ind0] + distSum[:, ind1]
			meanBack = distMean
			# distMean = np.concatenate((distMean, np.zeros((n, 1))), axis=1)
			# distMean = np.concatenate((distMean, np.zeros((1, n + 1))), axis=0)
			distMean = np.zeros((n + 1, n + 1))
			distMean[:n, :n] = meanBack
			for i in range(n):
				distMean[i, -1] = distMean[-1, i] = distSum[i, -1] / (nodes[i].leafCount * nodes[-1].leafCount)
			distMean[-1, -1] = np.inf
			tf = np.array((n + 1) * [True])
			tf[ind0] = tf[ind1] = False
			# distSum = np.delete(np.delete(distSum, ind, 0), ind, 1)
			# distMean = np.delete(np.delete(distMean, ind, 0), ind, 1)
			distSum = distSum[tf][:, tf]
			distMean = distMean[tf][:, tf]
			nodes.pop(ind1), nodes.pop(ind0)
			n -= 1
		return nodes[0]

	def getClusters(self, clusters: int):
		return self.root.getClusters(self.root.height - clusters + 1)

	def plotClusters(self, clusters: int, numbering=False, show=True):
		color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
		for i, cls in enumerate(self.getClusters(clusters)):
			pts = self.X[cls]
			plt.plot(pts[:, 0], pts[:, 1], color[i] + '.')
		if numbering:
			for i, x in enumerate(self.X):
				plt.text(x[0], x[1], f'${i}$', ha='left', va='top')
		if show:
			plt.show()

	def plotTree(self, show=True):
		if self.root is None:
			return
		self.root.plot()
		if show:
			plt.show()


def generateData(n=15):
	n = n // 3
	X = np.empty((3 * n, 2))
	X[:n, :] = np.random.randn(n, 2) / 4
	X[n:2 * n, :] = np.random.randn(n, 2) / 4 + np.array([1., 0.])
	X[2 * n:3 * n, :] = np.random.randn(n, 2) / 4 + np.array([0.5, np.math.sqrt(3) / 2])
	return np.round(X, 1)


# n = 15
# X = generateData()
# print(np.array2string(X.T, separator=','))
# plt.plot(X[:n // 3, 0], X[:n // 3, 1], 'r.')
# plt.plot(X[n // 3:2 * n // 3, 0], X[n // 3:2 * n // 3, 1], 'g.')
# plt.plot(X[2 * n // 3:n, 0], X[2 * n // 3:n, 1], 'b.')
# plt.show()

X = np.array([[0., 0.1, -0.2, 0., 0.4, 1.3, 0.6, 1.2, 0.9, 1.1, 0.7, 0.9, 0.4, 0.7, 0.5],
			  [-0.3, 0., 0.2, 0.2, -0.5, -0.3, 0.1, 0.2, -0.1, 0.1, 0.6, 0.6, 0.5, 1.1, 1.1]]).T
X = generateData(5 * 3)

start = time.time()
c = Cluster(X)
print(time.time() - start)
# print(c.root.left.dist(c.root.right, c.distTable))
plt.subplot(121)
c.plotTree(False)
plt.subplot(122)
plt.gca().set_aspect('equal')
c.plotClusters(3, True, True)
