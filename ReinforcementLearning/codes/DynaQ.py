import random
from typing import Union, Tuple

import numpy as np

"""
Example 8.1 Dyna Maze

"""
action_symbol = '←→↑↓'


def randomIndex(prob: np.ndarray):
	"""
	확률에 따라 index를 결정한다.
	pi=[0.1, 0.2, 0.3, 0.4]이면 p(index=0)=0.1, p(index=1)=0.2 등등
	:param prob: 확률
	:return: index
	"""
	num = np.random.rand()  # num in [0,1)
	s = 0
	for i in range(prob.shape[0]):
		s += prob[i]
		if num < s:
			return i
	# prob = [0, 0, ..., 0]인 경우
	raise RuntimeError


class Environment(object):
	def __init__(self):
		self.row = 6
		self.col = 9
		self.start = (2, 0)
		self.terminal = ((0, 8),)
		self.blocks = ((1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7))

	def applyAction(self, state: tuple, action: Union[int, str]) -> tuple:
		if state in self.terminal:
			return 0, None
		if isinstance(action, str):
			action = action_symbol.find(action)
		if action == 0:
			x, y = state[0], state[1] - 1
		elif action == 1:
			x, y = state[0], state[1] + 1
		elif action == 2:
			x, y = state[0] - 1, state[1]
		else:
			x, y = state[0] + 1, state[1]
		if (x, y) in self.terminal:
			return 1, (x, y)
		return 0, state if x < 0 or x >= self.row or y < 0 or y >= self.col or (x, y) in self.blocks else (x, y)


class QValueTable(object):
	"""
	table of q-values
	"""

	def __init__(self, row=6, col=9):
		self.row = 6
		self.col = 9
		self.table = np.random.rand(row, col, 4) * 0.01

	# self.table = np.zeros((row, col, 4))

	def getAction(self, state: tuple, epsilon: float = 0.0, delta: float = 0.0001) -> int:
		"""
		𝜖-greedy action at state
		:param state: state
		:param epsilon: 𝜖 for 𝜖-greedy
		:param delta: if |x-max| <= delta then x=max
		:return: selected action
		"""
		prob = np.zeros(4) + epsilon / 4
		tf = self.table[state] >= np.max(self.table[state]) - delta
		prob[tf] += (1 - epsilon) / np.count_nonzero(tf)
		return randomIndex(prob)

	def getMaxValue(self, state: tuple):
		return np.max(self.table[state])

	def update(self, state: tuple, action: int, reward: float, nextState: tuple, stepsize: float, discount: float):
		"""
		𝑞(𝑠,𝑎) ⟵ 𝑞(𝑠,𝑎) + 𝛼[𝑟 + 𝛾 max_𝑎 ⁡𝑞(𝑠′,𝑎) − 𝑞(𝑠,𝑎)]

		:param state: s
		:param action: a
		:param reward: r
		:param nextState: 𝑠′
		:param stepsize:𝛼
		:param discount: 𝛾
		:return:
		"""
		self.table[state][action] += stepsize * (reward + discount * self.getMaxValue(nextState) - self.table[state][action])

	def printPolicy(self, env: Environment, delta: float = 0.0001):
		symbol = np.array(list(action_symbol))
		words = []
		for i in range(self.row):
			for j in range(self.col):
				if (i, j) in env.terminal:
					words.append(f'[T]  ')
				elif (i, j) in env.blocks:
					words.append(f'[B]  ')
				else:
					m = np.max(self.table[i, j])
					tf = self.table[i, j] >= m - delta
					words.append(f'{"".join(symbol[tf]):5}')
			words.append('\n')
		print(''.join(words))


class Model(object):
	def __init__(self):
		self.dic: dict = {}

	def append(self, state: tuple, action: int, reward: float, nextState: tuple):
		self.dic[state, action] = (reward, nextState)

	def planning(self, qvt: QValueTable, stepSize: float, discount: float, iteration: int = 1):
		for i in range(iteration):
			(state, action), (reward, nextState) = random.choice(list(self.dic.items()))
			qvt.update(state, action, reward, nextState, stepSize, discount)


def main():
	stepsize = 0.1
	discount = 0.95
	epsilon = 0.1
	env = Environment()
	qvt = QValueTable()
	model = Model()
	epoch = 0
	while epoch < 100:
		state = env.start
		action = qvt.getAction(state, epsilon)
		reward, nextState = env.applyAction(state, action)
		epochSize = 0
		while nextState is not None:
			# if epochSize % 1000 == 0:
			# 	print('-', end='')
			model.append(state, action, reward, nextState)
			state = nextState
			action = qvt.getAction(state, epsilon)
			reward, nextState = env.applyAction(state, action)
			qvt.update(state, action, reward, nextState, stepsize, discount)
			epochSize += 1
		model.planning(qvt, stepsize, discount, iteration=50)
		# print('\nepoch', epoch, ', epoch size', epochSize)
		# qvt.printPolicy(env)
		epoch += 1
	print('\nepoch', epoch, ', epoch size', epochSize)
	qvt.printPolicy(env)


main()
