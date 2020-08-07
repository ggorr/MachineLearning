from typing import Union, Sequence, Tuple

import numpy as np

action_symbol = '←→↑↓'
action_letter = 'LRUD'


def random_index(prob: np.ndarray):
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
	@staticmethod
	def standard_prob(base_prob: float = 1.0):
		"""
		action에 대하여 next state를 선택할 확률
		해당 action에 base_prob + (1-base_prob)/4, 나머지 action에 (1-base_prob)/4의 확률을 배정
		:param base_prob: 1이면 deterministic
		:return: 확률 테이블
		"""
		return np.zeros((4, 4)) + (1 - base_prob) / 4 + base_prob * np.eye(4, 4)

	def __init__(self, row: int = 4, col: int = 4, terminal: tuple = None, prob: np.ndarray = None, base_prob: float = 1.0):
		"""
		:param row: # of rows
		:param col: # of columns
		:param prob: 확률 테이블. 편의상 모든 state에서 확률이 같다고 가정한다. 따라서 4x4 maxtrix. 그렇지 않으면 row x col x 4 x 4 matrix
		:param terminal: tuple of tuples i.e. tuple of terminal states
		:param base_prob: 확률 테이블 prob이 None이면 base_prob으로부터 확률을 계산
		"""
		self.row: int = row
		self.col: int = col
		# action을 실행하였을 때 실제 이동하는 방향에 대한 확률
		# 동쪽으로 이동하였다고 반드시 동쪽으로 이동하는 것은 아니다. 다른 방향으로 이동할 수도 있다
		self.prob: np.ndarray = Environment.standard_prob(base_prob) if prob is None else prob
		# state에서 direction 방향으로 움직일 때의 reward. 여기서는 기본값으로 항상 -1을 준다
		self.reward: np.ndarray = np.zeros((row, col, 4)) - 1
		self.terminal = ((0, 0), (row - 1, col - 1)) if terminal is None else terminal
		for t in self.terminal:
			self.reward[t][:] = 0.0

	def __move_to_direction(self, state: tuple, direction: int):
		"""
		각 방향으로 이동하였을 때의 state. action을 시행하면 확률적으로 direction이 결정된다.
		:param state:
		:param direction:
		:return:
		"""
		# grid 밖으로 나갈 수는 없다
		if direction == 0:  # left
			return state[0], state[1] if state[1] == 0 else state[1] - 1
		elif direction == 1:  # right
			return state[0], state[1] if state[1] + 1 == self.col else state[1] + 1
		elif direction == 2:  # up
			return state[0] if state[0] == 0 else state[0] - 1, state[1]
		else:  # down
			return state[0] if state[0] + 1 == self.row else state[0] + 1, state[1]

	def apply_action(self, state: tuple, action: int):
		"""
		state에서 action을 적용하여 다음 state를 얻는다.
		:param state:
		:param action:
		:return: reward와 다음 state. terminal state이면 None, None
		"""
		if state in self.terminal:
			return None, None
		# 실제로 이동할 방향을 결정한다
		direction = random_index(self.prob[action])
		return self.reward[state][direction], self.__move_to_direction(state, direction)

	def qvalue_from_stable(self, state: tuple, action: int, stable: np.ndarray, discount: float = 1.0) -> float:
		"""
		state value table로부터 action value를 계산한다
		본래
			q(s, a) ← ∑_(s′,r) p(s′,r│s,𝑎) [r + 𝛾 v(s′)]
		이지만 이 프로그램에서는
			q(s, a) ← ∑_s′ p(s′│s,𝑎) [r + 𝛾 v(s′)]
		이다. 이 값을 return한다.

		:param state: state to be evaluated
		:param action: action to be evaluated
		:param stable: state value table = state values
		:param discount: discount rate
		:return: action value ∑_s′ 𝑟 p(s′│s,𝑎) [r + 𝛾 v_𝜋(s′)]
		"""

		if state in self.terminal:
			return 0.0
		v = 0.0
		for direction in range(4):
			v += self.prob[action, direction] * (self.reward[state][direction] + discount * stable[self.__move_to_direction(state, direction)])
		return v

	def policy_from_stable(self, stable: np.ndarray, epsilon: float = 0.0, discount: float = 1.0, delta: float = 0.0001) -> np.ndarray:
		"""
		derives 𝜖-greedy policy from state values as policy improvement
		policy is greedy if 𝜖=0[DEFALUT]
		:param stable: state values
		:param epsilon: 𝜖 for 𝜖-greedy
		:param discount: discount rate
		:param delta: if |x-y| <= delta then x=y
		:return: the 𝜖-greedy policy derived from state values
		"""
		pi = np.zeros((self.row, self.col, 4)) + epsilon / 4
		v = np.empty(4)
		for i in range(self.row):
			for j in range(self.col):
				if (i, j) in self.terminal:
					pi[i, j, :] = 0.0
				else:
					for k in range(4):
						v[k] = self.qvalue_from_stable((i, j), k, stable, discount)
					tf = v >= np.max(v) - delta
					# maxima에 균등 분배
					pi[i, j][tf] += (1 - epsilon) / np.count_nonzero(tf)
		return pi

	def policy_from_qtable(self, qtable: np.ndarray, epsilon: float = 0.0, delta: float = 0.0001) -> np.ndarray:
		"""
		:param qtable: q values
		:param epsilon: 𝜖 for 𝜖-greedy
		:param delta: if |x-y| <= delta then x=y
		:return: the 𝜖-greedy policy derived from q
		"""
		pi = np.zeros((self.row, self.col, 4)) + epsilon / 4
		for i in range(self.row):
			for j in range(self.col):
				if (i, j) in self.terminal:
					pi[i, j, :] = 0.0
				else:
					tf = qtable[i, j] >= np.max(qtable[i, j]) - delta
					# maxima에 균등 분배
					pi[i, j][tf] += (1 - epsilon) / np.count_nonzero(tf)
		return pi

	def uniform_policy(self):
		pi = np.zeros((self.row, self.col, 4)) + 0.25
		for t in self.terminal:
			pi[t][:] = 0.0
		return pi


class Agent(object):
	def __init__(self, row: int = 4, col: int = 4, pi: np.ndarray = None):
		"""
		:param row: # of rows
		:param col: # of columns
		:param pi: policy, 곧 각 state에서 각 방향으로 이동할 확률
		"""
		self.row: int = row
		self.col: int = col
		self.pi: np.ndarray = pi

	def get_action(self, state: tuple):
		"""
		policy에 따라 action을 선택한다.
		:param state:
		:return: action, 0, 1, 2, 3
		"""
		return random_index(self.pi[state])

	def svalue_from_stable(self, state: tuple, stable: np.ndarray, env: Environment, discount: float = 1.0):
		"""
		state values로부터 state value를 계산한다.
		본래
			v(s) ⟵ ∑_𝑎 𝜋(𝑎│s) ∑_(s′,r) p(s′,𝑟│𝑠,𝑎)[r + 𝛾 v(s′)]
		이지만 이 프로그램에서는
			v(s) ⟵ ∑_𝑎 𝜋(𝑎│s) ∑_s′ p(s′│𝑠,𝑎)[r + 𝛾 v(s′)]
		:param state: state to be evaluated
		:param stable: state values
		:param env: environment
		:param discount: discount rate
		:return: state value ∑_𝑎 𝜋(𝑎│s) ∑_s′ p(s′│𝑠,𝑎)[r + 𝛾 v(s′)]
		"""
		v = 0.0
		for k in range(4):
			# v(s) ⟵ ∑_𝑎 𝜋(𝑎│s) ∑_s′ p(s′│𝑠,𝑎)[r + 𝛾 v(s′)]
			v += self.pi[state][k] * env.qvalue_from_stable(state, k, stable, discount)
		return v

	def stable_from_stable(self, stable: np.ndarray, env: Environment, discount: float = 1.0):
		"""
		evaluates state value table from state value table following given policy self.pi
		본래
			v(s) ⟵ ∑_𝑎 𝜋(𝑎│s) ∑_(s′,r) p(s′,𝑟│𝑠,𝑎)[r + 𝛾 v(s′)]
		이지만 이 프로그램에서는
			v(s) ⟵ ∑_𝑎 𝜋(𝑎│s) ∑_s′ p(s′│𝑠,𝑎)[r + 𝛾 v(s′)]
		:param stable: state values = state value table
		:param env: environment
		:param discount: discount rate
		:return: evaluated state values = state value table
		"""
		new_stable = np.empty_like(stable)
		for i in range(self.row):
			for j in range(self.col):
				new_stable[i, j] = self.svalue_from_stable((i, j), stable, env, discount)
		return new_stable

	def optimal_stable(self, env: Environment, stable: np.ndarray = None, discount: float = 1.0, err: float = 0.0001, display: tuple = ()):
		"""
		optimal state values for given policy self.pi
		:param stable: state values = state value table
		:param env: environment
		:param discount: discount rate
		:param err: evaluate until |v - v_previous| <= err
		:param display: epochs for print values
		:return: optimal state values and # of iteration
		"""
		# 원본 보존
		stable = np.zeros((self.row, self.col)) if stable is None else stable.copy()
		epoch = 0
		diff = np.inf
		while diff > err:
			if epoch in display:
				print(f'iteration {epoch}\n{stable}')
			stable_copy = stable.copy()
			for i in range(self.row):
				for j in range(self.col):
					stable[i, j] = self.svalue_from_stable((i, j), stable_copy, env, discount)
			diff = np.max(np.abs(stable - stable_copy))
			epoch += 1
		if epoch in display:
			print(f'iteration {epoch}\n{stable}')
		return stable, epoch

	def policy_string(self, delta: float = 0.0001):
		return Agent.policy_to_string(self.pi, delta)

	@staticmethod
	def policy_to_string(pi: np.ndarray, delta: float = 0.0001):
		"""
		:param pi: policy
		:param delta: if |x-y| <= delta then x=y
		:return:
		"""
		symbol = np.array(list(action_symbol))
		word = []
		for i in range(pi.shape[0]):
			for j in range(pi.shape[1]):
				m = np.max(pi[i, j])
				if m > 0:
					tf = pi[i, j] >= m - delta
					word.append(f'{"".join(symbol[tf]):5}')
				else:
					word.append(f'     ')
			word.append('\n')
		return ''.join(word)


class EpisodeGenerator(object):
	def __init__(self, env: Environment, agent: Agent):
		self.env: Environment = env
		self.agent: Agent = agent

	def start_state(self):
		"""
		:return: 시작 state
		"""
		state = self.env.terminal[0]
		while state in self.env.terminal:
			state = np.random.randint(self.env.row), np.random.randint(self.env.col)
		return state

	def policy_next(self, state: tuple):
		"""
		:param state: current state.
		:return: action과 그 action을 적용한 reward, state. state가 terminal이면 None, None, None
		"""
		if state in self.env.terminal:
			return None
		action = self.agent.get_action(state)
		reward, state = self.env.apply_action(state, action)
		return action, reward, state

	def policy_episode(self, state: Union[tuple, None] = None):
		"""
		:param state: 시작 state. None(DEFAULT)이면 시작 state를 무작위로 설정한다.
		:return: a full episode
		"""
		state = self.start_state() if state is None else state
		episode = [state]
		ars = self.policy_next(state)  # action, reward, state
		while ars is not None:
			episode += ars
			ars = self.policy_next(ars[2])  # action, reward, state
		return episode

	def q_next(self, qtable: np.ndarray, state: tuple, epsilon: float = 0.0, delta: float = 0.0001):
		"""
		chooses 𝜖-greedy action
		:param qtable: q-value table
		:param state: state
		:param epsilon: 𝜖 for 𝜖-greedy
		:param delta: if |x-y| <= delta then x=y
		:return:  action과 그 action을 적용한 reward, state. state가 terminal이면 None, None, None
		"""
		if state in self.env.terminal:
			return None
		prob = np.zeros(4) + epsilon / 4
		tf = qtable[state] >= np.max(qtable[state]) - delta
		prob[tf] += (1 - epsilon) / np.count_nonzero(tf)
		action = random_index(prob)
		reward, state = self.env.apply_action(state, action)
		return action, reward, state

	def q_episode(self, qtable: np.ndarray, state: tuple = None, epsilon: float = 0.0, delta: float = 0.0001):
		"""
		:param qtable: q-value table
		:param state: 시작 state. None(DEFAULT)이면 시작 state를 무작위로 설정한다.
		:param epsilon: 𝜖 for 𝜖-greedy
		:param delta: if |x-y| <= delta then x=y
		:return: a full episode
		"""
		state = self.start_state() if state is None else state
		episode = [state]
		ars = self.q_next(qtable, state, epsilon, delta)  # action, reward, state
		while ars is not None:
			episode += ars
			ars = self.q_next(qtable, ars[2], epsilon, delta)  # action, reward, state
		return episode

	@staticmethod
	def to_string(episode: list):
		"""
		converts episode to string
		:param episode:
		:return:
		"""
		return ''.join([action_symbol[x] if isinstance(x, int) else str(x) for x in episode])


def policy_evaluation():
	row, col = 4, 4
	# discount rate 𝛾
	discount = 1.0
	# deterministic environment, agent가 선택한 방향으로 이동한다
	env = Environment(row, col)
	#
	# env = Environment(row, col, terminal=((0, 0),))
	agent = Agent(row, col, env.uniform_policy())
	# 출력하는 iteration
	display = (0, 1, 2, 3, 10, 100)
	# 연속된 두 value의 차가 err 이하가 될때까지 evaluate한다
	stable, epoch = agent.optimal_stable(env, discount=discount, display=display, err=0.00000001)
	print(f'iteration {epoch}\n{stable}')


def policy_iteration():
	row, col = 4, 4
	# discount rate 𝛾
	discount = 1.0
	# deterministic environment, agent가 선택한 방향으로 이동한다
	env = Environment(row, col)
	agent = Agent(row, col, env.uniform_policy())
	# 출력하는 iteration
	display = (0, 1, 2, 3, 10, 100)
	epoch = 0
	pi_diff = np.inf
	while pi_diff > 0.0001:
		stable, _ = agent.optimal_stable(env, discount=discount)
		pi_copy = agent.pi
		# delta는 오차 범위. delta가 클수록 많은 방향이 검출됨
		agent.pi = env.policy_from_stable(stable, discount=discount, delta=0.0001)
		pi_diff = np.max(np.abs(agent.pi - pi_copy))
		if epoch in display:
			print(f'iteration {epoch}\n{stable}')
		epoch += 1


def value_iteration():
	"""
	state value table과 policy를 동시에 최적화한다.
	:return:
	"""
	row, col = 4, 4
	# deterministic environment, agent가 선택한 방향으로 이동한다
	env = Environment(row, col)
	# env = Environment(row, col, terminal=((0, 0),))
	# agent는 네 방향 중 무작위로 선택하여 이동한다
	# 이 정책으로 시작하여 optimal에 도달
	agent = Agent(row, col, env.uniform_policy())
	# 초기 state value는 모두 0으로
	stable = np.zeros((row, col))
	# discount rate 𝛾
	discount = 1.0
	# 출력하는 iteration
	display = (0, 1, 2, 3, 10, 100)
	# state value의 변화가 미미하면 종료
	# diff는 변화랼
	diff = np.inf
	epoch = 0
	while diff > 0.0001:
		if epoch in display:
			print(f'iteration {epoch}\n{stable}')
			print(agent.policy_string())
		stable_copy = stable.copy()
		# state values와 policy로부터 새로운 state values를 계산한다
		# 𝑣(𝑠) ⟵ ∑_𝑎 𝜋(𝑎│𝑠) ∑_𝑠′ 𝑝(𝑠│𝑠,𝑎) [𝑟 + 𝛾𝑣(𝑠′)]
		stable = agent.stable_from_stable(stable_copy, env, discount=discount)
		diff = np.max(np.abs(stable - stable_copy))
		# policy evaluation
		# delta는 오차 범위. delta가 클수록 많은 방향이 검출됨
		agent.pi = env.policy_from_stable(stable, discount=discount, delta=0.0001)
		epoch += 1
	print(f'iteration {epoch}\n{stable}')
	print(agent.policy_string())


def td0():
	"""
	given policy에 대하여 state value table을 계산한다
	v(s) ⟵ v(s) + 𝛼 [R + 𝛾 v(s') - v(s)]
	:return:
	"""
	row, col = 4, 4
	# deterministic environment, agent가 선택한 방향으로 이동한다
	env = Environment(row, col, terminal=((0, 0), (row - 1, col - 1)))
	#
	# env = Environment(row, col, terminal=((0, 0),))
	# agent는 네 방향 중 무작위로 선택하여 이동한다
	# 이 policy에 대한 state의 가치를 계산한다.
	agent = Agent(row, col, env.uniform_policy())
	# step size 𝛼
	step_size = 0.01
	# discount rate 𝛾
	discount = 1.0
	# state value는 모두 0으로 초기화
	stable = np.zeros((row, col))
	episode = EpisodeGenerator(env, agent)
	# 출력하는 iteration
	display = (0, 1, 2, 3, 10, 100)
	epoch = 0
	while epoch < 30000:
		if epoch in display:
			print(f'iteration {epoch}\n{stable}')
		state = episode.start_state()
		ars = episode.policy_next(state)  # action, reward, next_state
		while ars is not None:
			# v(s) ⟵ v(s) + 𝛼 [R + 𝛾 v(s') - v(s)]
			stable[state] += step_size * (ars[1] + discount * stable[ars[2]] - stable[state])
			state = ars[2]
			ars = episode.policy_next(state)  # action, reward, next_state
		epoch += 1
	print(f'iteration {epoch}\n{stable}')


def q_learning():
	row, col = 4, 4
	# stochastic environment
	# Environment.standard_prob(0.6)은 다음과 같다
	# [[.7, .1, .1, .1],
	#  [.1, .7, .1, .1],
	#  [.1, .1, .7, .1],
	#  [.1, .1, .1, .7]]
	# 따라서 left로 action을 취하면 다른 방향으로 이동할 확률이 각각 0.1이다.
	# right, up, down도 마찬가지
	env = Environment(row, col, terminal=((0, 0), (row - 1, col - 1)), prob=Environment.standard_prob(0.6))
	# policy-independent
	agent = Agent(row, col)
	# q는 랜덤하게 초기화
	# q = np.zeros((row, col, 4))
	# 이어도 비슷하다
	# qtable = np.random.rand(row, col, 4) - 1
	qtable = np.zeros((row, col, 4))
	# terminal state에서는 q-value(action value)가 0.0이다
	for t in env.terminal:
		qtable[t][:] = 0.0
	# action은 𝜖-greedy로 선택
	# 𝜖 for 𝜖-greedy
	epsilon = 0.1
	# step size 𝛼
	step_size = 0.01
	# discount rate 𝛾
	discount = 1.0
	gen = EpisodeGenerator(env, agent)
	# 출력하는 iteration
	display = (0, 1, 2, 3, 10, 100)
	epoch = 0
	while epoch < 10000:
		if epoch in display:
			print(f'iteration {epoch}\n{np.array2string(qtable, precision=2)}')
			pi = env.policy_from_qtable(qtable, epsilon=0.0, delta=0.01)
			print(Agent.policy_to_string(pi))
		state = gen.start_state()
		ars = gen.q_next(qtable, state, epsilon)  # action, reward, next_state
		while ars is not None:
			# 𝑞(𝑠, 𝑎) ⟵ 𝑞(𝑠, 𝑎) + 𝛼[𝑟 + 𝛾 max_𝑎′ ⁡𝑞(𝑠′, 𝑎′) − 𝑞(𝑠, 𝑎)]
			qtable[state][ars[0]] += step_size * (ars[1] + discount * np.max(qtable[ars[2]]) - qtable[state][ars[0]])
			state = ars[2]
			ars = gen.q_next(qtable, state, epsilon)  # action, reward, next_state
		epoch += 1
	print(f'iteration {epoch}\n{np.array2string(qtable, precision=2)}')
	# policy를 계산한다
	# delta는 오차 범위. delta가 크면 여러 방향이 검출된다
	pi = env.policy_from_qtable(qtable, epsilon=0.0, delta=0.01)
	print(Agent.policy_to_string(pi))


def episode0():
	# policy에 따라 episode 생성
	env = Environment()
	agent = Agent(pi=env.uniform_policy())
	gen = EpisodeGenerator(env, agent)
	epi = gen.policy_episode()
	print(epi)
	print(EpisodeGenerator.to_string(epi))


def episode1():
	# q에 따라 episode 생성
	env = Environment()
	agent = Agent()
	q = np.random.rand(4, 4, 4) - 1
	gen = EpisodeGenerator(env, agent)
	epi = gen.q_episode(q, epsilon=0.1)
	print(epi)
	print(EpisodeGenerator.to_string(epi))


# policy_evaluation()
# policy_iteration()
# value_iteration()
# td0()
q_learning()
# episode0()
# episode1()
