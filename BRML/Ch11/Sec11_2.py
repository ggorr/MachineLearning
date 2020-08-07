# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def Example11_2():
	def p(v, h, theta):
		return np.exp(-(v - theta * h) ** 2) / np.sqrt(np.pi)

	def plotLogLikelihood(v=2.75):
		x = np.linspace(np.floor(v / 2), np.ceil(v), 41)
		y = map(lambda t: np.log(p(v, 1, t) + p(v, 2, t)), x)
		plt.plot(x, y)
		plt.show()

	def em(v=np.float(2.75), theta=np.float(1.8)):
		while True:
			a, b = p(v, 1, theta), p(v, 2, theta)

			q2 = b / (a + b)
			newTheta = v * (2 * q2 + (1 - q2)) / (4 * q2 + (1 - q2))
			print q2, newTheta

			# q1 = a / (a + b)
			# newTheta = v * (2 * (1 - q1) + q1) / (4 * (1 - q1) + q1)
			# print q1, newTheta

			if np.abs(newTheta - theta) < 0.00001:
				return newTheta
			else:
				theta = newTheta

	plotLogLikelihood()
	print em()


def Example11_3():
	def likelihood(tt11, tt12, tt21, tt22):
		return np.log(tt11) + np.log(tt11 + tt12) + np.log(tt12 + tt22)

	EPSILON = 0.00000001
	t11, t12, t21, t22 = 0.25, 0.25, 0.25, 0.25
	lk = likelihood(t11, t12, t21, t22)
	while (True):
		q1_1 = t11 / (t11 + t12)  # p(x2=1 | x1=1)
		q2_1 = t12 / (t12 + t22)  # p(x1=1 | x2=2)
		t11_new = (1.0 + q1_1) / 3.0
		t12_new = (1 - q1_1 + q2_1) / 3.0
		t21_new = 0
		t22_new = (1 - q2_1) / 3.0
		# print 'sum = ', t11_new + t12_new + t21_new + t22_new
		lk_new = likelihood(t11_new, t12_new, t21_new, t22_new)

		# if np.abs(t11 - t11_new) < EPSILON and np.abs(t12 - t12_new) < EPSILON and np.abs(t22 - t22_new) < EPSILON:
		if np.abs(lk - lk_new) < EPSILON:
			print t11_new, t12_new, t21_new, t22_new, lk_new
			break
		t11 = t11_new
		t12 = t12_new
		t21 = t21_new
		t22 = t22_new
		lk = lk_new
		print t11, t12, t21, t22, lk


def Example11_4():
	EPSILON = 0.0000000001
	# data from figure 11.3
	# x^n = (s^n, c^n) for n = 1, ..., 7
	X = [[1, 1], [0, 0], [1, 1], [1, 0], [1, 1], [0, 0], [0, 1]]

	def p(a, c, s, theta_a_1, theta_s_1, theta_c_1):
		"""
		p(a, c, s) = p(c | a, s) p(a) p(s)
		    p(c | a, s) = theta_c(a, s)      if c = 1
		                  1 -  theta_c(a, s) if c = 0
		    p(a) = theta_a     if a = 1
		         = 1 - theta_a if a = 0
		    p(s) = theta_s     if s = 1
		           1 - theta_s if s = 0
		"""
		if c == 0:
			value = 1 - theta_c_1[a, s]
		else:
			value = theta_c_1[a, s]
		if a == 0:
			value *= 1 - theta_a_1
		else:
			value *= theta_a_1
		if s == 0:
			value *= 1 - theta_s_1
		else:
			value *= theta_s_1
		return value

	def logLikelihood(theta_a_1, theta_s_1, theta_c_1):
		"""
		log p(X | theta) = sum_{n=1}^{7} log p(x^n | theta)
		                 = sum_{n=1}^{7} log p(c^n, s^n | theta)
		                 = sum_{n=1}^{7} log (p(a=0, c^n, s^n | theta) + p(a=1, c^n, s^n | theta))
		"""
		value = np.float(0)
		for x in X:
			value += np.log(p(0, x[1], x[0], theta_a_1, theta_s_1, theta_c_1) + \
			                p(1, x[1], x[0], theta_a_1, theta_s_1, theta_c_1))
		return value

	def logLikelihood1(theta_a_1, theta_s_1, theta_c_1):
		value = np.float(0)
		for x in X:
			if x[0] == 0:
				v = 1 - theta_s_1
			else:
				v = theta_s_1
			if x[1] == 0:
				v *= (1 - theta_c_1[0, x[0]]) * (1 - theta_a_1) + (1 - theta_c_1[1, x[0]]) * theta_a_1
			else:
				v *= theta_c_1[0, x[0]] * (1 - theta_a_1) + theta_c_1[1, x[0]] * theta_a_1
			value += np.log(v)
		return value

	def em():
		"""
		참고할 것은 log likelihood를 제외한 다른 값은 초기값에 따라 수렴하는 값이 달라진다는 것이다.
		theta_a_1, theta_s_1, theta_c_1의 초기값을 바꾸면 수렴하는 값이 달라진다.
		다만 log likelihood의 값은 항상 같은 값으로 수렴한다.
		이유를 설명할 수 있겠는가?
		"""

		theta_a_1 = np.float(0.5)  # theta_a(a=1)
		theta_s_1 = np.float(0.5)  # theta_s(s=1), 4.0/7.0으로 시작해도 된다
		# [[theta_c(c=1 | a=0, s=0), theta_c(c=1 | a=0, s=1)],
		#  [theta_c(c=1 | a=1, s=0), theta_c(c=1 | a=1, s=1)]]
		theta_c_1 = np.array([[0.7, 0.4], [0.3, 0.9]])
		logLi = logLikelihood(theta_a_1, theta_s_1, theta_c_1)
		t = 1
		while (True):
			################# E-step #################
			# q(a | c=0, s=0) = p(a | c=0, s=0, theta)
			#                 = p(a, c=0, s=0 | theta) / (p(a=0, c=0, s=0 | theta) + p(a=1, c=0, s=0 | theta))
			#                 = p(c=0 | a, s=0, theta) p(a | theta) p(s=0 | theta) / ...
			#                 = p(c=0 | a, s=0, theta) p(a | theta) /
			#                            (p(c=0 | a=0, s=0, theta) p(a=0 | theta) + p(c=0 | a=1, s=0, theta) p(a=1 | theta))
			# and so on
			# For given theta, q(a=1 | c, s) = p(a=1 | c, s, theta)
			# q_a_1 = q(a=1 | c, s)
			#    [[q(a=1 | c=0, s=0). q(a=1 | c=0, s=1)],
			#     [q(a=1 | c=1, s=0). q(a=1 | c=1, s=1)]]
			# c = 0, s = 0
			# q00 = q(a=1 | c=0, s=0)
			q00 = (1 - theta_c_1[1, 0]) * theta_a_1 / ((1 - theta_c_1[0, 0]) * (1 - theta_a_1) + (1 - theta_c_1[1, 0]) * theta_a_1)
			# c = 0, s = 1
			# q01 = q(a=1 | c=0, s=1)
			q01 = (1 - theta_c_1[1, 1]) * theta_a_1 / ((1 - theta_c_1[0, 1]) * (1 - theta_a_1) + (1 - theta_c_1[1, 1]) * theta_a_1)
			# c = 1, s = 0
			# q10 = q(a=1 | c=1, s=0)
			q10 = theta_c_1[1, 0] * theta_a_1 / (theta_c_1[0, 0] * (1 - theta_a_1) + theta_c_1[1, 0] * theta_a_1)
			# c = 1, s = 1
			# q11 = q(a=1 | c=1, s=1)
			q11 = theta_c_1[1, 1] * theta_a_1 / (theta_c_1[0, 1] * (1 - theta_a_1) + theta_c_1[1, 1] * theta_a_1)
			q_a_1 = np.array([[q00, q01], [q10, q11]])

			################# M-step #################
			theta_s_1_new = np.float(4) / np.float(7)
			theta_a_1_new = 0
			for x in X:
				theta_a_1_new += q_a_1[x[1], x[0]]
			theta_a_1_new /= len(X)
			theta_c_1_new = np.zeros((2, 2))
			theta_c_1_new[0, 0] = (1 - q10) / (2 * (1 - q00) + (1 - q10))
			theta_c_1_new[0, 1] = 3 * (1 - q11) / (3 * (1 - q11) + (1 - q01))
			theta_c_1_new[1, 0] = q10 / (2 * q00 + q10)
			theta_c_1_new[1, 1] = 3 * q11 / (3 * q11 + q01)

			################# log likelihood #################
			logLi_new = logLikelihood(theta_a_1_new, theta_s_1_new, theta_c_1_new)

			print logLi, logLi_new
			print theta_a_1, theta_a_1_new
			print theta_s_1, theta_s_1_new
			print theta_c_1
			print theta_c_1_new
			print '=========================='

			if np.abs(theta_a_1_new - theta_a_1) < EPSILON and \
							np.abs(theta_s_1_new - theta_s_1) < EPSILON and \
							np.max(np.abs(theta_c_1_new - theta_c_1)) < EPSILON:
				break

			# if np.abs(logLi - logLi_new) < EPSILON:
			#	print 'iteration: ', t
			#	break
			theta_a_1 = theta_a_1_new
			theta_s_1 = theta_s_1_new
			theta_c_1 = theta_c_1_new
			logLi = logLi_new
			t += 1

		return theta_a_1, theta_s_1, theta_c_1

	a1, s1, c1 = em()
	print 'theta_a_1 =', a1
	print 'theta_s_1 =', s1
	print 'theta_c_1 =', c1
	print 'log likelihood =', logLikelihood(a1, s1, c1)

	######################## test ####################################
	# compute p(c=1 | s=1)
	# p(c=1 | s=1) = p(c=1, s=1) / p(s=1)
	#              = (p(a=0, c=1, s=1) + p(a=1, c=1, s=1)) / ...
	# num = p(0, 1, 1, a1, s1, c1) + p(1, 1, 1, a1, s1, c1)
	# den = p(0, 0, 1, a1, s1, c1) + p(1, 0, 1, a1, s1, c1) + p(0, 1, 1, a1, s1, c1) + p(1, 1, 1, a1, s1, c1)
	# print num / den

	####################### direct computation ############################
	# assume that there is no latent variable(asbestos)
	# parameters theta_s_1 = p(s=1), theta_c_1 = [p(c=1 | s=0), p(c=1 | s=1)]
	theta_s_1 = sum([x[1] for x in X]) / np.float(len(X))
	theta_c_1 = np.array([ \
		sum([1 for x in X if x[0] == 0 and x[1] == 1]) / np.float(sum([1 for x in X if x[0] == 0])), \
		sum([1 for x in X if x[0] == 1 and x[1] == 1]) / np.float(sum([1 for x in X if x[0] == 1]))])
	# print theta_s_1, theta_c_1

	logLi = 0
	for x in X:
		if x[0] == 0:
			logLi += np.log(1 - theta_c_1[0] if x[1] == 0 else theta_c_1[0]) + np.log(1 - theta_s_1)
		else:
			logLi += np.log(1 - theta_c_1[1] if x[1] == 0 else theta_c_1[1]) + np.log(theta_s_1)
	print 'log likelihood without hidden variable =', logLi


def Example11_4_Algorithm11_2():
	EPSILON = 0.0000000001
	# data from figure 11.3
	X = [[1, 1], [0, 0], [1, 1], [1, 0], [1, 1], [0, 0], [0, 1]]

	N = len(X)
	p_a_1 = np.float(0.5)  # p(a=1)
	p_s_1 = np.float(0.5)  # p(s=1)
	# [[p(c=1 | a=0, s=0), p(c=1 | a=0, s=1)],
	#  [p(c=1 | a=1, s=0), p(c=1 | a=1, s=1)]]
	p_c_1 = np.array([[0.7, 0.4], [0.3, 0.9]])

	p_a = np.array([1 - p_a_1, p_a_1])  # p(a)
	p_s = np.array([1 - p_s_1, p_s_1])  # p(s)
	p_c = np.zeros((2, 2, 2))  # p_c[c, a, s] = p(c | a, s)
	for a in range(2):
		for s in range(2):
			p_c[0, a, s] = 1 - p_c_1[a, s]  # p(c=0 | a, s)
			p_c[1, a, s] = p_c_1[a, s]  # p(c=1 | a, s)

	def p(a, c, s):
		"""
		:return: p(a, c, s) = p(c | a, s) p(a) p(s)
		"""
		return p_c[c, a, s] * p_a[a] * p_s[s]

	q = np.zeros((N, 2, 2, 2))  # n, a, c, s
	while True:
		# print 'p_a =', p_a
		# print 'p_s =', p_s
		# print 'p_c =', p_c
		for n in range(N):
			for a in range(2):
				for c in range(2):
					for s in range(2):
						# q^n_t(x) = p_t(h^n | v^n) delta(v, v^n)
						if [s, c] == X[n]:
							# q^n_t(x) = p_t(a | c^n, s^n)
							#          = p(a, c^n, s^n) / p(c^n, s^n)
							#          = p(a, c^n, s^n) / (p(a=0, c^n, s^n) + p(a=1, c^n, s^n))
							q[n, a, c, s] = p(a, c, s) / (p(0, c, s) + p(1, c, s))
						else:
							# q^n_t(x) = 0
							q[n, a, c, s] = 0
		p_a_new = np.zeros(2)
		p_s_new = np.zeros(2)
		for n in range(N):
			p_a_new[0] += q[n, 0, 0, 0] + q[n, 0, 0, 1] + q[n, 0, 1, 0] + q[n, 0, 1, 1]
			p_a_new[1] += q[n, 1, 0, 0] + q[n, 1, 0, 1] + q[n, 1, 1, 0] + q[n, 1, 1, 1]
			p_s_new[0] += q[n, 0, 0, 0] + q[n, 0, 1, 0] + q[n, 1, 0, 0] + q[n, 1, 1, 0]
			p_s_new[1] += q[n, 0, 0, 1] + q[n, 0, 1, 1] + q[n, 1, 0, 1] + q[n, 1, 1, 1]
		p_a_new /= 7
		p_s_new /= 7
		p_c_new = np.zeros((2, 2, 2))
		for c in range(2):
			for a in range(2):
				for s in range(2):
					num, den = 0, 0
					for n in range(N):
						num += q[n, a, c, s]
						den += q[n, a, 0, s] + q[n, a, 1, s]
					p_c_new[c, a, s] = num / den
		tf_a = np.abs(p_a[0] - p_a_new[0]) < EPSILON
		tf_s = np.abs(p_s[0] - p_s_new[0]) < EPSILON
		tf_c = True
		for a in range(2):
			for c in range(2):
				for s in range(2):
					if np.abs(p_c[c, a, s] - p_c_new[c, a, s]) >= EPSILON:
						tf_c = False
		if tf_a and tf_s and tf_c:
			break
		p_a = p_a_new
		p_s = p_s_new
		p_c = p_c_new
	print 'p_a =', p_a
	print 'p_s =', p_s
	print 'p_c =', p_c


if __name__ == "__main__":
	np.set_printoptions(threshold=np.nan, linewidth=np.nan)
	# Example11_2()
	# Example11_3()
	Example11_4()
	print '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-'
	Example11_4_Algorithm11_2()
