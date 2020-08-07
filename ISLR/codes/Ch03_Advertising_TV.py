import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def plot2d(X, y):
	plt.plot(X, y, '.')
	plt.show()


def linearRegression2d(X, y):
	lr = LinearRegression()
	lr.fit(X.reshape(-1, 1), y)

	################# information ##############################
	print('beta_0 =', lr.intercept_, ', beta_1 =', lr.coef_[0])
	hat_y = lr.predict(X.reshape(-1, 1))
	RSS = np.sum((y - hat_y) ** 2) # residual sum of squares
	print('RSS =', RSS)
	RSE = np.sqrt(RSS / (X.shape[0] - 2)) # residual standard error
	print('RSE =', RSE)
	TSS = np.sum((y-np.mean(y)) ** 2)
	R_square = 1 - RSS/TSS # R^2
	print(R_square)

	################## plot ################################
	plt.plot(X, y, '.')
	label = f'y = {lr.intercept_:0.5} + {lr.coef_[0]:0.5} * X'
	hat_y = lr.predict(X.reshape(-1, 1))
	plt.plot(X, hat_y, label=label)
	plt.legend()
	plt.show()


df = pd.read_csv('data/Advertising.csv', sep=',')  # DataFrame
tv = df.TV.values
radio = df.radio.values
newspaper = df.newspaper.values
sales = df.sales.values
plot2d(tv, sales)
linearRegression2d(tv, sales)
