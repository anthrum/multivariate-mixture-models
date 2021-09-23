import pandas as pd
import numpy as np
import random
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt

def normal_variance_mixture(miu, sigma, n, a, b):
    Z = stats.multivariate_normal.rvs(cov=np.identity(len(sigma)), size=n)  # nx2 matrix
    W = np.sqrt(stats.uniform.rvs(loc=0.5, scale=3, size=n))

    for i in range(n):
        Z[i, :] = Z[i, :] * W[i]

    A = np.linalg.cholesky(sigma)  # Cholesky Decomposition of sigma Matrix
    X = np.array(miu) + np.matmul(A, Z.transpose()).transpose()  # X = miu + wAZ0 (normal variance mixture)
    return X

SIGMA = np.array([[5,-2],[-2,8]])
MIU = [50,3]
N = 10_000
A = 0.5
B = 3

x = normal_variance_mixture(MIU, SIGMA, N, A, B)
y = mixture.GaussianMixture(100).fit(x)
print(y.sample(N)[0])
print(x)

print( 'x: \n \n', stats.mstats.describe(x), '\n')
print('y \n \n', stats.mstats.describe(y.sample(N)[0]),'\n')
x = pd.DataFrame(x)
y = pd.DataFrame(y.sample(N)[0])

#x.hist(bins=100)
#y.hist(bins=100)
#x.plot.scatter([0,],[1,], title = 'Original Sample', xlabel = 'M1', ylabel = 'M2')
#y.plot.scatter([0,],[1,], title = 'Sample from fitted gaussian mixture', xlabel = 'M1', ylabel = 'M2')
#plt.show()

sp500 = pd.read_csv('HistoricalData_1632437651658.csv')
gold = pd.read_csv('HistoricalData_1632440949562.csv').drop([0])

print(sp500)
print(gold)


gold1 = gold['Date']

# cleaning and fitting dates/index
comparison = sp500['Data'].reset_index(drop=True) == gold1.drop([len(gold1)],  axis = 0).reset_index(drop=True)
gold.index = np.arange(0, len(gold))
for i in range(len(comparison)):
    if comparison[i] == False:
        print(i)
        gold = gold.drop([i], axis = 0)
        break
    else:
        pass

gold.reset_index(drop=True, inplace=True)
print(gold)
print(sp500)
# number index matches

# Calculating returns (we're calculating daily log returns considering the difference between closing and opening prices
sp500_log_returns = np.log(sp500.iloc[:, 1]) - np.log(sp500.iloc[:, 3])
gold_log_returns = np.log(gold.iloc[:, 1]) - np.log(gold.iloc[:, 3])
returns = pd.DataFrame({'sp500 returns':sp500_log_returns,'gold returns': gold_log_returns})
print(returns)
#print(y)
z = mixture.GaussianMixture(100).fit(returns)