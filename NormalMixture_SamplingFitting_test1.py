import pandas as pd
import numpy as np
import random
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt

# For details on the function bellow check out https://github.com/anthrum/multivariate-mixture-models/blob/main/normal_variance_mixture.ipynb
def normal_variance_mixture(miu, sigma, n, a, b):
    Z = stats.multivariate_normal.rvs(cov=np.identity(len(sigma)), size=n)  # nx2 matrix
    W = np.sqrt(stats.uniform.rvs(loc=0.5, scale=3, size=n))

    for i in range(n):
        Z[i, :] = Z[i, :] * W[i]

    A = np.linalg.cholesky(sigma)  # Cholesky Decomposition of sigma Matrix
    X = np.array(miu) + np.matmul(A, Z.transpose()).transpose()  # X = miu + wAZ0 (normal variance mixture)
    return X

#parameters for original mixture model
SIGMA = np.array([[5,-2],[-2,8]])
MIU = [50,3]
N = 10_000
A = 0.5
B = 3


x = normal_variance_mixture(MIU, SIGMA, N, A, B)
#Fitting normal mixture model with 100 component distribution to the original with continuous uniform mixture rv.
y = mixture.GaussianMixture(100).fit(x)   
print(y.sample(N)[0])
print(x)

print( 'x: \n \n', stats.mstats.describe(x), '\n')
print('y \n \n', stats.mstats.describe(y.sample(N)[0]),'\n')
x = pd.DataFrame(x)
y = pd.DataFrame(y.sample(N)[0])

x.hist(bins=100)
y.hist(bins=100)
x.plot.scatter([0,],[1,], title = 'Original Sample', xlabel = 'M1', ylabel = 'M2')
y.plot.scatter([0,],[1,], title = 'Sample from fitted gaussian mixture', xlabel = 'M1', ylabel = 'M2')
plt.show()

# From the sample descriptions and graphical analysis, the two samples seem to be generated from quite similar distributions suggesting the fitted model might
# be an appropriate approximation.  
