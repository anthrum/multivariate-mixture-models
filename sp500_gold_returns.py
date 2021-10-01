import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt
import math
from statsmodels.distributions.empirical_distribution import ECDF
import useful_functions


sp500 = pd.read_csv('HistoricalData_SP500.csv')
gold = pd.read_csv('HistoricalData_Gold.csv').drop([0])

#print(sp500)
#print(gold)

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




# Calculating returns (we're calculating daily log returns considering the difference between closing and opening prices
sp500_log_returns = (np.log(sp500.iloc[:, 1]) - np.log(sp500.iloc[:, 3]))*100
gold_log_returns = (np.log(gold.iloc[:, 1]) - np.log(gold.iloc[:, 3]))*100

# Bivariate returns dataframe
returns = pd.DataFrame({'sp500 returns':sp500_log_returns,'gold returns': gold_log_returns})

print(returns)



ecdf_dict = useful_functions.bivariate_ecdf(returns, 'sp500 returns', 'gold returns')
ecdf_list = []
for point in returns.dropna().to_dict('records'):
    ecdf_list.append(ecdf_dict[(point['sp500 returns'], point['gold returns'])])
ecdf_array = np.array(ecdf_list)

#----------------------        --------------------------      -----------------------------          -----------------#
#3D histogram
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

hist, xedges, yedges  = np.histogram2d(sp500_log_returns, gold_log_returns, bins=30, density=True,
                      range=[[min(gold_log_returns),max(gold_log_returns)],
                             [min(sp500_log_returns),max(sp500_log_returns)]]
                      )

# Construct arrays for the anchor positions of the 400 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 400 bars.
dx = dy = 0.5
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
plt.show()

#Plotting synthesis plot
useful_functions.bivariate_synthesis_plot(returns.dropna())

x = stats.t.fit(gold_log_returns)
sample = stats.t(x[0],x[1],x[2]).rvs(125)
print(stats.kurtosis(sample))
print(stats.kurtosis(gold_log_returns))
print(x)
plt.hist(sample,bins = 10, density=True )
plt.hist(gold_log_returns, bins= 10, density=True, color='r')
plt.show()

kurtosis_diff = []
for i in range(100):
    x = stats.t.fit(gold_log_returns)
    sample = stats.t(x[0], x[1], x[2]).rvs(125)
    kurtosis_diff.append(stats.kurtosis(sample)-stats.kurtosis(gold_log_returns))

kurtosis_diff = np.array(kurtosis_diff)
print(kurtosis_diff.mean(), kurtosis_diff.std())
plt.hist(kurtosis_diff, density=True, bins=10)
plt.show()
# BIAS ESTIMATION OF KURTOSIS




#Tail scatter plots
useful_functions.tail_scatter(quantile=0.05,  df= returns,
             column_label2='gold returns', column_label1='sp500 returns', lower=True) # sp500 lower tail

useful_functions.tail_scatter(quantile=0.05,  df= returns,
             column_label1='gold returns', column_label2='sp500 returns', lower=True) # gold lower tail
useful_functions.tail_scatter(quantile=0.05,  df= returns,
             column_label2='gold returns', column_label1='sp500 returns', lower=False) # sp500 upper tail

useful_functions.tail_scatter(quantile=0.05,  df= returns,
             column_label1='gold returns', column_label2='sp500 returns', lower=False) # gold upper tail





