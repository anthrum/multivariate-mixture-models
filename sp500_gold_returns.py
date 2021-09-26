import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt
import math

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

gold.reset_index(drop=True, inplace=True)
print(gold)
print(sp500)
# number index matches

gold.reset_index(drop=True, inplace=True)
print(gold)
print(sp500)
# number index matches

# Calculating returns (we're calculating daily log returns considering the difference between closing and opening prices
sp500_log_returns = (np.log(sp500.iloc[:, 1]) - np.log(sp500.iloc[:, 3]))*100
gold_log_returns = (np.log(gold.iloc[:, 1]) - np.log(gold.iloc[:, 3]))*100

# Bivariate returns dataframe
returns = pd.DataFrame({'sp500 returns':sp500_log_returns,'gold returns': gold_log_returns})
print(returns)

#g_returns = returns['gold returns']
#s_returns = returns['sp500 returns']

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.hist(sp500_log_returns, bins = 100, density= True)
xmin, xmax = plt.xlim()
ax3.scatter(sp500_log_returns, gold_log_returns)
ax4.hist(gold_log_returns, bins = 100, orientation='horizontal', density= True)


# Plot the PDF.
sorted_gold = np.sort(gold_log_returns)
sorted_sp500 = np.sort(sp500_log_returns)

ax1.plot(sorted_gold,
         1 / (gold_log_returns.std() *np.sqrt(2 * np.pi)) *
         np.exp(- (sorted_gold - gold_log_returns.mean())**2 / (2 * gold_log_returns.std()**2)))

ax4.plot(1 / (sp500_log_returns.std() *np.sqrt(2 * np.pi)) *
         np.exp(- (sorted_sp500 - sp500_log_returns.mean())**2 / (2 * sp500_log_returns.std()**2)),
         sorted_sp500)


ax2.scatter(sorted_gold ,sorted_sp500 )
ax2.plot(sorted_gold,sorted_gold)
plt.show()


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

def tail_scatter(quantile, sample_array, df, column_label1, column_label2, lower):
    if lower == True:
        sorted_array = np.sort(sample_array)
        tail = sorted_array[:math.floor(quantile*len(sorted_array))]
    elif lower == False:
        sorted_array = np.sort(sample_array)[::-1]
        tail = sorted_array[:math.floor(quantile*len(sorted_array))]
    else:
        print('ERROR')

    observations_in_tail = []
    for element in tail:
        observations_in_tail.append(np.where(sample_array == element)[0][0])
    observations_in_tail = df.iloc[observations_in_tail]
    observations_in_tail.plot.scatter( column_label1,column_label2)
    plt.show()

tail_scatter(quantile=0.05, sample_array=sp500_log_returns, df= returns,
             column_label2='gold returns', column_label1='sp500 returns', lower=True)
tail_scatter(quantile=0.05, sample_array=gold_log_returns, df= returns,

             column_label1='gold returns', column_label2='sp500 returns', lower=True)
tail_scatter(quantile=0.05, sample_array=sp500_log_returns, df= returns,
             column_label2='gold returns', column_label1='sp500 returns', lower=False)

tail_scatter(quantile=0.05, sample_array=gold_log_returns, df= returns,
             column_label1='gold returns', column_label2='sp500 returns', lower=False)