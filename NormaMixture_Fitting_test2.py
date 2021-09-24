import pandas as pd
import numpy as np
import random
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt

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

z = mixture.GaussianMixture(100).fit(returns)
