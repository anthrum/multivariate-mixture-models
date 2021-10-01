import pandas as pd
import numpy as np
import random
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt

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

# Calculating returns (we're calculating daily log returns considering the difference between closing and opening prices
sp500_log_returns = (np.log(sp500.iloc[:, 1]) - np.log(sp500.iloc[:, 3]))*100
gold_log_returns = (np.log(gold.iloc[:, 1]) - np.log(gold.iloc[:, 3]))*100

# Bivariate returns dataframe
returns = pd.DataFrame({'sp500 returns':sp500_log_returns,'gold returns': gold_log_returns})
print(returns)

# Mixture Model Selection (lowest aic criterion)
components = 200
#CRITERION = 'bic'



# Generating simulated sample using the best fitted model
model = mixture.GaussianMixture(components).fit(returns)
print(
    f'Generating sample from normal mixture distribuiton with '
    f'{components} components')

generated = pd.DataFrame(
    #model.sample(len(gold.index))[0]
    model.sample(len(gold.index))[0])


# Descriptive statistics of the margins
print( 'observed sample: \n \n', stats.mstats.describe(returns), '\n')
print('generated sample \n \n', stats.mstats.describe(generated),'\n')

# Plotting
returns.hist(bins=30)
generated.hist(bins=30)

returns.plot.scatter(['sp500 returns'],['gold returns'], title = 'Original Sample', xlabel = 'sp500', ylabel = 'gold')
generated.plot.scatter([0,],[1,], title = 'Sample from fitted gaussian mixture', xlabel = 'M1', ylabel = 'M2')
plt.show()
print(model.get_params())