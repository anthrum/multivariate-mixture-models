import pandas as pd
import numpy as np
import random
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt

sp500 = pd.read_csv('HistoricalData_1632437651658.csv')
gold = pd.read_csv('HistoricalData_1632440949562.csv').drop([0])

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
MAX_COMPONENTS = 200
CRITERION = 'bic'

def norm_mixture_mod_select(max_components, info_criterion):
    scores = []
    if info_criterion == 'aic':
        for i in range(max_components):    #measures aic from 1 component to max_components
            scores.append(mixture.GaussianMixture(i + 1).fit(returns).aic(returns))
    elif info_criterion == 'bic':
        for i in range(max_components):    #measures bic from 1 component to max_components
            scores.append(mixture.GaussianMixture(i + 1).fit(returns).bic(returns))
    else:
        print("ERROR: unrecognized information criterion. Please insert 'aic' or 'bic' as information criterion")
    num_of_components = scores.index(min(scores)) + 1
    # Fitting model with the selected number of components
    model = mixture.GaussianMixture(num_of_components).fit(returns)
    return model, num_of_components

# Generating simulated sample using the best fitted model
print(
    f'Generating sample from normal mixture distribuiton with '
    f'{norm_mixture_mod_select(MAX_COMPONENTS, CRITERION)[1]} components')

generated = pd.DataFrame(
    norm_mixture_mod_select(MAX_COMPONENTS, CRITERION)[0]
        .sample(len(gold.index))[0]
    )

# Descriptive statistics of the margins
print( 'observed sample: \n \n', stats.mstats.describe(returns), '\n')
print('generated sample \n \n', stats.mstats.describe(generated),'\n')

# Plotting
returns.hist(bins=30)
generated.hist(bins=30)

returns.plot.scatter(['sp500 returns'],['gold returns'], title = 'Original Sample', xlabel = 'M1', ylabel = 'M2')
generated.plot.scatter([0,],[1,], title = 'Sample from fitted gaussian mixture', xlabel = 'M1', ylabel = 'M2')
plt.show()
