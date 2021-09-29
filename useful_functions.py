import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt
import math


def tail_scatter(quantile,  df, column_label1, column_label2, lower):
    print(df[{column_label1}])
    sorted_array = df.sort_values(by= column_label1, ascending= lower)[column_label1]
    tail = sorted_array[:math.floor(quantile*len(sorted_array))]

    print(sorted_array)
    print(tail)
    observations_in_tail = []
    for element in tail:
        observations_in_tail.append(np.where(df[{column_label1}] == element)[0][0])
    observations_in_tail = df.iloc[observations_in_tail]
    observations_in_tail.plot.scatter( column_label1, column_label2, title= f'{column_label1} tail')
    plt.show()



def bivariate_synthesis_plot(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.hist(df.iloc[:,0], bins = 100, density= True)
    xmin, xmax = plt.xlim()
    ax3.scatter(df.iloc[:,0], df.iloc[:,1])
    ax4.hist(df.iloc[:,1], bins = 100, orientation='horizontal', density= True)


    # Plot the PDF.
    column0_sorted = df.iloc[:,0].sort_values().to_numpy()
    column1_sorted= df.iloc[:,1].sort_values().to_numpy()

    ax1.plot(column0_sorted,
             1 / (df.iloc[:,0].std() *np.sqrt(2 * np.pi)) *
             np.exp(- (column0_sorted - df.iloc[:,0].mean())**2 / (2 * df.iloc[:,0].std()**2)))

    ax4.plot( 1 / (df.iloc[:, 1].std() * np.sqrt(2 * np.pi)) *
             np.exp(- (column1_sorted - df.iloc[:, 1].mean()) ** 2 / (2 * df.iloc[:, 1].std() ** 2)),
              column1_sorted)




    ax2.scatter(column1_sorted ,column0_sorted )
    ax2.plot(column0_sorted,column0_sorted)
    plt.show()