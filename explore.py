import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import wrangle as w
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway



def barplot(df, x, color):
    sns.barplot(data = df, x = x, y = 'charges', color = color)
    plt.title(f'Charges made based on {x}')
    plt.xlabel(x.capitalize())
    plt.ylabel('Charges')
    plt.show()



def welch_ttest(data,col):
    
    Null = "There is no difference in mean insurance charges between smokers and non smokers"
    Alternative = "There is a statistically significant difference in mean insurance charges between smokers and non smokers"

    alpha = 0.5
    smokers = data[data[col] == 'yes']['charges']
    non_smokers = data[data[col] == 'no']['charges']
    
    # Perform Welch's t-test
    t_stat, p_value = ttest_ind(smokers, non_smokers, equal_var=False)
    
    if p_value < alpha:        
        print("Reject the null hypothesis.")
        print(f"{col.capitalize()} status significantly affects insurance charges.")
    else:
        print("Fail to reject the null hypothesis.")
        print("No statistically significant difference detected.")
    
    
    print("Mean charges (Smokers):", smokers.mean())
    print("Mean charges (Non-Smokers):", non_smokers.mean())


def pearsonr_test (data, col):
    alpha = 0.05

    Null_Hypothesis= f'There is no linear correlation between {col.capitalize()} and insurance charges' 
    Alternative_Hypothesis = f'{col.capitalize()} is positively correlated with insurance charges.'
    
    r, p_value = pearsonr(x = data[col], y = data['charges'])

    if p_value < alpha:
        print("Reject the null hypothesis.")
        print(Alternative_Hypothesis)
    else:
        print("Fail to reject the null hypothesis.")
        print(Null_Hypothesis)
    
    print("Pearson r:", r)
    print("p-value:", p_value)



def regplot(data, x):

    sns.regplot(data = data, x = x, y = 'charges')
    plt.title(f'Insurance Charges vs {x.capitalize()}')
    plt.xlabel(x.capitalize())
    plt.ylabel('Charges')
    plt.show()


def anova_test (data, col):
    alpha = 0.05

    Null_Hypothesis= f'The mean insurance charges are equal across all {col.capitalize()}' 
    Alternative_Hypothesis = f'At least one {col.capitalize()} group has a different mean insurance charge.'

    groups= []
    
    for group in data[col].unique():
        data_point = data[data[col] == group]['charges']    
        
        groups.append(data_point)

    
    f_stat, p_value = f_oneway(*groups)

    if p_value < alpha:
        print("Reject the null hypothesis.")
        print(Alternative_Hypothesis)
    else:
        print("Fail to reject the null hypothesis.")
        print(Null_Hypothesis)
    
    print("F_stat:", f_stat)
    print("p-value:", p_value)
   