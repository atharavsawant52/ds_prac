# t-Test Code (prac4_ttest.py)

import numpy as np
from scipy import stats

sample1 = np.random.normal(10,2,30)
sample2 = np.random.normal(12,2,30)

t,p = stats.ttest_ind(sample1,sample2)

print("t-statistic:",t)
print("p-value:",p)


# 2️ Chi-Square Code (prac4_chi.py)

import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("prac4.csv")

table = pd.crosstab(df["Gender"],df["Result"])

chi,p,_,_ = chi2_contingency(table)

print("Chi-square:",chi)
print("p-value:",p)
