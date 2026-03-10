#Practical 5 – One Way ANOVA

from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from scipy import stats

# Sample data from 3 groups
group1 = [23,29,25,34,30]
group2 = [19,20,22,24,25]
group3 = [15,18,20,21,17]
group4 = [24,26,28,30,29]

f_stat, p_value = stats.f_oneway(group1, group2, group3, group4)

print("F-statistic:", f_stat)
print("p-value:", p_value)

#Post-Hoc Test

data = [23,29,25,19,20,22,15,18,20]
groups = ['A','A','A','B','B','B','C','C','C']

tukey = pairwise_tukeyhsd(data, groups)

print(tukey)