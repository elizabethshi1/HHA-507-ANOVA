# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 05:31:46 2021

@author: Elizabeth Shi
"""

import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, bartlett



url = "https://raw.githubusercontent.com/elizabethshi1/HHA-507-ANOVA/main/Medicare_Hospital_Spending_by_Claim.csv"
medicare = pd.read_csv(url)
medicare.info()

# 1 factor of state we have 50 level
medicare.State.value_counts()
len(medicare.State.value_counts())

# 1 factor of Period we have 4 level
medicare.Period.value_counts()
len(medicare.Period.value_counts())

# 1 factor for Claim type we have 8 level
medicare['Claim Type'].value_counts()
len(medicare['Claim Type'].value_counts())

hospitalspnding = medicare['Avg Spndg Per EP Hospital']

# 1 way anova 
# 1 DV - Avg Spndg Per EP Hospital
# 1 IV - period

#Is there a difference between "levels" of period and Avg Spndg Per EP Hospital?

# DV ~ C(IV) + C(IV)
model = smf.ols('hospitalspnding ~ C(Period)', data = medicare).fit()
stats.shapiro(model.resid)

# Charts

period1 = medicare[medicare['Period'] == '1 to 3 days Prior to Index Hospital Admission']
period2 = medicare[medicare['Period'] == 'During Index Hospital Admission']
period3 = medicare[medicare['Period'] == '1 through 30 days After Discharge from Index Hospital Admission']
period4 = medicare[medicare['Period'] == 'Complete Episode']

plt.hist(period1['Avg Spndg Per EP Hospital'])
plt.show()
plt.hist(period2['Avg Spndg Per EP Hospital'])
plt.show()
plt.hist(period3['Avg Spndg Per EP Hospital'])
plt.show()
plt.hist(period4['Avg Spndg Per EP Hospital'])
plt.show()

# Looks like the homogeneity of variabnce is met because there isn't much of a variance other than Complete Episode

stats.bartlett(period1['Avg Spndg Per EP Hospital'],
               period2['Avg Spndg Per EP Hospital'],
               period3['Avg Spndg Per EP Hospital'],
               period4['Avg Spndg Per EP Hospital'])

stats.f_oneway(period1['Avg Spndg Per EP Hospital'],
               period2['Avg Spndg Per EP Hospital'],
               period3['Avg Spndg Per EP Hospital'],
               period4['Avg Spndg Per EP Hospital'])

# # There is a small significant difference between the levels of period and the avg spending per EP hospital because the value of p is lower than 0.5

#Is there a difference between "levels" of Claim type and Avg Spndg Per EP Hospital?
# 1 way anova 
# 1 DV - Avg Spndg Per EP Hospital
# 1 IV - Claim Type

claim_type = medicare['Claim Type']

# DV ~ C(IV) + C(IV)
model2 = smf.ols('hospitalspnding ~ C(claim_type)', data = medicare).fit()
stats.shapiro(model.resid)

claim1 = medicare[medicare['Claim Type'] == 'Home Health Agency']
claim2 = medicare[medicare['Claim Type'] == 'Hospice']
claim3 = medicare[medicare['Claim Type'] == 'Skilled Nursing Facility']
claim4 = medicare[medicare['Claim Type'] == 'Durable Medical Equipment']
claim5 = medicare[medicare['Claim Type'] == 'Outpatient']
claim6 = medicare[medicare['Claim Type'] == 'Inpatient']
claim7 = medicare[medicare['Claim Type'] == 'Carrier']
claim8 = medicare[medicare['Claim Type'] == 'Total']

stats.bartlett(claim1['Avg Spndg Per EP Hospital'],
               claim2['Avg Spndg Per EP Hospital'],
               claim3['Avg Spndg Per EP Hospital'],
               claim4['Avg Spndg Per EP Hospital'],
               claim5['Avg Spndg Per EP Hospital'],
               claim6['Avg Spndg Per EP Hospital'],
               claim7['Avg Spndg Per EP Hospital'],
               claim8['Avg Spndg Per EP Hospital'])

stats.f_oneway(claim1['Avg Spndg Per EP Hospital'],
               claim2['Avg Spndg Per EP Hospital'],
               claim3['Avg Spndg Per EP Hospital'],
               claim4['Avg Spndg Per EP Hospital'],
               claim5['Avg Spndg Per EP Hospital'],
               claim6['Avg Spndg Per EP Hospital'],
               claim7['Avg Spndg Per EP Hospital'],
               claim8['Avg Spndg Per EP Hospital'])

# There is a small significant difference between the levels of claim type and the avg spending per EP hospital because the value of p is lower than 0.5
# It also seems to be a non-parametric as well


#Is there a difference between "levels" of States and Avg Spndg Per EP Hospital?
# 1 way anova 
# 1 DV - Avg Spndg Per EP Hospital
# 1 IV - States

# DV ~ C(IV) + C(IV)
model3 = smf.ols('hospitalspnding ~ C(State)', data = medicare).fit()
stats.shapiro(model.resid)


import statsmodels.stats.multicomp as mc
comp = mc.MultiComparison( medicare['Avg Spndg Per EP Hospital'], medicare['State'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())

tukey1way[3].describe(include = 'all')

# There does not look to be a large difference between the levels of States and the avg spnding per EP hospital because by looking at the dataframe
# It seems that most of the p-values differences are higher than .05














