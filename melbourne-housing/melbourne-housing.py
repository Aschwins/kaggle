# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 14:56:21 2017

@author: 1asch
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, skew
from scipy import stats

#loading the dataset
df = pd.read_csv('Melbourne_housing_FULL.csv')

# Making a column name list
columns = list(df.columns.values)
print("all_data size is : {}".format(df.shape))
df.head()

# Saving our response vector Price.
Price = df['Price']
print(np.mean(Price))
print(np.max(Price))

# Let's see what Price looks like!
sns.distplot(df['Price'].dropna())
plt.show()

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
df["Price"] = np.log1p(df["Price"])

#Check the new distribution 
sns.distplot(df["Price"].dropna() , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df["Price"].dropna())
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df["Price"].dropna(), plot=plt)
plt.show()

# Correlation Matrix
corrmat = df.dropna().corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Price')['Price'].index
cm = np.corrcoef(df[cols].dropna().values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

df = df.drop(df[(df['BuildingArea']>2000) | (df['Landsize']>30000)].index)

# Checking for missing data
df_na = (df.isnull().sum()/len(df))*100
df_na = df_na.drop(df_na[df_na==0].index).sort_values(ascending=False)[:30]
df_na.head(30)
missing_data = pd.DataFrame({'Missing Ratio': df_na})
missing_data.head(30)

nw = df[df['BuildingArea'].dropna()]

df["Date"] = pd.to_datetime(df["Date"],dayfirst=True)


for column in columns:
    if len(df[df[column] == 'nan'])>0:
        print(column)
        
