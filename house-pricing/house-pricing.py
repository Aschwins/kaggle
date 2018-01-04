# House pricing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import defaultdict
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression


# Importing the datasets
train = pd.read_csv('train.csv') #train.dtypes
test = pd.read_csv('test.csv')

# Making a column name list
columns = list(train.columns.values)

# Making a list of all categorical variables
categorical = list()
for i in columns:
    if train[i].dtype == object:
        categorical.append(i)
del i

# Making a list for numerical variables       
numerical = []
for e in columns:
    if e not in categorical:
        numerical.append(e)       
del e

test_Id = test['Id']
train_Id = train['Id']

# Let's put all the data together in one big data frame.
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.drop(['Id'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

# Taking care of missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

# Not all houses have pools, but it's not missing data.
all_data['PoolQC'] = all_data['PoolQC'].fillna("None")

# Not all houses have a Misc Feature, but it's not missing data...
all_data['MiscFeature'] = all_data['MiscFeature'].fillna("None")

# Not all houses have an Alley, but it's not missing data...
all_data['Alley'] = all_data['Alley'].fillna("None")

# Not all houses have a fence
all_data['Fence'] = all_data['Fence'].fillna("None")

# Not all houses have a fireplace...
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna("None")

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)    
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()




# Checking for missing data
totaltrain = train.isnull().sum().sort_values(ascending=False)
totaltest = test.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
percent_2 = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data_train = pd.concat([totaltrain, percent], axis=1, keys=['Total', 'Percent'])
missing_data_train.head(20)
missing_data_test = pd.concat([totaltest, percent_2], axis=1, keys=['Total', 'Percent'])
missing_data_test.head(20)




# Replacing NaN in string vectors with None (categorical NaN's)
train[categorical] = train[categorical].replace(np.nan, 'None')

# Replacing numerical NaN's with the mean of the column
imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
imputer = imputer.fit(train[numerical])
df = train.copy()
df[numerical] = imputer.transform(train[numerical])

# Now we have a dataset, df, without missing values.

# Encoding the categorical data, and adding dummy variables
dfe = pd.get_dummies(df, columns = categorical)

# Defining the explanatory variables and dependent variable
X = dfe.drop('SalePrice', axis=1)
y = dfe[['SalePrice']]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# For the test set
test[categorical] = test[categorical].replace(np.nan, 'None')
numerical.remove('SalePrice')
imputer = imputer.fit(test[numerical])
dftest = test.copy()
dfteste = pd.get_dummies(dftest, columns = categorical)
dftest[numerical] = imputer.transform(test[numerical])






