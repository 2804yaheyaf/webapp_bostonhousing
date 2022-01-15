import numpy as np
import matplotlib.pyplot as plt 
from sklearn import metrics
import pandas as pd  
import seaborn as sns 
from scipy import stats
# %matplotlib inline
# To prevent from warnings
import warnings
warnings.filterwarnings('ignore')
import requests
import json

print("Imported all libraries")

df=pd.read_csv("Boston Dataset.csv")
df.drop(columns=['Unnamed: 0'], axis=0, inplace=True)
# df.head()

# df.shape

## we will omit CHAS as its correlation with the target variable was negligible. other features having high skewness can be normalized.
df=df.drop(columns=["chas","black"]) # omitting blacks as it feels too racist

## taking only significant features( highly correlated ) for training model
cols=['rm','tax','ptratio','lstat','indus', 'medv']
df1=pd.DataFrame(df[cols])

# standardization
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()

# fit our data
scaled_cols = scalar.fit_transform(df1[cols])
scaled_cols = pd.DataFrame(scaled_cols, columns=cols)
# scaled_cols.head()

## since all the significant variables arent normally distributed, we will perform min max normalization on all of them

colm = ['tax', 'ptratio', 'lstat']
for col in colm:
    minimum = min(df1[col])
    maximum = max(df1[col])
    df1[col] = (df1[col] - minimum)/ (maximum - minimum)

## Splitting the predictor and the outcome variable
X = df1.drop(columns=['medv'], axis=1)
# X.head()

## outcome variable is 'medv' which is the output column
y=df1['medv']
# y.shape

from sklearn.ensemble import RandomForestRegressor
# creating object of Random Forest Regressor
model_rf = RandomForestRegressor()
from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Performing train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7,random_state=123)

## model training

model_rf.fit(X_train, y_train)

## saving model

import pickle
pickle.dump(model_rf, open( 'model_file' + ".pkl", "wb"))

# Loading model to compare the results
modell = pickle.load(open('model_file.pkl','rb'))
print(modell.predict([[6.5,0.4,0.6,0.3,12.0]]))