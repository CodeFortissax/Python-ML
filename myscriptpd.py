import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_decomposition, svm
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
import matplotlib as plt 

df = quandl.get('WIKI/GOOGL')

print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100
df['PCT_Change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
print(df.head())

forecast_col = df['Adj. Close']
(df.fillna(-99999, inplace=True))

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


X = np.array(df.drop(['label'], 1))
y = np.array(df.drop(['label']))

X = preprocessing.scale(X)
y = np.array(df['label'])




