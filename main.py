import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)  # Replace missing data points

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(["label"], 1))

y = np.array(df['label'])

X = preprocessing.scale(X)

y = np.array(df['label'])

print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = LinearRegression(n_jobs=10)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(accuracy)
