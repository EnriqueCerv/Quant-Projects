# %%
import numpy as np
import pandas as pd
import scipy as scp
import matplotlib.pyplot as plt
import yfinance as yf
import sklearn

from sklearn.ensemble import RandomForestClassifier
# %%
nvda = yf.Ticker('NVDA')
nvda = nvda.history(period='max')

nvda.index
nvda.plot.line(y = 'Close', use_index = True)

nvda['Tomorrow'] = nvda['Close'].shift(-1)
# nvda

nvda['Target'] = (nvda['Tomorrow'] > nvda['Close']).astype(int)
# nvda

nvda = nvda.loc['2000-01-01':].copy()
# nvda

model = RandomForestClassifier(n_estimators=100, min_samples_split=100)
train = nvda.iloc[:-100]
test = nvda.iloc[-100:]

predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])

from sklearn.metrics import precision_score
predictions = model.predict(test[predictors])
predictions = pd.Series(predictions, index = test.index)

score = precision_score(test['Target'], predictions)
print(score)

combined = pd.concat([test['Target'], predictions], axis = 1)
combined.plot()
# %%
aapl = yf.Ticker('AAPL')
aapl = aapl.history(period='max')

aapl.plot.line(y='Close', use_index = True)

aapl['Tomorrow'] = aapl['Close'].shift(-1)
# aapl

aapl['Target'] = (aapl['Tomorrow'] > aapl['Close']).astype(int)
# aapl

aapl = aapl.loc['2000-01-01':].copy()
# aapl

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100)
train = aapl.iloc[:-100]
test = aapl.iloc[-100:]

predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])

from sklearn.metrics import precision_score
predictions = model.predict(test[predictors])
predictions = pd.Series(predictions, index = test.index)

score = precision_score(test['Target'], predictions)
print(score)

combined = pd.concat([test['Target'], predictions], axis = 1)
combined.plot()

# %%

voo = yf.Ticker('VOO')
voo = voo.history(period='max')

voo.plot.line(y='Close', use_index = True)

voo['Tomorrow'] = voo['Close'].shift(-1)
# voo

voo['Target'] = (voo['Tomorrow'] > voo['Close']).astype(int)
# voo

voo = voo.loc['2011-01-01':].copy()
# voo

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100)
train = voo.iloc[:-100]
test = voo.iloc[-100:]

predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])

from sklearn.metrics import precision_score
predictions = model.predict(test[predictors])
predictions = pd.Series(predictions, index = test.index)

score = precision_score(test['Target'], predictions)
print(score)

combined = pd.concat([test['Target'], predictions], axis = 1)
combined.plot()

# %%
'''Getting everything into a function'''

def data_vis(df, index = 'Close'):
    # Receives DataFrame from yfinance
    return df.plot.line(y = index, use_index = True)

def data_prep(df, st):
    # Receives DataFrame from yfinance

    #Adds a column with the closing price of next day
    df['Tomorrow'] = df['Close'].shift(-1) 

    # Compares closing price of day n and n+1
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)

    #Makes a deep copy to not change the values of original data
    df = df.loc[st:].copy() 

    return df

def model_train(df, n_estimators = 100, min_samples_split = 100, train_test_split = 100):
    # Receives DataFrame from data_prep
    model = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split)
    train = df.iloc[:-train_test_split]
    test = df.iloc[-train_test_split:]

    predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
    model.fit(train[predictors], train['Target'])

    return model, train, test

def model_test(model, test):
    from sklearn.metrics import precision_score
    predictions = model.predict(test[predictors])
    predictions = pd.Series(predictions, index = test.index)

    score = precision_score(test['Target'], predictions)

    combined = pd.concat([test['Target'], predictions], axis = 1)

    return score, combined.plot()

# %% Teseting the function above
nvda = yf.Ticker('NVDA')
nvda = nvda.history(period='max')

data_vis(nvda)
nvda = data_prep(nvda, st = '2000-01-01')

model, train, test = model_train(nvda)
score = model_test(model, test)
score