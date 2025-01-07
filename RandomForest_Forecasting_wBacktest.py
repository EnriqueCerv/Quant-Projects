# %%
import numpy as np
import pandas as pd
import scipy as scp
import matplotlib.pyplot as plt
import yfinance as yf
import sklearn

from sklearn.ensemble import RandomForestClassifier
#%%
'''A simple model that uses data from yfinane to see if closing price of day n+1 is higher than closing price of day n'''

def data_vis(df, index = 'Close'):
    # Receives DataFrame from yfinance, plots with time
    return df.plot.line(y = index, use_index = True)

def data_prep(df, st = '2000-01-01'):
    # Receives DataFrame from yfinance, returns DataFrame with added column comparing closing prices in days n and n+1

    #Adds a column with the closing price of next day
    df['Tomorrow'] = df['Close'].shift(-1) 

    # Compares closing price of day n and n+1
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)

    #Makes a deep copy to not change the values of original data
    df = df.loc[st:].copy() 

    return df

def model_train(df, n_estimators = 100, min_samples_split = 100, train_test_split = 100):
    # Receives DataFrame from data_prep, returns the trained model and predictors, 
    # and the data used for training and to be used for testing

    # Model is RandomForest
    model = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split)

    # Split training and testing data
    train = df.iloc[:-train_test_split]
    test = df.iloc[-train_test_split:]

    # Training the model
    predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
    model.fit(train[predictors], train['Target'])

    return model, predictors, train, test

def model_test(model, predictors, test):
    # Receives model, predictors and testing DataFrame, returns the score of testing set with graph of Target and Predictions

    # Predicts the target on the testing set
    predictions = model.predict(test[predictors])
    predictions = pd.Series(predictions, index = test.index)

    # Caluclates score on predicted data points
    from sklearn.metrics import precision_score
    score = precision_score(test['Target'], predictions)
    
    # Creates DF of predictions and targets on testing data
    combined = pd.concat([test['Target'], predictions], axis = 1)

    return score, combined, combined.plot()

# %% Teseting the functiosn above
nvda = yf.Ticker('NVDA')
nvda = nvda.history(period='max')

data_vis(nvda)
nvda = data_prep(nvda, st = '2000-01-01')

model, predictors, train, test = model_train(nvda)
score = model_test(model, predictors, test)
score
# %% Onto some backtesting
'''Adds backtesting on the model'''

def predict(model, predictors, train, test):
    # Receives the model (RandomForest), the set of predictors, train and test data
    # Trains model on train data, predicts test data
    # Returns DataFrame of predictins and targets on teseting data

    model.fit(train[predictors], train['Target'])
    predictions = model.predict(test[predictors])
    predictions = pd.Series(predictions, index = test.index, name = 'Predictions')

    combined = pd.concat([test['Target'], predictions], axis = 1)
    return combined

def backtest(df, model, predictors, start=2500, step=250):
    # What happends to datapoints before the 2500th day?
    all_predictions = []

    for i in range(start, df.shape[0], step):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:i+step].copy()

        combined  = predict(model, predictors, train, test)
        all_predictions.append(combined)
        
    return pd.concat(all_predictions)
# %%
'''Testing the backtest function above'''

nvda = yf.Ticker('NVDA')
nvda = nvda.history(period='max')

data_vis(nvda)
nvda = data_prep(nvda, '2000-01-01')

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)
predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
# model, predictors, train, test = model_train(nvda)

predictions = backtest(nvda, model, predictors)

from sklearn.metrics import precision_score

ratio = predictions['Predictions'].value_counts() / predictions.shape[0]

print(ratio, precision_score(predictions['Target'], predictions['Predictions']))


# %%
''''Adds more predictors to the back test'''

def more_predictors(df, horizons = [2,5,100,250,1000]):
    # Receives DataFrame and horizons list, returns DataFrame with two new predictors:
    # ratio_col = close / average of preceeding horizon closes, for horizon in horizons
    # trend_col = sum of trends in preceeding horizon closes, for horizon in horizons

    new_predictors = []

    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()

        ratio_col = f'Close_Ratio_{horizon}'
        df[ratio_col] = df['Close'] / rolling_averages['Close']

        trend_col = f'Trend_Col_{horizon}'
        df[trend_col] = df.shift(1).rolling(horizon).sum()['Target']

        new_predictors += [ratio_col, trend_col]
    
    return df.dropna()

'''To test the above'''

nvda = yf.Ticker('NVDA')
nvda = nvda.history(period='max')
nvda = data_prep(nvda)
nvda_new = more_predictors(nvda)
nvda
nvda_new


# %%

'''Training a more stringent model'''

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)
horizons = [2,5,100,250,1000]
predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
for horizon in horizons:
    predictors += [f'Close_Ratio_{horizon}', f'Trend_Col_{horizon}']

def predict(model, predictors, train, test, threshold = 0.6):
    # Receives model, predictors, train, test, data. Returns DF of predicted and target points on testing data

    model.fit(train[predictors], train['Target'])

    # The predictions are now returned as probability in the range 0,1
    predictions = model.predict_proba(test[predictors])[:,1]

    # Predictions are 1 (increase from day n to n+1) iff probability is larger than 0.6
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 1

    # Join predictions and targets
    predictions = pd.Series(predictions, index = test.index, name = 'Predictions')
    combined = pd.concat ([test['Target'], predictions], axis = 1)

    return combined
# %%

'''Testing the new predict function'''

predictions = backtest(nvda_new, model, predictors)

ratio = predictions['Predictions'].value_counts() / predictions.shape[0]

print(ratio, precision_score(predictions['Target'], predictions['Predictions']))