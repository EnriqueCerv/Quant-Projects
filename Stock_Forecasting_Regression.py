# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# %%

def load_data(ticker: str, graph: bool = True, crop: bool = True) -> pd.DataFrame:
    df = yf.Ticker(ticker)
    df = df.history(period='max')
    
    # The last 100days crashed everything due to Trump admin, so im testing models withouth the crash
    if crop:
        df = df.iloc[:-100]

    if graph:
        df.plot(y='Close', use_index=True)
    
    return df


def target_prep_regression(df: pd.DataFrame) -> pd.DataFrame:
    # The regression target is to predict tomorrows closing price
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    return df

def train_test(df: pd.DataFrame,
    model_type: str,
    predictors: list = ['Close', 'Volume', 'Open', 'High', 'Low'],
    split: float = 0.8):

    # Data cannot be randomly split since it is time ordered, hence train = all but last 100 days, test = last 100 days
    # train = df[predictors].iloc[:-100]
    # train_target = df['Target'].iloc[:-100]
    # test = df[predictors].iloc[-100:]
    # test_target = df['Target'].iloc[-100:]
    X_train, X_test, y_train, y_test = train_test_split(df[predictors], df['Target'], shuffle=False, test_size=1-split)


    # Need to scale data for neural networks, no need for RF
    if model_type == 'NN':
        scaler = StandardScaler()
        # The train data gets scaled according to its mean and std
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=predictors)
        # The test data gets scaled according to the mean and std of the TRAINING data
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=predictors)

    return X_train, X_test, y_train, y_test

#%%

# The regression pipeline:

def regression_training(X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    model_type: str
    ):
    
    if model_type == 'RF':
        model = RandomForestRegressor(n_estimators=100, min_samples_split=100, max_depth=None)
    elif model_type == 'NN':
        model = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=500)
    
    model.fit(X_train, y_train)

    return model
    
def regression_eval(model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    graph: bool = True
    ):

    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, index = X_test.index, name = 'Predicted Target')
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    combined = pd.concat([y_test, y_pred], axis = 1)
    if graph:
        combined.plot(use_index=True)
    
    return y_pred, mae, mse, r2

def master_regressor(ticker: str, model_type: str, graph: bool = True, crop: bool = False):
    # Loading data and making target
    ticker_df = load_data(ticker, graph = graph, crop = crop)
    ticker_df = target_prep_regression(ticker_df)

    # Train test split, data needs to be split chronologically else future data affects training of past data
    X_train, X_test, y_train, y_test = train_test(ticker_df, model_type=model_type, predictors=['Volume', 'Open', 'High', 'Low'], split = 0.8)

    # Training
    model = regression_training(X_train, y_train, model_type=model_type)

    # Scores
    y_pred, mae, mse, r2 = regression_eval(model, X_test, y_test, graph=graph)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    return y_pred
# %%
