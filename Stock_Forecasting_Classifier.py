# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, accuracy_score

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


def target_prep_classifier(df: pd.DataFrame) -> pd.DataFrame:
    # The classifier target is to predict whether tomorrows closing price is higher than todays closing price
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    return df

def rolling_averages(df: pd.DataFrame, n_items: list = [2,5,50,200, 500]) -> tuple[pd.DataFrame, list]:
    # Adds new predictor columns based on rolling averages
    new_predictors = []

    for item in n_items:
        rolling_averages = df.rolling(item).mean()
        ratio_col = f'Close_Ratio_{item}'
        df[ratio_col] = df['Close'] / rolling_averages['Close']

        target_trend = f'Target_Trend_{item}'
        df[target_trend] = df.shift(1).rolling(item).sum()['Target']

        new_predictors += [ratio_col, target_trend]
    
    return df.dropna(), new_predictors

def train_test(df: pd.DataFrame,
    model_type: str,
    predictors: list,
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

# %%

# The classifier pipeline:

def classifier_training(X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    model_type: str
    ):
    
    if model_type == 'RF':
        model = RandomForestClassifier(n_estimators=100, min_samples_split=100)
    elif model_type == 'NN':
        model = MLPClassifier(hidden_layer_sizes=(100,))
    
    model.fit(X_train, y_train)

    return model

def classifier_eval(model, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame,
    graph: bool = True
    ):

    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, index = X_test.index, name = 'Predicted Target')
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    combined = pd.concat([y_test, y_pred], axis = 1)
    if graph:
        combined.plot()

    return y_pred, precision, accuracy

def master_classifier(ticker: str, model_type: str, graph: bool = True, crop: bool = True):
    # Loading data and making target
    ticker_df = load_data(ticker, graph = graph, crop=crop)
    ticker_df = target_prep_classifier(ticker_df)

    ticker_df, new_predictors = rolling_averages(ticker_df)
    predictors = ['Close', 'Volume', 'Open', 'High', 'Low'] + new_predictors
    
    # Train test split, data needs to be split chronologically else future data affects training of past data
    X_train, X_test, y_train, y_test = train_test(ticker_df, model_type=model_type, predictors=predictors)

    # Training
    model = classifier_training(X_train, y_train, model_type=model_type)

    # Scores
    y_pred, precision, accuracy = classifier_eval(model, X_test, y_test, graph=graph)

    print(f"Precision Score: {precision:.2f}")
    print(f"Accuracy Score: {accuracy:.2f}")

    return y_pred
# %%
