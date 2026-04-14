# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
# %%
# # # # # # # # # # # # #
# LOAD PORTFOLIO
# # # # # # # # # # # # #
def load_stock(ticker: str, period: str, plot: bool) -> pd.DataFrame:
    """
    Returns data frame of given ticker, also a plot of its close and returns if plot == True
    """
    df = yf.Ticker(ticker)
    df = df.history(period=period)
    df['Return'] = df['Close'].pct_change()
    
    if plot:
        ax = df.plot(y='Close', use_index=True, title=ticker, figsize = (10,5))
        df.plot(y='Return', ax= ax, secondary_y= True, alpha = 0.3, linewidth = 1, style = '--')
        ax.set_ylabel('Price')
        ax.right_ax.set_ylabel('Return')
    
    return df


tickers = ['VOO', 'AAPL', 'SMH', 'TSM', 'NVDA', 'MSFT', 'AMD', 'BOTZ', 'NLR']
tickers += ['LMT', 'RTX', 'GD', 'NOC']
tickers += ['AMZN', 'GOOG', 'TSLA', 'JPM', 'META']

data = {ticker: load_stock(ticker, period='5y', plot=False) for ticker in tickers}
returns = pd.DataFrame({ticker: data[ticker]['Return'] for ticker in tickers}).dropna(how='any')

# %%
if __name__ == '__main__':
    for ticker in tickers:
        load_stock(ticker, period='5y', plot=True)