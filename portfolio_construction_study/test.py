# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

# %%
def load_stock(ticker: str, period: str = 'max', plot: bool = True) -> pd.DataFrame:
    """
    Returns data frame of given ticker, also a plot of its close and returns if plot == True
    """
    df = yf.Ticker(ticker)
    df = df.history(period=period)
    df['Return'] = df['Close'].pct_change()
        
    return df

# Enrique global tickers, need to separate by exchange
tickers = ['VOO', 'AAPL', 'SMH', 'TSM', 'NVDA', 'MSFT', 'AMD', 'BOTZ', 'NLR']
defense_tickers = ['LMT', 'RTX', 'GD', 'NOC']
tickers += defense_tickers

big_companies = ['AMZN', 'GOOG', 'TSLA', 'JPM', 'META']
tickers += big_companies

# %%
data = {ticker: load_stock(ticker, period='5y', plot=True) for ticker in tickers}
returns = pd.DataFrame({ticker: data[ticker]['Return'] for ticker in data.keys()}).dropna(how='any')
cov, corr = returns.cov(), returns.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Return correlation')
plt.show()
# %%
dist_matrix = np.sqrt((1 - corr) / 2)

plt.figure(figsize=(10,8))
sns.heatmap(dist_matrix, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Corr dist')
plt.show()
# %%
from sklearn.metrics.pairwise import pairwise_distances

euc_dist = pairwise_distances(dist_matrix)
euc_dist = pd.DataFrame(euc_dist, index=returns.columns, columns=returns.columns)


plt.figure(figsize=(10,8))
sns.heatmap(euc_dist, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Euc dist')
plt.show()
# %%
def cluster(eucledian_distances):
    n = len(eucledian_distances)
    linkage = []
    for _ in range(n - 1):
        mask = np.eye(len(eucledian_distances), dtype=bool)
        masked = eucledian_distances.mask(mask)

        i, j = masked.stack().idxmin()
        new_label = i + '|' + j
        min_dist = masked.min().min()

        new_dist = eucledian_distances[[i,j]].min(axis=1)
        new_dist.name = new_label

        eucledian_distances = eucledian_distances.drop(index = [i,j], columns = [i,j])
        eucledian_distances[new_label] = new_dist.drop([i,j])
        eucledian_distances.loc[new_label] = new_dist.drop([i,j])
        eucledian_distances.loc[new_label, new_label] = 0

        linkage.append((i, j, min_dist))
    
    return eucledian_distances, linkage


clust, linkage = cluster(eucledian_distances=euc_dist)
linkage
# %%
def quasi_diagonlize(label, merge_info):
    if label not in merge_info:
        return [label]
    
    i, j, dist = merge_info[label]
    return quasi_diagonlize(i, merge_info) + quasi_diagonlize(j, merge_info)

root = f'{linkage[-1][0]}|{linkage[-1][1]}'
merge_info = {f'{i}|{j}': (i, j, dist) for i, j, dist in linkage}
quasi_sort = quasi_diagonlize(root, merge_info)


sorted_corr = corr.loc[quasi_sort, quasi_sort]
sorted_cov = cov.loc[quasi_sort, quasi_sort]
plt.figure(figsize=(10,8))
sns.heatmap(sorted_corr, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Return correlation')
plt.show()