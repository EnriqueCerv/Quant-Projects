# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from allocators.hrp import hrp_portfolio
from allocators.mean_variance import mean_variance_portfolio
from allocators.min_variance import min_variance_portfolio
# %%

portfolio_dict = {
    'hrp': hrp_portfolio,
    'mean_variance': mean_variance_portfolio,
    'min_variance': min_variance_portfolio
}


def backtest_cum_returns(
        returns: pd.DataFrame,
        lookback: int = 126,
        rebalance_freq: int = 21,
        cost_bps: float = 5,
        portfolio: str = 'hrp',
        **kwargs
):
    """
    Input: Returns, lookback window size, rebalance freqeuency, portfolio type
    Output: DataFrame of cumulative returns over entire time series
    """
    
    # Get rolling weights for each time windo
    rolling_weights = compute_rolling_weights(returns, lookback, rebalance_freq, portfolio, **kwargs)
    
    # Get optimized portfolio returns
    optimized_returns, avg_turnover = backtest_portfolio(returns, rolling_weights, rebalance_freq=rebalance_freq, cost_bps=cost_bps)

    return optimized_returns, avg_turnover


# %%

# # # # # # # # # # # # #
# Rolling portfolio optimizer
# # # # # # # # # # # # #
def compute_rolling_weights(
        returns: pd.DataFrame,
        lookback: int,
        rebalance_freq: int,
        portfolio: str,
        **kwargs
    ):
    """
    Input: Returns, lookback window size, rebalance freqeuency, portfolio type
    Output: DataFrame of optimal portfolio weights per time window
    """

    portfolio_function = portfolio_dict[portfolio]
    weights = []
    dates = []

    for t in range(lookback, len(returns), rebalance_freq):
        window = returns.iloc[t - lookback : t].dropna()

        if len(window) < lookback:
            continue

        w = portfolio_function(window, **kwargs)

        weights.append(w)
        dates.append(returns.index[t])

    return pd.DataFrame(weights, index = dates)


# # # # # # # # # # # # #
# Backtest
# # # # # # # # # # # # #
# def backtest_portfolio(
#         returns: pd.DataFrame,
#         weights: pd.DataFrame,
#         rebalance_freq: int = 21
#     ):
#     """
#     Input: Returns, weights of stock and rebalancing frequency
#     Output: DataFrame of optimized portfolio returns as given by optimized weights in each time period
#     """
#     portfolio_returns = []
#     for i in range(len(weights) - 1):
#         start = returns.index.get_loc(weights.index[i])
#         end = start + rebalance_freq

#         window_returns = returns.iloc[start:end]
#         w = weights.iloc[i].values

#         portfolio_returns.append(window_returns @ w)
    
#     return pd.concat(portfolio_returns)

# # # # # # # # # # # # #
# Backtest with turnover
# # # # # # # # # # # # #
def backtest_portfolio(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        rebalance_freq: int,
        cost_bps: float
    ):
    """
    Input: Returns, weights of stock and rebalancing frequency
    Output: DataFrame of optimized portfolio returns as given by optimized weights in each time period
            minus the turnover cost between rebalances
    """

    portfolio_returns = []
    turnovers = []
    old_weights = np.zeros(weights.shape[1])

    for i in range(len(weights) - 1):
        start = returns.index.get_loc(weights.index[i])
        end = start + rebalance_freq

        window_returns = returns.iloc[start:end].copy()
        new_weights = weights.iloc[i].values

        # Calculate rebalance cost
        turnover = np.sum(np.abs(new_weights - old_weights))
        cost = turnover * cost_bps / 10000
        window_returns.iloc[0] = window_returns.iloc[0] - cost # only deduct from first day 

        portfolio_returns.append(window_returns @ new_weights)
        turnovers.append(turnover)
        old_weights = new_weights
    
    return pd.concat(portfolio_returns), np.mean(turnovers)

# # # # # # # # # # # # #
# Eval backtest
# # # # # # # # # # # # #
def performance_stats(r, avg_turnover = None):
    return pd.Series({
        "Annual Return": (1 + r).prod() ** (252 / len(r)) - 1,
        "Annual Vol": r.std() * np.sqrt(252),
        "Sharpe": (r.mean() / r.std()) * np.sqrt(252),
        "Max Drawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(),
        "Avg Turnover": avg_turnover * 12 if avg_turnover else 0
    })
# %%

if __name__ == '__main__':
    from data import returns

    # Benchmarks 
    equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
    equal_portfolio_returns = returns @ equal_weights
    equal_stats = performance_stats(equal_portfolio_returns)
    print('Equal Portfolio stats')
    print(equal_stats, '\n')
    equal_portfolio_cum_returns = (1 + equal_portfolio_returns).cumprod()

    voo_returns = returns['VOO']
    voo_cum_returns = (1 + voo_returns).cumprod()
    voo_stats = performance_stats(equal_portfolio_returns)
    print('VOO stats')
    print(voo_stats, '\n')

    # Get returns and turnover
    cost_bps = 10
    
    # HRP
    hrp_returns, hrp_avg_turnover = backtest_cum_returns(
        returns,
        lookback=126, 
        rebalance_freq=21, 
        cost_bps = cost_bps,
        portfolio='hrp',
        lw = True, 
        plot = False
        )
    hrp_stats = performance_stats(hrp_returns, hrp_avg_turnover)
    print('HRP stats')
    print(hrp_stats, '\n')
    hrp_cum_returns = (1 + hrp_returns).cumprod()

    # Mean-variance
    mean_variance_returns, mean_variance_avg_turnover = backtest_cum_returns(
        returns, 
        lookback=126, 
        rebalance_freq=21, 
        cost_bps = cost_bps,
        portfolio='mean_variance',
        lambda_risk = 0.5, 
        lw = True, 
        shrink = 0.3, 
        plot = False
        )
    mean_variance_stats = performance_stats(mean_variance_returns, mean_variance_avg_turnover)
    print('Mean-variance stats')
    print(mean_variance_stats, '\n')
    mean_variance_cum_returns = (1 + mean_variance_returns).cumprod()
    
    # Min-variance
    min_variance_returns, min_variance_avg_turnover = backtest_cum_returns(
        returns, 
        lookback=126, 
        rebalance_freq=21, 
        cost_bps = cost_bps,
        portfolio='min_variance',
        lw = True,
        plot = False
    )
    min_variance_stats = performance_stats(min_variance_returns, min_variance_avg_turnover)
    print('Min-variance stats')
    print(min_variance_stats, '\n')
    min_variance_cum_returns = (1 + min_variance_returns).cumprod()

    # Plot comparison
    plt.figure(figsize=(10,6))
    hrp_cum_returns.plot(label='HRP portfolio')
    mean_variance_cum_returns.plot(label='Mean-variance portfolio')
    min_variance_cum_returns.plot(label='Min-variance portfolio')
    equal_portfolio_cum_returns.plot(label='Equal weight portfolio')
    voo_cum_returns.plot(label='VOO')
    plt.legend()
    plt.title('Out-of-sample backtest')
    plt.show()
