from statsmodels.api import add_constant, OLS
import pandas as pd
import numpy as np

def compute_spread(data, stock1: str, stock2: str):
    # lp = np.log(data)
    # S1 = lp[stock1]
    S1 = data[stock1]
    S1 = add_constant(S1)
    # S2 = lp[stock2]
    S2 = data[stock2]
    linear_regression = OLS(S2, S1).fit()
    S1 = S1[stock1]
    beta = linear_regression.params[stock1]
    spread = S2 - beta * S1
    return spread

def zscore_normalization(data):
    return (data - data.mean()) / data.std()

def ADF_test(spread):
    adf = adfuller(spread, maxlag=1)
    adf_pvalue = adf[1]
    return adf_pvalue

def compute_sharpe_ratio(porfolio: pd.DataFrame, risk_free_rate: float):
    # daily_return = np.log(porfolio / porfolio.shift(1))
    daily_return = porfolio.pct_change().dropna()
    trading_days_per_year = 252

    volatility = daily_return.std() * np.sqrt(trading_days_per_year)

    sharpe_ratio = (daily_return.mean() * trading_days_per_year - risk_free_rate) / volatility

    volatility = volatility.values[0]
    sharpe_ratio = sharpe_ratio.values[0]

    return volatility, sharpe_ratio

def compute_volatility(portfolio: pd.DataFrame):
    daily_return = portfolio.pct_change().dropna()
    trading_days_per_year = 252
    annualized_volatility = daily_return.std() * np.sqrt(trading_days_per_year)

    return annualized_volatility

def compute_maximal_drawdown(portfolio: pd.DataFrame):
    daily_return = portfolio.pct_change().dropna()
    cumulative_returns = (1 + daily_return).cumprod()
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cumulative_max) / cumulative_max
    maximal_drawdown = min(drawdown.iloc[1:].values)
    return maximal_drawdown[0]
