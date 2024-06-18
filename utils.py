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
    daily_return = porfolio.pct_change().dropna()
    mean_daily_return = daily_return.mean()
    std_daily_return = daily_return.std()

    # Annualize the mean daily return and standard deviation
    trading_days_per_year = 252  # Typical number of trading days in a year
    annualized_return = (1 + mean_daily_return) ** trading_days_per_year - 1
    annualized_std = std_daily_return * np.sqrt(trading_days_per_year)

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std

    return sharpe_ratio

def compute_volatility(portfolio: pd.DataFrame):
    daily_return = portfolio.pct_change().dropna()
    std_daily_return = daily_return.std()
    trading_days_per_year = 252
    annualized_volatility = std_daily_return * np.sqrt(trading_days_per_year)

    return annualized_volatility