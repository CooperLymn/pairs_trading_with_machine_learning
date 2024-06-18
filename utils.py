from statsmodels.api import add_constant, OLS

def compute_spread(data, stock1, stock2):
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

def compute_sharpe_ratio(portfolio):
    pass
