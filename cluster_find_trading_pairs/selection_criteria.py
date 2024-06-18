import pandas as pd
from statsmodels.api import add_constant, OLS
from statsmodels.tsa.stattools import coint, adfuller
import numpy as np
from hurst import compute_Hc
import pandas as pd

class PairsSelection:
    def __init__(self, leg1_name: str, leg1: pd.DataFrame, leg2_name: str, leg2: pd.DataFrame):
        self.leg1_name = leg1_name
        self.leg1 = leg1
        self.leg2_name = leg2_name
        self.leg2 = leg2

        self.coint_threshold = 0.02
        self.adf_threshold = 0.02
        self.min_half_life = 5
        self.max_half_life = 90
        self.hurst_threshold = 0.5
        self.min_zero_crossings = 18

    def coint_test(self):
        coint_t, coint_pvalue, _ = coint(self.leg1, self.leg2)
        self.coint_t = coint_t
        self.coint_pvalue = coint_pvalue

    def compute_spread(self):
        S1 = self.leg1
        S1 = add_constant(S1)
        S2 = self.leg2
        linear_regression = OLS(S2, S1).fit()
        self.hedge_ratio = linear_regression.params[self.leg1_name]
        spread = self.leg2 - self.hedge_ratio * self.leg1
        self.spread = spread

    def ADF_test(self):
        adf = adfuller(self.spread, maxlag=1)
        self.adf_pvalue = adf[1]

    def compute_hurst_exponent(self):
        """
        Calculates the Hurst Exponent of a given time series.
            - The sublinear function of time (as if the series is stationary) can be approximated by: tau^(2H), where
              tau is the time separating two measurements and H is the Hurst Exponent.
            - Hypothesis Test:

                - Hurst Exponent < 0.5: Time series is stationary.
                - Hurst Exponent = 0.5: Time series is a geometric random walk.
                - Hurst Exponent > 0.5: Time series is trending.
        """
        # lags = range(2, 200)
        # tau = [np.sqrt(np.std(np.subtract(self.spread[lag:], self.spread[:-lag]))) for lag in lags]
        # poly = np.polyfit(np.log(lags), np.log(tau), 1)
        #
        # hurst_exponent = poly[0] * 2.0
        # self.hurst_exponent = hurst_exponent

        h, c, data = compute_Hc(self.spread)
        self.hurst_exponent = h

    def compute_half_life(self):
        """
        Calculates the half-life of a given time series.
            - Measures how quickly a time series reverts to its mean.
            - Good predictor of the profitability or Sharpe ratio of a mean-reverting strategy.
        """
        time_series_lag = np.roll(self.spread, 1)
        time_series_lag[0] = 0
        ret = self.spread - time_series_lag
        ret.iloc[0] = 0
        time_series_lag2 = add_constant(time_series_lag)
        model = OLS(ret[1:], time_series_lag2[1:])
        res = model.fit()

        half_life = -np.log(2) / res.params.iloc[1]

        self.half_life = half_life

    def compute_zero_crossings(self):
        """
        Calculates the number of times a given time series crosses zero.
        """
        x = self.spread - self.spread.mean()
        zero_cross = 0
        # zero_cross = sum(1 for i, _ in enumerate(x) if (i + 1 < len(x)) if ((x.iloc[i] * x.iloc[i + 1] < 0) or (x.iloc[i] == 0)))
        for i, _ in enumerate(x):
            if i + 1 < len(x) and x.iloc[i] * x.iloc[i + 1] < 0:
                zero_cross += 1
            elif x.iloc[i] == 0:
                zero_cross += 1

        self.zero_cross = zero_cross


    def is_eligible(self):
        self.coint_test()
        if self.coint_pvalue > self.coint_threshold:
            return False

        self.compute_spread()
        self.ADF_test()
        if self.adf_pvalue > self.adf_threshold:
            return False

        self.compute_hurst_exponent()
        if self.hurst_exponent >= self.hurst_threshold:
            return False

        self.compute_half_life()
        if self.half_life > self.max_half_life or self.half_life < self.min_half_life:
            return False

        self.compute_zero_crossings()
        if self.zero_cross < self.min_zero_crossings:
            return False

        return True
