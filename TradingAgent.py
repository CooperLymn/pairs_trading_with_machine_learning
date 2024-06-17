from matplotlib import pyplot as plt
import pandas as pd
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

class PairsTradingAgent:
    def __init__(self, stock1, stock2, spread, entry=1.0, exit=0.5):
        # stock1, stock2, and spread are all Pandas DataFrames
        self.stock1 = stock1
        self.stock2 = stock2
        self.spread = spread
        self.ratio = stock1 / stock2

        self.upper_entry = entry
        self.upper_exit = exit
        self.lower_entry = -entry
        self.lower_exit = -exit

        self.n_share_stock1 = pd.DataFrame(index=self.spread.index, columns=["n_share_stock1"], dtype=float)
        self.n_share_stock1["n_share_stock1"] = 0.0
        self.n_share_stock2 = pd.DataFrame(index=self.spread.index, columns=["n_share_stock2"], dtype=float)
        self.n_share_stock2["n_share_stock2"] = 0.0
        self.cash = pd.DataFrame(index=self.spread.index, columns=["cash"], dtype=float)
        self.cash["cash"] = 0.0
        self.portfolio_value = pd.DataFrame(index=self.spread.index, columns=["portfolio_value"], dtype=float)
        self.portfolio_value["portfolio_value"] = 0.0

        self.trading_signals = pd.DataFrame(index=self.spread.index)

    def initialize(self):
        self.n_share_stock1["n_share_stock1"] = 0.0
        self.n_share_stock2["n_share_stock2"] = 0.0
        self.cash["cash"] = 0.0
        self.portfolio_value["portfolio_value"] = 0.0

    def plot_threshold(self):
        plt.figure(figsize=(13, 6))
        plt.plot()
        plt.plot(self.spread, label='spread')
        plt.axhline(self.spread["Spread"].mean(), color='black')
        plt.axhline(self.upper_entry, color='red')
        plt.axhline(self.lower_entry, color='red')
        plt.axhline(self.upper_exit, color='green')
        plt.axhline(self.lower_exit, color='green')
        plt.xlabel("Date")
        plt.ylabel("Standardized Spread")
        plt.legend()
        plt.show()

    def calculate_portfolio_value(self, time):
        value = 0
        value += self.cash.loc[time, "cash"]
        value += self.n_share_stock1.loc[time, "n_share_stock1"] * self.stock1.loc[time]
        value += self.n_share_stock2.loc[time, "n_share_stock2"] * self.stock2.loc[time]
        self.portfolio_value.loc[time, "portfolio_value"] = value

    def print_portfolio(self, time):
        print(f"Portfolio on {time}:")
        print(f"Number of stock1: {self.n_share_stock1.loc[time, 'n_share_stock1']} shares")
        print(f"Stock1 price: {self.stock1.loc[time]:.4f}")
        print(f"Number of stock2: {self.n_share_stock2.loc[time, 'n_share_stock2']} shares")
        print(f"Stock2 price: {self.stock2.loc[time]:.4f}")
        print(f"Amount of cash: {self.cash.loc[time, 'cash']}")
        print(f"Total value of asset: {self.portfolio_value.loc[time, 'portfolio_value']}")

    def long_position(self, time, amount=1.0, n_share=10.0):
        self.n_share_stock1.loc[time:, "n_share_stock1"] += amount / self.stock1.loc[time]
        self.n_share_stock2.loc[time:, "n_share_stock2"] -= amount / self.stock2.loc[time]
        # self.n_share_stock1.loc[time:, "n_share_stock1"] += n_share
        # self.n_share_stock2.loc[time:, "n_share_stock2"] -= n_share * self.ratio.loc[time]
        # self.cash.loc[time:, "cash"] -= n_share * self.stock1.loc[time]
        # self.cash.loc[time:, "cash"] += n_share * self.ratio.loc[time] * self.stock2.loc[time]

    def short_position(self, time, amount=1.0, n_share=10.0):
        self.n_share_stock1.loc[time:, "n_share_stock1"] -= amount / self.stock1.loc[time]
        self.n_share_stock2.loc[time:, "n_share_stock2"] += amount / self.stock2.loc[time]
        # self.n_share_stock1.loc[time:, "n_share_stock1"] -= n_share
        # self.n_share_stock2.loc[time:, "n_share_stock2"] += n_share * self.ratio.loc[time]
        # self.cash.loc[time:, "cash"] += n_share * self.stock1.loc[time]
        # self.cash.loc[time:, "cash"] -= n_share * self.ratio.loc[time] * self.stock2.loc[time]

    def clear_position(self, time):
        self.cash.loc[time:, "cash"] += self.n_share_stock1.loc[time, "n_share_stock1"] * self.stock1.loc[time]
        self.cash.loc[time:, "cash"] += self.n_share_stock2.loc[time, "n_share_stock2"] * self.stock2.loc[time]
        self.n_share_stock1.loc[time:, "n_share_stock1"] = 0
        self.n_share_stock2.loc[time:, "n_share_stock2"] = 0

    def generate_trading_signals(self):
        self.trading_signals['Long Entry'] = self.spread > self.upper_entry
        self.trading_signals['Short Entry'] = self.spread < self.lower_entry
        self.trading_signals['Long Exit'] = self.spread < self.upper_exit
        self.trading_signals['Short Exit'] = self.spread > self.lower_exit
        self.trading_signals['Action'] = 'None'

        current_potition = 'None'
        for ind in self.trading_signals.index:
            if current_potition != 'Long' and self.trading_signals.loc[ind, 'Long Entry']:
                self.trading_signals.loc[ind, "Action"] = 'Long Entry'
                current_potition = 'Long'
            if current_potition == 'Long' and self.trading_signals.loc[ind, 'Long Exit']:
                self.trading_signals.loc[ind, "Action"] = 'Long Exit'
                current_potition = 'None'
            if current_potition != 'Short' and self.trading_signals.loc[ind, 'Short Entry']:
                self.trading_signals.loc[ind, "Action"] = 'Short Entry'
                current_potition = 'Short'
            if current_potition == 'Short' and self.trading_signals.loc[ind, 'Short Exit']:
                self.trading_signals.loc[ind, "Action"] = 'Short Exit'
                current_potition = 'None'

    def print_trading_signals(self):
        for ind in self.trading_signals.index:
            if self.trading_signals.loc[ind, 'Action'] != 'None':
                print(f"{self.trading_signals.loc[ind, 'Action']} at {ind}")

    def backtesting(self):
        for ind in self.trading_signals.index:
            action = self.trading_signals.loc[ind, 'Action']
            if action == 'Long Entry':
                self.long_position(ind)
            elif action == 'Short Entry':
                self.short_position(ind)
            elif action == 'Long Exit' or action == 'Short Exit':
                self.clear_position(ind)

            self.calculate_portfolio_value(ind)

        return self.portfolio_value

    def set_threshold(self, entry, exit):
        self.upper_entry = entry
        self.upper_exit = exit
        self.lower_entry = -entry
        self.lower_exit = -exit

    def plot_portfolio_value(self):
        plt.figure(figsize=(13, 6))
        plt.plot()
        plt.plot(self.portfolio_value, label='portfolio_value')
        plt.legend

    def trading_simulation(self, entry, exit):
        self.initialize()
        self.set_threshold(entry, exit)
        self.generate_trading_signals()
        portfolio_value = self.backtesting()
        return portfolio_value