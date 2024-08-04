import numpy as np
from BrownianMotion import BrownianMotion
from numpy.polynomial import Polynomial

class LongstaffSchwartz():

    def __init__(self, S, K, r, sigma, M, n_days=250, trading_days=250, seed=6644) -> None:
        '''
        Parameters
        S              stock price 
        K              strike price
        r              risk-free interest rate
        sigma          volatility
        M              number of simulations or paths
        n_days         days / time periods until maturity
        trading_days   number of trading days in the year
        
        Returns
        American Option Price for these types of options
        '''

        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.M = M
        self.n_days = n_days
        self.trading_days = trading_days
        self.seed = seed
        self.prices = None

        BM = BrownianMotion(S, K, r, sigma, M, n_days)
        BM.simulate_stock_prices()
        self.paths = BM.prices.T
        self.time = np.linspace(0, n_days, n_days)
        self.dt = self.time[1] - self.time[0]

    def exercise_value(self, x):
        if self.K > self.S:
            return np.maximum(self.K - x, 0)
        else:
            return np.maximum(x - self.K, 0)
            
    def simulate_stock_prices(self, Nsteps=1):
        temp_results = []
        cashflow = self.exercise_value(self.paths[-1, :])

        # Iterating Backwards in Time
        for i in reversed(range(Nsteps, self.paths.shape[0] - 1)):
            df = np.exp(-self.r * self.dt)
            # Discounted cashflows to present value
            cashflow = cashflow * df
            # Get current stock price
            x = self.paths[i, :]
            # Exercise value
            exercise = self.exercise_value(x)
            itm = self.exercise_value(x) > 0
            # Estimate continuation at time t[i]
            fit_func = Polynomial.fit(x[itm], cashflow[itm], 3)
            # Estimate Continuation Value at time t[i]
            continuation = fit_func(x)
            # Exercise prefer index
            ex_index = itm & (exercise > continuation)
            # Early exercises cashflows
            cashflow[ex_index] = exercise[ex_index]
            
            temp_results.append(
                (x, cashflow, fit_func, exercise, continuation, ex_index))
       
        return np.round(np.average(cashflow * np.exp(-self.r * self.dt)), 4)