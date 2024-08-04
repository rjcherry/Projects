import numpy as np
from scipy.stats import norm

class BrownianMotion:
    
    def __init__(self, S, K, r, sigma, M, n_days=250, trading_days=250, seed=6644):
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
        Black Scholes Option Prices for these types of options
        - European Call
        - European Put
        '''
        
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.M = M
        self.n_days = n_days
        self.trading_days = trading_days
        self.seed = seed
        self.prices = []
        self.type = None
        
        self.T = self.n_days / self.trading_days
        self.dt = (self.n_days / self.trading_days) / self.trading_days
        
    def simulate_stock_prices(self):
        '''
        Return Geometric Brownian Motion Prices
        '''
        
        drift_coef = (self.r - 0.5 * self.sigma**2) * self.dt
        right_hand_side = self.sigma * np.sqrt(self.dt)

        prices = np.zeros((self.M, self.n_days))
        prices[:,0] += self.S
        
        Z = np.random.RandomState(self.seed).normal(0, 1, (self.M, self.n_days))
        
        for i in range(1, self.n_days):
            prices[:,i] = prices[:,i-1] * np.exp(drift_coef + right_hand_side * Z[:,i-1])
                        
        self.prices = prices
        
        return self
    
    def get_option_price(self, type=None):
        
        self.type = type
        
        if len(self.prices)==0:
            self.simulate_stock_prices()
        
        time_value = np.exp(-self.r*self.T)
        
        if self.type == "call":
            return time_value * np.maximum(self.prices[:,-1] - self.K, 0).mean()
        elif self.type == "put":
            return time_value * np.maximum(self.K - self.prices[:,-1], 0).mean()
        else:   
            return None