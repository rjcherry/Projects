import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, r, sigma, n_days, trading_days=250, type=None):
        '''
        Parameters
        S: stock price 
        K: strike price
        r: risk-free interest rate
        sigma: volatility
        T: time to maturity (assume 250 trading days)
        
        Returns
        Black Scholes Option Prices for these types of options
        - European Call
        - European Put
        '''
        
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.n_days = n_days
        self.trading_days=trading_days
        self.type=type
        
        self.T = self.n_days/self.trading_days
        
        self.d1 = (np.log(self.S/self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        
    def european_option(self, type=None):
        
        if type is not None:
            self.type = type
            
        if self.type=="call":
            return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        
        elif self.type=="put":
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)
        
        else:
            return None 