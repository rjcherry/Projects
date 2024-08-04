import numpy as np
from scipy.stats import norm

class Greeks:
    def __init__(self, S, K, r, sigma, n_days, trading_days=250, type=None):
        '''
        Parameters
        S: stock price 
        K: strike price
        r: risk-free interest rate
        sigma: volatility
        T: time to maturity (assume 250 trading days)
        
        Returns
        Greek Metrics
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
    
    def gamma(self):
        
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1) * 0.01

    def delta(self, type=None):
        
        if type is not None:
            self.type = type
        
        if self.type == 'call':
            return norm.cdf(self.d1)
        
        elif self.type == "put":
            return norm.cdf(self.d1) - 1
        
        else:
            return None 
        
    def rho(self, type=None):
        
        if type is not None:
            self.type = type
        
        if self.type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2, 0, 1) * 0.01
        
        elif self.type == "put":
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2, 0, 1) * 0.01
        
        else:
            return None  

    def theta(self, type=None):
        
        if type is not None:
            self.type = type
        
        if self.type == 'call':
            left_side = -self.r * self.S * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            right_side = (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        
            return (left_side - right_side)/self.trading_days
        
        elif self.type == "put":
            left_side =  self.r * self.S * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            right_side = (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) 
            
            return (left_side - right_side)/self.trading_days
        
        else:
            return None  