import numpy as np
from scipy.stats import norm

class Heston:
    
    def __init__(self, S, K, r, sigma, M, theta, rho, kappa=1.0, n_days=250, trading_days=250, seed=6644, type=None):
        
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.M = M
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.n_days = n_days
        self.trading_days=trading_days
        self.seed = seed
        self.prices=[]
        self.volatilty_paths=[]
        self.type=None
        
        self.T = self.n_days/self.trading_days
        self.dt = (self.n_days/self.trading_days)/self.trading_days
        
        self.V0 = sigma**2
        
    def simulate_stock_prices(self):
        
        # Initialize arrays
        
        prices = np.zeros((self.M, self.n_days + 1))
        prices[:,0] += self.S

        variances = np.zeros((self.M, self.n_days + 1))
        variances[:,0] += (self.sigma)

        # Generate correlated normals
        Z1 = np.random.RandomState(self.seed).normal(0, 1, (self.M, self.n_days + 1))
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * np.random.RandomState(self.seed).normal(0, 1, (self.M, self.n_days + 1))
        
        for i in range(1, self.n_days+1):
                        
            variances[:,i] = np.abs(variances[:,i - 1] + self.kappa * (self.theta - variances[:,i - 1]) * self.dt + self.sigma * np.sqrt(variances[:,i - 1] * self.dt) * Z1[:,i])
            prices[:,i] = prices[:,i - 1] * np.exp((self.r - 0.5 * variances[:,i - 1]) * self.dt + np.sqrt(variances[:,i - 1] * self.dt) * Z2[:,i])
        
        self.prices = prices
        self.volatilty_paths = variances

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