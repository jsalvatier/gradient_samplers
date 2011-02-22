'''
Created on Feb 13, 2011

@author: johnsalvatier
'''
import numpy as np
import pymc as pm
from multistep import MultiStep
from find_mode import find_mode

__all__ = ['HMCStep']

class HMCStep(MultiStep):
    """step method based on Hamiltonian dynamics. Primarily based off http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html
    
    """
    
    def __init__(self, stochastics,covariance = None, leapfrog_size = .3, leapfrog_n = 7, verbose = 0, tally = True  ):
        MultiStep.__init__(self, stochastics, verbose, tally)
        
        _, inv_hessian = find_mode(self)
        self.accept()
        
        if covariance is None:
            covariance  = inv_hessian
            
        self.covariance = covariance
        self.inv_covariance = np.linalg.inv(covariance)
        self.leapfrog_size = leapfrog_size
        self.leapfrog_n = leapfrog_n 
        self.zero = np.zeros(self.dimensions)
        
        self._metrop_ratios = []
        print self.vector
    
    
    def step(self):
        startp = self.logp_plus_loglike
        
        p = np.random.multivariate_normal(mean = self.zero ,cov = self.inv_covariance)
        start_p = p
        
        p = p - (self.leapfrog_size/2) * (-self.gradients_vector)
        
        for i in range(self.leapfrog_n): 
            
            self.consider(self.vector + self.leapfrog_size * np.dot(self.covariance, p))
            
            if i != self.leapfrog_n - 1:
                p = p - self.leapfrog_size * (-self.gradients_vector)
             
        p = p - (self.leapfrog_size/2) * (-self.gradients_vector)   
        
        p = -p 
            
        try: 
            log_metrop_ratio = (-startp) - (-self.logp_plus_loglike) + self.kenergy(start_p) - self.kenergy(p)
            self._metrop_ratios.append(log_metrop_ratio)
            
            if (np.isfinite(log_metrop_ratio) and 
                np.log(np.random.uniform()) < log_metrop_ratio):
                
                self.accept()
            else: 
                self.reject() 
                
        except pm.ZeroProbability:
            self.reject()    
            
    
    def kenergy (self, x):
        return .5 * np.dot(x,np.dot(self.covariance, x))
    
    @property
    def accept_ratios(self):
        return np.array(np.minimum(np.exp(self._metrop_ratios), 1))
        