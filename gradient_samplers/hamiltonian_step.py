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
    """
    Hamiltonian MCMC step method. 
    
    Based off Radford's review paper of the subject. Available here http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html
    
    Parameters
        ----------
        stochastics : iterable of stochastics
            the stochastics that should use this HMCStep
        step_size : float
            number of steps each trajectory should take
        step_size : float
            how far each HMC step should travel (think in terms of standard deviations
        covariance : (ndim , ndim) ndarray (where ndim is the total number of variables)
            covariance matrix for the HMC sampler to use. If None then will be estimated using the inverse hessian at the mode
        find_mode : bool
            whether to start the chain at the local minima of the distribution. If false, will start the simulation from the initial values of the stochastics
        verbose : int
        tally : bool"""
    def __init__(self, stochastics,step_count = 6, trajectory_length = 2., covariance = None, find_mode = True, verbose = 0, tally = True  ):
        MultiStep.__init__(self, stochastics, verbose, tally)
        
        if find_mode:
            _, inv_hessian = find_mode(self)
        self.accept()
        
        if covariance is None:
            if find_mode:
                covariance  = inv_hessian
            else :
                raise ValueError("can't estimate covariance without finding the mode")
            
        self.covariance = covariance
        self.inv_covariance = np.linalg.inv(covariance)
        self.step_size = trajectory_length/step_count
        self.step_count = step_count
        self.zero = np.zeros(self.dimensions)
        
        self._metrop_ratios = []
    
    
    def step(self):
        startp = self.logp_plus_loglike
        
        p = np.random.multivariate_normal(mean = self.zero ,cov = self.inv_covariance)
        start_p = p
        
        #use the leapfrog method
        
        p = p - (self.step_size/2) * (-self.gradients_vector) # half momentum update
        
        for i in range(self.step_count): 
            #alternate full variable and momentum updates
            self.consider(self.vector + self.step_size * np.dot(self.covariance, p))
            
            if i != self.step_count - 1:
                p = p - self.step_size * (-self.gradients_vector)
             
        p = p - (self.step_size/2) * (-self.gradients_vector)   # do a half step momentum update to finish off
        
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
        
