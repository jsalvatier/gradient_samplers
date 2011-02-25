'''
Created on Feb 13, 2011

@author: johnsalvatier
'''
import numpy as np
import pymc as pm
from multistep import MultiStep
import find_mode as fm

__all__ = ['HMCStep']

class HMCStep(MultiStep):
    """
    Hamiltonian MCMC/Hybrid Monte Carlo (HMC) step method. Works well on continuous variables for which
    the gradient of the log posterior can be calculated.
    
    Based off Radford's review paper of the subject 
    (http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html)
    
    Parameters
    ----------
    stochastics : single or iterable of stochastics
        the stochastics that should use this HMCStep
    step_size_scaling : float
        a scaling factor for the step sizes
    trajectory_length : float
        (roughly) how far each HMC step should travel (think in terms of standard deviations)
    covariance : (ndim , ndim) ndarray (where ndim is the total number of variables)
        covariance matrix for the HMC sampler to use.
        If None then will be estimated using the inverse hessian at the mode
    find_mode : bool
        whether to start the chain at the local minima of the distribution.
        If false, will start the simulation from the initial values of the stochastics
    
    
    Tuning advice:
     * General problems: try passing a better covariance matrix. For example,
        try doing a trial run and and then passing the empirical covariance matrix from
        that. 
     * optimal acceptance approaches .651 from above for high dimensional posteriors 
         (see Beskos 2010 esp. page 13). Target somewhat higher acceptance in pratice.
     * Low acceptance: try a lower step_size_scaling. 
     * Slow mixing: try significantly longer or shorter trajectory length (trajectories
        can double back).
     * Seems to sometimes get stuck in places for long periods: This is due to trajectory
         instability, try a smaller step size. Think of this as low acceptance in certain 
         areas. This is a sign that the sampler may give misleading results for small sample
         numbers in the areas with different stability limits (often the tails), so don't 
         ignore this if you care about those areas.
     * See section 4.2 of Radford's paper for more advice.
     
    Relevant Literature: 
    
    A. Beskos, N. Pillai, G. Roberts, J. Sanz-Serna, A. Stuart. "Optimal tuning of the Hybrid Monte-Carlo Algorithm" 2010. http://arxiv.org/abs/1001.4460
    G. Roberts. "MCMC using Hamiltonian dynamics" out of "Handbook of Markov Chain Monte Carlo" 2010. http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html    
    """
    
    optimal_acceptance = .651 #Beskos 2010
    
    def __init__(self, stochastics, step_size_scaling = .25, trajectory_length = 2., covariance = None, find_mode = True, verbose = 0, tally = True  ):
        MultiStep.__init__(self, stochastics, verbose, tally)
        
        self._tuning_info = ['acceptr']
        self._id = 'HMC'
        
        if find_mode:
            _, inv_hessian = fm.find_mode(self)
            self.accept()
        
        if covariance is None:
            if find_mode:
                covariance  = inv_hessian
            else :
                raise ValueError("can't estimate covariance without finding the mode")
            
        self.covariance = covariance
        self.inv_covariance = np.linalg.inv(covariance)
        self.step_size = step_size_scaling * self.dimensions**(1/4.)
        
        self.step_count = int(np.floor(trajectory_length / self.step_size))
        self.zero = np.zeros(self.dimensions)
        
    
    acceptr = 0.
    
    def step(self):
        start_logp = self.logp_plus_loglike
        
        # momentum scale proportional to inverse of parameter scale (basically sqrt(covariance))
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
            log_metrop_ratio = (-start_logp) - (-self.logp_plus_loglike) + self.kenergy(start_p) - self.kenergy(p)
            
            self.acceptr = np.minimum(np.exp(log_metrop_ratio), 1.)
            
            
            if (np.isfinite(log_metrop_ratio) and 
                np.log(np.random.uniform()) < log_metrop_ratio):
                
                self.accept()
            else: 
                self.reject() 
                
            a = 0
                
        except pm.ZeroProbability:
            self.reject()     
    
    def kenergy (self, x):
        return .5 * np.dot(x,np.dot(self.covariance, x))
    
    @staticmethod
    def competence(s):
        if pm.datatypes.is_continuous(s): 
            if False: # test ability to find gradient 
                return 2.5
        return 0
