'''
Created on Jan 11, 2010

@author: johnsalvatier
'''
from __future__ import division
import pymc as pm
import numpy as np

__all__ = ['MultiStep']

class MultiStep(pm.StepMethod):
    
    
    def __init__(self, stochastics, verbose = 0, tally = True):
        
        self.slices, self.dimensions = vectorize_stochastics(stochastics)
        
        # Initialize superclass
        pm.StepMethod.__init__(self, stochastics, tally=tally)
        
        if verbose is not None:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose 
           
    @property 
    def vector(self):
        vector = np.empty(self.dimensions)
        
        for stochastic in self.stochastics:
            vector[self.slices[str(stochastic)]] = np.ravel(stochastic.value)
            
        return vector  
    
    @property
    def gradients_vector(self):
        
        grad_logp = np.zeros(self.dimensions)
        for stochastic, logp_gradient in self.logp_gradient.iteritems():
                
            grad_logp[self.slices[str(stochastic)]] = np.ravel(logp_gradient)  

        return grad_logp

    def propose(self, proposal_vector):

        for stochastic in self.stochastics:

            stochastic.value = np.reshape(proposal_vector[self.slices[str(stochastic)]],  np.shape(stochastic.value))
            
            
    def revert (self):
        for stochastic in self.stochastics:
            stochastic.revert()
    
    def accept(self):
        self.under_consideration = False
        
    def reject(self):
        self.revert()
        self.under_consideration = False
    
    under_consideration = False
    
    def consider(self, vector):
        if self.under_consideration:
            self.revert()
            
        self.propose(vector)   
        
        self.under_consideration = True
        
def vectorize_stochastics(stochastics):
    """Compute the dimension of the sampling space and identify the slices
    belonging to each stochastic.
    """
    dimensions = 0
    slices = {}
    
    for s in stochastics:
        
        if isinstance(s.value, np.matrix):
            p_len = len(s.value.A.ravel())
        elif isinstance(s.value, np.ndarray):
            p_len = len(s.value.ravel())
        else:
            p_len = 1
            
        slices[str(s)] = slice(dimensions, dimensions + p_len)
        dimensions += p_len
        
    return slices, dimensions
