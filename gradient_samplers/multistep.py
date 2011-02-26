'''
Created on Jan 11, 2010

@author: johnsalvatier
'''
from __future__ import division
import pymc as pm
import numpy as np

__all__ = ['MultiStep']

class MultiStep(pm.StepMethod):
    """
    Base class for multi-stochastic step methods. Gives the values of 
    the stochastics a single ordered representation.
    
    Properties: 
        vector : ndarray(dimensions)
            the values of all the stochastics given in a standard 1-d format.
        
        gradients_vector : ndarray(dimensions)
            vector of the log posterior with respect to each variable;
            standard format. 
        slices : dict(string, slice)
            describes the standard format. Indexed by stochastic name.
        dimensions : int 
            the total number of values in the stochastics.
        
    Methods:
        consider(vector): sets the values to the vector. Must be accepted or
            rejected to count, otherwise the next consider() will simply 
            override the current value without recording anything. 
        accept(): 
        reject(): 
    """
    def __init__(self, stochastics, verbose = 0, tally = True):
        pm.StepMethod.__init__(self, stochastics,verbose, tally=tally)
                
        self.slices, self.dimensions = vectorize_stochastics(stochastics)
      
    @property 
    def vector(self):
        vector = np.empty(self.dimensions)
        
        for stochastic in self.stochastics:
            vector[self.slices[str(stochastic)]] = np.ravel(stochastic.value)
            
        return vector  
    
    @property
    def gradients_vector(self):
        
        grad_logp = np.empty(self.dimensions)
        for stochastic, logp_gradient in self.logp_gradient.iteritems():
                
            grad_logp[self.slices[str(stochastic)]] = np.ravel(logp_gradient)  

        return grad_logp
         
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
            
        for stochastic in self.stochastics:
            proposal_value = vector[self.slices[str(stochastic)]]
            
            if np.size(proposal_value) > 1:
                proposal_value = np.reshape(proposal_value,  np.shape(stochastic.value))
            
            stochastic.value = proposal_value   
        
        self.under_consideration = True
        
def vectorize_stochastics(stochastics):
    """Compute the dimension of the sampling space and identify the slices
    belonging to each stochastic.
    """
    dimensions = 0
    slices = {}
    
    for s in set(stochastics):
        
        if isinstance(s.value, np.matrix):
            p_len = len(s.value.A.ravel())
        elif isinstance(s.value, np.ndarray):
            p_len = len(s.value.ravel())
        else:
            p_len = 1
            
        slices[str(s)] = slice(dimensions, dimensions + p_len)
        dimensions += p_len
        
    return slices, dimensions
