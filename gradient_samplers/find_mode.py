'''
Created on Feb 13, 2011

@author: johnsalvatier
'''
import numpy as np
import pymc as pm
from scipy.optimize import fmin_bfgs

__all__ = ['find_mode']

def find_mode(step_method, disp = True):
    def logp(x):
        step_method.consider(x) 
        try:
            return  -step_method.logp_plus_loglike   
        except pm.ZeroProbability:
            return 300e100

    def grad_logp(x):
        step_method.propose(x)
        
        try:
            step_method.logp_plus_loglike 
        except pm.ZeroProbability:
            return step_method.zero
        
        return -step_method.gradients_vector
    
    results = fmin_bfgs(logp, step_method.vector, grad_logp, disp = disp, full_output = True)
    
    return results[0], results[3]
        