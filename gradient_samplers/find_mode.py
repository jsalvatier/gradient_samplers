'''
Created on Feb 13, 2011

@author: johnsalvatier
'''
import numpy as np
import pymc as pm
from scipy.optimize import fmin_bfgs, fmin_ncg

__all__ = ['find_mode']

def find_mode(step_method, disp = False):
    def logp(x):
        step_method.consider(x) 
        try:
            return  -step_method.logp_plus_loglike   
        except pm.ZeroProbability:
            return 300e100

    def grad_logp(x):
        step_method.consider(x)
        
        try:
            step_method.logp_plus_loglike 
        except pm.ZeroProbability:
            return np.zeros(step_method.dimensions)
        
        return -step_method.gradients_vector
    
    
    #fmin_ncg(logp, step_method.vector, grad_logp, disp = disp)
    
    fmin_bfgs(logp, step_method.vector, grad_logp, disp = disp)
    step_method.accept() 