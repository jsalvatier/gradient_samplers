'''
Created on Feb 13, 2011

@author: johnsalvatier
'''
import numpy as np
import pymc as pm
import numdifftools as nd

__all__ = ['find_mode']

def approx_hess(step_method, disp = True):
    
    def grad_logp(x):
        step_method.consider(x)
        
        return -step_method.gradients_vector
    
    #find the jacobian of the gradient function at the current position
    #this should be the hessian
    hess = nd.Jacobian(grad_logp)(step_method.vector)
    step_method.reject()
    return hess
    