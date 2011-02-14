'''
Created on Feb 14, 2011

@author: jsalvatier
'''
import pymc as pm
import gradient_samplers as gs

x = pm.Normal('x', mu = 0.0, tau = 10.0**-2, size = 3)
y = pm.Normal('y', mu = 0.0, tau = .1**-2, size = 1)

model = (x,y)

M = pm.MCMC(model)
M.use_step_method(gs.HMCStep, model)
M.isample(iter=2000, burn=0, thin=1)