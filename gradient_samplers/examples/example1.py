'''
Created on Feb 14, 2011

@author: jsalvatier
'''
import pymc as pm
import gradient_samplers as gs
import numpy as np 

sd = np.arange(-3,3, .2)
r = pm.Normal('x', mu = 0.0, tau = (10**sd)**-2)


data = 
y = 
model = (x,)

M = pm.MCMC(model)
M.use_step_method(gs.HMCStep, model, step_size_scaling = .15, trajectory_length = 2)
M.isample(iter=500, burn=0, thin=1)

acceptance = M.trace('HMC' + '_acceptr' )()
print np.mean(acceptance)
import pylab 
pylab.plot(acceptance)
pylab.show()

#gs.show_samples(gs.plot,x.trace())
gs.show_samples(gs.plot,x.trace())