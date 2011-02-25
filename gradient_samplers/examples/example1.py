'''
Created on Feb 14, 2011

@author: jsalvatier
'''
import pymc as pm
import gradient_samplers as gs
import numpy as np 

#x = pm.Normal('x', mu = 0.0, tau = 10.0**-2, size = 1)
y = pm.Normal('y', mu = 0.0, tau = np.array([.1, .2])**-2, size = 2)

model = (y,)

import pydevd 
pydevd.set_pm_excepthook()

M = pm.MCMC(model)
M.use_step_method(gs.HMCStep, model, step_size_scaling = .5, trajectory_length = .66)
M.isample(iter=50, burn=0, thin=1)

acceptance = M.trace('HMC' + '_acceptr' )()
print np.mean(acceptance)
print (acceptance)
import pylab 
pylab.plot(acceptance)
pylab.show()

gs.show_samples(gs.plot,x.trace())
gs.show_samples(gs.plot,y.trace())