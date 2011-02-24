'''
Created on Nov 25, 2009

@author: johnsalvatier
'''

from numpy import *
import pymc
import pylab as pl
import gradient_samplers as gs

"""
nonlinear regression. 

This is a problem from engineering. You have a water-steam heat exchanger, and you want to characterize how well
it transfers heat given a water flow rate. You can measure the water flow rate and you know the shape of the heat
exchanger, so you can calculate the Reynolds number for the water flow rate. You can also calculate the empirical
heat transfer coefficient as it operates.

The heat transfer coefficient (h) should be related to the Reynolds number (Re) by something like: 

h = a * Re** b + error

We can try to estimate a and b by collecting a bunch of data. Unfortunately mcmc sampling of the posterior can be
quite tricky. The parameters will be highly correlated in the posterior and nonlinearly. 
"""

#fake data
Re = arange(200, 3000, 25)
h_measured = 10.2 * Re ** .5 + random.normal(0, 55, size = len(Re))

#model
sd =pymc.Uniform('sd', lower = 5, upper = 100, value = 55.0) 
a = pymc.Uniform('a', lower = 0, upper = 100, value = 10.0)
b = pymc.Uniform('b', lower = .05, upper = 2.0, value = .5)

h = pymc.Normal('h', mu = a  * Re ** b, tau = sd **-2, value = h_measured, observed = True)

model = (sd, a, b)


#fit
M = pymc.MCMC(model)
M.use_step_method(gs.HMCStep, model) #compare to without HMCStep
M.isample(iter=2000, burn=0, thin=1)

#plot
gs.show_samples(gs.plot,a.trace())
gs.show_samples(gs.plot,b.trace())
gs.show_samples(gs.plot,sd.trace())

gs.show_samples(gs.hist,a.trace())
gs.show_samples(gs.hist,b.trace())
gs.show_samples(gs.hist,sd.trace())

pl.figure()
pl.hexbin(a.trace(), b.trace(), gridsize = 30)
pl.show()