'''
Created on Nov 25, 2009

@author: johnsalvatier
'''

from numpy import *
import pymc
from scipy import stats
import pylab as pl
import gradient_samplers as gs

ReData = arange(200, 3000, 25)
measured = 10.2 * (ReData )** .5 + stats.distributions.norm(mu = 0, scale = 55).rvs(len(ReData))

    
sd =pymc.Uniform('sd', lower = 5, upper = 100, value = 55.0) #pymc.Gamma("sd", 60 , beta =  2.0)

a = pymc.Uniform('a', lower = 0, upper = 100, value = 10.0)#pymc.Normal('a', mu =  10, tau = 5**-2)
b = pymc.Uniform('b', lower = .05, upper = 2.0, value = .5)

results = pymc.Normal('results', mu = a  * (ReData )** b, tau = sd **-2, value = measured, observed = True)

model = (sd, a, b)

M = pymc.MCMC(model)
M.use_step_method(gs.HMCStep, model)
M.isample(iter=2000, burn=0, thin=1)