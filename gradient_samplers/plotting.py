'''
Created on Feb 2, 2011

@author: jsalvatier
'''
import pylab as pyl
from pylab import hist, plot
import numpy as np

__all__ = ['hist', 'plot', 'show_samples']

def show_samples(plot_func, samples, start = 0, *args):
    
    variables_shape = samples.shape[1:]
    n_variables = np.product(variables_shape)
    rowcol = np.ceil(np.sqrt(n_variables))
    
    plot_num = 1
    
    pyl.figure()

    for i in range(n_variables):
                
        pyl.subplot(rowcol, rowcol, plot_num)
        
        indexes =  list(np.unravel_index(i , variables_shape))
        plot_func(samples[tuple([slice(start,None)] + indexes)], *args) 
        pyl.title(str(indexes))
        
        plot_num += 1
        
    pyl.show()