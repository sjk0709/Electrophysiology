import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode, solve_ivp, odeint
from scipy.optimize import curve_fit, least_squares

class Simulator:
    """
    
    """
    def __init__(self, model):
        '''
        '''        
        # Get model
        self.model = model
                        
        self.name = None        
        self.bcl = 1000
        self.vhold = 0  # -80e-3      -88.0145 
                        
#         self.times = np.linspace(0, self.bcl, 1000)  # np.arange(self._bcl)        
        self.V = -80.0
     
    # def set_times(self, times):
    #     self.times = times
    #     print("Times has been set.")

        
    def simulate(self, times, log=None, method='LSODA', max_step=None, default_time_unit='ms'):
        '''
        '''                       
        if default_time_unit == 's':
            self._time_conversion = 1.0
            default_unit = 'standard'            
        else:
            self._time_conversion = 1000.0
            default_unit = 'milli'

        t_span = [0, times.max() * self._time_conversion * 1e-3]
                   
        if method == 'LSODA':
            if max_step ==None:
                max_step = 8e-4 * self._time_conversion  
            self.solver = solve_ivp(self.model.differential_eq, t_span, y0=self.model.y0, args=self.model.params, t_eval=times, dense_output=True, 
                                    method='LSODA', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                                    max_step=max_step
                                    )
        if method == 'BDF':
            if max_step ==None:
                max_step = 1e-3 * self._time_conversion  
            self.solver = solve_ivp(self.model.differential_eq, t_span, y0=self.model.y0, args=self.model.params, t_eval=times, dense_output=True, 
                                    method='BDF', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                                    max_step=max_step, atol=1E-2, rtol=1E-4 
                                    )

        self.model.set_result(self.solver.t, self.solver.y, log)


        
if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))