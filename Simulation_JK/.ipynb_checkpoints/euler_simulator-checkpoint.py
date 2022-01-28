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
        self.dt = 1.
        self.record_time_step = 1

    # def set_times(self, times):
    #     self.times = times
    #     print("Times has been set.")

    def cal_dt(self, max_step):
        dt = 0.01
        if dt>max_step:
            dt = max_step
        return dt    
    
    def simulate(self, end_time, log=None, max_step=None, default_time_unit='ms'):
        '''
        '''                       
        current_time = 0
        end_time = 60        
        nt = end_time/dt

        V_current = -75.0
        n_current = 0.317
        m_current = 0.05
        h_current = 0.595
            
        current_y = self.model.y0
        
        times = [current_time]
        y_li = [self.model.y0]
        while current_time<=end_time:    
            
            # calculate time step
            dt = self.cal_dt(max_step)
            current_time += dt 
            
            # integration
            df = self.model.differential_eq(current_time, current_y)
            next_y = current_y + dt*df
            
            # update values
            current_y = next_y

        #     if current_time
            times.append(current_time)
            y_li.append(current_y)  
                                    )
        self.model.set_result(times, current_y, log)


        
if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))