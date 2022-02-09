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
        self.dt = 0.01
        self.record_time_step = 1

    # def set_times(self, times):
    #     self.times = times
    #     print("Times has been set.")

    def cal_dt(self, max_step):        
        if self.dt>max_step:
            self.dt = max_step
        return self.dt    
    
    def simulate(self, end_time, log=[], max_step=float('inf'), default_time_unit='ms'):
        '''
        '''                               
        # self.nt = end_time/dt
                   
        # initial values
        current_time = 0          
        current_y = np.array(self.model.y0)

        times = [current_time]
        y_li = [self.model.y0]
        
        while current_time<=end_time:    
            
            # calculate time step
            dt = self.cal_dt(max_step)
            current_time += dt 
            
            # integration
            df = self.model.differential_eq(current_time, current_y)
            
            next_y = current_y + dt*np.array(df)
            
            # update values
            current_y = next_y

        #     if current_time
            times.append(current_time)
            y_li.append(current_y)  

        self.model.set_result(np.array(times), np.array(y_li).T, log)


        
if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))