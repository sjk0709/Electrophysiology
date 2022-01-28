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

        
    def simulate(self, times, log=None, max_step=None, default_time_unit='ms'):
        '''
        '''                       
        if default_time_unit == 's':
            self._time_conversion = 1.0
            default_unit = 'standard'            
        else:
            self._time_conversion = 1000.0
            default_unit = 'milli'

        t_span = [0, times.max() * self._time_conversion * 1e-3]
        
        current_time = 0
        end_time = 60
        dt = 0.001 # ms
        print(end_time/dt)

        V_current = -75.0
        n_current = 0.317
        m_current = 0.05
        h_current = 0.595

        times = [0]
        AP = [V_current]
        record_time_step = 1
        while current_time<=end_time:    

            current_time += dt 

            df = self.model.differential_eq(current_time, [V_current, n_current, m_current, h_current])
            V_next = V_current + dt*df[0]
            n_next = n_current + dt*df[1]
            m_next = m_current + dt*df[2]
            h_next = h_current + dt*df[3]

            V_current = V_next
            n_current = n_next
            m_current = m_next
            h_current = h_next

        #     if current_time
            times.append(current_time)
            AP.append(V_next)    

        self.model.set_result(self.solver.t, self.solver.y, log)


        
if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))