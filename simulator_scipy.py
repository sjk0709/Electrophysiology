import os, sys
import copy
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode, solve_ivp, odeint
from scipy.optimize import curve_fit, least_squares

sys.path.append('./Protocols')
from pacing_protocol import PacingProtocol
import mod_protocols

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
        
     
    # def set_times(self, times):
    #     self.times = times
    #     print("Times has been set.")

    def pre_simulate(self, pre_step=5000, protocol=None, 
                           method='BDF', max_step=np.inf, atol=1E-6, rtol=1E-3, t_eval=None, default_time_unit='ms'):
        '''
        ''' 
        self.model.protocol = protocol           
        if protocol == 'Constant':
            self.model.protocol = mod_protocols.VoltageClampProtocol( [mod_protocols.VoltageClampStep(voltage=self.model.y0[0], duration=pre_step)] )
        else :
            self.model.protocol = PacingProtocol(level=-1, start=-10, length=-1, period=-1, multiplier=0, default_time_unit='ms')
        
        if default_time_unit == 's':
            self._time_conversion = 1.0
            default_unit = 'standard'            
        else:
            self._time_conversion = 1000.0
            default_unit = 'milli'
        
        t_span = (0, pre_step)   
        
        if method == 'LSODA':
            if max_step ==None:
                max_step = 8e-4 * self._time_conversion  
            self.solver = solve_ivp(self.model.response_diff_eq, t_span, y0=self.model.y0, t_eval=t_eval,
                                    dense_output=False, 
                                    method='LSODA', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                                    max_step=max_step, atol=atol, rtol=rtol )
        if method == 'BDF': 
            if max_step ==None:
                max_step = 1e-3 * self._time_conversion  
            self.solver = solve_ivp(self.model.response_diff_eq, t_span, y0=self.model.y0, t_eval=t_eval,
                                    dense_output=False, 
                                    method='BDF', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                                    max_step=max_step, atol=atol, rtol=rtol )
            
        self.model.y0 = copy.copy(self.solver.y[:,-1])
                
        
    def simulate(self, t_span : list, 
                       t_eval=None, log=None, method='BDF', max_step=np.inf, atol=1E-6, rtol=1E-3, default_time_unit='ms'):
        '''
        '''                       
        if default_time_unit == 's':
            self._time_conversion = 1.0
            default_unit = 'standard'            
        else:
            self._time_conversion = 1000.0
            default_unit = 'milli'
           
        if method == 'LSODA':
            if max_step ==None:
                max_step = 8e-4 * self._time_conversion  
            self.solver = solve_ivp(self.model.response_diff_eq, t_span, y0=self.model.y0, t_eval=t_eval,
                                    dense_output=False, 
                                    method='LSODA', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                                    max_step=max_step, atol=atol, rtol=rtol )
        if method == 'BDF': 
            if max_step ==None:
                max_step = 1e-3 * self._time_conversion  
            self.solver = solve_ivp(self.model.response_diff_eq, t_span, y0=self.model.y0, t_eval=t_eval,
                                    dense_output=True, 
                                    method='BDF', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                                    max_step=max_step, atol=atol, rtol=rtol )  # atol=1E-2, rtol=1E-4 
        
        self.model.set_result(self.solver.t, self.solver.y, log)


        
if __name__=='__main__':
    
    start_time = time.time()  
   

    print("--- %s seconds ---"%(time.time()-start_time))