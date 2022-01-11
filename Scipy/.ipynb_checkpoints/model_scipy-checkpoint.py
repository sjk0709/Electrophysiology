import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt

class ModelJK:
    """
    
    """
    def __init__(self, model, protocol):
        '''
        if pre_sim==0 : No pre simulation
        if pre_sim==1 : pre simulation
        if pre_sim==2 : constant pacing pre simulation
        '''
        
        self.model = model
        self.protocol = protocol
                
        self._membrane_V  = -75.0
        self._potassium_n = 0.317
        self._sodium_m    = 0.05
        self._sodium_h    = 0.595
                        
        self.name = None        
        self.bcl = 1000
        self.vhold = 0  # -80e-3      -88.0145 
                        
        self._times = np.linspace(0, self.bcl, 1000)  # np.arange(self._bcl)        
        # Get model
        self._model, self._protocol, self._script = myokit.load(model_path)
                
        self.pacing_constant_pre_simulate(self.vhold)
       
        self.simulation = myokit.Simulation(self._model, self._protocol)
#         self._simulation.set_tolerance(1e-12, 1e-14)
#         self._simulation.set_max_step_size(1e-5)
   
     
    def set_times(self, times):
        self._times = times
        print("Times has been set.")


    def pacing_constant_pre_simulate(self, vhold=0):     
        # 1. Create pre-pacing protocol                
        protocol = myokit.pacing.constant(vhold)        
        self._pre_simulation = myokit.Simulation(self._model, protocol)    
        self._init_state = self._pre_simulation.state()  
        self._pre_simulation.reset()
        self._pre_simulation.set_state(self._init_state)        
        self._pre_simulation.pre(self.bcl*100)
        #         self._pre_simulation.set_tolerance(1e-12, 1e-14)
        #         self._pre_simulation.set_max_step_size(1e-5)
    
        
    def simulate(self, times, params=None, default_time_unit='ms'):
        '''Solve activation and inactivation gate.
        '''                
        if default_time_unit == 's':
            self._time_conversion = 1.0
            default_unit = 'standard'
        else:
            self._time_conversion = 1000.0
            default_unit = 'milli'
        
        t_span = [0, times.max() * self._time_conversion * 1e-3]
                   
        self.solver = solve_ivp(self.differential_eq, t_span, y0=[self._membrane_V, self._potassium_n, self._sodium_m, self._sodium_h], args=params, t_eval=times, dense_output=True, 
                        method='LSODA', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                        max_step=8e-4   #   LSODA : max_step=8e-4*time_conversion  |  BDF : max_step=1e-3*time_conversion, atol=1E-2, rtol=1E-4   |
                        )
        
        self.times = self.solver.t
#         self.V = self.solver.y[0]
#         self.n = self.solver.y[1]
#         self.m = self.solver.y[2]
#         self.h = self.solver.y[3]
        return self.solver
    

        
if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))