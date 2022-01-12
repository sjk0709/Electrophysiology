import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt

import myokit
   
class Simulator:
    """
    
    """
    def __init__(self, model_path, protocol_def=None, vhold=0):
        '''
        if pre_sim==0 : No pre simulation
        if pre_sim==1 : pre simulation
        if pre_sim==2 : constant pacing pre simulation
        '''
        basename = os.path.basename(model_path)        
        self.name = os.path.splitext(basename)[0]                
        self.bcl = 1000
        self.vhold = vhold  # -80e-3      -88.0145 
                        
        self._times = np.linspace(0, self.bcl, 1000)  # np.arange(self._bcl)        
        # Get model
        self._model, self._protocol, self._script = myokit.load(model_path)
                
        self.pacing_constant_pre_simulate(self.vhold)
     
        if protocol_def != None:
            self._model, steps = protocol_def(self._model)
            self._protocol = myokit.Protocol()
            for f, t in steps:
                self._protocol.add_step(f, t)
    
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
    

    def simulate(self, extra_log=[], pre_sim=0):             

        self.simulation.reset()
        
        if pre_sim==1:
            self.simulation.pre(self.bcl*100)        
        if pre_sim==2:                                      
            self.simulation.set_state(self._init_state)
            self.simulation.set_state(self._pre_simulation.state())
                
        # Run simulation
        try:
            result = self.simulation.run(np.max(self._times),
                                          log_times = self._times,
                                          log = ['engine.time', 'membrane.V'] + extra_log,
                                         ).npview()
        except myokit.SimulationError:
            return float('inf')
            
        return result
          
    

        
if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))