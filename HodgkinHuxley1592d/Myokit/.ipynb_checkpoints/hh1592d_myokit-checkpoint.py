import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt

import myokit

import multiprocessing
from functools import partial 
from tqdm import tqdm

   
class HH1592d:
    """
    Hodgkin-Huxley 1952d
    """
    def __init__(self, model_path):
        
        self._bcl = 30
        # self._times = np.arange(self._bcl)
        self._times = np.linspace(0, self._bcl, 1000)
        
        # Get model
        self._model, self._protocol, self._script = myokit.load(model_path)
        self._simulation = myokit.Simulation(self._model, self._protocol)
          
#         self._simulation.set_tolerance(1e-12, 1e-14)
#         self._simulation.set_max_step_size(1e-5)
   
     
    def set_times(self, times):
        self._times = times
        print("Times has been set.")


    def simulate(self, extra_log=[], plot=False):             

        self._simulation.reset()
        # self._simulation.pre(self._bcl*100)
        
        # Run simulation
        try:
            result = self._simulation.run(np.max(self._times),
                                          log_times = self._times,
                                          log = ['engine.time', 'membrane.V'] + extra_log,
                                         ).npview()
        except myokit.SimulationError:
            return float('inf')

        # Display the result
        if plot:
            plt.figure()
            plt.suptitle('Hodgkin-Huxley 1952d')
            plt.plot(result['engine.time'], result['membrane.V'])
            plt.title('Membrane potential')
            plt.xlabel('Time (millisecond)')
            plt.ylabel('Membrane potential (millivolt)')
            #plt.legend("membrane", loc='best')
#             plt.savefig("./default.png")
            plt.show()
            
        return result
          
    

        
if __name__=='__main__':
    
    start_time = time.time()
    
    model_path = "../../../mmt-model-files/hh-1952d-modern.mmt"    
    hh = HH1592d(model_path) 
    times = np.arange(30)
    times = np.linspace(0,30,1000)
    hh.set_times(times)
    hh.simulate(plot=True)

    print("--- %s seconds ---"%(time.time()-start_time))