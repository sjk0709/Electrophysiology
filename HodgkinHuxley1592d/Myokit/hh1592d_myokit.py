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

            fig, ax = plt.subplots(figsize=(6,4))    
            fig.suptitle('Hodgkin-Huxley 1952d', fontsize=14)
            # ax.set_title('Simulation %d'%(simulationNo))
            plt.xlabel('Time (millisecond)')
            plt.ylabel('Membrane Potential (millivolt)')     
            ax.plot(result['engine.time'], result['membrane.V'], label='AP')   
            # textstr = "GNa : %1.4f\nGNaL : %1.4f\nGto : %1.4f\nPCa : %1.4f\nGKr : %1.4f\nGKs : %1.4f\nGK1 : %1.4f\nGf : %1.4f"%(GNa/g_fc[0], GNaL/g_fc[1], Gto/g_fc[2], PCa/g_fc[3], GKr/g_fc[4], GKs/g_fc[5], GK1/g_fc[6], Gf/g_fc[7])
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            #     ax.text(0.67, 0.60, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)    
            #     fig1 = plt.gcf()
            ax.legend()
            plt.show()
            # fig.savefig(os.path.join(result_folder, "AP.jpg"), dpi=100)
            
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