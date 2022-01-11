import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from functools import partial 
from tqdm import tqdm

from scipy.integrate import ode, solve_ivp, odeint
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
# import pickle
# import bisect


class Membrane():
    def __init__(self):
        self.C = 1  # [uF/cm^2] : The membrane capacitance
        self.V = -84.622        
    
    def dot_V(self, IK1, Ix1, INa, Isi, IStim):
        '''
        in [mV]
        Membrane potential
        '''
        return -(1.0/self.C) * (IK1 + Ix1 + INa + Isi - IStim)
    

class Stimulus():
    def __init__(self):
        self.amplitude = 25 # [uA/cm^2]
        
    def I(self, pace):
        return self.amplitude * pace
                
class INa():
    def __init__(self):
        self.m = 0.01
        self.h = 0.99
        self.j = 0.98
        
        self.gNaBar = 4   # [mS/cm^2]
        self.gNaC = 0.003 # [mS/cm^2]
        self.ENa = 50     # [mV]
          
    def I(self, m, h, j, V):        
        """
        in [uA/cm^2]
        The excitatory inward sodium current
        """                
        return (self.gNaBar * m**3 * h * j + self.gNaC) * (V - self.ENa)
        
        
class Isi():
    def __init__(self):
        self.d = 0.003
        self.f = 0.99
        self.Cai = 2e-7
        
        self.gsBar = 0.09    

   
    def I(self, d, f, Cai, V):        
        """
        in [uA/cm^2]
        The slow inward current, primarily carried by calcium ions. Called either
        "iCa" or "is" in the paper.
        """        
        Es = - 82.3 - 13.0287 * np.log(Cai)        
        return self.gsBar * d * f * (V - Es)         # in [uA/cm^2]
    

class IK1():
    def __init__(self):
        self.xx = 0
    
    def I(self, V):        
        """
        in [uA/cm^2]
        A time-independent outward potassium current exhibiting inward-going rectification
        """
        return 0.35 * (4 * (np.exp(0.04 * (V + 85)) - 1) / (np.exp(0.08 * (V + 53)) + np.exp(0.04 * (V + 53))) + 0.2 * (V + 23) / (1 - np.exp(-0.04 * (V + 23)))
    )
        
        
class Ix1():
    def __init__(self):
        self.x1 = 0.0004
    
    def I(self, x1, V):        
        """
        in [uA/cm^2]
        A voltage- and time-dependent outward current, primarily carried by potassium ions 
        """        
        return x1 * 0.8 * (np.exp(0.04 * (V + 77)) - 1) / np.exp(0.04 * (V + 35))

    
        
        
class BR1977():
    """    
    Beeler and Reuter 1977
    """
    def __init__(self, protocol):
        
        self.protocol = protocol
        
        self.membrane = Membrane()
        self.stimulus = Stimulus()        
        self.ina = INa()       
        self.isi = Isi()       
        self.ik1 = IK1()       
        self.ix1 = Ix1()               

        self.y0 = [self.membrane.V, self.ina.m, self.ina.h, self.ina.j, self.isi.d, self.isi.f, self.isi.Cai, self.ix1.x1]
        
    def differential_eq(self, t, y0):    
        V, ina_m, ina_h, ina_j, isi_d, isi_f, isi_Cai, ix1_x1 = y0
       
        # Stimulus
        face = self.protocol.pacing(t)
        IStim = self.stimulus.I(face)
        
        # INa
        alpha = (V + 47) / (1 - np.exp(-0.1 * (V + 47)))
        beta  = 40 * np.exp(-0.056 * (V + 72))
        dot_m =  alpha * (1 - ina_m) - beta * ina_m  # The activation parameter             
        
        alpha = 0.126 * np.exp(-0.25 * (V + 77))
        beta  = 1.7 / (1 + np.exp(-0.082 * (V + 22.5)))
        dot_h = alpha * (1 - ina_h) - beta * ina_h  # An inactivation parameter        
        
        alpha = 0.055 * np.exp(-0.25 * (V + 78)) / (1 + np.exp(-0.2 * (V + 78)))
        beta  = 0.3 / (1 + np.exp(-0.1 * (V + 32)))
        dot_j =  alpha * (1 - ina_j) - beta * ina_j  # An inactivation parameter
        
        INa = (self.ina.gNaBar * ina_m**3 * ina_h * ina_j + self.ina.gNaC) * (V - self.ina.ENa)
        
        
        # Isi        
        alpha = 0.095 * np.exp(-0.01 * (V + -5)) / (np.exp(-0.072 * (V + -5)) + 1)
        beta  = 0.07 * np.exp(-0.017 * (V + 44)) / (np.exp(0.05 * (V + 44)) + 1)
        dot_d = alpha * (1 - isi_d) - beta * isi_d
       
        alpha = 0.012 * np.exp(-0.008 * (V + 28)) / (np.exp(0.15 * (V + 28)) + 1)
        beta  = 0.0065 * np.exp(-0.02 * (V + 30)) / (np.exp(-0.2 * (V + 30)) + 1)
        dot_f =  alpha * (1 - isi_f) - beta * isi_f        
        
        Es = -82.3 - 13.0287 * np.log(isi_Cai)  # in [mV]
        Isi = self.isi.gsBar * isi_d * isi_f * (V - Es)         # in [uA/cm^2]
        dot_Cai = -1e-7 * Isi + 0.07 * (1e-7 - isi_Cai)
        
        # IK1
        IK1 = 0.35 * (4 * (np.exp(0.04 * (V + 85)) - 1) / (np.exp(0.08 * (V + 53)) + np.exp(0.04 * (V + 53))) + 0.2 * (V + 23) / (1 - np.exp(-0.04 * (V + 23))))
    
    
        # Ix1           
        alpha = 0.0005 * np.exp(0.083 * (V + 50)) / (np.exp(0.057 * (V + 50)) + 1)
        beta  = 0.0013 * np.exp(-0.06 * (V + 20)) / (np.exp(-0.04 * (V + 333)) + 1)
        dot_x1 =  alpha * (1.0 - ix1_x1) - beta * ix1_x1
        
        Ix1 = ix1_x1 * 0.8 * (np.exp(0.04 * (V + 77)) - 1) / np.exp(0.04 * (V + 35))
              
            
        # Membrane potential
        dot_V = -(1.0/self.membrane.C) * (IK1 + Ix1 + INa + Isi - IStim)     
    
        return [dot_V, dot_m, dot_h, dot_j, dot_d, dot_f, dot_Cai, dot_x1]
 
        


def main():
    start_time = time.time()

    print("--- %s seconds ---"%(time.time()-start_time))


if __name__ == '__main__':
    main()