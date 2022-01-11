import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from functools import partial 
from tqdm import tqdm

import matplotlib.pyplot as plt
# import pickle
# import bisect


class HH1592d():
    """
    Hodgkin, Huxley, 1952d, Journal of Physiology
    """
    def __init__(self, protocol):
        
        self.protocol = protocol
                
        self.membrane_V0  = -75.0
        self.potassium_n0 = 0.317
        self.sodium_m0    = 0.05
        self.sodium_h0    = 0.595
        self.y0 = [self.membrane_V0, self.potassium_n0, self.sodium_m0, self.sodium_h0]
      
    def differential_eq(self, t, y0):    
        V, n, m, h = y0
       
        # Potassium current
        Ek = -87 # [mV]
        GK_max = 36 # [mS/cm^2]
        self.IK = GK_max * n**4 * (V - Ek)  # [uA/cm^2] Current carried by potassium ions
        a1 = 0.01 * (-V - 65.0) / (np.exp((-V - 65.0) / 10.0) - 1)
        b1 = 0.125 * np.exp((-V - 75.0) / 80.0)
        dn = a1*(1-n) - b1*n

        # Sodium current      
        ENa = 40.0      # [mV]
        GNa_max = 120.0   # [mS/cm^2] 
        self.INa = GNa_max * m**3 * h * (V - ENa)  # desc: Current carried by Sodium ions in [uA/cm^2] 
        a2 = 0.1 * (-V - 50.0) / (np.exp((-V - 50.0) / 10.0) - 1)
        b2 = 4.0 * np.exp((-V - 75.0) / 18.0)
        dm = a2 * (1.0 - m) - b2 * m            
        a3 = 0.07 * np.exp((-V - 75.0) / 20.0)
        b3 = 1.0 / (np.exp((-V - 45.0) / 10.0) + 1.0)
        dh = a3 * (1.0 - h) - b3 * h

        # Leak current
        Eleak = -64.387
        GLeak_max = 0.3 # [mS/cm^2]
        self.ILeak = GLeak_max * (V - Eleak)

        # Stimulus
        Vhold = -60  # [mV] : A temporary holding potential
        A = 100 # An amplification factor for the holding current        
        self.i_stim = (V - Vhold) * A * self.protocol.pacing(t)

        # Membrane
        C = 1 # [uF/cm^2] : The membrane capacity per unit area (See table 3 in HH1952d)        
        dV = -(1/C) * (self.INa + self.IK + self.ILeak + self.i_stim)
                    
        return [dV, dn, dm, dh]




def main():
    start_time = time.time()

    print("--- %s seconds ---"%(time.time()-start_time))


if __name__ == '__main__':
    main()