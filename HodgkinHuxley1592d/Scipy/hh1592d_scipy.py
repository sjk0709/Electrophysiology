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


class HH1592d():
    """
    Myokit implementation of the model described in Hodgkin and Huxley's 1952
    paper [1], updated to use modern conventions.

    The modernization consisted of mirroring the gate functions around the
    point V=-75 to obtain the usual direction of depolarization. Additionally,
    the formulation V = E - Er was dropped and V is taken to be the action
    potential, not the difference in action potential.

    [1] A quantitative description of membrane current and its application to
    conduction and excitation in nerve.
    Hodgkin, Huxley, 1952d, Journal of Physiology
    """
    def __init__(self, protocol):
        
        self.protocol = protocol
                
        self._membrane_V  = -75.0
        self._potassium_n = 0.317
        self._sodium_m    = 0.05
        self._sodium_h    = 0.595
    
  
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
        self.V = self.solver.y[0]
        self.n = self.solver.y[1]
        self.m = self.solver.y[2]
        self.h = self.solver.y[3]
        return self.V
       
        
    def simulate_odeint(self, t, g,p1,p2,p3,p4,p5,p6,p7,p8):       
    
        def myode(ar, t):           
            a, r = ar        
            V = self.protocol.voltage_at_time(t)    
            k1 = p1*np.exp(p2*V)
            k2 = p3*np.exp(-p4*V)
            k3 = p5*np.exp(p6*V)
            k4 = p7*np.exp(-p8*V)
            tau_a = 1/(k1+k2)
            tau_r = 1/(k3+k4)
            a_inf = k1/(k1+k2)
            r_inf = k4/(k3+k4)    
            dot_a = (a_inf-a)/tau_a
            dot_r = (r_inf-r)/tau_r
            return [dot_a, dot_r]

        ar = odeint(myode, [self.open0, self.active0], t)
        a = ar[:, 0]
        r = ar[:, 1]            
        V = self.voltage(t)        
        IKr = g * a * r * (V - self.EK)
        return IKr
    
    def curve_fitting(self, times, data, p0, bounds=None, method=None):
        fit_p, pcov = curve_fit(self.simulate_odeint, times, data, p0=p0, bounds=bounds ,method=method)
        return fit_p
        


def main():
    hh = HH1592d()
    hh.simulate()


if __name__ == '__main__':
    main()