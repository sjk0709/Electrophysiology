import os # import listdir
import numpy as np
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt
# import pickle
# import bisect

class Kylie2017IKr():
    def __init__(self, protocol):
        self.protocol = protocol
                
        self.open0 = 0
        self.active0 = 1
        R = 8.314472 # [J/mol/K]
        T = 310     # [K]  # 36-37oC (BT)
        F = 9.64853415e4 #[C/mol]
        self.RTF = R * T / F        
        self.Ki = 110  # [mM]
        #Ki = 125 [mM]  # for iPSC solution
        self.Ko = 4    # [mM]
        #Ko = 3.75 [mM]  # for iPSC solution
        self.EK = self.RTF * np.log(self.Ko / self.Ki)  # in [V]
        
        self.g = 0.1524 * 1e3 # [pA/V]
        self.p1 = 2.26e-4 * 1e3 # [1/s]
        self.p2 = 0.0699 * 1e3  # [1/V]
        self.p3 = 3.45e-5 * 1e3 # [1/s]
        self.p4 = 0.05462 * 1e3 # [1/V]
        self.p5 = 0.0873 * 1e3  # [1/s]
        self.p6 = 8.91e-3 * 1e3 # [1/V]
        self.p7 = 5.15e-3 * 1e3 # [1/s]
        self.p8 = 0.03158 * 1e3 # [1/V]
    
        
    def voltage(self, times):
        '''Solve voltage            
        ''' 
        V_li = []
        for t in times:
            V_li.append( self.protocol.voltage_at_time(t) )                   
        return np.array(V_li)
    
    def get_rates(self, V, params=None):    
        if params is None:
            params = (self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8)
        k1 = params[0]*np.exp(params[1]*V)
        k2 = params[2]*np.exp(-params[3]*V)
        k3 = params[4]*np.exp(params[5]*V)
        k4 = params[6]*np.exp(-params[7]*V)
        
        return k1, k2, k3, k4
    
    def activation_inactivation_eq(self, t, y0, p1, p2, p3, p4, p5, p6, p7, p8):    
        a, r = y0    
        V = self.protocol.voltage_at_time(t)
        k1 = p1*np.exp(p2*V)
        k2 = p3*np.exp(-p4*V)
        k3 = p5*np.exp(p6*V)
        k4 = p7*np.exp(-p8*V)
        tau_a = 1.0/(k1+k2)
        tau_r = 1.0/(k3+k4)
        a_inf = k1/(k1+k2)
        r_inf = k4/(k3+k4) 
        da = (a_inf-a)/tau_a
        dr = (r_inf-r)/tau_r  
        return [da, dr]


    def simulate(self, times, t_span=[0, 20], params=None, default_time_unit='s'):
        '''Solve activation and inactivation gate.
        '''
        if default_time_unit == 's':
            time_conversion = 1.0
            default_unit = 'standard'
        else:
            time_conversion = 1000.0
            default_unit = 'milli'
        
        if params is None:
            params = (self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8)
        solver = solve_ivp(self.activation_inactivation_eq, t_span, y0=[self.open0, self.active0], args=params, dense_output=True, 
                        method='LSODA', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                        max_step=8e-4*time_conversion)
        
        self.V = self.voltage(times)        
        self.open, self.active = solver.sol(times)
        self.IKr = self.g * self.open * self.active * (self.V - self.EK)
        return self.IKr


def main():
    kylie = Kylie2017IKr()
    kylke.simulate()



if __name__ == '__main__':
    main()