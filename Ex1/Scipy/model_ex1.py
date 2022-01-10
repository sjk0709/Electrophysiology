import os # import listdir
import numpy as np
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt
# import pickle
# import bisect

class ModelEx1():
    def __init__(self, protocol=None):
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
                
        self.k1 = 250 # [1/s]
        self.k2 = 100  # [1/V]
                
    def voltage(self, times):
        '''Solve voltage            
        ''' 
        V_li = []
        for t in times:
            V_li.append( self.protocol.voltage_at_time(t) )                   
        return np.array(V_li)

    
    def model_eq(self, t, y0, k1, k2):    
        O, C = y0   
        V = np.sin(0.1*t)
        k1 = 2*np.exp(1.30*V)
        k2 = 8*np.exp(-1.40*V)
        dO = k1*C - k2*O
        dC = k2*O - k1*C
        return [dO, dC]


    def simulate(self, times, params=None, default_time_unit='ms'):
        '''Solve activation and inactivation gate.
        '''
        t_span = [0, times.max()]
        
        if default_time_unit == 's':
            time_conversion = 1.0
            default_unit = 'standard'
        else:
            time_conversion = 1000.0
            default_unit = 'milli'
        
        if params is None:
            params = (self.k1, self.k2)
        self.solver = solve_ivp(self.model_eq, t_span, y0=[self.open0, self.active0], args=params, t_eval=times, dense_output=True, 
                        method='BDF', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                        max_step=1e-3*time_conversion,  #   LSODA : 8e-4*time_conversion
                        atol=1E-2, rtol=1E-4 )
          
        self.O, self.C = self.solver.y[0], self.solver.y[1]
#         self.O, self.C = self.solver.sol(times)
        
        return self.O, self.C 


def main():
    kylie = Kylie2017IKr()
    kylke.simulate()



if __name__ == '__main__':
    main()