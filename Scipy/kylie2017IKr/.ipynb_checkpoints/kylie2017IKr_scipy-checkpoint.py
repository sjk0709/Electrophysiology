import os # import listdir
import numpy as np
from scipy.integrate import ode, solve_ivp, odeint
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
# import pickle
# import bisect

# Parameter range setting
parameter_ranges = []
dataset_dir = 'Kylie'
if 'Kylie' in dataset_dir:    
    if 'rmax600' in dataset_dir:
        # Kylie: rmax = 600 
        parameter_ranges = []
        parameter_ranges.append( [100, 500000] )
        parameter_ranges.append( [0.0001, 598] )
        parameter_ranges.append( [0.0001, 260] )
        parameter_ranges.append( [0.0001, 598] )
        parameter_ranges.append( [0.0001, 130] )
        parameter_ranges.append( [0.0001, 598] )
        parameter_ranges.append( [0.0001, 260] )
        parameter_ranges.append( [0.0001, 598] )
        parameter_ranges.append( [0.0001, 130] )
        print("Kylie-rmax600 dataset has been selected.")
    else :
        # Kylie
        parameter_ranges.append([100, 500000])
        parameter_ranges.append( [0.0001, 1000000])
        parameter_ranges.append( [0.0001, 384])
        parameter_ranges.append( [0.0001, 1000000] )
        parameter_ranges.append( [0.0001, 192] )
        parameter_ranges.append( [0.0001, 1000000] )
        parameter_ranges.append( [0.0001, 384] )
        parameter_ranges.append( [0.0001, 1000000] )
        parameter_ranges.append( [0.0001, 192] )
        print("Kylie dataset has been selected.")

elif 'RealRange' in dataset_dir:
        parameter_ranges.append([3134, 500000])                 # g
        parameter_ranges.append( [0.0001, 2.6152843264828003])  # p1
        parameter_ranges.append( [43.33271226094526, 259])      # p2
        parameter_ranges.append( [0.001, 0.5] )                 # p3
        parameter_ranges.append( [15, 75] )                     # p4
        parameter_ranges.append( [0.8, 410] )                   # p5
        parameter_ranges.append( [0.0001, 138.] )               # p6
        parameter_ranges.append( [1.0, 59] )                    # p7
        parameter_ranges.append( [1.6, 90] )                    # p8
        print("RealRange dataset has been selected.")

parameter_ranges = np.array(parameter_ranges)
print(parameter_ranges.shape)

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


    def simulate(self, times, params=None, default_time_unit='s'):
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
            params = (self.g, self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8)        
        self.solver = solve_ivp(self.activation_inactivation_eq, t_span, y0=[self.open0, self.active0], args=params[1:], t_eval=times, dense_output=True, 
                        method='BDF', # RK45 | LSODA | DOP853 | Radau | BDF | RK23
                        max_step=1e-3*time_conversion,  #   LSODA : max_step=8e-4*time_conversion  |  BDF : max_step=1e-3*time_conversion, atol=1E-2, rtol=1E-4   |
                        atol=1E-2, rtol=1E-4 )
        
        self.times = self.solver.t
        self.open = self.solver.y[0]
        self.active = self.solver.y[1]
        self.V = self.voltage(self.times)        
#         self.open, self.active = self.solver.sol(times)
        self.IKr = params[0] * self.open * self.active * (self.V - self.EK)
        return self.IKr
       
        
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
    kylie = Kylie2017IKr()
    kylke.simulate()


if __name__ == '__main__':
    main()