import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from functools import partial 
from tqdm import tqdm

# import pickle
# import bisect

sys.path.append('../')
from Protocols import pacing_protocol
from Lib import mod_trace as trace
        
        
class Membrane():
    def __init__(self):        
        self.V = -87.0
    
    def dot_V(self, Iions):
        '''
        in [mV]
        Membrane potential
        '''
        return -Iions
    
class Stimulus():
    def __init__(self):
        self.amplitude = -80 # [uA/cm^2]
        self.I = 0
        
    def cal_stimulation(self, pace):
        self.I = self.amplitude * pace     
        return self.I

class Phys():
    def __init__(self):
        '''
        The type of cell. Endo=0, Epi=1, Mid=2
        '''
        self.R = 8314           # [J/kmol/K] : Gas constant
        self.T = 310            # [K] : Temperature
        self.F = 96485          # [C/mol] : Faraday's constant
        self.RTF  = self.R*self.T/self.F
        self.FRT  = self.F/(self.R*self.T)
        self.FFRT = self.F*self.F/(self.R*self.T)  
        self.zna = 1
        self.zca = 2
        self.zk = 1

class Cell():
    def __init__(self):
        '''
        The type of cell. Endo=0, Epi=1, Mid=2
        '''
        self.phys = Phys()
        
        self.mode = 0
        self.L = 0.01                                                           # [cm] Cell length
        self.rad = 0.0011                                                       # [cm] cell radius
        self.vcell = 1000 * 3.14 * self.rad * self.rad * self.L                 # [uL] Cell volume
        self.Ageo = 2*3.14 * self.rad * self.rad + 2*3.14 * self.rad * self.L   # [cm^2] Geometric cell area
        self.Acap = 2 * self.Ageo                                               # [cm^2] Capacitative membrane area
        self.vmyo = 0.68 * self.vcell                                           # [uL] Volume of the cytosolic compartment
        self.vnsr = 0.0552 * self.vcell                                         # [uL] Volume of the NSR compartment
        self.vjsr = 0.0048 * self.vcell                                         # [uL] Volume of the JSR compartment
        self.vss = 0.02 * self.vcell                                            # [uL] Volume of the Submembrane space near the T-tubules
        self.AF = self.Acap / self.phys.F                                                 # F : Faraday's constant


class CaMK(): 
    def __init__(self):
        '''
        CaMKII signalling
        '''
        self.KmCaMK = 0.15
        self.aCaMK  = 0.05
        self.bCaMK  = 0.00068
        self.CaMKo  = 0.05
        self.KmCaM  = 0.0015

        # initial value
        self.CaMKt = 0.0125840447  # CaMKt=0
                
    def d_CaMKt(self, CaMKt, calcium_cass):
        '''          
        '''
        CaMKb  = self.CaMKo * (1.0 - self.CaMKt) / (1.0 + self.KmCaM / calcium_cass)
        CaMKa  = CaMKb + self.CaMKt
        self.f = 1 / (1 + self.KmCaMK / CaMKa) # Fraction of phosphorylated channels        
        return self.aCaMK * CaMKb * (CaMKb + CaMKt) - self.bCaMK * CaMKt        

class Sodium():
    '''
    Intracellular Sodium concentrations
    Page 18
    '''
    def __init__(self):        
        self.phys = Phys()
        self.cell = Cell()
        
        self.cm = 1.0
        
        # initial values
        self.Nai      = 7.268004498      # nai=7.268004498            2
        self.Na_ss    = 7.268089977      # nass=nai                   3

    def d_Nai(self, Nai, INa, INaL, INaCa, INaK, INab, diff_JdiffNa):
        INa_tot = INa + INaL + 3*INaCa + 3*INaK + INab 
        return (-INa_tot * self.cell.Acap * self.cm) / (self.phys.F * self.cell.vmyo) + diff_JdiffNa * self.cell.vss / self.cell.vmyo   
                
    def d_Na_ss(self, Na_ss, ical_ICaNa, inacass_INaCa_ss, diff_JdiffNa):        
        INa_ss_tot = ical_ICaNa + 3*inacass_INaCa_ss
        return -INa_ss_tot * self.cm * self.cell.Acap / (self.phys.F * self.cell.vss) - diff_JdiffNa
        
class Potassium():
    '''
    Intracellular Potassium concentrations
    Page 18
    '''
    def __init__(self):        
        self.phys = Phys()
        self.cell = Cell()
        
        self.cm = 1.0
        
        # initial values
        self.Ki    = 144.6555918      # ki=144.6555918             4
        self.K_ss  = 144.6555651      # kss=ki                     5

    def d_Ki(self, Ki, Ito, IKr, IKs, IK1, IKb, INaK, i_stim, diff_JdiffK):
        IK_tot = Ito + IKr + IKs + IK1 + IKb - 2 * INaK 
        return -(IK_tot + i_stim) * self.cm * self.cell.Acap / (self.phys.F * self.cell.vmyo) + diff_JdiffK * self.cell.vss / self.cell.vmyo
                
    def d_K_ss(self, K_ss, ical_ICaK, diff_JdiffK):       
        '''
        desc: Potassium concentration in the T-Tubule subspace
        ''' 
        IK_ss_tot = ical_ICaK
        return -IK_ss_tot * self.cm * self.cell.Acap / (self.phys.F * self.cell.vss) - diff_JdiffK

class Calcium():
    '''
    Intracellular Calcium concentrations and buffers
    Page 18
    '''
    def __init__(self, cell_mode):        
        self.phys = Phys()
        self.cell = Cell()
        self.cell_mode = cell_mode
        
        self.cm = 1.0
        cmdnmax_b = 0.05
        self.cmdnmax = cmdnmax_b  
        if self.cell_mode == 1:
            self.cmdnmax = 1.3*cmdnmax_b
        
        self.kmcmdn  = 0.00238
        self.trpnmax = 0.07
        self.kmtrpn  = 0.0005
        self.BSRmax  = 0.047
        self.KmBSR   = 0.00087
        self.BSLmax  = 1.124
        self.KmBSL   = 0.0087
        self.csqnmax = 10.0
        self.kmcsqn  = 0.8
        
        # initial values
        self.Cai = 8.6e-05 
        self.cass = 8.49e-05  
        self.Ca_nsr = 1.619574538
        self.Ca_jsr = 1.571234014 

    def d_Cai(self, Cai, ipca_IpCa, icab_ICab, inaca_INaCa , serca_Jup, diff_Jdiff):
        '''
        desc: Intracellular Calcium concentratium
        in [mmol/L]
        '''
        ICa_tot = ipca_IpCa + icab_ICab - 2*inaca_INaCa 
        a = self.kmcmdn + Cai
        b = self.kmtrpn + Cai
        Bcai = 1 / (1 + self.cmdnmax * self.kmcmdn / (a*a) + self.trpnmax * self.kmtrpn / (b*b))        
        return Bcai * (-ICa_tot * self.cm * self.cell.Acap / (2*self.phys.F*self.cell.vmyo) - serca_Jup  * self.cell.vnsr / self.cell.vmyo + diff_Jdiff * self.cell.vss / self.cell.vmyo )
                
    def d_cass(self, cass, ICaL, INaCa_ss, ryr_Jrel, diff_Jdiff):       
        '''
        desc: Calcium concentratium in the T-Tubule subspace
        in [mmol/L]
        ''' 
        ICa_ss_tot = ICaL - 2 * INaCa_ss
        a = self.KmBSR + cass
        b = self.KmBSL + cass
        Bcass = 1 / (1 + self.BSRmax * self.KmBSR / (a*a) + self.BSLmax * self.KmBSL / (b*b))
        return Bcass * (-ICa_ss_tot * self.cm * self.cell.Acap / (2*self.phys.F*self.cell.vss) + ryr_Jrel * self.cell.vjsr / self.cell.vss - diff_Jdiff )
    
    def d_Ca_nsr(self, Ca_nsr, serca_Jup, trans_flux_Jtr):       
        '''
        desc: Calcium concentration in the NSR subspace
        in [mmol/L]
        '''         
        return serca_Jup - trans_flux_Jtr * self.cell.vjsr / self.cell.vnsr
    
    def d_Ca_jsr(self, Ca_jsr, trans_flux_Jtr, ryr_Jrel):       
        '''
        desc: Calcium concentration in the JSR subspace
        in [mmol/L]
        '''         
        Bcajsr = 1.0 / (1.0 + self.csqnmax * self.kmcsqn / (self.kmcsqn + Ca_jsr)^2)                
        return Bcajsr * (trans_flux_Jtr - ryr_Jrel)

class Extra():
    '''
  
    '''
    def __init__(self):        
        self.Nao = 140 # [mmol/L] : Extracellular Na+ concentration
        self.Cao = 1.8 # [mmol/L] : Extracellular Ca2+ concentration
        self.Ko  = 5.4 # [mmol/L] : Extracellular K+ concentration

class Nernst():
    '''
    
    '''
    def __init__(self):        
        self.phys = Phys()
        self.extra = Extra()
    
    def nernst(self, Nai, Ki):
        ENa = self.phys.RTF * np.log(self.extra.Nao / Nai)      # in [mV]  desc: Reversal potential for Sodium currents
        EK = self.phys.RTF * np.log(self.extra.Ko / Ki)      # in [mV]  desc: Reversal potential for Potassium currents
        PKNa = 0.01833          # desc: Permeability ratio K+ to Na+
        EKs = self.phys.RTF * np.log((self.extra.Ko + PKNa * self.extra.Nao) / (Ki + PKNa * Nai)) # desc: Reversal potential for IKs  in [mV]

                
class INa():
    def __init__(self):
        self.m = 0.01
        self.h = 0.99
        self.j = 0.98
        
        self.gNaBar = 4   # [mS/cm^2]
        self.gNaC = 0.003 # [mS/cm^2]
        self.ENa = 50     # [mV]
                
    def dot_m(self, m, V):
        '''
        The activation parameter       
        '''
        alpha = (V + 47) / (1 - np.exp(-0.1 * (V + 47)))
        beta  = 40 * np.exp(-0.056 * (V + 72))        
        return alpha * (1 - m) - beta * m  # 
    
    def dot_h(self, h, V):
        '''
        An inactivation parameter   
        '''
        alpha = 0.126 * np.exp(-0.25 * (V + 77))
        beta  = 1.7 / (1 + np.exp(-0.082 * (V + 22.5)))
        return alpha * (1 - h) - beta * h  # 
    
    def dot_j(self, j, V):
        '''
        An inactivation parameter
        '''
        alpha = 0.055 * np.exp(-0.25 * (V + 78)) / (1 + np.exp(-0.2 * (V + 78)))
        beta  = 0.3 / (1 + np.exp(-0.1 * (V + 32)))
        return alpha * (1 - j) - beta * j  # 
            
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

    def dot_d(self, d, V):
        alpha = 0.095 * np.exp(-0.01 * (V + -5)) / (np.exp(-0.072 * (V + -5)) + 1)
        beta  = 0.07 * np.exp(-0.017 * (V + 44)) / (np.exp(0.05 * (V + 44)) + 1)
        return alpha * (1 - d) - beta * d
        
    def dot_f(self, f, V):
        alpha = 0.012 * np.exp(-0.008 * (V + 28)) / (np.exp(0.15 * (V + 28)) + 1)
        beta  = 0.0065 * np.exp(-0.02 * (V + 30)) / (np.exp(-0.2 * (V + 30)) + 1)
        return alpha * (1 - f) - beta * f
        
    def dot_Cai(self, Isi, Cai):
        '''
        desc: The intracellular Calcium concentration
        in [mol/L]
        '''
        return -1e-7 * Isi + 0.07 * (1e-7 - Cai)
   
    def I(self, d, f, Cai, V):        
        """
        in [uA/cm^2]
        The slow inward current, primarily carried by calcium ions. Called either
        "iCa" or "is" in the paper.
        """        
        Es = - 82.3 - 13.0287 * np.log(Cai)        # in [mV]
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
        
    def dot_x1(self, x1, V):
        alpha = 0.0005 * np.exp(0.083 * (V + 50)) / (np.exp(0.057 * (V + 50)) + 1)
        beta  = 0.0013 * np.exp(-0.06 * (V + 20)) / (np.exp(-0.04 * (V + 333)) + 1)
        return alpha * (1 - x1) - beta * x1
    
    def I(self, x1, V):        
        """
        in [uA/cm^2]
        A voltage- and time-dependent outward current, primarily carried by potassium ions 
        """        
        return x1 * 0.8 * (np.exp(0.04 * (V + 77)) - 1) / np.exp(0.04 * (V + 35))
    
        
        
class ORD2017():
    """    
    Beeler and Reuter 1977
    """
    def __init__(self, protocol=None):

        self.name = "BR1977"
        
        self.current_response_info = trace.CurrentResponseInfo(protocol)
        
        self.protocol = protocol
        
        self.membrane = Membrane()
        self.stimulus = Stimulus()        
        self.ina = INa()       
        self.isi = Isi()       
        self.ik1 = IK1()       
        self.ix1 = Ix1()               

        self.y0 = [self.membrane.V, self.ina.m, self.ina.h, self.ina.j, self.isi.d, self.isi.f, self.isi.Cai, self.ix1.x1]
        self.params = []

    def set_result(self, t, y, log=None):
        self.times =  t
        self.V = y[0]    
        self.m = y[1]
        self.h = y[2]
        self.j = y[3]
        self.d = y[4]
        self.f = y[5]
        self.Cai = y[6]
        self.x1 = y[7]               
        
    def differential_eq(self, t, y):    
        V, m, h, j, d, f, Cai, x1 = y
        
        # INa        
        INa = self.ina.I(m, h, j, V)
        dot_m = self.ina.dot_m(m, V)
        dot_h = self.ina.dot_h(h, V)
        dot_j = self.ina.dot_j(j, V)        
                
        # Isi        
        Isi = self.isi.I(d, f, Cai, V)
        dot_d = self.isi.dot_d(d, V)       
        dot_f = self.isi.dot_f(f, V)       
        dot_Cai = self.isi.dot_Cai(Isi, Cai)       
            
        # IK1
        IK1 = self.ik1.I(V)        
    
        # Ix1           
        Ix1 = self.ix1.I(x1, V)
        dot_x1 = self.ix1.dot_x1(x1, V)
                   
        # Membrane potential        
        dot_V = self.membrane.dot_V(IK1 + Ix1 + INa + Isi - self.stimulus.I)
        
        if self.current_response_info:
            current_timestep = [
                trace.Current(name='I_Na', value=INa),
                trace.Current(name='I_si', value=Isi),
                trace.Current(name='I_K1', value=IK1),
                trace.Current(name='I_x1', value=Ix1),               
            ]
            self.current_response_info.currents.append(current_timestep)
            
        return [dot_V, dot_m, dot_h, dot_j, dot_d, dot_f, dot_Cai, dot_x1]
    
    
    def response_diff_eq(self, t, y):
        
        if type(self.protocol) == pacing_protocol.PacingProtocol :
            if self.protocol.type=='AP':            
                face = self.protocol.pacing(t)
                self.stimulus.cal_stimulation(face) # Stimulus    
            
            elif self.protocol.type=='VC':
                y[0] = self.protocol.voltage_at_time(t)

        else:
            y[0] = self.protocol.get_voltage_at_time(t)
                    
        return self.differential_eq(t, y)
   



def main():
    start_time = time.time()

    print("--- %s seconds ---"%(time.time()-start_time))


if __name__ == '__main__':
    main()