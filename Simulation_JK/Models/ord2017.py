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
sys.path.append('../Lib')
from Protocols import pacing_protocol
import mod_trace as trace
        
        
class Membrane():
    def __init__(self):        
        self.V = -87.0
        self.y0 = [self.V]

    def d_V(self, Iions):
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
    def __init__(self, F=96485):
        '''
        The type of cell. Endo=0, Epi=1, Mid=2
        '''                
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
        self.AF = self.Acap / F                                                 # F : Faraday's constant

class Extra():
    '''
  
    '''
    def __init__(self):        
        self.Nao = 140 # [mmol/L] : Extracellular Na+ concentration
        self.Cao = 1.8 # [mmol/L] : Extracellular Ca2+ concentration
        self.Ko  = 5.4 # [mmol/L] : Extracellular K+ concentration


class Sodium():
    '''
    Intracellular Sodium concentrations
    Page 18
    '''
    def __init__(self):        
        
        # initial values
        self.Nai      = 7.268004498      # nai=7.268004498            2
        self.Na_ss    = 7.268089977      # nass=nai                   3

        self.y0 = [self.Nai, self.Na_ss]

    def diff_eq(self, Nai : float, Na_ss : float, 
                INa, INaL, INaCa, INaK, INab, ical_ICaNa, INaCa_ss, 
                diff): 
        cm = 1.0
        INa_tot = INa + INaL + 3*INaCa + 3*INaK + INab 
        d_Nai = (-INa_tot * cell.Acap * cm) / (phys.F * cell.vmyo) + diff.JdiffNa * cell.vss / cell.vmyo   
    
        INa_ss_tot = ical_ICaNa + 3*INaCa_ss
        d_Na_ss = -INa_ss_tot * cm * cell.Acap / (phys.F * cell.vss) - diff.JdiffNa

        return [d_Nai, d_Na_ss]
        
class Potassium():
    '''
    Intracellular Potassium concentrations
    Page 18
    '''
    def __init__(self):        
                
        self.cm = 1.0
        
        # initial values
        self.Ki    = 144.6555918      # ki=144.6555918             4
        self.K_ss  = 144.6555651      # kss=ki                     5

        self.y0 = [self.Ki, self.K_ss]

    def diff_eq(self, Ki, K_ss, 
                Ito, IKr, IKs, IK1, IKb, INaK, i_stim, ical_ICaK,
                diff):
        cm = 1.0
        
        IK_tot = Ito + IKr + IKs + IK1 + IKb - 2 * INaK 
        d_Ki = -(IK_tot + i_stim) * cm * cell.Acap / (phys.F * cell.vmyo) + diff.JdiffK * cell.vss / cell.vmyo

        IK_ss_tot = ical_ICaK
        d_K_ss = -IK_ss_tot * cm * cell.Acap / (phys.F * cell.vss) - diff.JdiffK

        return [d_Ki, d_K_ss]
    
class Calcium():
    '''
    Intracellular Calcium concentrations and buffers
    Page 18
    '''
    def __init__(self):        
        
        # initial values
        self.Cai = 8.6e-05 
        self.cass = 8.49e-05  
        self.Ca_nsr = 1.619574538
        self.Ca_jsr = 1.571234014 
        
        self.y0 = [self.Cai, self.cass, self.Ca_nsr, self.Ca_jsr]

    def diff_eq(self, Cai, cass, Ca_nsr, Ca_jsr, ipca, icab, inaca, ical, inacass, ryr, serca, diff, trans_flux):
        cm = 1.0
        cmdnmax_b = 0.05
        cmdnmax = cmdnmax_b  
        if cell.mode == 1:
            cmdnmax = 1.3*cmdnmax_b        
        kmcmdn  = 0.00238
        trpnmax = 0.07
        kmtrpn  = 0.0005
        BSRmax  = 0.047
        KmBSR   = 0.00087
        BSLmax  = 1.124
        KmBSL   = 0.0087
        csqnmax = 10.0
        kmcsqn  = 0.8

        '''
        desc: Intracellular Calcium concentratium
        in [mmol/L]
        '''
        ICa_tot = ipca.IpCa + icab.ICab - 2*inaca.INaCa 
        a = kmcmdn + Cai
        b = kmtrpn + Cai
        Bcai = 1 / (1 + cmdnmax * kmcmdn / (a*a) + trpnmax * kmtrpn / (b*b))        
        d_Cai = Bcai * (-ICa_tot * cm * cell.Acap / (2*phys.F*cell.vmyo) - serca.Jup  * cell.vnsr / cell.vmyo + diff.Jdiff * cell.vss / cell.vmyo )
                
        '''
        desc: Calcium concentratium in the T-Tubule subspace
        in [mmol/L]
        ''' 
        ICa_ss_tot = ical.ICaL - 2 * inacass.INaCa_ss
        a = KmBSR + cass
        b = KmBSL + cass
        Bcass = 1 / (1 + BSRmax * KmBSR / (a*a) + BSLmax * KmBSL / (b*b))
        d_cass = Bcass * (-ICa_ss_tot * cm * cell.Acap / (2*phys.F*cell.vss) + ryr.Jrel * cell.vjsr / cell.vss - diff.Jdiff )
            
        '''
        desc: Calcium concentration in the NSR subspace
        in [mmol/L]
        '''         
        d_Ca_nsr = serca.Jup - trans_flux.Jtr * cell.vjsr / cell.vnsr
            
        '''
        desc: Calcium concentration in the JSR subspace
        in [mmol/L]
        '''         
        Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (kmcsqn + Ca_jsr)^2)                
        d_Ca_jsr = Bcajsr * (trans_flux.Jtr - ryr.Jrel)

        return [d_Cai, d_cass, d_Ca_nsr, d_Ca_jsr]

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

        self.y0 = [self.CaMKt]
                
    def d_CaMKt(self, CaMKt, cass):
        '''          
        '''
        CaMKb  = self.CaMKo * (1.0 - self.CaMKt) / (1.0 + self.KmCaM / cass)

        d_CaMKt = self.aCaMK * CaMKb * (CaMKb + CaMKt) - self.bCaMK * CaMKt        

        CaMKa  = CaMKb + CaMKt
        self.f = 1 / (1 + self.KmCaMK / CaMKa) # Fraction of phosphorylated channels        
        
        return [d_CaMKt]

class Nernst():
    '''
    
    '''
    def __init__(self):        
        
        self.PKNa = 0.01833          # desc: Permeability ratio K+ to Na+
            
    def nernst(self, Nai, Ki):
        self.ENa = phys.RTF * np.log(extra.Nao / Nai)      # in [mV]  desc: Reversal potential for Sodium currents
        self.EK = phys.RTF * np.log(extra.Ko / Ki)      # in [mV]  desc: Reversal potential for Potassium currents        
        self.EKs = phys.RTF * np.log((extra.Ko + self.PKNa * extra.Nao) / (Ki + self.PKNa * Nai)) # desc: Reversal potential for IKs  in [mV]


class INa():
    '''
    INa :: Fast Sodium current
    Page 6
    
    The fast sodium current is modelled using a Hodgkin-Huxley type formulation
    including activation (m), slow and fast components of inactivation (h) and
    recovery from inactivation (j). The slow component of inactivation and
    recovery from inactivation have an alternative formulation for CaMKII-
    phosphorylated channels.
    '''
    def __init__(self):        
        
        # initial values    
        self.m = 0.007344121102   # m=0                        10
        self.hf = 0.6981071913     # hf=1                       11
        self.hs = 0.6980895801     # hs=1                       12
        self.j = 0.6979908432     # j=1                        13
        self.hsp = 0.4549485525     # hsp=1                      14
        self.jp = 0.6979245865     # jp=1                       

        self.y0 = [self.m, self.hf, self.hs, self.j, self.hsp, self.jp]

        #: Maximum conductance of INa channels
        self.GNa = 75.0        
            
    def diff_eq(self, V, m, hf, hs, j, hsp, jp, camk, nernst):
        '''
        Activation gate for INa channels
        '''
        mtD1 = 6.765
        mtD2 = 8.552
        mtV1 = 11.64
        mtV2 = 34.77
        mtV3 = 77.42
        mtV4 = 5.955    
        mssV1 = 39.57
        mssV2 = 9.871
        self.tm  = 1.0 / (mtD1 * np.exp((V + mtV1) / mtV2) + mtD2 * np.exp(-(V + mtV3) / mtV4)) # desc: Time constant for m-gate   in [ms]
        mss  = 1.0 / (1.0 + np.exp(-(V + mssV1)/mssV2))  # desc: Steady state value for m-gate 
        
        d_m = (mss - m) / tm           

        hssV1 = 82.9 
        hssV2 = 6.086 
        shift_INa_inact = 0.0
        hss = 1.0 / (1.0 + np.exp((V + hssV1-shift_INa_inact) / hssV2))   # desc: Steady-state value for h-gate
        thf = 1.0 / (1.432e-5 * np.exp(-(V + 1.196 - shift_INa_inact) / 6.285) + 6.1490 * np.exp((V + 0.5096 - shift_INa_inact) / 20.27)) # desc: Time constant for fast development of inactivation in INa   in [ms]
        ths = 1 / (0.009794 * np.exp(-(V + 17.95-shift_INa_inact) / 28.05) + 0.3343 * np.exp((V + 5.7300 - shift_INa_inact) / 56.66))  # desc: Time constant for slow development of inactivation in INa  in [ms]
        Ahf = 0.99 # : Fraction of INa channels with fast inactivation
        Ahs = 1.0 - Ahf # : Fraction of INa channels with slow inactivation
        
        d_hf = (hss - hf) / thf   # desc: Fast componennt of the inactivation gate for INa channels

        d_hs = (hss - hs) / ths   # desc: Slow componennt of the inactivation gate for non-phosphorylated INa channels
        
        h = Ahf * hf + Ahs * hs   # desc: Inactivation gate for INa
        tj = 2.038 + 1 / (0.02136 * np.exp(-(V + 100.6 - shift_INa_inact) / 8.281) + 0.3052 * np.exp((V + 0.9941 - shift_INa_inact) / 38.45)) # desc: Time constant for j-gate in INa in [ms]
        jss = hss  # desc: Steady-state value for j-gate in INa

        d_j = (jss - j) / tj # desc: Recovery from inactivation gate for non-phosphorylated INa channels

        # Phosphorylated channels
        thsp = 3 * ths # desc: Time constant for h-gate of phosphorylated INa channels  in [ms]
        hssp = 1 / (1 + np.exp((V + 89.1 - shift_INa_inact) / 6.086)) # desc: Steady-state value for h-gate of phosphorylated INa channels
        
        d_hsp = (hssp - hsp) / thsp # desc: Slow componennt of the inactivation gate for phosphorylated INa channels

        hp = Ahf * hf + Ahs * hsp # desc: Inactivation gate for phosphorylated INa channels
        tjp = 1.46 * tj # desc: Time constant for the j-gate of phosphorylated INa channels     in [ms]
        
        d_jp = (jss - jp) / tjp # desc: Recovery from inactivation gate for phosphorylated INa channels
        
        # Current        
        fINap = camk.f
        INa = self.GNa * (V - nernst.ENa) * m^3 * ((1 - fINap) * h * j + fINap * hp * jp) # in [uA/uF]  desc: Fast sodium current
    
        return [d_m, d_hf, d_hs, d_j, d_hsp, d_jp], INa

class INaL():
    '''
    INaL :: Late component of the Sodium current
    Page 7
    '''
    def __init__(self):        
        
        # initial values            
        self.mL         = 0.0001882617273  # mL=0                       16
        self.hL         = 0.5008548855     # hL=1                       17
        self.hLp        = 0.2693065357     # hLp=1                      

        self.y0 = [self.mL, self.hL, self.hLp]

        #: Maximum conductance
        self.GNaL_b = 0.019957499999999975        
            
    def diff_eq(self, V, mL, hL, hLp, camk, nernst, ina):
        mLss = 1.0 / (1.0 + np.exp(-(V + 42.85) / 5.264)) # desc: Steady state value of m-gate for INaL
        tmL = ina.tm

        d_mL = (mLss - mL) / tmL # desc: Activation gate for INaL

        thL = 200.0 # [ms] : Time constant for inactivation of non-phosphorylated INaL channels
        hLss = 1.0 / (1.0 + np.exp((V + 87.61) / 7.488))  # desc: Steady-state value for inactivation of non-phosphorylated INaL channels
        
        d_hL = (hLss - hL) / thL # desc: Inactivation gate for non-phosphorylated INaL channels

        hLssp = 1.0 / (1.0 + np.exp((V + 93.81) / 7.488)) # desc: Steady state value for inactivation of phosphorylated INaL channels
        thLp = 3 * thL # in [ms] desc: Time constant for inactivation of phosphorylated INaL channels

        d_hLp = (hLssp - hLp) / thLp  # desc: Inactivation gate for phosphorylated INaL channels
                
        # Current
        GNaL = self.GNaL_b
        if cell.mode == 1:
            GNaL = GNaL_b*0.6  # desc: Adjustment for different cell types            
        fINaLp = camk.f
        INaL = GNaL * (V - nernst.ENa) * mL * ((1 - fINaLp) * hL + fINaLp * hLp)
    
        return [d_mL, d_hL, d_hLp], INaL




class Ryr():
    '''
    '''
    def __init__(self):
        self.Jrel = None

        # initial values        
        self.Jrelnp      = 2.5e-7           # Jrelnp=0
        self.Jrelp       = 3.12e-7          # Jrelp=0                    

        self.y0 = [self.Jrelnp, self.Jrelp]
        
    def diff_eq(self, V, Jrelnp, Jrelp, Ca_jsr, ICaL, camk):

        bt=4.75
        a_rel=0.5*bt
        Jrel_inf_temp = a_rel * -ICaL / (1 + (1.5 / Ca_jsr)^8)
        Jrel_inf = Jrel_inf_temp
        if (cell.mode == 2):
            Jrel_inf = Jrel_inf_temp * 1.7            
        tau_rel_temp = bt / (1.0 + 0.0123 / Ca_jsr)
        tau_rel = tau_rel_temp
        if (tau_rel_temp < 0.001):
            tau_rel = 0.001
                        
        d_Jrelnp = (Jrel_inf - Jrelnp) / tau_rel                       

        btp = 1.25*bt
        a_relp = 0.5*btp
        Jrel_temp = a_relp * -ICaL / (1 + (1.5 / Ca_jsr)^8)
        Jrel_infp = Jrel_temp
        if cell.mode == 2:
            Jrel_infp = Jrel_temp * 1.7
        tau_relp_temp = btp / (1 + 0.0123 / Ca_jsr)
        tau_relp = tau_relp_temp
        if tau_relp_temp < 0.001:
            tau_relp = 0.001
        
        d_Jrelp = (Jrel_infp - Jrelp) / tau_relp            
            
        fJrelp = camk.f
        Jrel_scaling_factor = 1.0

        self.Jrel = Jrel_scaling_factor * (1 - fJrelp) * Jrelnp + fJrelp * Jrelp # desc: SR Calcium release flux via Ryanodine receptor  in [mmol/L/ms]

        return [d_Jrelnp, d_Jrelp]

class Diff():
    '''
    Diffusion fluxes
    Page 16
    '''
    def __init__(self):
        self.JdiffNa = None
        self.JdiffK  = None
        self.Jdiff   = None

    def calculate(self, Nai, Na_ss, Ki, K_ss, Cai, cass):
        self.JdiffNa = (Na_ss - Nai) / 2   # (sodium.Na_ss - sodium.Nai) / 2
        self.JdiffK  = (K_ss  - Ki)  / 2  # (potassium.K_ss  - potassium.Ki)  / 2
        self.Jdiff   = (cass - Cai) / 0.2     # (calcium.cass - calcium.Cai) / 0.2

class Serca():
    '''
    Jup :: Calcium uptake via SERCA pump
    Page 17
    '''
    def __init__(self):
        self.Jup = None

    def calculate(self, Cai, Ca_nsr, Ca_jsr, camk):
        upScale = 1.0
        if (cell.mode == 1):
            upScale = 1.3
        Jupnp = upScale * (0.004375 * Cai / (Cai + 0.00092))
        Jupp  = upScale * (2.75 * 0.004375 * Cai / (Cai + 0.00092 - 0.00017))
        fJupp = camk.f
        Jleak = 0.0039375 * Ca_nsr / 15.0 # in [mmol/L/ms]
        Jup_b = 1.0
        self.Jup = Jup_b * ((1.0 - fJupp) * Jupnp + fJupp * Jupp - Jleak) # desc: Total Ca2+ uptake, via SERCA pump, from myoplasm to nsr in [mmol/L/ms]

class Trans_flux():
    '''
    '''
    def __init__(self):
        self.Jtr = None
        
    def calculate(self, calcium_Ca_nsr, calcium_Ca_jsr):
        self.Jtr = (calcium_Ca_nsr - calcium_Ca_jsr) / 100.0   # desc: Ca2+ translocation from nsr to jsr    in [mmol/L/ms]
        

        
class ORD2017():
    """    
    O'Hara-Rudy CiPA v1.0 (2017)
    """
    def __init__(self, protocol=None):

        global phys
        phys = Phys()
        global cell 
        cell = Cell(phys.F)
        global extra 
        extra = Extra()
               
        
        self.name = "ORD2017"
        
        self.current_response_info = trace.CurrentResponseInfo(protocol)
        
        self.protocol = protocol
        
        self.membrane = Membrane()
        self.stimulus = Stimulus()      

        self.nernst = Nernst()
        self.camk = CaMK()  
        
        self.sodium = Sodium()
        self.potassium = Potassium()      
        self.calcium = Calcium()

        self.ina = INa()
        self.inal = INaL()

        self.ryr = Ryr()
        self.serca = Serca()
        self.diff = Diff()
        self.trans_flux = Trans_flux()

        self.y0 = self.membrane.y0 + self.camk.y0 + self.sodium.y0 + self.potassium.y0 + self.calcium.y0 + self.ina.y0 + self.inal.y0 + self.ryr.y0
        self.params = []

    def set_result(self, t, y, log=None):
        self.times =  t
        self.V = y[0]    
                 
        
    def differential_eq(self, t, y):    
        V, CaMKt, Nai, Na_ss, Ki, K_ss, Cai, cass, Ca_nsr, Ca_jsr, m, hf, hs, j, hsp, jp, mL, hL, hLp = y

        # Nernst  
        self.nernst.nernst(Nai, Ki)        
        
        # CaMKt
        d_CaMKt_li = self.camk.d_CaMKt(CaMKt, cass)

        # INa        
        d_INa_li, INa = self.ina.diff_eq(V, m, hf, hs, j, hsp, jp, self.camk, self.nernst)
                        
        # INaL
        d_INaL_li, INaL = self.inal.diff_eq(V, mL, hL, hLp, self.camk, self.nernst, self.ina)
  

        d_ryr_li = self.ryr.diff_eq(V, Jrelnp, Jrelp, Ca_jsr, ICaL, self.camk)
        self.diff.calculate(Nai, Na_ss, Ki, K_ss, Cai, cass)        
        self.serca.calculate(Cai, Ca_nsr, Ca_jsr, self.camk)
        self.trans_flux.calculate(Ca_nsr, Ca_jsr)

        d_sodium_li = self.sodium.diff_eq( Nai, Na_ss, 
                                           INa, INaL, INaCa, INaK, INab, ical_ICaNa, INaCa_ss,
                                           self.diff)

        d_potassium_li = self.potassium.diff_eq(Ki, K_ss, 
                                                Ito, IKr, IKs, IK1, IKb, INaK, i_stim, ical_ICaK,
                                                self.diff)

        d_calcium_li = self.calcium.diff_eq(Cai, cass, Ca_nsr, Ca_jsr, 
                                            self.ipca, self.icab, self.inaca, self.ical, self.inacass, 
                                            self.ryr, self.serca, self.diff, self.trans_flux)       

                   
        # Membrane potential        
        I_ion = INa + INaL
        d_V = self.membrane.d_V( I_ion + self.stimulus.I )
        
        if self.current_response_info:
            current_timestep = [
                trace.Current(name='INa', value=INa),
                trace.Current(name='INaL', value=INaL),                
            ]
            self.current_response_info.currents.append(current_timestep)
            
        return [d_V] + d_CaMKt_li + d_INa_li + d_INaL_li + d_ryr_li + d_sodium_li + d_potassium_li + d_calcium_li
    
    
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
    ord2017 = ORD2017(None)    
    print("--- %s seconds ---"%(time.time()-start_time))


if __name__ == '__main__':
    main()