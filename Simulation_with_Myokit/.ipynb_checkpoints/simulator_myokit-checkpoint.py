import os, sys
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import myokit
   


class Simulator:
    """
    
    """
    def __init__(self, model_path, protocol_def=None, pre_sim=2):
        '''
        if pre_sim==0 : No pre simulation
        if pre_sim==1 : pre simulation
        if pre_sim==2 : constant pacing pre simulation
        '''
        self.protocol_def = protocol_def
        basename = os.path.basename(model_path)        
        self.name = os.path.splitext(basename)[0]                
        self.bcl = 1000
        self.pre_sim = pre_sim  # 0:default  | 1 : pre simulation  | 2: pre simulation with constant pace
        self.vhold = 0  # -80e-3      -88.0145 
                        
        self.times = np.linspace(0, self.bcl, 1000)  # np.arange(self._bcl)        
        # Get model
        self.model, self._protocol, self._script = myokit.load(model_path)


        self.protocol_total_duration = 0
        if self.protocol_def != None:
            self.model, steps = self.protocol_def(self.model)
            self._protocol = myokit.Protocol()            
            for f, t in steps:
                self._protocol.add_step(f, t)
                self.protocol_total_duration += t
            self.vhold = steps[0][0]

        if self.pre_sim==2:
            self.pacing_constant_pre_simulate(self.vhold)
                
        self.simulation = myokit.Simulation(self.model, self._protocol)
#         self._simulation.set_tolerance(1e-12, 1e-14)
#         self._simulation.set_max_step_size(1e-5)
           
    # def set_times(self, times):
    #     self.times = times
    #     print("Times has been set.")

    def set_simulation_params(self, parameters):
        '''
        parameters : dictionary
        '''            
        for key, value in parameters.items():        
            self.simulation.set_constant(key, value)        
        

    def pacing_constant_pre_simulate(self, vhold=0):     
        # 1. Create pre-pacing protocol                
        protocol = myokit.pacing.constant(vhold)        
        self._pre_simulation = myokit.Simulation(self.model, protocol)    
        self._init_state = self._pre_simulation.state()  
        self._pre_simulation.reset()
        self._pre_simulation.set_state(self._init_state)        
        self._pre_simulation.pre(self.bcl*100)
        #         self._pre_simulation.set_tolerance(1e-12, 1e-14)
        #         self._pre_simulation.set_max_step_size(1e-5)
    

    def simulate(self, times, extra_log=[]):      
            
        self.simulation.reset()
        
        if self.pre_sim==1:
            self.simulation.pre(self.bcl*100)        
        if self.pre_sim==2:                                      
            self.simulation.set_state(self._init_state)
            self.simulation.set_state(self._pre_simulation.state())
                
        dt = times[1]-times[0]
        # Run simulation
        try:
            result = self.simulation.run(np.max(times),
                                          log_times = times,
                                          log = ['engine.time', 'membrane.V'] + extra_log,
                                         ).npview()
        except myokit.SimulationError:
            return float('inf')
            
        return result


    def gen_dataset(self, gen_params, datasetNo=1):
        '''
        type = 'AP' or 'I" 
        params = {
            'times': 1,                    
            'log_li' : [],
            'nData' : 10000,                         
            'dataset_dir' :   './dataset/,
            'data_file_name' :  'current,
            'scale' : 2,
        }  
        '''
        random.seed(datasetNo * 84)
        np.random.seed(datasetNo * 86)
    
        print("-----Dataset%d generation starts.-----"%(datasetNo))
        
        d = None              
        result_li = []
        param_li = []
        current_nData = 0
        
        simulation_error_count = 0
        with tqdm(total = gen_params['nData']) as pbar: 
            while (current_nData < gen_params['nData']):                
                g_fc_inv = np.random.uniform(0.0000001, 1, 8)                      
                if current_nData==0 and datasetNo==1:
                    g_fc_inv = np.ones(8)     
                g_fc = 1.0/g_fc_inv       # np.ones(6)
                
                Gfc = {                    
                    'ina.GNafc' : g_fc[0],
                    'inal.GNaLfc' : g_fc[1],
                    'ito.Gtofc' : g_fc[2],
                    'ical.PCafc' : g_fc[3],
                    'ikr.GKrfc' : g_fc[4],
                    'iks.GKsfc' : g_fc[5],
                    'ik1.GK1fc' : g_fc[6],
                    'if.Gffc' : g_fc[7]    
                } 
                self.set_simulation_params(Gfc)                
                log_li = ['membrane.V']
                if len(log_li)>0:
                    log_li = gen_params['log_li']
                try :
                    d = self.simulate( gen_params['times'], extra_log=gen_params['log_li'])                           
                    temp = []

                    for log in log_li :                                              
                        temp.append(d[log][::gen_params['scale']]) ###################
                    result_li.append( np.array(temp) ) 
                    param_li.append( g_fc_inv )
                    current_nData+=1                    
                except :
                    simulation_error_count += 1
    #                 print("There is a simulation error.")
                    continue
                pbar.update(1) 
  
        result_li = np.array(result_li, dtype=np.float32)        
        param_li = np.array(param_li, dtype=np.float32)        
                
        np.save(os.path.join(gen_params['dataset_dir'], 'times'), d['engine.time'].astype(np.float32)[::gen_params['scale']] )#######################3
        np.save(os.path.join(gen_params['dataset_dir'], 'voltage_protocol'), d['membrane.V'].astype(np.float32)[::gen_params['scale']] )#########################
        np.save(os.path.join(gen_params['dataset_dir'], "%s%d"%(gen_params['data_file_name'], datasetNo) ), result_li)
        np.save(os.path.join(gen_params['dataset_dir'], 'parameter%d'%(datasetNo) ), param_li )
        
        result_li = []
        param_li = []
            
        print("=====Dataset%d generation End.===== %d simulation errors occured.====="%(datasetNo, simulation_error_count))       
    
    
          
    

        
if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))