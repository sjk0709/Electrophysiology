"""Contains classes to store the result of a genetic algorithm run.

Additionally, the classes in this module allow for figure generation.
"""

from abc import ABC
import copy
import enum
import math
import random
from typing import Dict, List, Union
from os import listdir, mkdir

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

import ga_configs
import mod_protocols as protocols
import mod_trace as trace
import mod_kernik as kernik


#############################################
from scipy.integrate import ode, solve_ivp
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import bisect

sys.path.append('../')
from Protocols.pacing_protocol import PacingProtocol
import simulator_scipy

# Models
from Models.br1977 import BR1977
from Models.ord2011JK_v1 import ORD2011
#############################################



def get_model_response_JK( model, protocol, prestep=None):    
    
    simulator = simulator_scipy.Simulator(model)     
    
    if prestep == None:
        print("There is no pre-step simulation.")
    elif prestep == 'pre':
        print("There is no pre-step simulation.")
    else:
        simulator.pre_simulate( pre_step=prestep, protocol='constant' )
    
    simulator.simulate( [0, protocol.get_voltage_change_endpoints()[-1]] , max_step=0.5, atol=1e-08, rtol=1e-10) 
    
    command_voltages = [protocol.get_voltage_at_time(t) for t in simulator.model.times]

    # print("dd11",len(model.current_response_info.currents))
    
    simulator.model.current_response_info = trace.CurrentResponseInfo()
    if len(simulator.solver.y) < 200:
        list(map(simulator.model.differential_eq, simulator.solver.t, simulator.solver.y.transpose()))
    else:
        list(map(simulator.model.differential_eq, simulator.solver.t, simulator.solver.y))               
    
    # print("dd22",len(model.current_response_info.currents))
    
    tr = trace.Trace(protocol,
                     cell_params=None,
                     t=simulator.model.times,
                     y=command_voltages,  # simulator.model.V,
                     command_voltages=command_voltages,
                     current_response_info=model.current_response_info,
                     default_unit=None)
    
    return tr
