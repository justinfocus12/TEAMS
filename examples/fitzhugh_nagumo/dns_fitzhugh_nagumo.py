import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import sys
import glob
import copy as copylib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
sys.path.append('../..')
from fitzhugh_nagumo import FitzhughNagumoODE,FitzhughNagumoSDE
from ensemble import Ensemble
import forcing
from algorithms_fitzhugh_nagumo import FitzhughNagumoODEDirectNumericalSimulation as L96ODEDNS, FitzhughNagumoSDEDirectNumericalSimulation as FHNSDEDNS
import utils

def dns_multiparams():
    seed_incs = [0] # In theory we could make an unraveled array of (F4,seed)
    Ds = [0.01,0.1,0.25,0.5]
    epsilons = [0.01,0.1,1.0]
    return seed_incs,Ds,epsilons


def dns_paramset(i_param):
    # Organize the array of parameters as well as the output files 
    # Minimal labels to differentiate them 
    expt_labels = [r'$D=%g$'%(D) for D in Ds]
    expt_abbrvs = [(r'Deq%g'%(D)).replace('.','p') for D in Ds]
    config_dynsys = FitzhughNagumoSDE.default_config()
    config_dynsys['D'] = Ds[i_param]

    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_param], # add to seed_min
        'max_member_duration_phys': 50.0,
        'num_chunks_max': 100,
        })

    config_analysis = dict({
        # return statistics analysis
        'return_stats': dict({
            'time_block_size_phys': 12,
            'spinup_phys': 30,
            'k_roll_step': 4, # step size for augmenting Lorenz96 with rotational symmetry 
            })
        # Other possible parameters: method used for fitting GEV, threshold for GPD, ...
        })

    return config_dynsys,config_algo,config_analysis,expt_labels[i_param],expt_abbrvs[i_param]
