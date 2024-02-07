from abc import ABC, abstractmethod
import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import copy as copylib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from lorenz96 import Lorenz96
from ensemble import Ensemble
import forcing



class EnsembleAlgorithm(ABC):
    def __init__(self, config):
        self.derive_parameters(config)
        return
    @abstractmethod
    def derive_parameters(self, config):
        pass
    @abstractmethod
    def take_first_step(self, ens):
        pass
    @abstractmethod
    def take_next_step(self, ens):
        # Based on the current state of the ensemble, provide the arguments (icandf, obs_fun, parent) to give to ens.branch_or_plant. Don't modify ens right here.
        pass

# TODO make a global acquisition algorithm and a local acquisition algorithm, for some higher-level algorithm to manage in tandem

class PertGrowthMeasuringAlgorithm(EnsembleAlgorithm):
    # Algorithm to split off from initial condition 
    def derive_parameters(self, config):
        # Determine the random input space
        frc_type = config['frc']['type']
        if frc_type == 'white':
            self.frc_dim = 1
            self.frc_dtype = 'uint64'
        elif frc_type == 'impulsive': 
            # input dimension must match that of ens.dynsys
            self.frc_dim = config['frc']['impulse']['impulse_dim']
            self.frc_type = 'float64'
        # Determine branching number
        self.branch_number = config['branch_number'] # How many different members to spawn from the same initial condition
        self.branch_interval = config['branch_interval'] # How long to wait between consecutive splits
        self.branch_duration = config['branch_duration'] # How long to run each branch
        return
    def take_first_step(self, ens):
        # The ensemble should be empty
        assert ens.memgraph.number_of_nodes() == 0
        obs_fun = lambda t,x: None
        if icandf is None:
            icandf = ens.dynsys.default_icandf()
        
        ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=None)

    

