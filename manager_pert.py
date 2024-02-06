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
    def __init__(self, savedir, config):
        self.savedir = savedir
        makedirs(self.savedir, exist_ok=True)
        self.derive_parameters(config)
        return
    @abstractmethod
    def derive_parameters(self, config):
        pass
    @abstractmethod
    def take_next_step(self, ens):
        # Based on the current state of the ensemble, update its state by planting, branching, or terminating
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
        self.branches = config['branches']
        self.branch_interval = config['branch_interval']
        self.branch_duration = config['branch_duration']
        return
    

