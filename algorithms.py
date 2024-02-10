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
    def __init__(self, config, ens, seed):
        self.ens = ens
        self.rng = default_rng(seed) # In case ANY randomness is involved
        self.derive_parameters(config)
        return
    @abstractmethod
    def derive_parameters(self, config):
        pass
    @abstractmethod
    def take_first_step(self):
        pass
    @abstractmethod
    def take_next_step(self):
        # Based on the current state of the ensemble, provide the arguments (icandf, obs_fun, parent) to give to ens.branch_or_plant. Don't modify ens right here.
        pass

# TODO make a global acquisition algorithm and a local acquisition algorithm, for some higher-level algorithm to manage in tandem

class InitCondPertAlgo(EnsembleAlgorithm):
    # Algorithm to split off from initial condition 
    # Only for ODE system...or maybe make this more general and have a class hierarchy mirroring that of DynamicalSystem...
    def derive_parameters(self, config):
        # Determine the random input space
        self.seed_min,self.seed_max = config['seed_min'],config['seed_max']
        # Determine branching number
        self.branches_per_point = config['branches_per_point'] # How many different members to spawn from the same initial condition
        self.interbranch_interval = config['interbranch_interval'] # How long to wait between consecutive splits
        self.branch_duration = config['branch_duration'] # How long to run each branch
        self.num_branch_points = config['num_branch_points'] # but include the possibility for extension
        self.trunk_duration = self.ens.dynsys.t_burnin + self.branch_interval * self.num_branch_points
        # Most likely all subclasses will derive from this 
        return
    @abstractmethod
    def obs_fun(self, t, x):
        # We'll want to save out various observable functions of interest for post-analysis
        # TODO start out with some mandatory observables, like time horizons, for easy metadata analysis
        return
    @abstractmethod
    def generate_next_perturbation(self):
        # This depends on the kind of ODE and the type of sampling we want to do 
        pass
    def take_first_step(self, saveinfo):
        # The ensemble should be empty
        assert self.ens.memgraph.number_of_nodes() == 0
        icandf = self.ens.dynsys.generate_default_icandf(0,self.trunk_duration)
        ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=None)
        # Increment the tracker for the current branch etc
        self.next_branch_point = 0
        self.next_branch = 0 # The local index within the bunch
        self.next_branch_time = self.ens.dynsys.t_burnin
        self.terminable = False
        return
    def take_next_step(self, saveinfo):
        if self.next_branch_point >= self.num_branch_points:
            self.terminable = True
            return
        if self.next_branch >= self.branches_per_point:


    

