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
from ensemble import Ensemble
import forcing



class EnsembleAlgorithm(ABC):
    def __init__(self, config, ens, seed):
        self.ens = ens
        self.seed_init = seed
        self.rng = default_rng(self.seed_init) 
        self.terminate = False
        self.derive_parameters(config)
        return
    @staticmethod
    @abstractmethod
    def label_from_config(config):
        pass
    @abstractmethod
    def derive_parameters(self, config):
        pass
    @abstractmethod
    def take_next_step(self):
        # Based on the current state of the ensemble, provide the arguments (icandf, obs_fun, parent) to give to ens.branch_or_plant. Don't modify ens right here.
        pass

# TODO make a global acquisition algorithm and a local acquisition algorithm, for some higher-level algorithm to manage in tandem

class PeriodicBranching(EnsembleAlgorithm):
    def derive_parameters(self, config):
        self.seed_min,self.seed_max = config['seed_min'],config['seed_max']
        # Determine branching number
        self.branches_per_point = config['branches_per_point'] # How many different members to spawn from the same initial condition
        tu = self.ens.dynsys.dt_save
        self.interbranch_interval = int(config['interbranch_interval_phys']/tu) # How long to wait between consecutive splits
        self.branch_duration = int(config['branch_duration_phys']/tu) # How long to run each branch
        self.num_branch_points = config['num_branch_points'] # but include the possibility for extension
        self.trunk_duration = self.ens.dynsys.t_burnin + self.interbranch_interval * self.num_branch_points
        # Most likely all subclasses will derive from this 
        self.obs_dict = dict({key: [] for key in self.obs_dict_names()})
        return
    @staticmethod
    def label_from_config(config):
        abbrv_population = (
                r"bpp%d_ibi%.1f_bd%.1f"%(
                    config["branches_per_point"],
                    config["interbranch_interval_phys"],
                    config["branch_duration_phys"]
                    )
                ).replace(".","p")
        abbrv = '_'.join([
            'PeBr',
            abbrv_population,
            ])
        label = 'Periodic branching'
        return abbrv,label
    @abstractmethod
    def obs_dict_names(self):
        pass
    @abstractmethod
    def obs_fun(self, t, x):
        # We'll want to save out various observable functions of interest for post-analysis
        # TODO start out with some mandatory observables, like time horizons, for easy metadata analysis
        # Must return a dictionary
        return
    def append_obs_dict(self, obs_dict_new):
        for name in self.obs_dict_names():
            self.obs_dict[name].append(obs_dict_new[name])
        return 
    def plot_obs_spaghetti(self, obs_name, branch_point_subset=None):
        if branch_point_subset is None:
            branch_point_subset = np.arange(self.next_branch_point) # TODO include the branch point of each child as an observable function
        tu = self.ens.dynsys.dt_save
        fig,ax = plt.subplots(figsize=(12,5))
        for bp in branch_point_subset:
            child_subset = 1 + np.arange(bp * self.branches_per_point, (bp+1) * self.branches_per_point)
            for child in child_subset:
                pass

        return


    @abstractmethod
    def generate_next_icandf(self):
        # This depends on the kind of ODE and the type of sampling we want to do 
        pass
    def take_next_step(self, saveinfo):
        if self.ens.memgraph.number_of_nodes() == 0:
            icandf = self.ens.dynsys.generate_default_icandf(0,self.trunk_duration)
            parent = None
            # Increment the tracker for the current branch etc
            self.next_branch_point = 0
            self.next_branch = 0 # The local index within the bunch
            self.next_branch_time = self.ens.dynsys.t_burnin
            self.terminate = (self.num_branch_points <= 0) or (self.branches_per_point <= 0) # AFTER the current round
        else:
            icandf = self.generate_next_icandf()
            parent = 0 # TODO: if the main trunk has a chain of members, choose the parent whos branching time is appropriate
            if self.next_branch < self.branches_per_point - 1:
                self.next_branch += 1
            elif self.next_branch_point < self.num_branch_points - 1:
                self.next_branch_point += 1
                self.next_branch_time += self.interbranch_interval
                self.next_branch = 0
                self.rng = default_rng(self.seed_init) # Every new branch point will receive the same sequence of random numbers
            else:
                self.terminate = True
        obs_dict_new = self.ens.branch_or_plant(icandf, self.obs_fun, saveinfo, parent=parent)
        self.append_obs_dict(obs_dict_new)
        return
        
class ODEPeriodicBranching(PeriodicBranching):
    # where the system of interest is an ODE
    def generate_next_icandf(self):
        # Determine the initial time
        parent = 0
        init_time = self.next_branch_time
        parent_t,parent_x = self.ens.dynsys.load_trajectory(self.ens.traj_metadata[parent], tspan=[init_time]*2)
        print(f"{parent_t = }, {parent_x = }")
        impulse = self.rng.normal(size=self.ens.dynsys.impulse_dim)
        icandf = dict({
            'init_cond': parent_x[0],
            'frc': forcing.ImpulsiveForcing([init_time], [impulse], init_time+self.branch_duration-1)
            })
        return icandf

class SDEPeriodicBranching(PeriodicBranching):
    # where the system of interest is an SDE driven by white noise
    def generate_next_icandf(self):
        # Determine the initial time
        parent = 0
        init_time = self.next_branch_time
        parent_t,parent_x = self.ens.dynsys.load_trajectory(self.ens.traj_metadata[parent], tspan=[init_time]*2)
        print(f"{parent_t = }, {parent_x = }")
        seed = self.rng.integers(low=self.seed_min,high=self.seed_max)
        frc_imp = forcing.ImpulsiveForcing([init_time], [np.zeros(self.ens.dynsys.ode.impulse_dim)], init_time+self.branch_duration-1)
        frc_white = forcing.WhiteNoiseForcing([init_time], [seed], init_time+self.branch_duration-1)
        icandf = dict({
            'init_cond': parent_x[0],
            'frc': forcing.SuperposedForcing([frc_imp,frc_white]),
            })
        return icandf

# TODO analysis of spreading rates


    

