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
        self.branches_per_group = config['branches_per_group'] # How many different members to spawn from the same initial condition
        tu = self.ens.dynsys.dt_save
        # How long can each member run for, out of storage considerations? 
        self.max_member_duration = int(config['max_member_duration_phys']/tu)
        self.interbranch_interval = int(config['interbranch_interval_phys']/tu) # How long to wait between consecutive splits
        self.branch_duration = int(config['branch_duration_phys']/tu) # How long to run each branch
        self.num_branch_groups = config['num_branch_groups'] # but include the possibility for extension
        self.trunk_duration = self.ens.dynsys.t_burnin + self.interbranch_interval * (self.num_branch_groups) + self.branch_duration
        print(f'{self.trunk_duration = }')
        # Most likely all subclasses will derive from this 
        self.obs_dict = dict({key: [] for key in self.obs_dict_names()})

        self.init_cond = None
        self.init_time = 0
        return
    def set_init_cond(self, init_time, init_cond):
        self.init_time = init_time
        self.init_cond = init_cond
        return
    @staticmethod
    def label_from_config(config):
        abbrv_population = (
                r"bpg%d_ibi%.1f_bd%.1f_mmd%.1f"%(
                    config["branches_per_group"],
                    config["interbranch_interval_phys"],
                    config["branch_duration_phys"],
                    config['max_member_duration_phys']
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
        pass
    @abstractmethod
    def generate_icandf_from_parent(self, parent, branch_time, duration):
        pass
    def append_obs_dict(self, obs_dict_new):
        for name in self.obs_dict_names():
            self.obs_dict[name].append(obs_dict_new[name])
        return 
    def plot_obs_spaghetti(self, obs_name, branch_group_subset=None):
        if branch_group_subset is None:
            branch_group_subset = np.arange(self.branching_state['next_branch_group']) # TODO include the branch point of each child as an observable function
        tu = self.ens.dynsys.dt_save
        fig,ax = plt.subplots(figsize=(12,5))
        for bp in branch_group_subset:
            child_subset = 1 + np.arange(bp * self.branches_per_group, (bp+1) * self.branches_per_group)
            for child in child_subset:
                pass

        return
    def take_next_step(self, saveinfo):
        if self.terminate:
            return
        if self.ens.memgraph.number_of_nodes() == 0:
            # Initialize the state of the branching algorithm
            # Assume that a branch duration is no longer than max_mem_duration
            self.branching_state = dict({
                'next_branch_group': 0,
                'next_branch': 0,
                'next_branch_time': self.init_time + self.ens.dynsys.t_burnin,
                'trunk_lineage': [],
                'trunk_lineage_init_times': [],
                'trunk_lineage_fin_times': [],
                })
            duration = min(self.trunk_duration, self.max_member_duration)
            branching_state_update = dict({
                'trunk_lineage': [0],
                'trunk_lineage_init_times': [self.init_time],
                'trunk_lineage_fin_times': [self.init_time + duration],
                })

            # Keep track of trunk length
            parent = None
            icandf = self.ens.dynsys.generate_default_icandf(self.init_time,self.init_time+duration)
            if self.init_cond is not None:
                icandf['init_cond'] = self.init_cond
        elif self.branching_state['trunk_lineage_fin_times'][-1] < self.init_time + self.trunk_duration: # TODO make this more flexible; we could start branching as soon as the burnin time is exceeded
            print(f'{self.branching_state = }')
            print(f'{self.ens.root_dir = }')
            print(f'{self.ens.memgraph.number_of_nodes() = }')
            parent = self.branching_state['trunk_lineage'][-1]
            parent_init_time,parent_fin_time = self.ens.get_member_timespan(parent)
            print(f'{parent_init_time = }, {parent_fin_time = }')
            duration = min(self.max_member_duration, self.init_time+self.trunk_duration-parent_fin_time)
            print(f'{duration = }')
            icandf = self.generate_icandf_from_parent(parent, parent_fin_time, duration)
            print(f'{icandf = }')
            branching_state_update = dict({
                'trunk_lineage': self.branching_state['trunk_lineage'] + [self.ens.memgraph.number_of_nodes()],
                'trunk_lineage_init_times': self.branching_state['trunk_lineage_init_times'] + [parent_fin_time],
                'trunk_lineage_fin_times': self.branching_state['trunk_lineage_fin_times'] + [parent_fin_time + duration],
                })
        else:
            # decide whom to branch off of 
            trunk_segment_2branch = np.searchsorted(self.branching_state['trunk_lineage_fin_times'], self.branching_state['next_branch_time'], side='left')
            print(f'{self.branching_state = }')
            print(f'{trunk_segment_2branch = }')
            parent = self.branching_state['trunk_lineage'][trunk_segment_2branch]
            icandf = self.generate_icandf_from_parent(parent, self.branching_state['next_branch_time'], self.branch_duration)
            branching_state_update = dict()
            if self.branching_state['next_branch'] < self.branches_per_group - 1:
                branching_state_update['next_branch'] = self.branching_state['next_branch'] + 1
            elif self.branching_state['next_branch_group'] < self.num_branch_groups - 1:
                branching_state_update['next_branch_group'] = self.branching_state['next_branch_group'] + 1
                branching_state_update['next_branch_time'] = self.branching_state['next_branch_time'] + self.interbranch_interval
                branching_state_update['next_branch'] = 0
                self.rng = default_rng(self.seed_init) # Every new branch point will receive the same sequence of random numbers
            else:
                self.terminate = True
        obs_dict_new = self.ens.branch_or_plant(icandf, self.obs_fun, saveinfo, parent=parent)
        self.append_obs_dict(obs_dict_new)
        self.branching_state.update(branching_state_update)
        return
        
class ODEPeriodicBranching(PeriodicBranching):
    # where the system of interest is an ODE
    def generate_icandf_from_parent(self, parent, branch_time, duration):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent < branch_time <= fin_time_parent
        parent_t,parent_x = self.ens.dynsys.load_trajectory(self.ens.traj_metadata[parent], self.ens.root_dir, tspan=[branch_time]*2)
        impulse = self.rng.normal(size=self.ens.dynsys.impulse_dim)
        icandf = dict({
            'init_cond': parent_x[0],
            'frc': forcing.ImpulsiveForcing([branch_time], [impulse], branch_time+duration)
            })
        return icandf



class SDEPeriodicBranching(PeriodicBranching):
    # where the system of interest is an SDE driven by white noise
    def generate_icandf_from_parent(self, parent, branch_time, duration):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent < branch_time <= fin_time_parent
        parent_t,parent_x = self.ens.dynsys.load_trajectory(self.ens.traj_metadata[parent], self.ens.root_dir, tspan=[branch_time]*2)
        seed = self.rng.integers(low=self.seed_min,high=self.seed_max)
        frc_imp = forcing.ImpulsiveForcing([branch_time], [np.zeros(self.ens.dynsys.ode.impulse_dim)], branch_time+self.branch_duration)
        frc_white = forcing.WhiteNoiseForcing([branch_time], [seed], branch_time+self.branch_duration)
        icandf = dict({
            'init_cond': parent_x[0],
            'frc': forcing.SuperposedForcing([frc_imp,frc_white]),
            })
        return icandf

# TODO analysis of spreading rates


    

