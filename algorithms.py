from abc import ABC, abstractmethod
import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from scipy.stats import linregress 
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
    # --------------- Post-analysis functions --------------------
    def get_tree_subset(self, branch_group):
        all_init_times,all_fin_times = self.ens.get_all_timespans()
        print(f'\n{all_init_times = }')
        split_time = self.init_time + self.ens.dynsys.t_burnin + branch_group*self.interbranch_interval
        print(f'{split_time = }')
        nmem = self.ens.get_nmem()
        mems_nontrunk = np.setdiff1d(range(nmem), self.branching_state['trunk_lineage'])
        mems_branch = [mem for mem in mems_nontrunk if self.branch_times[mem] == split_time]
        #mems_branch = len(self.branching_state['trunk_lineage']) + branch_group*self.branches_per_group + np.arange(self.branches_per_group)
        tidxs_branch = []
        for i_mem,mem in enumerate(mems_branch):
            print(f'{mem = }, {next(self.ens.memgraph.predecessors(mem)) = }')
            print(f'{self.ens.traj_metadata[mem]["icandf"]["init_cond"] = }')
            tidxs_branch.append(split_time - all_init_times[mem] + np.arange(self.branch_duration))

        i_mem_trunk_init = np.searchsorted(self.branching_state['trunk_lineage_init_times'], split_time, side='right') - 1
        i_mem_trunk_fin = np.searchsorted(self.branching_state['trunk_lineage_fin_times'], split_time+self.branch_duration, side='right')
        mems_trunk = self.branching_state['trunk_lineage'][i_mem_trunk_init:i_mem_trunk_fin+1]
        time = split_time + np.arange(self.branch_duration, dtype=int)
        tidx_trunk = split_time - all_init_times[mems_trunk[0]] + np.arange(self.branch_duration)
        return time,mems_trunk,tidx_trunk,mems_branch,tidxs_branch
    def compute_pairwise_funs_local(self, pair_funs, branch_group):
        time,mems_trunk,tidx_trunk,mems_branch,tidxs_branch = self.get_tree_subset(branch_group)
        pair_list = []
        for mem0 in mems_trunk:
            pair_list.append(self.ens.compute_pairwise_observables(pair_funs, mem0, mems_branch))
        pairs = dict()
        pair_names = list(pair_funs.keys())
        for pair_name in pair_names:
            pairs[pair_name] = np.zeros((len(mems_branch), len(time)))
            for i_mem1,mem1 in enumerate(mems_branch):
                print(f'{pair_list[0][pair_name][i_mem1] = }')
                pairs[pair_name][i_mem1,:] = np.concatenate([d[pair_name][i_mem1] for d in pair_list])
        return time,pairs
    def analyze_pert_growth(self, pert_growth_dict):
        split_times,rmses,rmsds,dists = [pert_growth_dict[key] for key in ['split_times','rmses','rmsds','dists']]
        dist_names = list(rmses.keys())
        ngroups,nbranches,ntimes = dists[dist_names[0]].shape
        lyapunov_exponents = dict()
        for dist_name in dist_names:
            lyapunov_exponents[dist_name] = np.zeros(ngroups)
            for group in range(ngroups):
                tmax = np.where(rmses[dist_name][group] >= 0.25*rmsds[dist_name])[0][0]
                linmod = linregress(np.arange(tmax), np.log(rmses[dist_name][group][:tmax]))
                lyapunov_exponents[dist_name][group] = linmod.slope/self.ens.dynsys.dt_save
        return lyapunov_exponents
    def measure_pert_growth(self, dist_funs):
        # Save a statistical analysis of RMSE growth to a specified directory
        dist_names = list(dist_funs['tdep'].keys())
        ngroups = self.branching_state['next_branch_group']+1
        split_times = np.zeros(ngroups, dtype=int)
        print(f'{split_times = }')
        dists = dict({dist_name: np.zeros((ngroups, self.branches_per_group, self.branch_duration)) for dist_name in dist_names})
        rmses = dict({dist_name: np.zeros((ngroups, self.branch_duration)) for dist_name in dist_names})
        for branch_group in range(ngroups):
            print(f'About to compute distances for {branch_group = }')
            time,dists_local = self.compute_pairwise_funs_local(dist_funs['tdep'], branch_group)
            split_times[branch_group] = time[0]
            rmse_local = dict({key: np.sqrt(np.mean(val**2, axis=0)) for (key,val) in dists_local.items()})
            for dist_name in dist_names:
                dists[dist_name][branch_group,:,:] = dists_local[dist_name].copy()
                rmses[dist_name][branch_group,:] = np.sqrt(np.mean(dists_local[dist_name]**2, axis=0))
        # Compute RMSD, using only the last two branch members
        mems_trunk = self.branching_state['trunk_lineage']
        if len(mems_trunk) == 1:
            mems_rmsd = [len(mems_trunk)-1]*2
        else:
            mems_rmsd = [len(mems_trunk)-i for i in [1,2]]
        rmsds = self.ens.compute_pairwise_observables(dist_funs['rmsd'], mems_rmsd[0], mems_rmsd[1:])
        for dist_name in dist_names:
            rmsds[dist_name] = np.mean(rmsds[dist_name])
        # Also compute other summary stats besides RMSE, like Lyapunov exponents and changeover times from exponential to diffusive growth. 
        pert_growth = dict({'split_times': split_times, 'rmses': rmses, 'rmsds': rmsds, 'dists': dists})
        return pert_growth

    def plot_pert_growth(self, pert_growth_dict, lyap_dict, fndict, labels=None, abbrvs=None, logscale=True):
        split_times,rmses,rmsds,dists = [pert_growth_dict[key] for key in ['split_times','rmses','rmsds','dists']]

        dist_names = list(rmses.keys())
        ngroups,nbranches,ntimes = dists[dist_names[0]].shape
        tu = self.ens.dynsys.dt_save
        if labels is None: labels = dict({dist_name: f'Dist(CTRL,PERT) ({dist_name})' for dist_name in dist_names})
        if abbrvs is None: abbrvs = dict({dist_name: dist_name for dist_name in dist_names})
        for branch_group in range(ngroups):
            time = split_times[branch_group] + np.arange(rmses[dist_names[0]].shape[1])
            for dist_name in dist_names:
                fig,ax = plt.subplots(figsize=(12,5))
                for i_mem1 in range(nbranches):
                    ax.plot(time*tu, dists[dist_name][branch_group,i_mem1,:], color='tomato',)
                hrmse, = ax.plot(time*tu, rmses[dist_name][branch_group,:], color='black', label='RMSE')
                hrmsd = ax.axhline(rmsds[dist_name],label='RMSD', color='black', linestyle='--')
                hlyap, = ax.plot(time*tu, np.minimum(rmsds[dist_name], rmses[dist_name][branch_group,0]*np.exp(lyap_dict[dist_name][branch_group]*(time-time[0])*tu)), color='dodgerblue')
                ax.legend(handles=[hrmse,hrmsd,hlyap])
                ax.set_ylabel(labels[dist_name])
                ax.set_xlabel('time')
                if logscale: ax.set_yscale('log')
                fig.savefig(fndict[dist_name][branch_group], **pltkwargs)
                plt.close(fig)
        for dist_name in dist_names:
            fig,ax = plt.subplots()
            ax.plot(split_times,lyap_dict[dist_name],color='black',marker='o')
            ax.set_xlabel('split_time')
            ax.set_ylabel('Lyapunov exponent')
            fig.savefig(fndict[dist_name]['lyap_exp'], **pltkwargs)
            plt.close(fig)
        # TODO
        # 1. Aggregate the RMSE into one plot to measure error growth averaged and separately 
        time = np.arange(ntimes)
        for dist_name in dist_names:
            fig,ax = plt.subplots()
            for branch_group in range(ngroups):
                ax.plot(time, rmses[dist_name][branch_group,:], color=plt.cm.rainbow(branch_group/(ngroups-1)))
            ax.axhline(rmsds[dist_name], color='black', linestyle='--')
            ax.set_xlabel('time')
            if logscale: ax.set_yscale('log')
            fig.savefig(fndict[dist_name]['rmse'], **pltkwargs)
            plt.close(fig)
        # 2. Make this a classmethod
        return
    def plot_obs_spaghetti(self, obs_funs, branch_group, plotdir, ylabels=None, titles=None, abbrvs=None):
        print(f'\n\nPlotting group {branch_group}')
        obs_names = list(obs_funs.keys())
        if titles is None: titles = dict({obs_name: '' for obs_name in obs_names})
        if ylabels is None: ylabels = dict({obs_name: '' for obs_name in obs_names})
        if abbrvs is None: abbrvs = dict({obs_name: obs_name for obs_name in obs_names})
        # Plot all the observables from a single group of branches, along with the control 
        # TODO recover split times from init_times. Better yet, track it in the algorithm to begin with
        # Get all timespans
        time,mems_trunk,tidx_trunk,mems_branch,tidxs_branch = self.get_tree_subset(branch_group)
        print(f'{mems_branch = }')
        tu = self.ens.dynsys.dt_save

        obs_dict_branch = self.ens.compute_observables(obs_funs, mems_branch)
        obs_dict_trunk = self.ens.compute_observables(obs_funs, mems_trunk)

        for obs_name in obs_names:
            print(f'============== Plotting observable {obs_name} ============= ')
            fig,ax = plt.subplots(figsize=(12,5))
            # For trunk, restrict to the times of interest
            hctrl, = ax.plot(time*tu, np.concatenate(obs_dict_trunk[obs_name])[tidx_trunk], linestyle='--', color='black', linewidth=2, zorder=1, label='CTRL')
            for i_mem,mem in enumerate(mems_branch):
                print(f'{mem = }, {next(self.ens.memgraph.predecessors(mem)) = }')
                print(f'{self.ens.traj_metadata[mem]["icandf"]["init_cond"] = }')
                hpert, = ax.plot(time*tu, obs_dict_branch[obs_name][i_mem][tidxs_branch[i_mem]], linestyle='-', color='tomato', linewidth=1, zorder=0, label='PERT')
            #ax.axvline(split_time*tu, color='tomato')
            ax.legend(handles=[hctrl,hpert])
            ax.set_xlabel('time')
            ax.set_ylabel(ylabels[obs_name])
            ax.set_title(titles[obs_name])
            #ax.set_xlim([time[0],time[-1]+1])
            fig.savefig(join(plotdir,r'%s_group%d.png'%(abbrvs[obs_name],branch_group)), **pltkwargs)
            plt.close(fig)
        return
    def take_next_step(self, saveinfo):
        if self.terminate:
            return
        if self.ens.get_nmem() == 0:
            # Initialize the state of the branching algorithm
            # Initialize a list of splitting times for each member
            self.branch_times = []
            # Assume that a branch duration is no longer than max_mem_duration
            self.branching_state = dict({
                'on_trunk': True,
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
            if duration >= self.trunk_duration:
                branching_state_update['on_trunk'] = False
            branch_times_update = self.init_time

            # Keep track of trunk length
            parent = None
            icandf = self.ens.dynsys.generate_default_icandf(self.init_time,self.init_time+duration)
            print(f'In first if stateemnt: {icandf["init_cond"] = }')
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
            if parent_fin_time + duration >= self.init_time + self.trunk_duration:
                branching_state_update['on_trunk'] = False
            branch_times_update = parent_fin_time
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
                self.rng = default_rng(self.seed_init+1) # Every new branch point will receive the same sequence of random numbers
            else:
                self.terminate = True
            branch_times_update = self.branching_state['next_branch_time']
        # --------------- Run the trajectory -----------
        obs_dict_new = self.ens.branch_or_plant(icandf, self.obs_fun, saveinfo, parent=parent)
        # ----------------------------------------------
        self.append_obs_dict(obs_dict_new)
        self.branching_state.update(branching_state_update)
        self.branch_times.append(branch_times_update)
        return
        
class ODEPeriodicBranching(PeriodicBranching):
    # where the system of interest is an ODE
    def generate_icandf_from_parent(self, parent, branch_time, duration):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent <= branch_time <= fin_time_parent
        if branch_time == init_time_parent:
            init_cond = self.ens.traj_metadata[parent]['icandf']['init_cond']
        else:
            parent_t,parent_x = self.ens.dynsys.load_trajectory(self.ens.traj_metadata[parent], self.ens.root_dir, tspan=[branch_time]*2)
            init_cond = parent_x[0]
        if self.branching_state['on_trunk']:
            impulse = np.zeros(self.ens.dynsys.impulse_dim)
        else:
            impulse = self.rng.normal(size=self.ens.dynsys.impulse_dim)
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.OccasionalVectorForcing(branch_time, branch_time+duration, [branch_time], [impulse])
            })
        return icandf



class SDEPeriodicBranching(PeriodicBranching):
    # where the system of interest is an SDE driven by white noise
    def generate_icandf_from_parent(self, parent, branch_time, duration):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent <= branch_time <= fin_time_parent
        mdp = self.ens.traj_metadata[parent]
        if branch_time == init_time_parent:
            init_cond = mdp['icandf']['init_cond']
        else:
            parent_t,parent_x = self.ens.dynsys.load_trajectory(mdp, self.ens.root_dir, tspan=[branch_time]*2)
            init_cond = parent_x[0]
        if self.branching_state['on_trunk']:
            init_rngstate = mdp['fin_rngstate']
            frc_reseed = forcing.OccasionalReseedForcing(branch_time, branch_time+duration, [], [])
        else:
            seed = self.rng.integers(low=self.seed_min,high=self.seed_max)
            init_rngstate = default_rng(seed=seed).bit_generator.state 
            frc_reseed = forcing.OccasionalReseedForcing(branch_time, branch_time+duration, [branch_time], [seed])
        frc_vector = forcing.OccasionalVectorForcing(branch_time, branch_time+duration, [], [])
        icandf = dict({
            'init_cond': parent_x[0],
            'init_rngstate': init_rngstate,
            'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
            })
        return icandf



    

