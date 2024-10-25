from abc import ABC, abstractmethod, abstractclassmethod
from collections import deque
import pprint
import numpy as np
from numpy.random import default_rng
import pickle
import networkx as nx
from scipy import sparse as sps
from scipy.stats import linregress 
from scipy.special import logsumexp, softmax
from os.path import join, exists
from os import makedirs
import sys
import copy as copylib
from matplotlib import pyplot as plt, rcParams
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from ensemble import Ensemble
import forcing
import utils



class EnsembleAlgorithm(ABC):
    def __init__(self, config, ens):
        self.ens = ens
        self.seed_min,self.seed_max = config['seed_min'],config['seed_max']
        self.seed_init = config['seed_min'] + config['seed_inc_init'] 
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

class DirectNumericalSimulation(EnsembleAlgorithm):
    # This is unfinished and maybe should never be used. 
    @staticmethod
    def label_from_config(config):
        abbrv = r'DNS_si%d'%(config['seed_inc_init'])
        label = r'DNS ($\delta$seed %g)'%(config['seed_inc_init'])
        return abbrv,label
    def derive_parameters(self, config):
        tu = self.ens.dynsys.dt_save
        # The following size parameters should be adjustable
        self.max_member_duration = int(config['max_member_duration_phys']/tu)
        self.num_chunks_max = config['num_chunks_max']
        self.init_cond = None
        self.init_time = 0
        return
    @abstractmethod
    def generate_icandf_from_parent(self, parent):
        pass
    def set_init_cond(self, init_time, init_cond):
        self.init_time = init_time
        self.init_cond = init_cond
        return
    def set_capacity(self, num_chunks_max, max_member_duration_phys):
        num_new_chunks = num_chunks_max - self.ens.get_nmem()
        if num_new_chunks > 0:
            self.terminate = False
        self.max_member_duration = int(max_member_duration_phys/self.ens.dynsys.dt_save)
        self.num_chunks_max = num_chunks_max
        return
    def take_next_step(self, saveinfo):
        nmem = self.ens.get_nmem()
        self.terminate = (self.terminate or (nmem >= self.num_chunks_max))
        if self.terminate:
            return
        if nmem == 0:
            parent = None
            icandf = self.ens.dynsys.generate_default_icandf(self.init_time,self.init_time+self.max_member_duration)
        else:
            parent = nmem-1
            icandf = self.generate_icandf_from_parent(parent)
        obs_fun = lambda t,x: None
        self.ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        return
    # ---------------- Quantitative analysis ----------------------------------------
    def get_member_subset(self, tspan): # returned member timespan excludes tspan[0] by convention
        all_starts,all_ends = self.ens.get_all_timespans()
        first_mem = np.where(all_starts <= tspan[0])[0][-1]
        print(f'{all_starts,all_ends = }')
        if all_ends[-1] >= tspan[1]:
            last_mem = np.where(all_ends >= tspan[1])[0][0]
        else:
            last_mem = len(all_ends)-1
        print(f'{last_mem = }')
        time = 1 + np.arange(tspan[0],min(all_ends[-1],tspan[1]))
        tidx = time - all_starts[first_mem] - 1
        memset = np.arange(first_mem,last_mem+1)
        return time,memset,tidx
    def compute_return_stats(self, obs_funs2concat, time_block_size, spinup, outfile):
        nmem = self.ens.get_nmem()
        init_time = spinup
        fin_time = init_time + time_block_size * int((self.ens.get_member_timespan(nmem-1)[1] - init_time) / time_block_size)
        tspan = [init_time, fin_time]
        time,memset,tidx = self.get_member_subset(tspan)
        #obs = np.concatenate(tuple(self.ens.compute_obser
        f = [[] for fun in obs_funs2concat]
        for mem in memset:
            f_mem = self.ens.compute_observables(obs_funs2concat, mem)
            for i_fun,fun in enumerate(obs_funs2concat):
                f[i_fun].append(f_mem[i_fun])
        fconcat = np.concatenate(tuple(
            np.concatenate(tuple(f[i_fun][i_mem] for i_mem in range(len(memset))))
            for i_fun in range(len(obs_funs2concat))))
        bin_lows,hist,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale = utils.compute_returnstats_and_histogram(fconcat, time_block_size)
        np.savez(
                outfile, 
                bin_lows=bin_lows,
                hist=hist,
                rtime=rtime,
                logsf=logsf,
                rtime_gev=rtime_gev,
                logsf_gev=logsf_gev,
                shape=shape,
                loc=loc,
                scale=scale,
                )
        if rtime[-1] == rtime[-2]:
            print(f'{hist = }')
            print(f'{rtime = }')
        return
     
    # ------------------ Plotting -----------------------------
    def plot_obs_segment(self, obs_fun, tspan, fig, ax, **linekwargs):
        time,memset,tidx = self.get_member_subset(tspan)
        print(f'{memset = }')
        tu = self.ens.dynsys.dt_save
        obs_seg = np.concatenate(tuple(self.ens.compute_observables([obs_fun], mem)[0] for mem in memset))[tidx]
        h, = ax.plot(time*tu, obs_seg, **linekwargs)
        return h
    @classmethod
    def plot_return_curves(cls, return_stats_filename, fig, ax, **linekwargs):
        rst = np.load(return_stats_filename)
        rtime = rst['rtime']
        rlev = rst['bin_lows']
        h, = ax.plot(rtime,rlev,**linekwargs)
        ax.set_ylim([rlev[np.argmax(rtime>0)],2*rlev[-1]-rlev[-2]])
        ax.set_xlabel('Return time')
        ax.set_ylabel('Return level')
        ax.set_xscale('log')
        return h
    @classmethod
    def plot_histogram(cls, return_stats_filename, fig, ax, orientation='vertical', **linekwargs):
        rst = np.load(return_stats_filename)
        bin_lows = rst['bin_lows']
        hist = rst['hist']
        if orientation == 'vertical':
            h, = ax.plot(bin_lows,hist,**linekwargs)
            ax.set_yscale('log')
        else:
            h, = ax.plot(hist[hist>0],bin_lows[hist>0],**linekwargs)
            ax.set_xscale('log')
        return h 





class PeriodicBranching(EnsembleAlgorithm):
    def derive_parameters(self, config):
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
                r"bpg%d_ibi%.1f_bd%.1f_mmd%.1f_si%d"%(
                    config["branches_per_group"],
                    config["interbranch_interval_phys"],
                    config["branch_duration_phys"],
                    config['max_member_duration_phys'],
                    config['seed_inc_init']
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
    def generate_icandf_from_parent(self, parent, branch_time):
        pass
    def append_obs_dict(self, obs_dict_new):
        for name in self.obs_dict_names():
            self.obs_dict[name].append(obs_dict_new[name])
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
            print(f'self.branching_state = ')
            pprint.pprint(self.branching_state)
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
            print(f'self.branching_state = ')
            pprint.pprint(self.branching_state)
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
    # --------------- Post-analysis functions --------------------
    # Utility functions for collecting all trajectories (branches) from a particular branch group
    def get_tree_subset(self, branch_group):
        all_init_times,all_fin_times = self.ens.get_all_timespans()
        split_time = self.init_time + self.ens.dynsys.t_burnin + branch_group*self.interbranch_interval
        nmem = self.ens.get_nmem()
        mems_nontrunk = np.setdiff1d(range(nmem), self.branching_state['trunk_lineage'])
        mems_branch = [mem for mem in mems_nontrunk if self.branch_times[mem] == split_time]
        #mems_branch = len(self.branching_state['trunk_lineage']) + branch_group*self.branches_per_group + np.arange(self.branches_per_group)
        tidxs_branch = []
        for i_mem,mem in enumerate(mems_branch):
            tidxs_branch.append(split_time - all_init_times[mem] + np.arange(self.branch_duration))

        i_mem_trunk_init = np.searchsorted(self.branching_state['trunk_lineage_init_times'], split_time, side='right') - 1
        i_mem_trunk_fin = np.searchsorted(self.branching_state['trunk_lineage_fin_times'], split_time+self.branch_duration, side='left')
        mems_trunk = self.branching_state['trunk_lineage'][i_mem_trunk_init:i_mem_trunk_fin+1]
        time = 1 + split_time + np.arange(self.branch_duration, dtype=int)
        tidx_trunk = split_time - all_init_times[mems_trunk[0]] + np.arange(self.branch_duration)
        return time,mems_trunk,tidx_trunk,mems_branch,tidxs_branch
    def compute_pairwise_fun_local(self, pairwise_fun, branch_group):
        # These should be time-dependent functions
        time,mems_trunk,tidx_trunk,mems_branch,tidxs_branch = self.get_tree_subset(branch_group)
        print(f'{branch_group = }')
        print(f'{mems_trunk = }')
        print(f'{mems_branch = }')
        print(f'{time[[0,1,-2,-1]] = }')
        pairwise_fun_vals_list = []
        for mem0 in mems_trunk:
            pairwise_fun_vals_list.append(
                    self.ens.compute_pairwise_observables([pairwise_fun], mem0, mems_branch)[0])
        # Repackage into the right shape
        pairwise_fun_vals_array = np.zeros((len(mems_branch), len(time)))
        for i_mem1,mem1 in enumerate(mems_branch):
            pairwise_fun_vals_array[i_mem1,:] = np.concatenate(tuple(d[i_mem1] for d in pairwise_fun_vals_list))[tidx_trunk]
        if np.any(np.isnan(pairwise_fun_vals_array)):
            raise Exception(f'{np.mean(np.isnan(pairwise_fun_vals_array), axis=0) = }')
        return time,pairwise_fun_vals_array
    # ************** Dispersion characteristics ****************
    def measure_running_max(self, obs_fun, runmax_file, figfile_prefix, label='', abbrv=''):
        ngroups = self.branching_state['next_branch_group']+1
        split_times = np.zeros(ngroups, dtype=int)
        tu = self.ens.dynsys.dt_save
        print(f'{split_times = }')
        running_max_branch = np.zeros((ngroups, self.branches_per_group, self.branch_duration)) 
        running_max_trunk = np.zeros((ngroups, self.branch_duration)) 
        for group in range(ngroups):
            print(f'About to compute running maxes for {group = }')
            time,mems_trunk,tidx_trunk,mems_branch,tidxs_branch = self.get_tree_subset(group)

            obs_branch = np.array([self.ens.compute_observables([obs_fun], mem)[0][tidxs_branch[i_mem]] for (i_mem,mem) in enumerate(mems_branch)])
            obs_trunk = np.concatenate(tuple(self.ens.compute_observables([obs_fun], mem)[0] for mem in mems_trunk))[tidx_trunk]
            split_times[group] = time[0] - 1
            running_max_branch[group,:,:] = np.maximum.accumulate(obs_branch, axis=1)
            running_max_trunk[group,:] = np.maximum.accumulate(obs_trunk)

        running_max_std = np.std(running_max_branch, axis=1)
        running_max_mean = np.mean(running_max_branch, axis=1)
        # TODO measure maxima in terms of return period. This might be a task for meta-analysis.
        runmax_stats = dict(
                split_times = split_times,
                running_max_branch = running_max_branch, 
                running_max_trunk = running_max_trunk, 
                running_max_mean = running_max_mean,
                running_max_std = running_max_std,
                )
        np.savez(runmax_file, **runmax_stats)
        # Plot 
        ngroups,nbranches,ntimes = running_max_branch.shape
        for group in range(ngroups):
            fig,axes = plt.subplots(nrows=2,figsize=(6,8),sharex=True)
            time = np.arange(ntimes)
            ax = axes[0]
            for branch in range(nbranches):
                ax.plot(time*tu, running_max_branch[group,branch], color='tomato')
            ax.plot(time*tu, running_max_mean[group,:], color='dodgerblue')
            ax.plot(time*tu, running_max_trunk[group,:], color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('')
            ax.xaxis.set_tick_params(which='both',labelbottom=True)
            ax.set_ylabel('Running maxes')
            ax.set_title(label)
            ax = axes[1]
            ax.plot(time*tu, running_max_std[group,:], color='dodgerblue')
            ax.set_xlabel(r'time since split (%g)'%(split_times[group]*tu))
            ax.set_ylabel('Std. of running max')
            fig.savefig(r'%s_bg%d.png'%(figfile_prefix,group), **pltkwargs)
            plt.close(fig)
        return

    def measure_dispersion(self, dist_fun, satfracs, outfile):
        # Measure the distance of every member from the control (according to a given function), as well as the fractional saturation times
        ngroups = self.branching_state['next_branch_group']+1
        split_times = np.zeros(ngroups, dtype=int)
        dists = np.zeros((ngroups, self.branches_per_group, self.branch_duration)) 
        for branch_group in range(ngroups):
            print(f'About to compute distances for {branch_group = }')
            time,dists[branch_group,:,:] = self.compute_pairwise_fun_local(dist_fun, branch_group)
            split_times[branch_group] = time[0] - 1
        print(f'{split_times = }')
        rmses = np.sqrt(np.mean(dists**2, axis=1))
        rmsd = np.sqrt(np.mean(rmses[:,-1]**2))
        # Finite-size Lyapunov analysis (at fixed fractions of saturation)
        nfracs = len(satfracs)
        fsle = np.nan*np.ones((ngroups,nfracs))
        elfs = np.nan*np.ones((ngroups,nfracs))
        diff_pows = np.nan*np.ones((ngroups,nfracs))
        for group in range(ngroups):
            log_rmse = np.log(rmses[group,:])
            i_time_prev = 0 
            for i_frac,frac in enumerate(satfracs):
                exceedances = np.where(rmses[group,:] >= frac*rmsd)[0]
                if len(exceedances) > 0:
                    elfs[group,i_frac] = exceedances[0]
                    # Measure diffusive growth
                    tidx = np.arange(i_time_prev, exceedances[0], dtype=int)
                    if len(tidx) > 0:
                        diff_pows[group,i_frac] = linregress(
                                np.log(time[tidx]-split_times[group]), 
                                np.log(rmses[group,tidx])
                                ).slope
                        fsle[group,i_frac] = linregress(
                                time[tidx]-split_times[group], np.log(rmses[group,tidx])
                                ).slope
                        i_time_prev = tidx[-1]
        dispersion_stats = dict(
                split_times = split_times,
                dists = dists,
                rmses = rmses,
                rmsd = rmsd,
                satfracs = satfracs, 
                elfs = elfs,
                fsle = fsle,
                diffusive_powers = diff_pows,
                )
        np.savez(outfile, **dispersion_stats)
        return dispersion_stats 
    @classmethod
    def analyze_pert_growth_meta(cls, pert_growth_list):
        # Compare characteristics of perturbation growth as a function of various independent variables
        fracs = np.array([1/8,1/4,3/8,1/2])
        nfrac = len(fracs)
        nexpt = len(pert_growth_list)
        dist_names = list(pert_growth_list[0]['rmsds'].keys())
        print(f'{dist_names = }')
        t2fracsat = dict() 
        for dist_name in dist_names:
            t2fracsat[dist_name] = np.nan*np.ones((nexpt,nfrac))
            for i_pg,pg in enumerate(pert_growth_list):
                rmse = pg['rmses'][dist_name] # dims (group,time)
                rmsd = pg['rmsds'][dist_name] 
                ngroups,ntimes = rmse.shape
                for i_frac,frac in enumerate(fracs):
                    t2fracsat[dist_name][i_pg,i_frac] = np.mean(np.argmax(rmse/rmsd > frac, axis=1))
        return fracs,t2fracsat
    def plot_dispersion(self, dispersion_metrics, figfile_prefix, groups2plot=None, ylabel='', title='', logscale=False):
        # TODO add in the fractional saturation times 
        tu = self.ens.dynsys.dt_save
        split_times = dispersion_metrics['split_times']
        dists = dispersion_metrics['dists']
        rmses = dispersion_metrics['rmses']
        rmsd = dispersion_metrics['rmsd']
        satfracs = dispersion_metrics['satfracs']
        elfs = dispersion_metrics['elfs']
        fsle = dispersion_metrics['fsle']
        diff_pows = dispersion_metrics['diffusive_powers']
        print(f'{rmses.max() = }, {rmsd = }')
        ngroups,nbranches,ntimes = dists.shape
        time = np.arange(ntimes) # local time 
        if groups2plot is None:
            groups2plot = np.arange(ngroups, dtype=int)
        for group in groups2plot:
            fig,ax = plt.subplots()
            for i_mem1 in range(nbranches):
                ax.plot(time*tu, dists[group,i_mem1,:], color='tomato',)
            hrmse, = ax.plot(time*tu, rmses[group,:], color='black', label='RMSE')
            ax.axhline(rmsd, color='black', linestyle='--', label='RMSD')
            # Exponential growth model
            i_time_prev = 0
            for i_sf,sf in enumerate(satfracs):
                tidx = np.arange(i_time_prev,elfs[group,i_sf], dtype=int)
                #hpow, = ax.plot(time[tidx]*tu, rmses[group][tidx[0]] * (time/time[tidx[0]])**diff_pows[group,i_sf], color='dodgerblue', label='diffusive')
                if len(tidx) > 0:
                    hexp, = ax.plot(
                            time[tidx]*tu, 
                            rmses[group,time[tidx[0]]] * np.exp(
                                (time[tidx]-time[tidx[0]]) * 
                                fsle[group,i_sf]), 
                            color='limegreen', label='exp. growth')
                    #ax.axvline((time[tidx[-1]]-time[0])*tu, color='black', linewidth=0.5)
                    ax.axhline(rmsd*satfracs[i_sf], color='black', linewidth=0.5)
                    i_time_prev = tidx[-1]
            ax.legend(handles=[hrmse,hexp])
            ax.set_xlabel(r'time since split (%g)'%(split_times[group]*tu))
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if logscale: ax.set_yscale('log')
            figfile = r'%s_group%d.png'%(figfile_prefix,group)
            fig.savefig(figfile, **pltkwargs)
            plt.close(fig)
        return 
    def plot_pert_growth(self, split_times, dists, thalfsat, diff_expons, lyap_expons, rmses, rmsd, plot_dir, plot_suffix, logscale=False):
        ngroups,nbranches,ntimes = dists.shape
        tu = self.ens.dynsys.dt_save
        for branch_group in range(min(3,ngroups)):
            time = split_times[branch_group] + np.arange(ntimes)
            print(f'{time = }')
            fig,ax = plt.subplots(figsize=(12,5))
            for i_mem1 in range(nbranches):
                ax.plot(time*tu, dists[branch_group,i_mem1,:], color='tomato',)
            hrmse, = ax.plot(time*tu, rmses[branch_group,:], color='black', label='RMSE')
            hrmsd = ax.axhline(rmsd,label='RMSD', color='black', linestyle='--')
            hlyap, = ax.plot(time[1:]*tu, np.minimum(rmsd, rmses[branch_group,1]*np.exp(lyap_expons[branch_group]*(time[1:]-time[1]))), color='dodgerblue', label="Exponential growth")
            hdiff, = ax.plot(time[1:]*tu, np.minimum(rmsd, rmses[branch_group,1]*(time[1:]-time[1])**diff_expons[branch_group]), color='green', label="Diffusive growth")
            hthalf = ax.axvline((time[0]+thalfsat[branch_group])*tu, color='gray', label=r'$\frac{1}{2}$-saturation time')
            ax.legend(handles=[hrmse,hrmsd,hlyap,hdiff,hthalf])
            ax.set_ylabel('Distance')
            ax.set_xlabel('time')
            if logscale: ax.set_yscale('log')
            fig.savefig(join(plot_dir,r'pert_growth_group%d_%s.png'%(branch_group,plot_suffix)), **pltkwargs)
            plt.close(fig)
        return
    @classmethod 
    def plot_pert_growth_meta(cls, indep_var_list, fracsat, t2fracsat, savefile, ivlabel, tu=1, fracsat_ref=None, t2fracsat_ref=None):
        dist_names = list(t2fracsat.keys())
        colors = {dist_name: plt.cm.Set1(i_dist_name) for (i_dist_name,dist_name) in enumerate(dist_names)}
        fig,axes = plt.subplots(nrows=len(fracsat),figsize=(6,3*len(fracsat)),sharex=True,gridspec_kw={'hspace': 0.25})
        handles = []
        for dist_name,t in t2fracsat.items():
            print(f'{dist_name = }')
            print(f'{t = }')
            order = np.argsort(indep_var_list)
            for i_frac,frac in enumerate(fracsat):
                ax = axes[i_frac]
                h, = ax.plot(np.array(indep_var_list)[order], t[order,i_frac]*tu, label=dist_name, marker='o', color=colors[dist_name])
                if i_frac == 0: handles.append(h)
                ax.set_title(f'Time to {frac:g} of saturation')
                ax.set_xlabel('')
        if (fracsat_ref is not None) and (t2fracsat_ref is not None):
            for dist_name,t in t2fracsat_ref.items():
                print(f'{dist_name = }')
                print(f'{t = }')
                for i_frac,frac in enumerate(fracsat):
                    ax = axes[i_frac]
                    h = ax.axhline(t[0,i_frac]*tu, color=colors[dist_name], linewidth=3, linestyle='--')
        axes[0].legend(handles=handles,loc=(0,1.25),title='Field for distance')
        axes[-1].set_xlabel(ivlabel)
        fig.savefig(savefile, **pltkwargs)
        plt.close(fig)
        return
    def plot_observable_spaghetti(self, obs_fun, branch_group, outfile, ylabel='', title=''):
        print(f'\n\nPlotting group {branch_group}')
        # Get all timespans
        time,mems_trunk,tidx_trunk,mems_branch,tidxs_branch = self.get_tree_subset(branch_group)
        tu = self.ens.dynsys.dt_save

        obs_branch = [self.ens.compute_observables([obs_fun], mem)[0] for mem in mems_branch]
        obs_trunk = np.concatenate([self.ens.compute_observables([obs_fun], mem)[0] for mem in mems_trunk])
        fig,ax = plt.subplots()
        # For trunk, restrict to the times of interest
        hctrl, = ax.plot(time*tu, obs_trunk[tidx_trunk], linestyle='--', color='black', linewidth=2, zorder=1, label='CTRL')
        for i_mem,mem in enumerate(mems_branch):
            hpert, = ax.plot(time*tu, obs_branch[i_mem][tidxs_branch[i_mem]], linestyle='-', color='tomato', linewidth=1, zorder=0, label='PERT')
        #ax.axvline(split_time*tu, color='tomato')
        ax.legend(handles=[hctrl,hpert])
        ax.set_xlabel('time')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        #ax.set_xlim([time[0],time[-1]+1])
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return

class ODEDirectNumericalSimulation(DirectNumericalSimulation):
    def generate_icandf_from_parent(self, parent):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        parent_t,parent_x = self.ens.dynsys.load_trajectory(self.ens.traj_metadata[parent], self.ens.root_dir, tspan=[fin_time_parent]*2)
        init_cond = parent_x[0]
        impulse = np.zeros(self.ens.dynsys.impulse_dim)
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.OccasionalVectorForcing(fin_time_parent, fin_time_parent+self.max_member_duration, [fin_time_parent], [impulse])
            })
        return icandf

class SDEDirectNumericalSimulation(DirectNumericalSimulation):
    def generate_icandf_from_parent(self, parent):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        mdp = self.ens.traj_metadata[parent]
        parent_t,parent_x = self.ens.dynsys.load_trajectory(mdp, self.ens.root_dir, tspan=[fin_time_parent]*2)
        init_cond = parent_x[0]
        init_rngstate = mdp['fin_rngstate']
        frc_reseed = forcing.OccasionalReseedForcing(fin_time_parent, fin_time_parent+self.max_member_duration, [], [])
        frc_vector = forcing.OccasionalVectorForcing(fin_time_parent, fin_time_parent+self.max_member_duration, [], [])
        icandf = dict({
            'init_cond': parent_x[0],
            'init_rngstate': init_rngstate,
            'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
            })
        return icandf

        
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
            'init_cond': init_cond,
            'init_rngstate': init_rngstate,
            'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
            })
        return icandf


# --------------------- end PeriodicBranching section -------------------------

class AncestorGenerator(EnsembleAlgorithm):
    # From one single initial condition (maybe even a coldstart), spawn off a branching ensemble long enough to generate totally independent samples
    def __init__(self, uic_time, uic, config, ens):
        # uic = universal initial condition
        self.uic_time = uic_time
        self.uic = uic 
        super().__init__(config, ens)
        return
    @abstractmethod
    def generate_icandf_from_uic(self):
        # Put a random seed or small perturbation into an integration that will start from the universal common ancestor
        pass
    @abstractmethod
    def generate_icandf_from_buick(self, parent): 
        # Initialize an integration from a burnt-in condition (kicked)
        pass
    @staticmethod
    def label_from_config(config):
        abbrv = r'AnGe_si%d_Tbrn%g_Thrz%g'%(
                config['seed_inc_init'],
                config['burnin_time_phys'],
                config['time_horizon_phys'],
                )
        abbrv = abbrv.replace('.','p')
        label = r'Ancestor Generator (burn-in %g, horizon %g)'%(config['burnin_time_phys'],config['time_horizon_phys'])
        return abbrv,label
    def derive_parameters(self, config):
        tu = self.ens.dynsys.dt_save
        self.burnin_time = int(round(config['burnin_time_phys']/tu))
        self.time_horizon = int(round(config['time_horizon_phys']/tu)) # Time to run after burnin
        # Capacity parameters (mutable, in case we want to extend the dataset later)
        self.num_buicks = config['num_buicks'] 
        self.branches_per_buick = config['branches_per_buick']
        return
    def set_capacity(self, num_buicks, branches_per_buick):
        num_new_buicks = num_buicks - self.branching_state['num_buicks_generated']
        num_new_branches = [branches_per_buick - self.branching_state['num_branches_generated'][i] for i in range(self.branching_state['num_buicks_generated'])]
        if num_new_buicks < 0 or min(num_new_branches) < 0:
            raise Exception(f'{num_new_buicks = }, {num_new_branches = }')
        if num_new_buicks > 0 or max(num_new_branches) > 0:
            self.terminate = False
        self.num_buicks = num_buicks
        self.branches_per_buick = branches_per_buick
        return
    def take_next_step(self, saveinfo):
        if self.terminate:
            return
        nmem = self.ens.get_nmem()
        if nmem == 0:
            self.branching_state = dict({
                'num_buicks_generated': 0,
                'num_branches_generated': [],
                'generation_0': [], # List of all burn-in trajectories
                })
            parent = None
        # If we've generated fewer than all the requested initial conditions, go ahead and generate them 
        if self.branching_state['num_buicks_generated'] < self.num_buicks: 
            icandf = self.generate_icandf_from_uic()
            branching_state_update = dict({
                'num_buicks_generated': self.branching_state['num_buicks_generated'] + 1,
                'num_branches_generated': self.branching_state['num_branches_generated'] + [0],
                'generation_0': self.branching_state['generation_0'] + [nmem],
                })
            parent = None
        elif min(self.branching_state['num_branches_generated']) < self.branches_per_buick:
            # Find first branch with a deficit
            first_underbranched_buick = np.argmax(np.array(self.branching_state['num_branches_generated']) < self.branches_per_buick)
            parent = self.branching_state['generation_0'][first_underbranched_buick]
            icandf = self.generate_icandf_from_buick(parent=parent) 
            nbg = self.branching_state['num_branches_generated'].copy()
            nbg[first_underbranched_buick] += 1
            branching_state_update = dict({
                'num_branches_generated': nbg,
                })
        else:
            self.terminate = True
            return

        # ---------------------- Run the new trajectory -----------------------------
        obs_fun = lambda t,x: None
        _ = self.ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        # ---------------------------------------------------------------------------
        self.branching_state.update(branching_state_update)
        return
    # ------------------ Post-analysis methods -----------------
    def measure_dispersion(self, dist_fun, buicks=None):
        if buicks is None:
            buicks = np.arange(self.branching_state['num_buicks_generated'], dtype=int)
        print(f'{buicks = }')
        num_buicks = len(buicks)
        dists = []
        for i_buick in buicks:
            print(f'{i_buick = }')
            gen0mem = self.branching_state['generation_0'][i_buick]
            num_branches = self.branching_state['num_branches_generated'][i_buick]
            branches = tuple(self.ens.memgraph.successors(gen0mem))
            dists.append(np.zeros((int(num_branches*(num_branches-1)/2), self.time_horizon)))
            i_pair = 0
            for i0_branch in range(1,num_branches):
                branch0 = branches[i0_branch]
                other_branches = [b for b in branches[:i0_branch]]
                print(f'{branch0 = }; {other_branches = }')
                dists_to_branch0 = self.ens.compute_pairwise_observables([dist_fun], branch0, other_branches)[0]
                dists[i_buick][i_pair:i_pair+i0_branch,:] = dists_to_branch0
                i_pair += i0_branch
        # TODO rigorous size checking
        return dists

    # ------------------ Plotting methods ----------------------
    def plot_observable_spaghetti(self, obs_fun, mems2plot, outfile, time_horizon=None, ylabel='', title=''):
        tu = self.ens.dynsys.dt_save
        fig,ax = plt.subplots(figsize=(12,4))
        for mem in mems2plot:
            memobs = self.ens.compute_observables([obs_fun], mem)[0]
            init_time,fin_time = self.ens.get_member_timespan(mem)
            time = np.arange(init_time+1,fin_time+1)
            if time_horizon is not None:
                time = time[:time_horizon]
                memobs = memobs[:time_horizon]
            ax.plot(time*tu, memobs)
        ax.set_xlabel(r'Time [days]')
        ax.set(ylabel=ylabel, title=title)
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    def plot_observable_spaghetti_burnin(self, obs_fun, outfile, ylabel='', title=''):
        mems2plot = self.branching_state['generation_0']
        self.plot_observable_spaghetti(obs_fun, mems2plot, outfile, ylabel=ylabel, title=title)
        return
    def plot_observable_spaghetti_branching(self, obs_fun, family, outfile, time_horizon=None, ylabel='', title=''):
        mems2plot = tuple(self.ens.memgraph.successors(self.branching_state['generation_0'][family]))
        self.plot_observable_spaghetti(obs_fun, mems2plot, outfile, time_horizon=time_horizon, ylabel=ylabel, title=title)
        return
    def plot_score_distribution_branching(self, score_fun, buick, outfile, label=''):
        # Assume the score is the maximum of some scalar observable 
        # Plot a timeseries on the left and a sideways histogram on the right 
        tu = self.ens.dynsys.dt_save
        fig,axes = plt.subplots(ncols=2, figsize=(20,5), width_ratios=[3,1], sharey=False)
        mems2plot = list(self.ens.memgraph.successors(self.branching_state['generation_0'][buick]))
        print(f'{mems2plot = }')
        init_time,fin_time = self.ens.get_member_timespan(mems2plot[0])
        time = np.arange(init_time+1,fin_time+1)
        max_scores = np.zeros(len(mems2plot))
        max_score_timings = np.zeros(len(mems2plot), dtype=int)
        ax = axes[0]
        for i_mem,mem in enumerate(mems2plot):
            memscore = self.ens.compute_observables([score_fun], mem)[0]
            ax.plot(time*tu, memscore, color='black')
            max_scores[i_mem] = np.nanmax(memscore)
            print(f'{max_scores[i_mem] = }')
            max_score_timings[i_mem] = time[np.nanargmax(max_scores[i_mem])]
            ax.plot(max_score_timings[i_mem]*tu, max_scores[i_mem], marker='x', markerfacecolor="None", markeredgecolor='black')
        ax.set_xlabel('Time')
        ax.set_title(label)
        ax = axes[1]
        hist,bin_edges = np.histogram(max_scores, bins=10)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        ax.plot(bin_centers, hist, marker='o', color='black')
        ax.set_xlabel(label)

        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    # ************** Dispersion characteristics ****************
    def plot_gev_convergence(self, obs_fun):
        # Do block maximum method, per initial condition, over increasing-length intervals, and hope that the local GEV parameters converge first to something local and then to the global GEV parameters
        #TODO
        # ... or maybe rely less on GEV as a parameter, and focus on mean, variance, skew etc.
        # Basically this is another way to quantify the mixing time. A test of whether mixing time w.r.t. different observables is very different
        pass
    def measure_running_max(self, obs_fun, runmax_file, figfile_prefix, label='', abbrv='', precomputed=False, num2plot=12):
        buicks = np.array([i for i in range(min(num2plot,self.branching_state['num_buicks_generated'])) if self.branching_state['num_branches_generated'][i] == self.branches_per_buick])
        print(f'{len(buicks) = }, {self.branches_per_buick = }')
        tu = self.ens.dynsys.dt_save
        time = self.uic_time + np.arange(self.burnin_time+1, self.burnin_time+self.time_horizon+1)
        if precomputed:
            runmax_stats = np.load(runmax_file)
            buicks,running_max,running_max_mean,running_max_std,current_std = [runmax_stats[key] for key in 'buicks,running_max,running_max_mean,running_max_std,current_std'.split(',')]
        else:
            running_max = np.zeros((len(buicks), self.branches_per_buick, self.time_horizon)) 
            current_std = np.zeros((len(buicks), self.time_horizon))
            for buick in buicks:
                print(f'About to compute running maxes for {buick = }')
                print(f'Computed the time vector (how hard could that be?)')
                mems = list(self.ens.memgraph.successors(self.branching_state['generation_0'][buick]))
                print(f'Computed the graph successors: {mems = }')
                obs = []
                print(f'Computing observables for mem ...')
                for i_mem,mem in enumerate(mems):
                    obs.append(self.ens.compute_observables([obs_fun], mem)[0])
                    print(f'{mem},', end='')
                print(f'\n')
                obs = np.array(obs)
                print(f'{obs[0,:] = }')
                running_max[buick,:,:] = np.maximum.accumulate(np.where((np.isnan(obs)==0), obs, -np.inf), axis=1)
                current_std[buick,:] = np.std(obs, axis=0)
                print(f'{running_max[buick,:,-1] = }')

            running_max = np.where(np.isfinite(running_max), running_max, np.nan)
            running_max_std = np.std(running_max, axis=1)
            running_max_mean = np.mean(running_max, axis=1)
            print(f'{running_max[:,:,-1] = }')
            runmax_stats = dict(
                    buicks = buicks,
                    running_max = running_max, 
                    running_max_mean = running_max_mean,
                    running_max_std = running_max_std,
                    current_std = current_std,
                    )
            np.savez(runmax_file, **runmax_stats)
        # Plot 
        nbuicks,nbranches,ntimes = running_max.shape
        idx_time = np.unique(np.power(2, np.linspace(np.log2(1),np.log2(ntimes-1),9)).astype(int))
        ylim_runmax = [np.nanmin(running_max),np.nanmax(running_max)]
        ylim_runmaxstd = [0,np.nanmax(running_max_std)]
        fig_glob,axes_glob = plt.subplots(figsize=(15,3), nrows=2, sharex='col', height_ratios=[2,1])
        for buick in range(nbuicks):
            fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(20,6),sharex='col',height_ratios=[2,1],width_ratios=[3,1])
            ax = axes[0,0]
            ax_glob = axes_glob[0]
            for branch in range(nbranches):
                hbranch, = ax.plot(time*tu, running_max[buick,branch], color='tomato', label=r'Running max')
                ax_glob.plot(time*tu, running_max[buick,branch], color='tomato', label=r'Running max')
            hmean, = ax.plot(time*tu, running_max_mean[buick,:], color='dodgerblue', label=r'Mean running max')
            ax_glob.plot(time*tu, running_max_mean[buick,:])
            ax.legend(handles=[hbranch,hmean])
            ax.set_xlabel('')
            ax.xaxis.set_tick_params(which='both',labelbottom=True)
            ax.set_title(label)
            ax.set_ylim(ylim_runmax)
            ax = axes[1,0]
            ax_glob = axes_glob[1]
            hstd, = ax.plot(time*tu, current_std[buick,:], color='cyan', label='Std.')
            hmaxstd, = ax.plot(time*tu, running_max_std[buick,:], color='black', label='Std. running max')
            ax_glob.plot(time*tu, running_max_std[buick,:], label='Std. running max')
            ax.legend(handles=[hstd,hmaxstd])
            ax.set_xlabel(r'Time')
            ax.set_ylim(ylim_runmaxstd)

            # Plot complementary CDF 
            handles = []
            for i in range(len(idx_time)):
                t = time[idx_time[i]]
                color = plt.cm.rainbow(i/len(idx_time))
                rm = running_max[buick,:,idx_time[i]]
                hist,bin_edges = np.histogram(rm[np.isfinite(rm)], bins=10)
                prob_exc = np.cumsum(hist[::-1])[::-1] / self.branches_per_buick
                axes[0,0].axvline(t*tu, color=color)
                h, = axes[0,1].plot(prob_exc,bin_edges[:-1],color=color,label=r'$t=%g$'%(t*tu))
                handles.append(h)
            ax = axes[0,1]
            ax.set_xscale('log')
            ax.set_xlabel('Exc. Prob.')
            ax.set_ylim(ylim_runmax)
            ax.xaxis.set_tick_params(which='both',labelbottom=True)
            axes[1,1].axis('off')
            fig.savefig(r'%s_buick%d.png'%(figfile_prefix,buick), **pltkwargs)
            plt.close(fig)
        fig_glob.savefig(r'%s_allbuicks.png'%(figfile_prefix), **pltkwargs)
        plt.close(fig_glob)
        # Plot aggregated CDFs across BUICKs
        nrows = int(np.ceil(np.sqrt(len(idx_time))))
        fig,axes = plt.subplots(ncols=nrows, nrows=nrows, figsize=(6*nrows,6*nrows), gridspec_kw={'hspace': 0.2}, sharex=True, sharey=True)
        for i in range(len(idx_time)):
            ax = axes.flat[i]
            t = idx_time[i]
            color = plt.cm.rainbow(i/len(idx_time))
            for buick in range(nbuicks):
                rm = running_max[buick,:,idx_time[i]]
                hist,bin_edges = np.histogram(rm[np.isfinite(rm)], bins=20)
                prob_exc = np.cumsum(hist[::-1])[::-1] / self.branches_per_buick
                hsingle, = ax.plot(prob_exc,bin_edges[:-1],color=color,label=r'Single BUICK')
            rm = running_max[:,:,idx_time[i]]
            hist,bin_edges = np.histogram(rm[np.isfinite(rm)], bins=20)
            prob_exc = np.cumsum(hist[::-1])[::-1] / (nbuicks * self.branches_per_buick)
            hagg, = ax.plot(prob_exc,bin_edges[:-1],color='black',linestyle='--',linewidth=3,label=r'Aggregated BUICKs')
            ax.set_xscale('log')
            ax.set_title(r'$t=%g$'%(t*tu))
            ax.set_xlabel(r'Prob. Exc.')
            ax.set_ylabel(r'Running max')
        fig.savefig(r'%s_allbuicks.png'%(figfile_prefix), **pltkwargs)
        plt.close(fig)
        return

class ODEAncestorGenerator(AncestorGenerator):
    # where the system of interest is an ODE driven by impulses
    @classmethod
    def default_init(cls, config, ens):
        uic_time = 0
        uic = ens.dynsys.generate_default_init_cond(uic_time)
        return cls(uic_time, uic, config, ens)
    def generate_icandf_from_uic(self):
        imp = self.rng.normal(size=(self.ens.dynsys.impulse_dim,))
        icandf = dict({
            'init_cond': self.uic,
            #'init_rngstate': init_rngstate,
            'frc': forcing.OccasionalVectorForcing(self.uic_time, self.uic_time+self.burnin_time, [self.uic_time], [imp]),
            })
        return icandf
    def generate_icandf_from_buick(self, parent):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        print(f'{init_time_parent = }, {fin_time_parent = }')
        mdp = self.ens.traj_metadata[parent]
        parent_t,parent_x = self.ens.dynsys.load_trajectory(mdp, self.ens.root_dir, tspan=[fin_time_parent]*2)
        init_cond = parent_x[0]
        seed = self.rng.integers(low=self.seed_min,high=self.seed_max)
        imp = self.rng.normal(size=self.ens.dynsys.impulse_dim,)
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.OccasionalVectorForcing(fin_time_parent, fin_time_parent+self.time_horizon, [fin_time_parent], [imp]),
            })
        return icandf

class SDEAncestorGenerator(AncestorGenerator):
    # where the system of interest is an SDE driven by white noise
    @classmethod
    def default_init(cls, config, ens):
        uic_time = 0
        uic = ens.dynsys.generate_default_init_cond(uic_time)
        return cls(uic_time, uic, config, ens)
    def generate_icandf_from_uic(self):
        seed = self.rng.integers(low=self.seed_min,high=self.seed_max)
        init_rngstate = default_rng(seed=seed).bit_generator.state 
        frc_reseed = forcing.OccasionalReseedForcing(self.uic_time, self.uic_time+self.burnin_time, [], [])
        frc_vector = forcing.OccasionalVectorForcing(self.uic_time, self.uic_time+self.burnin_time, [], [])
        icandf = dict({
            'init_cond': self.uic,
            'init_rngstate': init_rngstate,
            'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
            })
        return icandf
    def generate_icandf_from_buick(self, parent):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        print(f'{init_time_parent = }, {fin_time_parent = }')
        mdp = self.ens.traj_metadata[parent]
        parent_t,parent_x = self.ens.dynsys.load_trajectory(mdp, self.ens.root_dir, tspan=[fin_time_parent]*2)
        init_cond = parent_x[0]
        init_rngstate = mdp['fin_rngstate']
        seed = self.rng.integers(low=self.seed_min,high=self.seed_max)
        frc_reseed = forcing.OccasionalReseedForcing(fin_time_parent, fin_time_parent+self.time_horizon, [fin_time_parent], [seed])
        frc_vector = forcing.OccasionalVectorForcing(fin_time_parent, fin_time_parent+self.time_horizon, [], [])
        icandf = dict({
            'init_cond': init_cond,
            'init_rngstate': init_rngstate,
            'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
            })
        return icandf



class TEAMS(EnsembleAlgorithm):
    def __init__(self, init_times, init_conds, config, ens):
        self.set_init_conds(init_times, init_conds) # Unlike for general Algorithms, an initial condition is mandatory
        super().__init__(config, ens)
        return
    @classmethod
    #@abstractmethod
    def initialize_from_ancestorgenerator(cls, angel, buicks):
        # angel is an instance of AncestorGenerator
        pass
    def derive_parameters(self, config):
        self.num_levels_max = config['num_levels_max']
        self.num_members_max = config['num_members_max']
        self.num_active_families_min = config['num_active_families_min']
        tu = self.ens.dynsys.dt_save
        self.time_horizon = int(round(config['time_horizon_phys']/tu))
        self.buffer_time = int(round(config['buffer_time_phys']/tu)) # Time between the end of one interval and the beginning of the next, when generating the initial ensemble. Add this to the END of ancestral trajectories. 
        self.advance_split_time = int(round(config['advance_split_time_phys']/tu))
        self.advance_split_time_max = int(round(config['advance_split_time_max_phys']/tu)) # Determines how many 'nan's to put at the start of each trajectory, in order to compare different deltas fairly 
        print(f'{self.advance_split_time = }')
        self.split_landmark = config['split_landmark'] # either 'max' or 'thx'
        self.inherit_perts_after_split = config['inherit_perts_after_split']
        self.population_size = config['population_size']
        self.drop_sched = config['drop_sched']
        self.drop_rate = config['drop_rate']
        self.birth_sched = config['birth_sched']
        return
    def set_capacity(self, num_levels_max, num_members_max):
        print(f'Resetting capacity; before, {self.terminate = }')
        num_new_levels = num_levels_max - len(self.branching_state['score_levels'])
        num_new_members = num_members_max - self.ens.get_nmem()
        num_active_members = len(self.branching_state['members_active'])
        num_active_families = len(set((sorted(nx.ancestors(self.ens.memgraph, ma) | {ma}))[0] for ma in self.branching_state['members_active'])) 
        if (
                (num_new_levels > 0) and 
                (num_new_members > 0) and 
                (num_active_members > 0) and 
                (num_active_families >= self.num_active_families_min)
                ):
            self.num_levels_max = num_levels_max
            self.num_members_max = num_members_max
            self.terminate = False
        print(f'Resetting capacity; after, {self.terminate = }')
        return
    def set_init_conds(self, init_times, init_conds):
        self.init_times = init_times
        self.init_conds = init_conds
        return
    @staticmethod
    def label_from_config(config):
        if config['drop_sched'] == 'num':
            drop_label = r'$\kappa=%d$'%(config['drop_rate']),
            drop_abbrv = r'kill%d'%(config['drop_rate'])
        elif config['drop_sched'] == 'frac':
            drop_label = r'$\kappa=%.2fN'%(config['drop_rate']),
            drop_abbrv = r'kill%gN'%(config['drop_rate'])
        elif config['drop_sched'] == 'frac_once_then_num':
            drop_frac_init,drop_num = config['drop_rate']
            drop_label = r'$\kappa=%.2fN$ then $%d$'%(drop_frac_init,drop_num)
            drop_abbrv = r'kill%.2fNthen%d'%(drop_frac_init,drop_num)

        if config['birth_sched'] == 'const_pop':
            birth_label = r'$\beta=N-\kappa$'
            birth_abbrv = 'kpop'
        elif config['birth_sched'] == 'one_birth':
            birth_label = r'$\beta=1$'
            birth_abbrv = '1birth'
        elif config['birth_sched'] == 'cull_once_then_const_pop':
            birth_label = '$\beta=1$ then $N-\kappa$'
            birth_abbrv = 'cullthenkpop'


        abbrv = (
                r'TEAMS_N%d_T%g_ast%gb4%s_%s_%s_ipas%d'%(
                    config['population_size'],
                    config['time_horizon_phys'],
                    config['advance_split_time_phys'],
                    config['split_landmark'],
                    drop_abbrv,
                    birth_abbrv,
                    int(config['inherit_perts_after_split']),
                    )
                ).replace('.','p')
        split_landmark_label = {'lmx': 'loc. max', 'gmx': 'glob. max', 'thx': 'lev. cross.'}[config['split_landmark']]
        label = r'TEAMS ($N=%d,%s,T=%g,\delta=%g$ before %s)'%(
                    config['population_size'],
                    drop_label,
                    config['time_horizon_phys'],
                    config['advance_split_time_phys'],
                    split_landmark_label,
                    )

        return abbrv, label 
    @abstractmethod
    def score_components(self, t, x):
        # Something directly computable from the system state. Return a dictionary
        pass
    @abstractmethod
    def score_combined(self, t, sccomps):
        # Scalar score used for splitting, which is derived from sccomp; e.g., a time average
        pass
    @abstractmethod
    def merge_score_components(self, comps0, comps1, nsteps2prepend):
        pass
    @abstractmethod
    def generate_icandf_from_parent(self, parent, branch_time):
        pass
    def take_next_step(self, saveinfo):
        if self.terminate:
            return
        if self.ens.get_nmem() < self.population_size:
            if self.ens.get_nmem() == 0:
                self.branching_state = dict({
                    'score_components_tdep': [],
                    'scores_tdep': [],
                    'scores_max': [],
                    'scores_max_timing': [],
                    'score_levels': [-np.inf], 
                    'goals_at_birth': [],
                    'members_active': [],
                    'parent_queue': deque(),
                    'init_cond_queue': deque(), 
                    'log_weights': [],
                    'multiplicities': [],
                    'branch_times': [],
                    })
                for i_ic in range(self.population_size):
                    self.branching_state['init_cond_queue'].append(i_ic % len(self.init_conds))
                    # TODO possibly ingest pre-simulated initial ensemble members 

            i_ic = self.branching_state['init_cond_queue'].popleft() #init_cond
            parent = None
            init_time = self.init_times[i_ic]
            icandf = self.ens.dynsys.generate_default_icandf(init_time,init_time+self.time_horizon+self.buffer_time,seed=self.rng.integers(low=self.seed_min,high=self.seed_max))
            icandf['init_cond'] = self.init_conds[i_ic] 
            branch_time = init_time
            log_active_weight_old = -np.inf
        else:
            #print(f'self.branching_state = ')
            #pprint.pprint({bskey: bsval for (bskey,bsval) in self.branching_state.items() if bskey not in ['scores_tdep','score_components_tdep']})
            parent = self.branching_state['parent_queue'].popleft()
            ancestor = sorted(nx.ancestors(self.ens.memgraph, parent) | {parent})[0]
            init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
            init_time_ancestor,fin_time_ancestor = self.ens.get_member_timespan(ancestor)
            assert self.branching_state['scores_max'][parent] > self.branching_state['score_levels'][-1]
            score_parent = self.branching_state['scores_tdep'][parent]
            #print(f'{score_parent = }')
            level = self.branching_state['score_levels'][-1]
            exceedance_tidx_parent = np.where(score_parent > level)[0]
            nonexceedance_tidx_parent = np.where(score_parent <= level)[0]
            if self.split_landmark == 'thx':
                landmark_ti = exceedance_tidx_parent[0] 
            elif self.split_landmark == 'lmx': # local max
                ti_lower = exceedance_tidx_parent[0]
                if len(nonexceedance_tidx_parent) > 0 and nonexceedance_tidx_parent[-1] > ti_lower:
                    ti_upper = nonexceedance_tidx_parent[np.where(nonexceedance_tidx_parent > ti_lower)[0][0]]
                else:
                    ti_upper = len(score_parent) - 1 # Don't allow splitting at the very last time point
                landmark_ti = ti_lower + np.nanargmax(score_parent[ti_lower:ti_upper])
                #print(f'{score_parent[ti_lower:ti_upper] = }')
            elif self.split_landmark == 'gmx': # global max
                landmark_ti = np.nanargmax(score_parent)
            else:
                raise Exception(f'Unsupported choice of {self.split_landmark = }')
            branch_ti = min(landmark_ti + 1 - self.advance_split_time, exceedance_tidx_parent[0] + 1)
            branch_time = init_time_ancestor + branch_ti 
            print(f'{score_parent[landmark_ti] = }, {score_parent[branch_ti] = }')
            print(f'branch time: original {branch_time}', end=', ')
            branch_time = max(init_time_ancestor, min(fin_time_parent-1, branch_time))
            branch_ti = branch_time - init_time_ancestor 
            print(f'modified to {branch_time}')
            icandf = self.generate_icandf_from_parent(parent, branch_time)
            memact = self.branching_state['members_active']
            log_active_weight_old = logsumexp([self.branching_state['log_weights'][ma] for ma in memact], b=[self.branching_state['multiplicities'][ma] for ma in memact])

        # ---------------- Run the new trajectory --------------
        new_score_components = self.ens.branch_or_plant(icandf, self.score_components, saveinfo, parent=parent)
        # ----------------------------------------------------------------------

        # Update the state
        new_mem = self.ens.get_nmem() - 1
        init_time_new,fin_time_new = self.ens.get_member_timespan(new_mem)
        # Concatenate with parent's score 
        if parent is not None:
            print(f'{new_score_components[0].shape = }')
            print(f'should be: {fin_time_parent - branch_time}')
            new_score_components = self.merge_score_components(new_mem, new_score_components)
            print(f'After merging, {new_score_components[0].shape = }')
            t0 = init_time_ancestor
        else:
            t0 = init_time_new

        new_score_combined = self.score_combined(new_score_components)
        new_score_combined[:self.advance_split_time_max] = np.nan
        if parent is not None:
            #print(f'{np.abs(self.branching_state["scores_tdep"][parent] - new_score_combined) = }')
            print(f'{branch_ti = }')
            nnidx = np.where(np.isfinite(self.branching_state["scores_tdep"][parent][:branch_ti]))[0]
            if len(nnidx) > 0:
                assert np.all(self.branching_state["scores_tdep"][parent][nnidx] == new_score_combined[nnidx]) 
        new_score_max = np.nanmax(new_score_combined[:self.time_horizon-1])
        self.branching_state['goals_at_birth'].append(self.branching_state['score_levels'][-1])
        self.branching_state['branch_times'].append(branch_time)
        self.branching_state['score_components_tdep'].append(new_score_components)
        self.branching_state['scores_tdep'].append(new_score_combined)
        self.branching_state['scores_max'].append(new_score_max)
        if self.split_landmark == 'thx' and self.advance_split_time == 0:
            assert new_score_max > self.branching_state['score_levels'][-1]
        self.branching_state['scores_max_timing'].append(1+t0+np.nanargmax(new_score_combined)) 
        success = (new_score_max > self.branching_state['score_levels'][-1])
        memact = self.branching_state['members_active']
        # Update the weights
        if parent is None: # still building the initial population of ancestors
            self.branching_state['log_weights'].append(0.0)
        else:
            logZ = np.log1p(np.exp(self.branching_state['log_weights'][parent] - log_active_weight_old))
            print(f'{logZ = }')
            for ma in self.branching_state['members_active']:
                self.branching_state['log_weights'][ma] -= logZ
            self.branching_state['log_weights'].append(self.branching_state['log_weights'][parent])
        if success:
            self.branching_state['members_active'].append(new_mem)
            self.branching_state['multiplicities'].append(1)
        else:
            self.branching_state['multiplicities'].append(0)
            self.branching_state['multiplicities'][parent] += 1


        # Raise level? 
        if len(self.branching_state['parent_queue']) == 0 and self.ens.get_nmem() >= self.population_size:
            self.raise_level_replenish_queue()
            # otherwise, the external caller will raise the level
        return
    def raise_level_replenish_queue(self):
        print(f'Raising level/replenishing queue')
        assert len(self.branching_state['parent_queue']) == 0
        scores_active = np.array([self.branching_state['scores_max'][ma] for ma in self.branching_state['members_active']])
        # Keep track of reasons for extinction
        termination_reasons = dict({
            'extinction': False,
            'level_limit': False,
            'member_limit': False,
            'ancestor_diversity': False,
            })
        nmem = self.ens.get_nmem()
        families_active = set((sorted(nx.ancestors(self.ens.memgraph, ma) | {ma}))[0] for ma in self.branching_state['members_active'])
        if nmem >= self.population_size: # Past the startup phase
            order = np.argsort(scores_active)
            num_leq = np.cumsum([self.branching_state['multiplicities'][order[j]] for j in range(len(order))])
            num2drop = 1
            if "num" == self.drop_sched:
                num2drop = max(num2drop, self.drop_rate)
            elif "frac" == self.drop_sched:
                num2drop = max(num2drop, int(round(self.drop_rate * len(scores_active)))) # TODO figure out whether we should count multiplicities in this fraction 
            elif "frac_once_then_num" == self.drop_sched:
                if len(self.branching_state['score_levels']) == 1: # this is the first time raising the level
                    num2drop = max(num2drop, int(round(self.drop_rate[0] * len(scores_active))))
                else:
                    num2drop = 1


            next_level = scores_active[order[np.where(num_leq >= num2drop)[0][0]]]
            self.branching_state['score_levels'].append(next_level)
            # Check termination conditions
            if next_level >= scores_active[order[-1]]:
                termination_reasons['extinction'] = True
            if (len(self.branching_state['score_levels']) >= self.num_levels_max):
                termination_reasons['level_limit'] = True
            if nmem >= self.num_members_max:
                termination_reasons['member_limit'] = True
            if len(families_active) < self.num_active_families_min:
                termination_reasons['ancestor_diversity'] = True

            # Re-populate the parent queue
            members2drop = [ma for ma in self.branching_state['members_active'] if self.branching_state['scores_max'][ma] <= next_level]
            num2drop_actual = len(members2drop)
            self.branching_state['members_active'] = [ma for ma in self.branching_state['members_active'] if self.branching_state['scores_max'][ma] > next_level]
        if not termination_reasons['extinction']:
            parent_pool = self.rng.permutation(np.concatenate(tuple([parent]*self.branching_state['multiplicities'][parent] for parent in self.branching_state['members_active']))) # TODO consider weighting parents' occurrence in this pool by weight
            lenpp = len(parent_pool)
            if "const_pop" == self.birth_sched:
                deficit = self.population_size - len(self.branching_state['members_active'])
            elif "one_birth" == self.birth_sched: # could generalize to a constant number of births 
                deficit = 1
            elif "cull_once_then_const_pop" == self.birth_sched:
                if len(self.branching_state['score_levels']) == 2: # this is the first time raising the level
                    deficit = 1
                else:
                    deficit = num2drop_actual
            for i in range(deficit):
                self.branching_state['parent_queue'].append(parent_pool[i % lenpp])
            print(f'The replenished queue is {self.branching_state["parent_queue"] = }')
        self.terminate = any(list(termination_reasons.values()))
        print(f'{termination_reasons = }')
        return
    # ----------------------- Plotting functions --------------------------------
    def plot_observable_spaghetti(self, obs_fun, ancestor, special_descendant=None, outfile=None, obs_label='', title='', is_score=False):
        tu = self.ens.dynsys.dt_save
        nmem = self.ens.get_nmem()
        N = self.population_size
        descendants = list(nx.descendants(self.ens.memgraph,ancestor))
        lineage = list(sorted(nx.ancestors(self.ens.memgraph,special_descendant) | {special_descendant}))
        if special_descendant is None:
            mems2plot = [ancestor] + descendants
        else:
            mems2plot = lineage
        t0,t1 = self.ens.get_member_timespan(ancestor)
        print(f'{t0 = }, {t1 = }')
        fig,axes = plt.subplots(ncols=2,figsize=(12,3),width_ratios=[3,1],sharey=is_score)
        for i_mem,mem in enumerate(mems2plot):
            tinit,tfin = self.ens.get_member_timespan(mem)
            #obs = self.ens.compute_observables([obs_fun], mem)[0]
            obs = self.ens.compute_observables_along_lineage([obs_fun], mem, merge_as_scalars=True)[0]
            print(f'{len(obs) = }, {tinit = }, {tfin = }, {t0 = }')
            assert tfin-t0 == len(obs)
            if mem == ancestor:
                linekwargs = {'color': 'black', 'linestyle': '--', 'linewidth': 2, 'zorder': 1}
            else:
                linekwargs = {'color': plt.cm.turbo((i_mem+1)/len(mems2plot)), 'linestyle': '-', 'linewidth': 1.5, 'zorder': 0}
            ax = axes[0]
            h, = ax.plot((np.arange(tinit+1,tfin+1)-t0)*tu, obs[tinit-t0:], **linekwargs)
            tbr = self.branching_state['branch_times'][mem]
            tmx = self.branching_state['scores_max_timing'][mem]
            print(f'{t0 = }, {tinit = }, {tbr = }, {tmx = }, {tmx*tu = }')
            if mem != ancestor:
                ax.plot([(tbr-t0)*tu,(tmx-t0)*tu], [obs[tmx-t0-1]]*2, marker='o', **linekwargs)
            #ax.plot((tbr-t0+1)*tu, obs[(tbr+1)-(tinit+1)], markerfacecolor="None", markeredgecolor=kwargs['color'], markeredgewidth=3, marker='o')
            ax = axes[1]
            ax.scatter([max(0,mem-N)], [self.branching_state['scores_max'][mem]], ec=linekwargs['color'], fc='none', marker='o', s=80, lw=2,)
        ax = axes[0]
        ax.set_xlabel('Time')
        if is_score:
            ylabel = r'Score $R(X(t))$'
        else:
            ylabel = obs_label
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax = axes[1]
        ax.plot(np.arange(nmem-N), self.branching_state['goals_at_birth'][N:], color='gray', linestyle='--')
        ax.scatter([max(0,mem-N) for mem in [ancestor]+descendants], [self.branching_state['scores_max'][mem] for mem in [ancestor]+descendants], marker='.', color='gray', )
        # TODO also overlay the full ascent of levels
        ax.set_xlabel('Generation')
        ylabel = r'$\max_t\{R(X(t))\}$'
        ax.set_ylabel(ylabel)
        #ax.set_xlim([time[0],time[-1]+1])
        if outfile is not None:
            fig.savefig(outfile, **pltkwargs)
            plt.close(fig)
        return fig, axes
    # ------------------------ Analysis functions -------------------
    def collect_ancdesc_pairs_byscore(self, anc_min, anc_max, desc_min, desc_max):
        B = self.ens.construct_descent_matrix().tocsr()
        scores = np.array(self.branching_state['scores_max'])
        anc_flag = (scores >= anc_min)*(scores < anc_max)
        desc_flag = (scores >= desc_min)*(scores < desc_max)
        C = sps.diags(anc_flag, dtype=bool) @ B @ sps.diags(desc_flag, dtype=bool)
        ancs,descs = C.nonzero()
        return ancs,descs
    @staticmethod
    def measure_plot_boost_distribution(config_algo, algs, figfile, alpha=0.1, param_display=''):
        # TODO add a histogram of scores, and repeat this whole process for timings as well. This helps to diagnose how much delta is creating new maxima vs. building upon old ones 
        # Joint distributions of scores, and score timings, between ancestor and descendant
        # ---------------- Calculate TEAMS statistics -------------------
        #Iterate through alg objects first to collect scores and define bin edges
        N = config_algo['population_size']
        original_ancs_only = True
        Nalg = len(algs)
        sclim = np.array([np.inf,-np.inf])
        sc_anc = [] # here 'ancestor' can include intermediates along the family tree
        sc_desc = [] 
        logw_anc = []
        logw_desc = [] # descendant weights
        for i_alg,alg in enumerate(algs):
            sclim[0] = min(sclim[0],np.min(alg.branching_state['scores_max']))
            sclim[1] = max(sclim[1],np.max(alg.branching_state['scores_max']))
            # Compute everyone's descendants by raising the matrix to higher powers
            B = alg.ens.construct_descent_matrix().toarray()
            Nanc = N if original_ancs_only else alg.ens.get_nmem()
            sc_anc_new = list(alg.branching_state['scores_max'][:Nanc])
            logw_anc_new = list(alg.branching_state['log_weights'][:Nanc])
            sc_desc_new = []
            logw_desc_new = []
            for anc in range(Nanc):
                desc = np.where(B[anc,:] == 1)[0]
                sc_desc_new.append(np.array([alg.branching_state['scores_max'][i] for i in desc]))
                logw_desc_new.append(np.array([alg.branching_state['log_weights'][i] for i in desc]))
            sc_anc += sc_anc_new
            logw_anc += logw_anc_new
            sc_desc += sc_desc_new
            logw_desc += logw_desc_new
        sc_anc = np.array(sc_anc)
        logw_anc = np.array(logw_anc)
        bins = np.linspace(sclim[0]-1e-10,sclim[1]+1e-10,30)
        binwidth = bins[1] - bins[0]
        anc2bin = ((sc_anc - bins[0])/binwidth).astype(int)
        print(f'{anc2bin.shape = }')
        # Determine quantiles to plot 
        alphas = np.array([0.5,0.25,0.1])
        lowers_wted = np.nan*np.ones((len(alphas),len(bins)-1))
        uppers_wted = np.nan*np.ones((len(alphas),len(bins)-1))
        means_wted = np.nan*np.ones(len(bins)-1)
        lowers_unif = np.nan*np.ones((len(alphas),len(bins)-1))
        uppers_unif = np.nan*np.ones((len(alphas),len(bins)-1))
        means_unif = np.nan*np.ones(len(bins)-1)
        bidx2plot = []
        fig,axes = plt.subplots(ncols=2,figsize=(10,5))
        for b in range(len(bins)-1):
            ancs_b = np.where(anc2bin == b)[0]
            if len(ancs_b) > 0:
                print(f'{ancs_b.min() = }')
                bidx2plot.append(b)
                scores_b = np.concatenate([sc_anc[ancs_b]] + [sc_desc[a] for a in ancs_b])
                logw_b = np.concatenate([logw_anc[ancs_b]] + [logw_desc[a] for a in ancs_b])
                logw_b -= logsumexp(logw_b)
                axes[0].scatter((bins[b]+binwidth/2) * np.ones(len(scores_b)), scores_b, color='red', marker='.', zorder=2, s=rcParams['lines.markersize'] ** 2 / 3 * np.exp(logw_b - max(logw_b)))
                axes[1].scatter((bins[b]+binwidth/2) * np.ones(len(scores_b)), scores_b, color='red', marker='.', zorder=2, s=rcParams['lines.markersize'] ** 2 / 3)
                print(f'{logw_b = }')
                print(f'{scores_b = }')

                means_wted[b] = np.exp(logsumexp(logw_b, b=scores_b) - logsumexp(logw_b))
                means_unif[b] = np.mean(scores_b)
                for i_alpha,alpha in enumerate(alphas):
                    lowers_wted[i_alpha,b] = utils.weighted_quantile(scores_b, alpha/2, logw_b, logscale=True)
                    uppers_wted[i_alpha,b] = utils.weighted_quantile(scores_b, 1-alpha/2, logw_b, logscale=True)
                    lowers_unif[i_alpha,b] = np.quantile(scores_b, alpha/2)
                    uppers_unif[i_alpha,b] = np.quantile(scores_b, 1-alpha/2)
        axes[0].plot(bins[bidx2plot]+binwidth/2, means_wted[bidx2plot], color='black', linewidth=2, marker='o')
        axes[1].plot(bins[bidx2plot]+binwidth/2, means_unif[bidx2plot], color='black', linewidth=2, marker='o')
        for i_alpha,alpha in enumerate(alphas):
            axes[0].fill_between(bins[bidx2plot]+binwidth/2, lowers_wted[i_alpha,bidx2plot],uppers_wted[i_alpha,bidx2plot],color='gray',alpha=1-i_alpha/len(alphas),zorder=-i_alpha-1)
            axes[1].fill_between(bins[bidx2plot]+binwidth/2, lowers_unif[i_alpha,bidx2plot],uppers_unif[i_alpha,bidx2plot],color='gray',alpha=1-i_alpha/len(alphas),zorder=-i_alpha-1)
        for ax in axes:
            ax.axline((0,0),slope=1,color='black',linestyle='--')
            ax.set_xlabel('Ancestor score')
            ax.set_ylabel('Descendant score')
            ax.set_xlim(sclim)
            ax.set_ylim(sclim)
        axes[0].set_title('Weighted')
        axes[1].set_title('Unweighted')
        fig.suptitle(r'$\delta=%g$'%(config_algo['advance_split_time_phys']))
        fig.savefig(figfile, **pltkwargs)
        print(f'{figfile = }')
        plt.close(fig)
        
    @staticmethod
    def measure_plot_score_distribution(config_algo, algs, scmax_dns, returnstats_file, figfileh, figfilev, alpha=0.05, param_display=''):
        N_dns = len(scmax_dns)
        print(f'{N_dns = }')
        time_horizon_effective = config_algo['time_horizon_phys'] - config_algo['advance_split_time_max_phys']
        sf2rt = lambda sf: utils.convert_sf_to_rtime(sf, time_horizon_effective) 
        # ---------------- Calculate TEAMS statistics -------------------
        #Iterate through alg objects first to collect scores and define bin edges
        sclim = [np.min(scmax_dns),np.max(scmax_dns)]
        scmaxs,logws,mults = ([] for i in range(3))
        Ns_init,Ns_fin = (np.zeros(len(algs),dtype=int) for i in range(2))
        for i_alg,alg in enumerate(algs):
            scmax,logw,mult = (alg.branching_state[s] for s in 'scores_max,log_weights,multiplicities'.split(','))
            scmaxs.append(scmax)
            logws.append(logw)
            mults.append(mult)
            Ns_init[i_alg] = alg.population_size
            Ns_fin[i_alg] = alg.ens.get_nmem()
            assert int(round(np.exp(logsumexp(logw,b=mult)))) == Ns_init[i_alg]
            sclim[0],sclim[1] = min(sclim[0],np.min(scmax)),max(sclim[1],np.max(scmax))
        N_teams_init = np.sum(Ns_init)
        N_teams_fin = np.sum(Ns_fin)
        bin_edges = np.linspace(sclim[0]-1e-10,sclim[1]+1e-10,16)
        hist_dns,_ = np.histogram(scmax_dns, bins=bin_edges, density=False)
        # determine bounds on measurable return periods 
        logw_pooled = np.concatenate(logws)
        mults_pooled = np.concatenate(mults)
        scmaxs_pooled = np.concatenate(scmaxs)
        finite_idx = np.where(np.isfinite(logw_pooled) * (mults_pooled >= 1))
        logw_pooled -= logsumexp(logw_pooled[finite_idx], b=mults_pooled[finite_idx])
        ccdf_min = np.exp(logw_pooled[np.argmax(scmaxs_pooled)])
        ccdf_max = 0.5 # arbitrary
        logccdf_grid = np.linspace(np.log(ccdf_max), np.log(ccdf_min), 30)
        rt_grid = sf2rt(np.exp(logccdf_grid))
        # Now put the scores from separate runs into this common set of bins
        hists_init,hists_fin_unif,hists_fin_wted,ccdfs_init,ccdfs_fin_unif,ccdfs_fin_wted = (np.zeros((len(algs),len(bin_edges)-1)) for i in range(6))
        rlevs_init,rlevs_fin = (np.zeros((len(algs), len(logccdf_grid))) for i in range(2))
        boost_family_mean = np.zeros(len(algs))
        boost_population = np.zeros(len(algs))
        for i_alg,alg in enumerate(algs):
            # return periods as function of return levels
            hists_init[i_alg],_ = np.histogram(scmaxs[i_alg][:alg.population_size], bins=bin_edges, density=False)
            hists_fin_unif[i_alg,:],_ = np.histogram(scmaxs[i_alg], bins=bin_edges, density=False)
            hists_fin_wted[i_alg,:],_ = np.histogram(scmaxs[i_alg], bins=bin_edges, weights=mults[i_alg]*np.exp(logws[i_alg]), density=False)
            ccdfs_init[i_alg] = utils.pmf2ccdf(hists_init[i_alg],bin_edges)
            ccdfs_fin_wted[i_alg,:] = utils.pmf2ccdf(hists_fin_wted[i_alg],bin_edges)
            ccdfs_fin_unif[i_alg,:] = utils.pmf2ccdf(hists_fin_unif[i_alg],bin_edges)
            # return levels as function of return periods
            xord,logccdf_emp = utils.compute_logsf_empirical_with_multiplicities(scmaxs[i_alg],logw=logws[i_alg],mults=mults[i_alg])
            rlevs_fin[i_alg,:] = np.interp(logccdf_grid[::-1], logccdf_emp[::-1], xord[::-1])[::-1]
            xord,logccdf_emp = utils.compute_logsf_empirical_with_multiplicities(scmaxs[i_alg][:alg.population_size],logw=np.zeros(alg.population_size),mults=np.ones(alg.population_size, dtype=int))
            rlevs_init[i_alg,:] = np.interp(logccdf_grid[::-1], logccdf_emp[::-1], xord[::-1])[::-1]
            # Calculate gains
            A = alg.ens.construct_descent_matrix().toarray().astype(int)
            print(f'{np.min(A) = }, {np.max(A) = }')
            print(f'{np.sum(A,axis=1) = }')
            desc_scores = A * np.array(scmaxs[i_alg]) 
            maxboosts = np.maximum(0, np.max(desc_scores,axis=1) - np.array(scmaxs[i_alg]))
            print(f'{np.min(maxboosts) = }, {np.max(maxboosts) = }')
            boost_family_mean[i_alg] = np.mean(maxboosts)
            boost_population[i_alg] = np.max(maxboosts)
        hist_init = np.sum(hists_init, axis=0)
        hist_fin_unif = np.sum(hists_fin_unif, axis=0)
        hist_fin_wted = np.sum(hists_fin_wted, axis=0)
        ccdf_init,ccdf_init_lower,ccdf_init_upper = utils.pmf2ccdf(hist_init,bin_edges,return_errbars=True,alpha=alpha)
        ccdf_fin_wted = utils.pmf2ccdf(hist_fin_wted,bin_edges)
        ccdf_fin_wted_lower = np.quantile(np.nan_to_num(ccdfs_fin_wted,nan=0), alpha/2, axis=0)
        ccdf_fin_wted_upper = np.quantile(np.nan_to_num(ccdfs_fin_wted,nan=0), 1-alpha/2, axis=0)
        ccdf_fin_wted_lower = np.where(ccdf_fin_wted_lower==0, np.nan, ccdf_fin_wted_lower)
        ccdf_fin_wted_upper = np.where(ccdf_fin_wted_upper==0, np.nan, ccdf_fin_wted_upper)
        ccdf_fin_unif = utils.pmf2ccdf(hist_fin_unif,bin_edges)
        # rlev

        xord_pooled,logccdf_emp_pooled = utils.compute_logsf_empirical_with_multiplicities(np.concatenate(scmaxs), logw=np.concatenate(logws), mults=np.concatenate(mults))

        rlev_fin_pooled = np.interp(logccdf_grid[::-1], logccdf_emp_pooled[::-1], xord_pooled[::-1])[::-1]
        # DNS 
        xord_dns,logccdf_dns_emp = utils.compute_logsf_empirical_with_multiplicities(scmax_dns)
        rlev_dns = np.interp(logccdf_grid[::-1], logccdf_dns_emp[::-1], xord_dns[::-1])[::-1]

        # put error bars on TEAMS by bootstrapping
        rng_boot = default_rng(45839)
        n_boot = 5000
        idx_alg_boot = rng_boot.choice(np.arange(len(algs)), replace=True, size=(n_boot,len(algs)))
        N_dns_boot_init = int(N_teams_init*config_algo['time_horizon_phys']/time_horizon_effective)
        print(f'{N_dns_boot_init = }')
        idx_dns_boot_init = rng_boot.choice(np.arange(N_dns), replace=True, size=(n_boot, N_dns_boot_init))
        N_dns_boot_fin = int(N_teams_fin*config_algo['time_horizon_phys']/time_horizon_effective)
        print(f'{N_dns_boot_fin = }')
        idx_dns_boot_fin = rng_boot.choice(np.arange(N_dns), replace=True, size=(n_boot, N_dns_boot_fin))
        print(f'{idx_dns_boot_init[1] = }')
        print(f'{idx_dns_boot_fin[1] = }')
        rlevs_dns_boot_init  = np.nan*np.ones((n_boot, len(logccdf_grid)))
        rlevs_dns_boot_fin  = np.nan*np.ones((n_boot, len(logccdf_grid)))
        ccdf_fin_wted_boot = np.nan*np.ones((n_boot,len(bin_edges)-1))
        rlevs_fin_pooled_boot = np.nan*np.ones((n_boot, len(logccdf_grid)))
        for i_boot in range(n_boot):
            # ||||||||||| horizontal ||||||||||
            hist_fin_wted_boot = np.sum(hists_fin_wted[idx_alg_boot[i_boot,:]],axis=0)
            ccdf_fin_wted_boot[i_boot,:] = utils.pmf2ccdf(hist_fin_wted_boot,bin_edges)
            # ------- vertical ----------
            # TEAMS
            scmax_boot = np.concatenate([scmaxs[i] for i in idx_alg_boot[i_boot]])
            logw_boot = np.concatenate([logws[i] for i in idx_alg_boot[i_boot]])
            mults_boot = np.concatenate([mults[i] for i in idx_alg_boot[i_boot]])
            xord,logccdf_emp = utils.compute_logsf_empirical_with_multiplicities(scmax_boot,logw=logw_boot,mults=mults_boot)
            rlevs_fin_pooled_boot[i_boot,:] = np.interp(logccdf_grid[::-1], logccdf_emp[::-1], xord[::-1], left=np.nan, right=np.nan)[::-1]
            # DNS 
            xord,logccdf_emp = utils.compute_logsf_empirical_with_multiplicities(scmax_dns[idx_dns_boot_init[i_boot]])
            rlevs_dns_boot_init[i_boot,:] = np.interp(logccdf_grid[::-1], logccdf_emp[::-1], xord[::-1], left=np.nan, right=np.nan)[::-1]
            xord,logccdf_emp = utils.compute_logsf_empirical_with_multiplicities(scmax_dns[idx_dns_boot_fin[i_boot]])
            rlevs_dns_boot_fin[i_boot,:] = np.interp(logccdf_grid[::-1], logccdf_emp[::-1], xord[::-1], left=np.nan, right=np.nan)[::-1]
            
        ccdf_fin_wted_pooled_lower = np.nanquantile(ccdf_fin_wted_boot,alpha/2,axis=0)
        ccdf_fin_wted_pooled_upper = np.nanquantile(ccdf_fin_wted_boot,1-alpha/2,axis=0)


        # --------------------- Tally costs ------------------------
        cost_teams_init = N_teams_init * (config_algo['time_horizon_phys'] - config_algo['advance_split_time_max_phys'] + config_algo['advance_split_time_phys'])
        cost_teams_fin = N_teams_fin/N_teams_init * cost_teams_init
        cost_dns = N_dns * (config_algo['time_horizon_phys'] - config_algo['advance_split_time_max_phys'])
        # Get DNS stats, comparing either to a single TEAMS run or the aggregate in cost 
        ccdf_dns,ccdf_dns_sep_lower,ccdf_dns_sep_upper = utils.pmf2ccdf(hist_dns,bin_edges,return_errbars=True,alpha=alpha,N_errbars=int(N_dns * cost_teams_fin/cost_dns * 1/len(algs)))
        _,ccdf_dns_pooled_lower,ccdf_dns_pooled_upper = utils.pmf2ccdf(hist_dns,bin_edges,return_errbars=True,alpha=alpha,N_errbars=int(N_dns * cost_teams_fin/cost_dns))

        print(f'{ccdf_dns = }')
        # Collect in a dictionary and store 
        returnstats = dict({
            'bin_edges': bin_edges,
            # Separate TEAMS runs
            'hists_init': hists_init,
            'hists_fin_wted': hists_fin_wted,
            'hists_fin_unif': hists_fin_unif,
            'ccdfs_init': ccdfs_init,
            'ccdfs_fin_wted': ccdf_fin_wted,
            'ccdfs_fin_unif': ccdf_fin_unif,
            'boost_family_mean': boost_family_mean,
            'boost_population': boost_population,
            # Pooled TEAMS runs
            'hist_init': hist_init,
            'hist_fin_wted': hist_fin_wted,
            'hist_fin_unif': hist_fin_unif,
            'ccdf_init': ccdf_init,
            'ccdf_init_lower': ccdf_init_lower,
            'ccdf_init_upper': ccdf_init_upper,
            'ccdf_fin_wted': ccdf_fin_wted,
            'ccdf_fin_wted_lower': ccdf_fin_wted_lower,
            'ccdf_fin_wted_upper': ccdf_fin_wted_upper,
            'ccdf_fin_unif': ccdf_fin_unif,
            'ccdf_fin_wted_pooled_lower': ccdf_fin_wted_pooled_lower,
            'ccdf_fin_wted_pooled_upper': ccdf_fin_wted_pooled_upper,
            # Inverted
            'logccdf_grid': logccdf_grid,
            'rlevs_fin': rlevs_fin,
            'rlev_fin_pooled': rlev_fin_pooled,
            'rlevs_fin_pooled_boot': rlevs_fin_pooled_boot,
            'rlevs_dns_boot_init': rlevs_dns_boot_init,
            'rlevs_dns_boot_fin': rlevs_dns_boot_fin,
            # DNS
            'hist_dns': hist_dns,
            'ccdf_dns': ccdf_dns,
            'ccdf_dns_sep_lower': ccdf_dns_sep_lower,
            'ccdf_dns_sep_upper': ccdf_dns_sep_upper,
            'ccdf_dns_pooled_lower': ccdf_dns_pooled_lower,
            'ccdf_dns_pooled_upper': ccdf_dns_pooled_upper,
            # Scalars
            'cost_teams_init': cost_teams_init,
            'cost_teams_fin': cost_teams_fin,
            'cost_dns': cost_dns,
            'time_horizon_effective': time_horizon_effective,
            })

        np.savez(returnstats_file, **returnstats)

        # ------------- Plot -------------------
        teams_abbrv = 'TEAMS' if algs[0].advance_split_time>0 else 'AMS'
        # ++++ left-hand text label +++
        cost_display = '\n'.join([
            r'%s cost:'%(teams_abbrv),
            r'%.1E'%(cost_teams_fin/len(algs)),
            r'$\times$ %d runs'%(len(algs)),
            r'$=$%.1E'%(cost_teams_fin),
            r' ',
            r'DNS cost:',
            r'%.1E'%(cost_dns)
            ])
        display = '\n'.join([param_display,'',cost_display])
        sf2rt = lambda sf: utils.convert_sf_to_rtime(sf, returnstats['time_horizon_effective'])

        def cliprlev(rlev_curve):
            i0 = np.argmax(rlev_curve)
            rlev_clipped = np.copy(rlev_curve)
            rlev_clipped[i0+1:] = np.NaN
            return rlev_clipped


        figh,axesh = plt.subplots(ncols=3, figsize=(18,4), sharex=False, sharey=True)
        figv,axesv = plt.subplots(ncols=3, figsize=(18,4), sharex=False, sharey=True) # vertically oriented errbars

        for axes in (axesh,axesv):
            axes[0].text(-0.3,0.5,display,fontsize=15,transform=axes[0].transAxes,horizontalalignment='right',verticalalignment='center')

        # ++++ Column 0: individual curves ++++

        ax = axesv[0]
        for i_alg,alg in enumerate(algs):
            ax.plot(rt_grid, cliprlev(rlevs_init[i_alg]), color='dodgerblue', linestyle='-', linewidth=1, alpha=0.5, label=r'Init')
            ax.plot(rt_grid, cliprlev(rlevs_fin[i_alg]), color='red', linestyle='-', linewidth=1, alpha=0.5, label=teams_abbrv)
        ax.plot(rt_grid, rlev_dns, color='black')
        ax.set_ylabel(r'Return level')
        ax.set_title(r'Single %s runs'%(teams_abbrv))
        ax.set_xscale('log')
        # TODO add DNS line 


        ax = axesh[0]
        # DNS, with equal-cost errorbars to compare to single DNS runs
        hdns, = ax.plot(sf2rt(ccdf_dns), bin_edges[:-1], marker='.', color='black', label=r'DNS (cost %.1E)'%(cost_dns))
        for i_alg,alg in enumerate(algs):
            # Initialization
            hinit_sep, = ax.plot(sf2rt(ccdfs_init[i_alg]),bin_edges[:-1],color='dodgerblue',linestyle='-',linewidth=1,alpha=0.5,label=r'Init')
            # Final (weighted)
            hfin_wted_sep, = ax.plot(sf2rt(ccdfs_fin_wted[i_alg]),bin_edges[:-1],color='red',linestyle='-',linewidth=1,alpha=0.5,label=teams_abbrv)
        #ax.fill_betweenx(bin_edges[:-1],sf2rt(ccdf_fin_wted_lower),sf2rt(ccdf_fin_wted_upper),fc='red',ec='none',zorder=-1,alpha=0.5)
        ax.set_ylabel(r'Return level')
        ax.set_title(r'Single %s runs'%(teams_abbrv))

        # ++++ Column 1: pooled curves ++++
        ax = axesh[1]
        errbars_init_flag = False
        # DNS again, this time accounting for total cost 
        hdns, = ax.plot(sf2rt(ccdf_dns), bin_edges[:-1], color='black', label=r'DNS')
        ax.fill_betweenx(bin_edges[:-1], sf2rt(ccdf_dns_pooled_lower), sf2rt(ccdf_dns_pooled_upper), fc='gray', ec='none', zorder=-1, alpha=0.5)
        # Initialization
        hinit, = ax.plot(sf2rt(ccdf_init), bin_edges[:-1], marker='.', color='dodgerblue', label=r'Init.')
        if errbars_init_flag:
            ax.fill_betweenx(bin_edges[:-1],sf2rt(ccdf_init_lower),sf2rt(ccdf_init_upper),fc='dodgerblue',ec='none',zorder=-1,alpha=0.5)
        # Final TEAMS (weighted)
        hfin_wted, = ax.plot(sf2rt(ccdf_fin_wted), bin_edges[:-1], marker='.', color='red', label=teams_abbrv)
        ax.fill_betweenx(bin_edges[:-1],sf2rt(ccdf_fin_wted_pooled_lower),sf2rt(ccdf_fin_wted_pooled_upper),fc='red',ec='none',zorder=-1,alpha=0.25)
        ax.legend(handles=[hinit,hfin_wted,hdns],bbox_to_anchor=(1,0),loc='lower right')
        ax.set_ylabel('')
        ax.yaxis.set_tick_params(which='both',labelbottom=True)
        ax.set_title('Pooled results')

        xlim = [returnstats['time_horizon_effective'],5*sf2rt(min(np.nanmin(ccdf_dns),np.nanmin(ccdf_fin_wted)))]
        ylim = [bin_edges[np.argmax(sf2rt(ccdf_dns) > xlim[0])],bin_edges[-1]]
        for ax in axesh[:2]:
            ax.set_xscale('log')
            print(f'{xlim = }')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(r'Return time')

        ax = axesv[1]
        ax.plot(rt_grid, cliprlev(rlev_fin_pooled), color='red')
        qlo = np.nanquantile(rlevs_fin_pooled_boot, alpha/2, axis=0)
        qhi = np.nanquantile(rlevs_fin_pooled_boot, 1-alpha/2, axis=0)
        ax.fill_between(rt_grid, cliprlev(2*rlev_fin_pooled-qhi), cliprlev(2*rlev_fin_pooled-qlo), color='red', alpha=0.25, zorder=-1)
        ax.plot(rt_grid, cliprlev(rlev_dns), color='black')
        qlo_dns = np.nanquantile(rlevs_dns_boot_fin, alpha/2, axis=0)
        qhi_dns = np.nanquantile(rlevs_dns_boot_fin, 1-alpha/2, axis=0)
        ax.fill_between(rt_grid, cliprlev(2*rlev_fin_pooled-qhi), cliprlev(2*rlev_fin_pooled-qlo), color='red', alpha=0.25, zorder=-1)
        ax.fill_between(rt_grid, cliprlev(2*rlev_dns-qhi_dns), cliprlev(2*rlev_dns-qlo_dns), color='gray', alpha=0.5, zorder=-2)
        ax.set_xscale('log')

        # ++++ Column 2: Histograms ++++
        for ax in (axesh[2],axesv[2]):
            ax.plot(hist_dns, bin_edges[:-1], color='black')
            ax.plot(hist_init, bin_edges[:-1], color='dodgerblue')
            ax.plot(hist_fin_unif, bin_edges[:-1], color='red')
            ax.yaxis.set_tick_params(which='both',labelbottom=True)
            ax.set_xscale('log')
            ax.set_ylim(ylim)
            ax.set_xlabel('Counts')
            ax.set_title('Score histograms')

        figh.savefig(figfileh, **pltkwargs)
        plt.close(figh)

        figv.savefig(figfilev, **pltkwargs)
        plt.close(figv)


        return 


# --------------- ITEAMS, where I stands for {initial condition-based, individual, whatever it stands for in Apple because the Apple doesn't fall far from the tree} ---------


class ITEAMS(EnsembleAlgorithm):
    # TODO Allow to reseed at multiple times, or for limited time 
    def __init__(self, init_time, init_cond, config, ens):
        self.set_init_cond(init_time, init_cond) # Unlike for general Algorithms, an initial condition is mandatory
        super().__init__(config, ens)
        return
    @classmethod
    @abstractmethod
    def initialize_from_ancestorgenerator(cls, angel, family):
        # angel is an instance of AncestorGenerator
        pass
    def derive_parameters(self, config):
        self.autonomy = config['autonomy'] # True if this single family is isolated, False if part of a team.
        self.buick = config['buick'] # Might be irrelevant; depends on whether a buick was used to initialize
        self.num_levels_max = config['num_levels_max']
        tu = self.ens.dynsys.dt_save
        self.time_horizon = int(round(config['time_horizon_phys']/tu))
        self.buffer_time = int(round(config['buffer_time_phys']/tu)) # Time between the end of one interval and the beginning of the next, when generating the initial ensemble. Add this to the END of ancestral trajectories. 
        self.advance_split_time = int(round(config['advance_split_time_phys']/tu))
        self.split_landmark = config['split_landmark'] # either 'max' or 'thx'
        self.population_size = config['population_size']
        self.num2drop = config['num2drop']
        return
    def set_init_cond(self, init_time, init_cond):
        self.init_time = init_time
        self.init_cond = init_cond
        return
    @staticmethod
    def label_from_config(config):
        abbrv = (
                r'ITEAMS_N%d_T%g_ast%gb4%s_drop%d_si%d_buick%d'%(
                    config['population_size'],
                    config['time_horizon_phys'],
                    config['advance_split_time_phys'],
                    config['split_landmark'],
                    config['num2drop'],
                    config['seed_inc_init'],
                    config['buick'], # This option might be irrelevant 
                    )
                ).replace('.','p')
        label = 'ITEAMS'
        return abbrv, label 
    @abstractmethod
    def score_components(self, t, x):
        # Something directly computable from the system state. Return a dictionary
        pass
    @abstractmethod
    def score_combined(self, t, sccomps):
        # Scalar score used for splitting, which is derived from sccomp; e.g., a time average
        pass
    @abstractmethod
    def generate_icandf_from_parent(self, parent, branch_time):
        pass
    def take_next_step(self, saveinfo):
        if self.terminate:
            return
        if self.ens.get_nmem() < self.population_size:
            if self.ens.get_nmem() == 0:
                self.branching_state = dict({
                    'scores_tdep': [],
                    'scores_max': [],
                    'scores_max_timing': [],
                    'score_levels': [-np.inf], 
                    'goals_at_birth': [],
                    'members_active': [],
                    'parent_queue': deque(),
                    'log_weights': [],
                    'multiplicities': [],
                    'branch_times': [],
                    })

            parent = None
            icandf = self.ens.dynsys.generate_default_icandf(self.init_time,self.init_time+self.time_horizon+self.buffer_time,seed=self.rng.integers(low=self.seed_min,high=self.seed_max))
            icandf['init_cond'] = self.init_cond
            log_active_weight_old = -np.inf
            branch_time = self.init_time
        else:
            print(f'self.branching_state = ')
            pprint.pprint({bskey: bsval for (bskey,bsval) in self.branching_state.items() if bskey != 'scores_tdep'})
            parent = self.branching_state['parent_queue'].popleft() # this might not be the actual source of initial conditions!
            ancestor = sorted(nx.ancestors(self.ens.memgraph, parent) | {parent})[0]
            init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
            init_time_ancestor,fin_time_ancestor = self.ens.get_member_timespan(ancestor)
            assert self.branching_state['scores_max'][parent] > self.branching_state['score_levels'][-1]
            score_parent = self.branching_state['scores_tdep'][parent]
            level = self.branching_state['score_levels'][-1]
            exceedance_tidx_parent = np.where(score_parent > level)[0]
            nonexceedance_tidx_parent = np.where(score_parent <= level)[0]
            if self.split_landmark == 'thx':
                landmark_ti = exceedance_tidx_parent[0] 
            elif self.split_landmark == 'lmx': # local max
                ti_lower = exceedance_tidx_parent[0]
                if len(nonexceedance_tidx_parent) > 0 and nonexceedance_tidx_parent[-1] > ti_lower:
                    ti_upper = nonexceedance_tidx_parent[np.where(nonexceedance_tidx_parent > ti_lower)[0][0]]
                else:
                    ti_upper = len(score_parent) - 1 # Don't allow splitting at the very last time point
                landmark_ti = ti_lower + np.nanargmax(score_parent[ti_lower:ti_upper])
                #print(f'{score_parent[ti_lower:ti_upper] = }')
                print(f'{np.nanmax(score_parent[ti_lower:ti_upper]) = }')
            elif self.split_landmark == 'gmx': # global max
                landmark_ti = np.nanargmax(score_parent)
            else:
                raise Exception(f'Unsupported choice of {self.split_landmark = }')
            branch_ti = min(landmark_ti - self.advance_split_time, exceedance_tidx_parent[0])
            branch_time = init_time_parent + branch_ti 
            print(f'{score_parent[landmark_ti] = }, {score_parent[branch_ti] = }')
            print(f'branch time: original {branch_time}', end=', ')
            branch_time = max(init_time_ancestor, min(fin_time_parent-1, branch_time))
            print(f'modified to {branch_time}')
            icandf = self.generate_icandf_from_parent(parent, branch_time)
            memact = self.branching_state['members_active']
            log_active_weight_old = logsumexp([self.branching_state['log_weights'][ma] for ma in memact], b=[self.branching_state['multiplicities'][ma] for ma in memact])

        # ---------------- Run the new trajectory --------------
        new_score_components = self.ens.branch_or_plant(icandf, self.score_components, saveinfo, parent=parent)
        # ----------------------------------------------------------------------

        # Update the state
        new_mem = self.ens.get_nmem() - 1
        init_time_new,fin_time_new = self.ens.get_member_timespan(new_mem)
        new_score_combined = self.score_combined(new_score_components)
        new_score_max = np.nanmax(new_score_combined[:self.time_horizon-1])
        self.branching_state['goals_at_birth'].append(self.branching_state['score_levels'][-1])
        self.branching_state['branch_times'].append(branch_time)
        self.branching_state['scores_tdep'].append(new_score_combined)
        self.branching_state['scores_max'].append(new_score_max)
        self.branching_state['scores_max_timing'].append(1+init_time_new+np.nanargmax(new_score_combined)) 
        success = (new_score_max > self.branching_state['score_levels'][-1])
        memact = self.branching_state['members_active']
        # Update the weights
        if parent is None:
            self.branching_state['log_weights'].append(0.0)
        else:
            if success:
                logZ = np.log1p(np.exp(self.branching_state['log_weights'][parent] - log_active_weight_old))
                print(f'{logZ = }')
                for ma in self.branching_state['members_active']:
                    self.branching_state['log_weights'][ma] -= logZ
            self.branching_state['log_weights'].append(self.branching_state['log_weights'][parent])
        if success:
            self.branching_state['members_active'].append(new_mem)
            self.branching_state['multiplicities'].append(1)
        else:
            self.branching_state['multiplicities'].append(0)
            self.branching_state['multiplicities'][parent] += 1


        # Raise level? TODO allow the next level to be set by an external meta-manager, in between calls to take_next_step 
        if len(self.branching_state['parent_queue']) == 0 and self.autonomy and self.ens.get_nmem() >= self.population_size:
            self.raise_level_replenish_queue()
            # otherwise, the external caller will raise the level
        return
    def raise_level_replenish_queue(self):
        print(f'Raising level/replenishing queue')
        assert len(self.branching_state['parent_queue']) == 0
        scores_active = np.array([self.branching_state['scores_max'][ma] for ma in self.branching_state['members_active']])
        if True and self.ens.get_nmem() >= self.population_size: # Past the startup phase
            order = np.argsort(scores_active)
            num_leq = np.cumsum([self.branching_state['multiplicities'][order[j]] for j in range(len(order))])
            next_level = scores_active[order[np.where(num_leq >= self.num2drop)[0][0]]]
            self.branching_state['score_levels'].append(next_level)
            if (len(self.branching_state['score_levels']) >= self.num_levels_max) or (next_level >= scores_active[order[-1]]):
                self.terminate = True
            # Re-populate the parent queue
            self.branching_state['members_active'] = [ma for ma in self.branching_state['members_active'] if self.branching_state['scores_max'][ma] > next_level]
        if not self.terminate:
            parent_pool = self.rng.permutation(np.concatenate(tuple([parent]*self.branching_state['multiplicities'][parent] for parent in self.branching_state['members_active']))) # TODO consider weighting parents' occurrence in this pool by weight
            lenpp = len(parent_pool)
            deficit = self.population_size - len(self.branching_state['members_active'])
            for i in range(deficit):
                self.branching_state['parent_queue'].append(parent_pool[i % lenpp])
            print(f'The replenished queue is {self.branching_state["parent_queue"] = }')
        return
    # ----------------------- Plotting functions --------------------------------
    def plot_observable_spaghetti(self, obs_fun, outfile, ylabel='', title='', is_score=False):
        print(f'******************* \n \t {self.branching_state["branch_times"] = } \n *************')
        # Get all timespans
        tu = self.ens.dynsys.dt_save
        nmem = self.ens.get_nmem()
        N = self.population_size
        obs = [self.ens.compute_observables([obs_fun], mem)[0] for mem in range(nmem)]
        fig,axes = plt.subplots(ncols=2,figsize=(20,5),width_ratios=[3,1],sharey=is_score)
        ax = axes[0]
        for mem in range(nmem):
            if mem < N:
                kwargs = {'color': 'black', 'linestyle': '--', 'linewidth': 2, 'zorder': 0}
            else:
                kwargs = {'color': plt.cm.rainbow((mem-N)/(nmem-N)), 'linestyle': '-', 'linewidth': 1, 'zorder': 1}
            tinit,tfin = self.ens.get_member_timespan(mem)
            h, = ax.plot(np.arange(tinit+1,tfin+1)*tu, obs[mem], **kwargs)
            tbr = self.branching_state['branch_times'][mem]
            tmx = self.branching_state['scores_max_timing'][mem]
            ax.plot(tbr*tu, obs[mem][tmx-(tinit+1)], markerfacecolor="None", markeredgecolor=kwargs['color'], markeredgewidth=3, marker='o')
            ax.plot(tmx*tu, obs[mem][tmx-(tinit+1)], markerfacecolor="None", markeredgecolor=kwargs['color'], markeredgewidth=3, marker='x')
        ax.set_xlabel('time')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax = axes[1]
        ax.scatter(np.arange(N), self.branching_state['scores_max'][:N], c='black', marker='o')
        ax.scatter(np.arange(N,nmem,1), self.branching_state['scores_max'][N:nmem], c=plt.cm.rainbow(np.arange(nmem-N)/(nmem-N)), marker='o')
        ax.plot(np.arange(nmem), self.branching_state['goals_at_birth'], color='gray', linestyle='--')
        ax.set_xlabel('Generation')
        ax.set_ylabel('')
        #ax.set_xlim([time[0],time[-1]+1])
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return

class SDETEAMS(TEAMS):
    @staticmethod
    def choose_buicks_for_initialization(config, angel):
        if 'buick_choices' in config.keys():
            buick_choices = config['buick_choices']
        else:
            assert angel.num_buicks >= config['population_size'] # TODO allow repetition
            rng_buick_choice = default_rng(seed=config['seed_min'] + config['seed_inc_init'])
            buick_choices = rng_buick_choice.choice(np.arange(angel.num_buicks, dtype=int), size=config['population_size'], replace=False)
        return buick_choices
    @classmethod
    def initialize_from_ancestorgenerator(cls, angel, config, ens):
        init_conds = []
        init_times = []
        buick_choices = cls.choose_buicks_for_initialization(config, angel)
        for b in buick_choices:  
            parent = angel.branching_state['generation_0'][b]
            init_time_parent,fin_time_parent = angel.ens.get_member_timespan(parent)
            mdp = angel.ens.traj_metadata[parent]
            parent_t,parent_x = angel.ens.dynsys.load_trajectory(mdp, angel.ens.root_dir, tspan=[fin_time_parent]*2)
            init_conds.append(parent_x[0])
            init_times.append(fin_time_parent)
        return cls(init_times, init_conds, config, ens)
    @classmethod
    def initialize_from_coldstart(cls, config, ens):
        init_conds = []
        init_times = []
        rng = default_rng(seed=config['seed_min'] + config['seed_inc_init'])
        tu = ens.dynsys.dt_save
        for i in range(config['population_size']):
            frc_reseed = forcing.OccasionalReseedForcing(0, ens.dynsys.t_burnin, [0], [rng.integers(low=config['seed_min'],high=config['seed_max'])])
            frc_vector = forcing.OccasionalVectorForcing(0, ens.dynsys.t_burnin, [], [])
            icandf = dict({
                'init_cond': ens.dynsys.generate_default_init_cond(0),
                'init_rngstate': rng.bit_generator.state,
                'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
                })
            obs_fun = lambda t,x: x
            saveinfo = {'filename': f'spinup_{i}.npz'} # TODO remove .nc
            metadata,x = ens.dynsys.run_trajectory(icandf, obs_fun, saveinfo, ens.root_dir)
            init_conds.append(x[-1,:])
            init_times.append(ens.dynsys.t_burnin)
        return cls(init_times, init_conds, config, ens)

    def generate_icandf_from_parent(self, requested_parent, branch_time):
        # Set a new seed for the branching time 
        new_seed = self.rng.integers(low=self.seed_min,high=self.seed_max)
        init_rngstate = default_rng(seed=new_seed).bit_generator.state 
        reseed_times = [branch_time] 
        seeds = [new_seed]
        parent = requested_parent
        reseed_times_uplim = np.inf # Upper limit on where to inherit new seeds
        searching_back = True
        while searching_back: 
            init_time,fin_time = self.ens.get_member_timespan(parent)
            if self.inherit_perts_after_split:
                # Extract the relevant forcing sequence from the parent
                mdp = self.ens.traj_metadata[parent]
                parent_seed_frcs = [frc for frc in mdp['icandf']['frc'].frc_list if isinstance(frc,forcing.OccasionalReseedForcing)]
                assert len(parent_seed_frcs) == 1
                frc = parent_seed_frcs[0]
                # Inherit an appropriate subsequence
                idx_seeds_to_inherit = [
                        i for i in range(len(frc.reseed_times)) 
                        if (
                            (branch_time < frc.reseed_times[i] < reseed_times_uplim)
                            and
                            (frc.reseed_times[i] not in reseed_times) # more direct ancestors get priority for implanting their seeds into the new child 
                            )
                        ]
                if len(idx_seeds_to_inherit) > 0:
                    ft = frc.get_forcing_times()
                    reseed_times += [ft[i] for i in idx_seeds_to_inherit] 
                    seeds += [frc.seeds[i] for i in idx_seeds_to_inherit]
                    reseed_times_uplim = frc.reseed_times[idx_seeds_to_inherit[0]]
            searching_back = (init_time > branch_time)
            if searching_back:
                parent = next(self.ens.memgraph.predecessors(parent))
        mdp = self.ens.traj_metadata[parent]
        if branch_time == init_time:
            init_cond = mdp['icandf']['init_cond']
        else:
            parent_t,parent_x = self.ens.dynsys.load_trajectory(mdp, self.ens.root_dir, tspan=[branch_time]*2)
            init_cond = parent_x[0]
        print(f'{reseed_times = }, \n{seeds = }')
        if len(reseed_times) > 1:
            print(f'MULTIPLE RESEEDS!!')
            if not self.inherit_perts_after_split:
                raise Exception('Without ipas, supposed to have only one seed')
        order = np.argsort(reseed_times)
        frc_reseed = forcing.OccasionalReseedForcing(branch_time, fin_time, [reseed_times[order[i]] for i in range(len(reseed_times))], [seeds[order[i]] for i in range(len(reseed_times))])
        # TODO share more of the following forcings with the child
        frc_vector = forcing.OccasionalVectorForcing(branch_time, fin_time, [], [])
        icandf = dict({
            'init_cond': init_cond,
            'init_rngstate': init_rngstate,
            'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
            })
        return icandf

class SDEITEAMS(ITEAMS):
    @classmethod
    def initialize_from_ancestorgenerator(cls, angel, config, ens):
        parent = angel.branching_state['generation_0'][config['buick']]

        init_time_parent,fin_time_parent = angel.ens.get_member_timespan(parent)
        mdp = angel.ens.traj_metadata[parent]
        parent_t,parent_x = angel.ens.dynsys.load_trajectory(mdp, angel.ens.root_dir, tspan=[fin_time_parent]*2)
        init_cond = parent_x[0]
        return cls(fin_time_parent, init_cond, config, ens)
    def generate_icandf_from_parent(self, parent, branch_time):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent <= branch_time < init_time_parent + self.time_horizon
        mdp = self.ens.traj_metadata[parent]
        if branch_time == init_time_parent:
            init_cond = mdp['icandf']['init_cond']
        else:
            parent_t,parent_x = self.ens.dynsys.load_trajectory(mdp, self.ens.root_dir, tspan=[branch_time]*2)
            init_cond = parent_x[0]
        seed = self.rng.integers(low=self.seed_min,high=self.seed_max)
        init_rngstate = default_rng(seed=seed).bit_generator.state 
        frc_reseed = forcing.OccasionalReseedForcing(branch_time, fin_time_parent, [branch_time], [seed])
        frc_vector = forcing.OccasionalVectorForcing(branch_time, fin_time_parent, [], [])
        icandf = dict({
            'init_cond': init_cond,
            'init_rngstate': init_rngstate,
            'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
            })
        return icandf
            




    



    

