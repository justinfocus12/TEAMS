from abc import ABC, abstractmethod
from collections import deque
import pprint
import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from scipy.stats import linregress 
from scipy.special import logsumexp, softmax
from os.path import join, exists
from os import makedirs
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
        abbrv = 'DNS_si%d'%(config['seed_inc_init'])
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
    def set_simulation_capacity(self, num_chunks_max, max_member_duration_phys):
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
        last_mem = np.where(all_ends >= tspan[1])[0][0]
        time = 1 + np.arange(tspan[0],tspan[1])
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
        bin_lows,hist,rtime,logsf = utils.compute_returnstats_and_histogram(fconcat, time_block_size)
        np.savez(
                outfile, 
                bin_lows=bin_lows,
                hist=hist,
                rtime=rtime,
                logsf=logsf)
        if rtime[-1] == rtime[-2]:
            print(f'{hist = }')
            print(f'{rtime = }')
        return
     
    # ------------------ Plotting -----------------------------
    def plot_obs_segment(self, obs_fun, tspan, fig, ax, **linekwargs):
        time,memset,tidx = self.get_member_subset(tspan)
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
        print(f'{time = }')
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
    # ************** Perturbation growth ****************
    def measure_dispersion(self, dist_fun, dispfile):
        # Save a statistical analysis of RMSE growth to a specified directory
        ngroups = self.branching_state['next_branch_group']+1
        split_times = np.zeros(ngroups, dtype=int)
        print(f'{split_times = }')
        dists = np.zeros((ngroups, self.branches_per_group, self.branch_duration)) 
        for branch_group in range(ngroups):
            print(f'About to compute distances for {branch_group = }')
            time,dists_local = self.compute_pairwise_fun_local(dist_fun, branch_group)
            split_times[branch_group] = time[0]
            dists[branch_group,:,:] = dists_local.copy()
        rmses = np.sqrt(np.mean(dists**2, axis=1))
        rmsd = np.sqrt(np.mean(rmses[:,-1]**2))
        np.savez(
                dispfile,
                split_times = split_times,
                dists = dists,
                rmses = rmses,
                rmsd = rmsd,
                )
        return 
    @staticmethod
    def compute_elfs_and_fsle(satfracs, dispfile, outfile):
        # elfs = elapsed lagtime until fractional saturation
        disp_data = np.load(dispfile)
        split_times = disp_data['split_times']
        dists = disp_data['dists']
        rmses = disp_data['rmses']
        rmsd = disp_data['rmsd']
        ngroups,nbranches,ntimes = dists.shape
        nfracs = len(satfracs)
        # Three measures of error growt 
        lyap_expons = np.nan*np.ones((ngroups,nfracs)) # RMSE ~ exp(lyap_expon*t)
        diff_pows = np.nan*np.ones((ngroups,nfracs)) # RMSE ~ t**(diff_pows)
        elfs = np.zeros((ngroups,nfracs), dtype=int)
        for group in range(ngroups):
            log_rmse = np.log(rmses[group,:])
            time_prev = 1 # beginning of interval over which to measure growth
            for i_frac,frac in enumerate(satfracs):
                elfs[group,i_frac] = np.where(rmses[group,:] >= frac*rmsd)[0][0]
                # Measure diffusive growth
                tidx = np.arange(time_prev,elfs[group,i_frac], dtype=int)
                if len(tidx) > 0:
                    diff_pows[group,i_frac] = linregress(
                            np.log(tidx), 
                            np.log(rmses[group,tidx])
                            ).slope
                    lyap_expons[group,i_frac] = linregress(
                            tidx, np.log(rmses[group,tidx])
                            ).slope
                    time_prev = tidx[-1]
        np.savez(
                outfile,
                satfracs = satfracs, 
                elfs = elfs,
                lyapunov_exponents = lyap_expons,
                diffusive_powers = diff_pows,
                )
        return 
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
    def plot_dispersion_allgroups(self, dispfile, satfractime_file, outfile, title=''):
        fig,axes = plt.subplots(ncols=2,figsize=(6,4), width_ratios=[2,1])
        tu = self.ens.dynsys.dt_save
        disp_data = np.load(dispfile)
        split_times = disp_data['split_times']
        dists = disp_data['dists']
        rmses = disp_data['rmses']
        rmsd = disp_data['rmsd']
        print(f'{rmses.max() = }, {rmsd = }')
        satfractime_data = np.load(satfractime_file)
        satfracs = satfractime_data['satfracs']
        elfs = satfractime_data['elfs']
        lyap_expons = satfractime_data['lyapunov_exponents']
        diff_pows = satfractime_data['diffusive_powers']
        fig,axes = plt.subplots(nrows=2,figsize=(6,10), sharex=True)
        handles = []
        for i_sf,sf in enumerate(satfracs):
            color = plt.cm.viridis(i_sf/len(satfracs))
            ax = axes[0]
            h, = ax.plot(split_times, lyap_expons[:,i_sf], color=color, marker='.', label=r'$f=%g$'%(sf))
            handles.append(h)
            ax.set_ylabel(r'$\lambda(f)$')
            ax = axes[1]
            ax.plot(split_times, elfs[:,i_sf], color=color, marker='.')
            ax.set_ylabel(r'$\tau(f)$')
        axes[0].legend(handles=handles)
        axes[0].set_title(title)
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)



        

    def plot_dispersion_onegroup(self, group, dispfile, satfractime_file, outfile, ylabel='', title='', logscale=False):
        # TODO add in the fractional saturation times 
        tu = self.ens.dynsys.dt_save
        disp_data = np.load(dispfile)
        split_times = disp_data['split_times']
        dists = disp_data['dists']
        rmses = disp_data['rmses']
        rmsd = disp_data['rmsd']
        print(f'{rmses.max() = }, {rmsd = }')
        satfractime_data = np.load(satfractime_file)
        satfracs = satfractime_data['satfracs']
        elfs = satfractime_data['elfs']
        lyap_expons = satfractime_data['lyapunov_exponents']
        diff_pows = satfractime_data['diffusive_powers']
        ngroups,nbranches,ntimes = dists.shape
        time = np.arange(ntimes) # local time 
        fig,ax = plt.subplots()
        for i_mem1 in range(nbranches):
            ax.plot(time[1:]*tu, dists[group,i_mem1,1:], color='tomato',)
        hrmse, = ax.plot(time[1:]*tu, rmses[group,1:], color='black', label='RMSE')
        ax.axhline(rmsd, color='black', linestyle='--', label='RMSD')
        # Exponential growth model
        time_prev = 1
        for i_sf,sf in enumerate(satfracs):
            tidx = np.arange(time_prev,elfs[group,i_sf])
            #hpow, = ax.plot(time[tidx]*tu, rmses[group][tidx[0]] * (time/time[tidx[0]])**diff_pows[group,i_sf], color='dodgerblue', label='diffusive')
            hexp, = ax.plot(
                    time[tidx]*tu, 
                    rmses[group,time[tidx[0]]] * np.exp(
                        (time[tidx]-time[tidx[0]]) * 
                        lyap_expons[group,i_sf]), 
                    color='limegreen', label='exponential')
            ax.axvline((time[tidx[-1]]-time[0])*tu, color='black', linewidth=0.5)
            time_prev = tidx[-1]
        ax.legend(handles=[hrmse,hexp])
        ax.set_xlabel(r'time since %g'%(split_times[group]*tu))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logscale: ax.set_yscale('log')
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        print(f'{outfile = }')
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
    def plot_obs_spaghetti(self, obs_fun, branch_group, outfile, ylabel='', title='', abbrv=''):
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
            'frc': forcing.OccasionalVectorForcing(fin_time_parent, fin_time_parent+self.chunk_size, [fin_time_parent], [impulse])
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

class ITEAMS(EnsembleAlgorithm):
    # TEAMS starting from a fixed initial condition. The TEAMS algorithm may wrap this, or just be similar; TBD
    def __init__(self, init_time, init_cond, config, ens, seed):
        self.set_init_cond(init_time, init_cond) # Unlike for general Algorithms, an initial condition is mandatory
        super().__init__(config, ens, seed)
        return
    def derive_parameters(self, config):
        self.autonomy = config['autonomy'] # True if this single family is isolated, False if part of a team.
        self.num_levels_max = config['num_levels_max']
        tu = self.ens.dynsys.dt_save
        self.time_horizon = int(round(config['time_horizon_phys']/tu))
        self.buffer_time = int(round(config['buffer_time_phys']/tu)) # Time between the end of one interval and the beginning of the next, when generating the initial ensemble. Add this to the END of ancestral trajectories. 
        self.advance_split_time = int(round(config['advance_split_time_phys']/tu))
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
                r'N%d_T%g_ast%g_drop%d_si%d'%(
                    config['population_size'],
                    config['time_horizon_phys'],
                    config['advance_split_time_phys'],
                    config['num2drop'],
                    config['seed_inc_init'],
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
            icandf = self.ens.dynsys.generate_default_icandf(self.init_time,self.init_time+self.time_horizon+self.buffer_time)
            icandf['init_cond'] = self.init_cond
            log_active_weight_old = -np.inf
            branch_time = self.init_time
        else:
            print(f'self.branching_state = ')
            pprint.pprint(self.branching_state)
            parent = self.branching_state['parent_queue'].popleft()
            init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
            assert self.branching_state['scores_max'][parent] > self.branching_state['score_levels'][-1]
            first_exceedance_time_parent = init_time_parent + np.where(self.branching_state['scores_tdep'][parent]  > self.branching_state['score_levels'][-1])[0][0]
            # TODO correct the branch timing
            branch_time = first_exceedance_time_parent - self.advance_split_time #TODO
            print(f'{branch_time = }')
            if branch_time < init_time_parent:
                branch_time = init_time_parent
                print(f'...modified to {branch_time = }')
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
        self.branching_state['scores_max_timing'].append(init_time_new+np.nanargmax(new_score_combined))
        self.branching_state['branch_times'].append(branch_time)
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
        if len(self.branching_state['parent_queue']) == 0 and self.autonomy:
            self.raise_level_replenish_queue()
            # otherwise, the external caller will raise the level
        return
    def raise_level_replenish_queue(self):
        assert len(self.branching_state['parent_queue']) == 0
        scores_active = np.array([self.branching_state['scores_max'][ma] for ma in self.branching_state['members_active']])
        if len(scores_active) > 1: # Past the startup phase
            order = np.argsort(scores_active)
            num_leq = np.cumsum([self.branching_state['multiplicities'][order[j]] for j in range(len(order))])
            next_level = scores_active[order[np.where(num_leq >= self.num2drop)[0][0]]]
            self.branching_state['score_levels'].append(next_level)
            if (len(self.branching_state['score_levels']) >= self.num_levels_max) or (next_level >= scores_active[order[-1]]):
                self.terminate = True
            # Re-populate the parent queue
            self.branching_state['members_active'] = [ma for ma in self.branching_state['members_active'] if self.branching_state['scores_max'][ma] > next_level]
        parent_pool = self.rng.permutation(np.concatenate(tuple([parent]*self.branching_state['multiplicities'][parent] for parent in self.branching_state['members_active']))) # TODO consider weighting parents' occurrence in this pool by weight
        lenpp = len(parent_pool)
        deficit = self.population_size - len(self.branching_state['members_active'])
        for i in range(deficit):
            self.branching_state['parent_queue'].append(parent_pool[i % lenpp])
        print(f'The replenished queue is {self.branching_state["parent_queue"] = }')
        return
    # ----------------------- Plotting functions --------------------------------
    def plot_obs_spaghetti(self, obs_fun, plotdir, ylabel='', title='', abbrv='', is_score=False):
        # Get all timespans
        tu = self.ens.dynsys.dt_save
        nmem = self.ens.get_nmem()
        obs = [self.ens.compute_observables([obs_fun], mem)[0] for mem in range(nmem)]
        # TODO update this generic plotting function to extract time span from the metadata
        print(f'{obs[0] = }')
        fig,axes = plt.subplots(ncols=2,figsize=(20,5),width_ratios=[3,1],sharey=is_score)
        ax = axes[0]
        for mem in range(nmem):
            if mem == 0:
                kwargs = {'color': 'black', 'linestyle': '--', 'linewidth': 2, 'zorder': 1}
            else:
                kwargs = {'color': plt.cm.rainbow(mem/nmem), 'linestyle': '-', 'linewidth': 1, 'zorder': 0}
            tinit,tfin = self.ens.get_member_timespan(mem)
            h, = ax.plot(np.arange(tinit,tfin)*tu, obs[mem], **kwargs)
            tbr = self.branching_state['branch_times'][mem]
            tmx = self.branching_state['scores_max_timing'][mem]
            ax.plot(tbr*tu, obs[mem][tmx-tinit], markerfacecolor="None", markeredgecolor=kwargs['color'], markeredgewidth=3, marker='o')
            ax.plot(tmx*tu, obs[mem][tmx-tinit], markerfacecolor="None", markeredgecolor=kwargs['color'], markeredgewidth=3, marker='x')
        ax.set_xlabel('time')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax = axes[1]
        ax.scatter(np.arange(nmem), self.branching_state['scores_max'], color='gray', marker='o')
        ax.plot(np.arange(nmem), self.branching_state['goals_at_birth'], color='gray', linestyle='--')
        ax.set_xlabel('Generation')
        ax.set_ylabel('')
        #ax.set_xlim([time[0],time[-1]+1])
        fig.savefig(join(plotdir,r'spaghetti_%s.png'%(abbrv)), **pltkwargs)
        plt.close(fig)
        return


class SDEITEAMS(ITEAMS):
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
            'init_cond': parent_x[0],
            'init_rngstate': init_rngstate,
            'frc': forcing.SuperposedForcing([frc_vector,frc_reseed]),
            })
        return icandf

            




    



    

