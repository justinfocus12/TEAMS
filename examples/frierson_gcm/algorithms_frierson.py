import numpy as np
from numpy.random import default_rng
from scipy.special import logsumexp,softmax
import xarray as xr
from matplotlib import pyplot as plt, rcParams 
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
from os.path import join, exists, basename, relpath
from os import mkdir, makedirs
import sys
import shutil
import glob
import subprocess
import resource
import pickle
import copy as copylib
from importlib import reload

sys.path.append("../..")
print(f'Now starting to import my own modules')
import utils; #reload(utils)
import ensemble; #reload(ensemble)
from ensemble import Ensemble
import forcing; #reload(forcing)
import algorithms; #reload(forcing)
import frierson_gcm; #reload(frierson_gcm)
from frierson_gcm import FriersonGCM


class FriersonGCMPeriodicBranching(algorithms.PeriodicBranching):
    def obs_dict_names(self):
        return ['total_rain','column_water_vapor','surface_pressure']
    def obs_fun(self, t, ds):
        lat = 45.0
        lon = 180.0
        obs = dict()
        for key in self.obs_dict_names():
            obs[key] = getattr(self.ens.dynsys, key)(ds).sel(dict(lat=lat,lon=lon),method='nearest')
        return obs
    def generate_icandf_from_parent(self, parent, branch_time, duration):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent <= branch_time <= fin_time_parent
        print(f'{init_time_parent = }, {branch_time = }, {fin_time_parent = }')
        if branch_time < fin_time_parent:
            init_cond = self.ens.traj_metadata[parent]['icandf']['init_cond']
            init_time = init_time_parent
        else:
            init_cond = self.ens.traj_metadata[parent]['filename_restart']
            init_time = fin_time_parent
        fin_time = branch_time + duration
        new_seed = self.rng.integers(low=self.seed_min, high=self.seed_max)
        if self.ens.dynsys.pert_type == 'SPPT':
            if init_time_parent < branch_time < fin_time_parent:
                reseed_times = [init_time,branch_time]
                seeds = [self.ens.traj_metadata[parent]['icandf']['frc'].seeds[0], new_seed]
            else:
                reseed_times = [branch_time]
                seeds = [new_seed] # TODO if possible, when on trunk, continue the random number generator
        else:
            if self.branching_state['on_trunk']:
                reseed_times = []
                seeds = []
            else:
                reseed_times = [branch_time]
                seeds = [new_seed]
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.OccasionalReseedForcing(init_time, fin_time, reseed_times, seeds),
            })
        return icandf

class FriersonGCMAncestorGenerator(algorithms.AncestorGenerator):
    def generate_icandf_from_uic(self):
        init_cond = self.uic
        init_time = self.uic_time
        fin_time = init_time + self.burnin_time
        reseed_times = [init_time]
        seeds = [self.rng.integers(low=self.seed_min, high=self.seed_max)]
        icandf = dict({
            'init_cond': self.uic,
            'frc': forcing.OccasionalReseedForcing(init_time, fin_time, reseed_times, seeds),
            })
        return icandf
    def generate_icandf_from_buick(self, parent):
        init_cond = self.ens.traj_metadata[parent]['filename_restart']
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        init_time = fin_time_parent
        fin_time = init_time + self.time_horizon
        reseed_times = [init_time]
        seeds = [self.rng.integers(low=self.seed_min, high=self.seed_max)]
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.OccasionalReseedForcing(init_time, fin_time, reseed_times, seeds),
            })
        return icandf

class FriersonGCMTEAMS(algorithms.TEAMS):
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
        assert angel.num_buicks >= config['population_size'] # TODO allow repetition
        buick_choices = cls.choose_buicks_for_initialization(config, angel)
        #self.buick_choices = buick_choices
        for b in range(config['population_size']):  
            parent = angel.branching_state['generation_0'][b]
            init_time_parent,fin_time_parent = angel.ens.get_member_timespan(parent)
            init_cond = relpath(
                    join(angel.ens.root_dir, angel.ens.traj_metadata[parent]['filename_restart']),
                    ens.root_dir)
            init_conds.append(init_cond)
            init_times.append(fin_time_parent)
        return cls(init_times, init_conds, config, ens)
    def derive_parameters(self, config):
        # Parameterize the score function in a simple way: the components will be area-averages of fields over specified regions. The combined score will be a linear combination.
        self.score_params = dict({
            'components': dict()
            })
        for compkey,compval in config['score_components'].items():
            sccomp = dict({'observable': compval['observable']}) # name of the observable function
            sccomp['roi'] = dict()
            for dim in ['lat','lon','pfull']:
                if dim in compval['roi'].keys():
                    if isinstance(compval['roi'][dim],list):
                        sccomp['roi'][dim] = slice(compval['roi'][dim][0],sccomp['roi'][dim][1])
                    else:
                        sccomp['roi'][dim] = compval['roi'][dim]
            sccomp['tavg'] = compval['tavg']
            sccomp['weight'] = compval['weight']
            self.score_params['components'][compkey] = sccomp.copy()
        super().derive_parameters(config)
        return
    def score_components(self, t, ds):
        scores = []
        for compkey,compval in self.score_params['components'].items():
            field = self.ens.dynsys.sel_from_roi(
                        getattr(self.ens.dynsys, compval['observable'])(ds),
                        compval['roi'])
            scores.append(field.mean(dim=set(field.dims) - {'time'}))
        return xr.concat(scores, dim='component').assign_coords(component=list(self.score_params['components'].keys()))
    def score_combined(self, sccomps):
        score = np.zeros(sccomps.time.size)
        total_weight = 0.0
        for compkey,compval in self.score_params['components'].items():
            conv = np.convolve(
                    np.ones(compval['tavg'])/compval['tavg'],
                    sccomps.sel(component=compkey).to_numpy(),
                    mode='full')[:sccomps['time'].size]
            conv[:(compval['tavg']-1)] = np.nan
            score += compval['weight']*conv
            total_weight += compval['weight']
        score /= total_weight
        score[:self.advance_split_time] = np.nan
        return score
    def merge_score_components(self, mem_leaf, score_components_leaf):
        # The child always starts from the same restart as the ancestor, so no merging necessary
        return score_components_leaf
    @staticmethod
    def label_from_config(config):
        abbrv_population,label_population = algorithms.TEAMS.label_from_config(config)
        # Append a code for the score
        obsprop = FriersonGCM.observable_props()
        comp_labels = []
        for compkey,compval in config['score_components'].items():
            roi_abbrv,roi_label = FriersonGCM.label_from_roi(compval['roi'])
            comp_label = r'%s%stavg%gd'%(
                    obsprop[compval['observable']]['abbrv'],
                    roi_abbrv,
                    compval['tavg'], 
                    )
            comp_labels.append(comp_label)
        abbrv_score = '_'.join(comp_labels) 
        abbrv = '_'.join([
            abbrv_population,
            abbrv_score,
            ])
        label = ', '.join([
            label_population,
            ])
        return abbrv,label
    def generate_icandf_from_parent(self, parent, branch_time):
        # Replicate all parent seeds occurring before branch time
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent <= branch_time < init_time_parent + self.time_horizon + self.buffer_time == fin_time_parent
        print(f'{init_time_parent = }, {branch_time = }, {fin_time_parent = }')
        init_cond = self.ens.traj_metadata[parent]['icandf']['init_cond']
        init_time = init_time_parent
        fin_time = init_time + self.time_horizon #+ self.buffer_time
        new_seed = self.rng.integers(low=self.seed_min, high=self.seed_max)
        # TODO consider carefully whether we need to distinguish procedure based on SPPT vs. other kinds of forcing
        if init_time_parent < branch_time:
            pfrc = self.ens.traj_metadata[parent]['icandf']['frc']
            reseed_times = []
            seeds = []
            # Replicate parent's seeds up until the branch time
            for i_rst,rst in enumerate(pfrc.reseed_times):
                if rst < branch_time:
                    reseed_times.append(rst)
                    seeds.append(pfrc.seeds[i_rst])
            # Put in a new seed for the branch time
            reseed_times.append(branch_time)
            seeds.append(new_seed)
        else:
            reseed_times = [branch_time]
            seeds = [new_seed] 
        # TODO If parent also has seeds following the branch time, MAYBE copy those too, to make use of useful forcing discovered by the parent 
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.OccasionalReseedForcing(init_time, fin_time, reseed_times, seeds),
            })
        return icandf
    @staticmethod
    def plot_boost_composites(algs, config_analysis, plotdir, param_suffix):
        for boost_size in config_analysis['composites']['boost_sizes']:
            for anc_score in config_analysis['composites']['anc_scores']:
                anc_min = anc_score - config_analysis['composites']['score_tolerance']/2
                anc_max = anc_score + config_analysis['composites']['score_tolerance']/2
                desc_min = anc_min + boost_size
                desc_max = anc_max + boost_size
                for (field_name,field_props) in config_analysis['fields_2d'].items():
                    fs_anc = []
                    fs_desc = []
                    logw_anc = []
                    logw_desc = []
                    print(f'--------------Compositing fields {field_name} ------------')
                    for i_alg,alg in enumerate(algs):
                        print(f'{i_alg = }, {alg.ens.get_nmem() = }')
                        ancs,descs = alg.collect_ancdesc_pairs_byscore(anc_min,anc_max,desc_min,desc_max)
                        print(f'{len(ancs) = }, {len(descs) = }')
                        for i_mem,mem in enumerate(np.concatenate((ancs, descs))):
                            print(f'{mem = }, {alg.branching_state["scores_max"][mem] = }')
                            tinit,tfin = alg.ens.get_member_timespan(mem)
                            tmx = alg.branching_state['scores_max_timing'][mem]
                            fun = lambda ds: field_props['fun'](ds).isel(time=tmx-tinit-1,drop=True)
                            f_new = alg.ens.compute_observables([fun], mem, compute=True)[0]
                            logw_new = alg.branching_state['log_weights'][mem]
                            if i_mem < len(ancs):
                                fs_anc.append(f_new)
                                logw_anc.append(logw_new)
                            else:
                                fs_desc.append(f_new)
                                logw_desc.append(logw_new)
                    # Compute average 
                    logw_anc = np.array(logw_anc)
                    logw_anc -= logsumexp(logw_anc)
                    logw_desc = np.array(logw_desc)
                    logw_desc -= logsumexp(logw_desc)
                    print(f'{logw_anc = }, {logw_desc = }')
                    f_mean_anc = (
                            xr.concat([fs_anc[i] * np.exp(logw_anc[i]) for i in range(len(fs_anc))], dim='member')
                            .assign_coords(member=np.arange(len(fs_anc)))
                            .sum(dim='member')
                            )
                    f_mean_desc = (
                            xr.concat([fs_desc[i] * np.exp(logw_desc[i]) for i in range(len(fs_desc))], dim='member')
                            .assign_coords(member=np.arange(len(fs_desc)))
                            .sum(dim='member')
                            )
                    f_diff_mean = f_mean_desc - f_mean_anc
                    f_diff_std = np.sqrt(
                            xr.concat([(fs_desc[i]-fs_anc[i]-f_diff_mean)**2 * np.exp(logw_desc[i]) for i in range(len(fs_desc))], dim='member')
                            .assign_coords(member=np.arange(len(fs_desc)))
                            .sum(dim='member')
                            )
                    vmin,vmax = min(f_mean_anc.min().item(),f_mean_desc.min().item()),max(f_mean_anc.max().item(),f_mean_desc.max().item())

                    fig,axes = plt.subplots(ncols=2, nrows=2, figsize=(16,8), sharey=True, sharex=True)
                    ax = axes[0,0]
                    xr.plot.pcolormesh(f_mean_anc,x='lon',y='lat',cmap=field_props['cmap'],ax=ax,vmin=vmin,vmax=vmax,cbar_kwargs={'orientation': 'vertical','label': None})
                    ax.set_title(r'Ancestors (scores %g-%g)'%(anc_min,anc_max))
                    ax = axes[0,1]
                    xr.plot.pcolormesh(f_mean_desc,x='lon',y='lat',cmap=field_props['cmap'],ax=ax,vmin=vmin,vmax=vmax,cbar_kwargs={'orientation': 'vertical','label': None})
                    ax.set_title(r'Descendants (scores %g-%g)'%(desc_min,desc_max))
                    ax = axes[1,0]
                    xr.plot.pcolormesh(f_diff_mean,x='lon',y='lat',cmap=field_props['cmap'],ax=ax,cbar_kwargs={'orientation': 'vertical','label': None})
                    ax.set_title("Mean diff.")
                    ax = axes[1,1]
                    xr.plot.pcolormesh(f_diff_std,x='lon',y='lat',cmap=field_props['cmap'],ax=ax,cbar_kwargs={'orientation': 'vertical','label': None})
                    ax.set_title("Std. diff.")
                    for ax in axes.flat:
                        ax.set(xlabel='',ylabel='')
                    fig.suptitle(r'%s composite (%d samples)'%(field_props['label'],len(fs_anc)),x=0.5, y=1.00, ha='center', va='bottom')
                    figfile = (r'composite_boost%gplus%g_%s_%s'%(anc_score,boost_size,field_props['abbrv'],param_suffix)).replace('.','p')
                    fig.savefig(join(plotdir,r'%s.png'%(figfile)),**pltkwargs)
                    plt.close(fig)
        return









class FriersonGCMITEAMS(algorithms.ITEAMS):
    @classmethod
    def initialize_from_ancestorgenerator(cls, angel, config, ens):

        init_time_parent,fin_time_parent = angel.ens.get_member_timespan(parent)
        init_cond = relpath(
                join(angel.ens.root_dir, angel.ens.traj_metadata[parent]['filename_restart']),
                ens.root_dir)
        return cls(fin_time_parent, init_cond, config, ens)
    def derive_parameters(self, config):
        # Parameterize the score function in a simple way: the components will be area-averages of fields over specified regions. The combined score will be a linear combination.
        self.score_params = dict({
            'components': dict()
            })
        for compkey,compval in config['score_components'].items():
            sccomp = dict({'observable': compval['observable']}) # name of the observable function
            sccomp['roi'] = dict()
            for dim in ['lat','lon','pfull']:
                if dim in compval['roi'].keys():
                    if isinstance(compval['roi'][dim],list):
                        sccomp['roi'][dim] = slice(compval['roi'][dim][0],sccomp['roi'][dim][1])
                    else:
                        sccomp['roi'][dim] = compval['roi'][dim]
            sccomp['tavg'] = compval['tavg']
            sccomp['weight'] = compval['weight']
            self.score_params['components'][compkey] = sccomp.copy()
        super().derive_parameters(config)
        return
    def score_components(self, t, ds):
        scores = []
        for compkey,compval in self.score_params['components'].items():
            field = self.ens.dynsys.sel_from_roi(
                        getattr(self.ens.dynsys, compval['observable'])(ds),
                        compval['roi'])
            scores.append(field.mean(dim=set(field.dims) - {'time'}))
        return xr.concat(scores, dim='component').assign_coords(component=list(self.score_params['components'].keys()))
    def score_combined(self, sccomps):
        score = np.zeros(sccomps.time.size)
        total_weight = 0.0
        for compkey,compval in self.score_params['components'].items():
            conv = np.convolve(
                    np.ones(compval['tavg'])/compval['tavg'],
                    sccomps.sel(component=compkey).to_numpy(),
                    mode='full')[:sccomps['time'].size]
            conv[:(compval['tavg']-1)] = np.nan
            score += compval['weight']*conv
            total_weight += compval['weight']
        score /= total_weight
        return score
    def merge_score_components(self, comps0, comps1, nsteps2prepend):
        return xr.concat(comps0.isel(time=slice(None,nsteps2prepend)),comps1,dim='time')
    @staticmethod
    def label_from_config(config):
        abbrv_population,label_population = algorithms.ITEAMS.label_from_config(config)
        # Append a code for the score
        obsprop = FriersonGCM.observable_props()
        comp_labels = []
        for compkey,compval in config['score_components'].items():
            roi_abbrv,roi_label = FriersonGCM.label_from_roi(compval['roi'])
            comp_label = r'%s%stavg%gd'%(
                    obsprop[compval['observable']]['abbrv'],
                    roi_abbrv,
                    compval['tavg'], 
                    )
            comp_labels.append(comp_label)
        abbrv_score = '_'.join(comp_labels) 
        abbrv = '_'.join([
            'ITEAMS',
            abbrv_population,
            abbrv_score,
            ])
        label = ', '.join([
            label_population,
            ])
        return abbrv,label
    def generate_icandf_from_parent(self, parent, branch_time):
        # Replicate all parent seeds occurring before branch time
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent <= branch_time < init_time_parent + self.time_horizon + self.buffer_time == fin_time_parent
        print(f'{init_time_parent = }, {branch_time = }, {fin_time_parent = }')
        init_cond = self.ens.traj_metadata[parent]['icandf']['init_cond']
        init_time = init_time_parent
        fin_time = init_time + self.time_horizon #+ self.buffer_time
        new_seed = self.rng.integers(low=self.seed_min, high=self.seed_max)
        # TODO consider carefully whether we need to distinguish procedure based on SPPT vs. other kinds of forcing
        if init_time_parent < branch_time:
            pfrc = self.ens.traj_metadata[parent]['icandf']['frc']
            reseed_times = []
            seeds = []
            # Replicate parent's seeds up until the branch time
            for i_rst,rst in enumerate(pfrc.reseed_times):
                if rst < branch_time:
                    reseed_times.append(rst)
                    seeds.append(pfrc.seeds[i_rst])
            # Put in a new seed for the branch time
            reseed_times.append(branch_time)
            seeds.append(new_seed)
        else:
            reseed_times = [branch_time]
            seeds = [new_seed] 
        # TODO If parent also has seeds following the branch time, MAYBE copy those too, to make use of useful forcing discovered by the parent 
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.OccasionalReseedForcing(init_time, fin_time, reseed_times, seeds),
            })
        return icandf

class FriersonGCMDirectNumericalSimulation(algorithms.DirectNumericalSimulation):
    def generate_icandf_from_parent(self, parent):
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        init_time = fin_time_parent
        fin_time = init_time + self.max_member_duration
        icandf = dict({
            'init_cond': self.ens.traj_metadata[parent]['filename_restart'],
            'frc': forcing.OccasionalReseedForcing(init_time, fin_time, [init_time], [self.rng.integers(low=self.seed_min,high=self.seed_max)]) # TODO gracefully continue seed from previous
            })
        return icandf
