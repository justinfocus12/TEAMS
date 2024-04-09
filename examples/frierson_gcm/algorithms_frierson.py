import numpy as np
from numpy.random import default_rng
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
    @classmethod
    def initialize_from_ancestorgenerator(cls, angel, config, ens):
        init_conds = []
        init_times = []
        assert angel.num_buicks >= config['population_size'] # TODO allow repetition
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
