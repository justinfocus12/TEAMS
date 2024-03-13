i = 0
print(f'--------------Beginning imports-------------')
import numpy as np
print(f'{i = }'); i += 1
from numpy.random import default_rng
print(f'{i = }'); i += 1
import xarray as xr
print(f'{i = }'); i += 1
from matplotlib import pyplot as plt, rcParams 
print(f'{i = }'); i += 1
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
print(f'{i = }'); i += 1
from os.path import join, exists, basename, relpath
print(f'{i = }'); i += 1
from os import mkdir, makedirs
print(f'{i = }'); i += 1
import sys
print(f'{i = }'); i += 1
import shutil
print(f'{i = }'); i += 1
import glob
print(f'{i = }'); i += 1
import subprocess
print(f'{i = }'); i += 1
import resource
print(f'{i = }'); i += 1
import pickle
print(f'{i = }'); i += 1
import copy as copylib
print(f'{i = }'); i += 1
from importlib import reload

sys.path.append("../..")
print(f'Now starting to import my own modules')
import utils; reload(utils)
print(f'{i = }'); i += 1
import ensemble; reload(ensemble)
from ensemble import Ensemble
print(f'{i = }'); i += 1
import forcing; reload(forcing)
print(f'{i = }'); i += 1
import algorithms; reload(forcing)
print(f'{i = }'); i += 1
import frierson_gcm; reload(frierson_gcm)
from frierson_gcm import FriersonGCM
print(f'{i = }'); i += 1

class FriersonGCMITEAMS(algorithms.ITEAMS):
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
                    if isinstance(compval[dim],list):
                        sccomp['roi'][dim] = slice(compval[dim][0],sccomp[dim][1])
                    else:
                        sccomp['roi'][dim] = sccomp[dim]
            sccomp['roi'] = roi.copy()
            sccomp['tavg'] = compval['tavg']
            sccomp['weight'] = compval['weight']
            self.score_params['components'][compkey] = sccomp.copy()
        super().__derive_parameters(config)
        return
    def score_components(self, t, ds):
        scores = []
        for compkey,compval in self.score_params['components'].items():
            field = self.ens.dynsys.sel_from_roi(
                        getattr(self.ens.dynsys, compval['observable'])(ds),
                        compval['roi'])
            scores.append(field.mean(dim=set(field.dims) - {'time'}))
        return xr.concat(scores, dim='component').assign_coords(component=list(self.score_params['components'].keys()))
    def score_combined(self, t, sccomps):
        # In principle, this could get arbitrarily complicated. 
        score = np.zeros(len(t))
        total_weight = 0.0
        for compkey,compval in self.score_params['components'].items():
            conv = np.convolve(
                    np.ones(compval['tavg'])/compval['tavg'],
                    sccomps.sel(component=compkey).to_numpy(),
                    mode='full')[:sccomps['time'].size-(compval['tavg']+1)]
            conv[:compval['tavg']-1] = np.nan
            score += compval['weight']*conv
            total_weight += compval['weight']
        score /= total_weight
        return score
    def label_from_score(config):
        obsprop = FriersonGCM.observable_props()
        comp_labels = []
        for compkey,compval in config['score_components'].items():
            comp_label = r'%s%stavg%d'%(
                    obsprop[compkey]['abbrv'],
                    FriersonGCM.label_from_roi(compval['roi']),
                    compval['tavg']
                    )
            comp_labels.append(comp_label)
        return '_'.join(comp_labels)
    def generate_icandf_from_parent(self, parent, branch_time):
        # Replicate all parent seeds occurring before branch time
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent <= branch_time < fin_time_parent
        print(f'{init_time_parent = }, {branch_time = }, {fin_time_parent = }')
        init_cond = self.ens.traj_metadata[parent]['icandf']['init_cond']
        init_time = init_time_parent
        fin_time = self.time_horizon + self.buffer_time
        new_seed = self.rng.integers(low=self.seed_min, high=self.seed_max)
        if self.ens.dynsys.pert_type == 'SPPT':
            if init_time_parent < branch_time:
                reseed_times = []
                for # TODO
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





