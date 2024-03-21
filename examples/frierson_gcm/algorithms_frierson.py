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

