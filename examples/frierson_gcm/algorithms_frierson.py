# Instantiation of EnsembleMember class on Frierson GCM

import numpy as np
from numpy.random import default_rng
from scipy.special import softmax
import xarray as xr
import dask
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
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

import sys
sys.path.append("../..")
from ensemble import Ensemble
from dynamicalsystem import DynamicalSystem
import forcing
import algorithms
import frierson_observables as frobs
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
        seed = self.rng.integers(low=self.seed_min, high=self.seed_max)
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.ContinuousTimeForcing(init_time, fin_time, [branch_time], [seed]),
            })
        return icandf

def test_periodic_branching(nproc):
    tododict = dict({
        'run_pebr':                1,
        'plot_pebr':               0,
        })
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm"
    date_str = "2024-02-21"
    sub_date_str = "0/PeBr"
    print(f'About to generate default config')
    config_gcm = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'branches_per_group': 4, 
        'interbranch_interval_phys': 10.0,
        'branch_duration_phys': 20.0,
        'num_branch_groups': 10,
        'max_member_duration_phys': 25.0,
        })
    seed = 849582 # TODO make this a command-line argument
    param_abbrv_algo,param_label_algo = FriersonGCMPeriodicBranching.label_from_config(config_algo)
    algdir = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
    root_dir = algdir
    makedirs(algdir, exist_ok=True)
    alg_filename = join(algdir,'alg.pickle')

    init_cond_dir = join(scratch_dir, date_str, sub_date_str)
    init_time = int(xr.open_mfdataset(join(init_cond_dir,'mem19.nc'),decode_times=False)['time'].load()[-1].item())
    init_cond = relpath(join(init_cond_dir,'restart_mem19.cpio'), root_dir)
        
    if tododict['run_pebr']:
        if exists(alg_filename):
            alg = pickle.load(open(alg_filename, 'rb'))
        else:
            gcm = FriersonGCM(config_gcm)
            ens = Ensemble(gcm, root_dir=root_dir)
            alg = FriersonGCMPeriodicBranching(config_algo, ens, seed)
            alg.set_init_cond(init_time,init_cond)

        alg.ens.dynsys.set_nproc(nproc)
        alg.ens.set_root_dir(root_dir)
        while not alg.terminate:
            mem = alg.ens.memgraph.number_of_nodes()
            print(f'----------- Starting member {mem} ----------------')
            saveinfo = dict({
                # Temporary folder
                'temp_dir': f'mem{mem}',
                # Ultimate resulting filenames
                'filename_traj': f'mem{mem}.nc',
                'filename_restart': f'restart_mem{mem}.cpio',
                })
            alg.take_next_step(saveinfo)
            pickle.dump(alg, open(alg_filename, 'wb'))
    if tododict['plot_pebr']:
        plot_dir = join(algdir,'plots')
        makedirs(plot_dir,exist_ok=True)

        alg = pickle.load(open(join(algdir,'alg.pickle'),'rb'))
        ens = alg.ens
        obslib = ens.dynsys.observable_props()
        obs2plot = ['temperature','total_rain','column_water_vapor','surface_pressure'][1:]
        lat = 45.0
        lon = 180.0
        pfull = 500.0

        # Plot local observables for all members
        obs_vals = dict({obs: [] for obs in obs2plot})
        for mem in range(ens.memgraph.number_of_nodes()):
            dsmem = xr.open_mfdataset(join(ens.root_dir,ens.traj_metadata[mem]['filename_traj']), decode_times=False)
            for obs in obs2plot:
                memobs = getattr(ens.dynsys, obs)(dsmem).sel(dict(lat=lat,lon=lon),method='nearest').compute()
                if 'pfull' in memobs.dims:
                    memobs = memobs.sel(pfull=pfull,method='nearest')
                obs_vals[obs].append(memobs)
        for obs in obs2plot:
            fig,ax = plt.subplots(figsize=(20,5))
            handles = []
            for mem in range(ens.memgraph.number_of_nodes()):
                h, = xr.plot.plot(obs_vals[obs][mem], x='time', label=f'm{mem}', marker='o')
                handles.append(h)
            ax.legend(handles=handles)
            ax.set_title(obslib[obs]['label'])
            fig.savefig(join(plot_dir,f'{obslib[obs]["abbrv"]}.png'),**pltkwargs)

    return

if __name__ == "__main__":
    print(f'Got into Main')
    nproc = int(sys.argv[1])
    print(f'{nproc = }')
    test_periodic_branching(nproc)
