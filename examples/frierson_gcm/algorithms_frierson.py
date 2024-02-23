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
import utils
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
        'run_pebr':                0,
        'plot_pebr': dict({
            'observables':    1,
            'divergence':     0,
            'response':       0,
            }),
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
        'branches_per_group': 16, 
        'interbranch_interval_phys': 10.0,
        'branch_duration_phys': 20.0,
        'num_branch_groups': 20,
        'max_member_duration_phys': 25.0,
        })
    seed = 849582 # TODO make this a command-line argument
    param_abbrv_algo,param_label_algo = FriersonGCMPeriodicBranching.label_from_config(config_algo)
    algdir = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
    root_dir = algdir
    makedirs(algdir, exist_ok=True)
    alg_filename = join(algdir,'alg.pickle')

    init_cond_dir = '/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm/2024-02-21/0/DNS/resT21_abs1_pert0p001'
    init_time = int(xr.open_mfdataset(join(init_cond_dir,'mem24.nc'),decode_times=False)['time'].load()[-1].item())
    init_cond = relpath(join(init_cond_dir,'restart_mem24.cpio'), root_dir)
        
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
    if utils.find_true_in_dict(tododict['plot_pebr']):
        plotdir = join(algdir,'plots')
        makedirs(plotdir,exist_ok=True)

        alg = pickle.load(open(join(algdir,'alg.pickle'),'rb'))
        print(f'{alg.trunk_duration = }')
        print(f'{alg.branching_state["trunk_lineage_fin_times"] = }')
        if tododict['plot_pebr']['observables']:
            obsprop = alg.ens.dynsys.observable_props()
            obs_names = ['temperature','total_rain','column_water_vapor','surface_pressure']
            lat = 45.0
            lon = 180.0
            pfull = 500.0
            obs_funs = dict()
            obs_abbrvs = dict()
            obs_labels = dict()
            for obs_name in obs_names:
                def obs_fun(ds):
                    field = getattr(alg.ens.dynsys, obs_name)(ds).sel(lat=lat, lon=lon, method='nearest')
                    if 'pfull' in field.dims:
                        field = field.sel(pfull=pfull, method='nearest')
                    return field
                obs_funs[obs_name] = obs_fun
                obs_abbrvs[obs_name] = obsprop[obs_name]['abbrv']
                obs_labels[obs_name] = obsprop[obs_name]['label']
            for branch_group in range(alg.branching_state['next_branch_group']):
                alg.plot_obs_spaghetti(obs_funs, branch_group, plotdir, labels=obs_labels, abbrvs=obs_abbrvs)


        if tododict['plot_pebr']['divergence']:
            # Plot distance from trunk for all branches, in terms of (u,v) coordinates
            trulin = alg.branching_state['trunk_lineage']
            for mem in trulin:
                print(f'{mem = } in trunk; {alg.ens.get_member_timespan(mem) = }')
            dist_region = dict(lat=slice(lat-10,lat+10),lon=slice(lon-10,lon+10))
            ds_trunk = (
                    xr.open_mfdataset([
                        join(ens.root_dir,ens.traj_metadata[mem]['filename_traj'])
                        for mem in alg.branching_state['trunk_lineage']],
                        decode_times=False,
                        data_vars=["ucomp","vcomp"])
                    .sel(dist_region)
                    .sel(pfull=pfull,method='nearest')
                    )
            print(f'{ds_trunk.time.values = }')
            dist2trunk = dict()
            for mem in np.setdiff1d(np.arange(ens.memgraph.number_of_nodes()), alg.branching_state['trunk_lineage']):
                dsmem = (
                        xr.open_dataset(
                            join(ens.root_dir,ens.traj_metadata[mem]['filename_traj']),
                            decode_times=False)
                        .sel(dist_region)
                        .sel(pfull=pfull,method='nearest')
                        )
                dsmem.drop_vars(np.setdiff1d(list(dsmem.data_vars.keys()),['ucomp','vcomp']))
                common_time = np.intersect1d(dsmem.time.to_numpy(), ds_trunk.time.to_numpy())
                if len(common_time) > 0:
                    timesel = dict(time=common_time)
                    print(f'{ds_trunk["ucomp"].sel(timesel).shape = }')
                    print(f'{dsmem["ucomp"].sel(timesel).shape = }')
                    dist2trunk[mem] = np.sqrt((
                        (dsmem['ucomp'].sel(timesel) - ds_trunk['ucomp'].sel(timesel))**2 + 
                        (dsmem['vcomp'].sel(timesel) - ds_trunk['vcomp'].sel(timesel))**2
                        )
                        .sum(dim=['lat','lon']))

            fig,ax = plt.subplots()
            for (mem,dist) in dist2trunk.items():
                xr.plot.plot(dist, x='time', label=f'm{mem}')
            ax.set_yscale('log')
            ax.set_title('Distance to trunk')
            ax.set_xlabel('Time')
            ax.set_ylabel('Euclidean distance')
            fig.savefig(join(plotdir,f'dist2trunk.png'), **pltkwargs)
            plt.close(fig)

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
                fig.savefig(join(plotdir,f'{obslib[obs]["abbrv"]}.png'),**pltkwargs)





        


    return

if __name__ == "__main__":
    print(f'Got into Main')
    nproc = int(sys.argv[1])
    print(f'{nproc = }')
    test_periodic_branching(nproc)
