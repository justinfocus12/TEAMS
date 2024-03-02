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

def test_periodic_branching(nproc,pert_type):
    tododict = dict({
        'run_pebr':                0,
        'analyze_pebr': dict({
            'measure_pert_growth':           1,
            'analyze_pert_growth':           1,
            }),
        'plot_pebr': dict({
            'observables':    0,
            'pert_growth':    1,
            'response':       0,
            }),
        })
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm"
    date_str = "2024-03-01"
    sub_date_str = "0/PeBr"
    print(f'About to generate default config')
    config_gcm = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)
    config_gcm['pert_type'] = pert_type
    if pert_type == 'SPPT':
        config_gcm['SPPT']['tau_sppt'] = 6.0 * 3600.0 # units are seconds
        config_gcm['SPPT']['std_sppt'] = 0.1
    config_gcm['remove_temp'] = 1
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'branches_per_group': 8, 
        'interbranch_interval_phys': 5.0,
        'branch_duration_phys': 10.0,
        'num_branch_groups': 12,
        'max_member_duration_phys': 30.0,
        })
    param_abbrv_algo,param_label_algo = FriersonGCMPeriodicBranching.label_from_config(config_algo)
    seed = 849582 # TODO make this a command-line argument

    # Set up directories
    dirdict = dict({
        'alg': join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
        })
    dirdict['analysis'] = join(dirdict['alg'],'analysis')
    dirdict['plots'] = join(dirdict['alg'],'plots')
    for dirname in list(dirdict.values()):
        makedirs(dirname, exist_ok=True)

    # Enumerate filenames
    fndict = dict({
        'alg': dict({
            'alg': join(dirdict['alg'],'alg.pickle'),
            }),
        'analysis': dict({
            'pert_growth': join(dirdict['analysis'],'pert_growth.pickle'),
            'lyap_exp': join(dirdict['analysis'],'lyap_exp.pickle')
            })
        })
    fndict['plots'] = dict()
    dist_names = ['temperature','column_water_vapor','surface_pressure','total_rain']
    for dist_name in dist_names:
        fndict['plots'][dist_name] = dict({'rmse': join(dirdict['plots'],f'rmse_dist{dist_name}')})
        fndict['plots'][dist_name]['lyap_exp'] = join(dirdict['plots'], f'lyap_exp_dist{dist_name}')
        for branch_group in range(config_algo['num_branch_groups']):
            fndict['plots'][dist_name][branch_group] = join(dirdict['plots'],f'pert_growth_bg{branch_group}_dist{dist_name}.png')

    root_dir = dirdict['alg']
    alg_filename = join(dirdict['alg'],'alg.pickle')
    # TODO write config to file, too 


    init_cond_dir = f'/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm/2024-02-29/0/DNS/resT21_abs1_frc{pert_type}'
    init_time = int(xr.open_mfdataset(join(init_cond_dir,'mem20.nc'),decode_times=False)['time'].load()[-1].item())
    init_cond = relpath(join(init_cond_dir,'restart_mem20.cpio'), root_dir)
        
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
            mem = alg.ens.get_nmem()
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
    if utils.find_true_in_dict(tododict['analyze_pebr']):
        alg = pickle.load(open(fndict['alg']['alg'], 'rb'))
        if tododict['analyze_pebr']['measure_pert_growth']:
            obsprop = alg.ens.dynsys.observable_props()
            lat = 45.0
            lon = 180.0
            pfull = 500.0
            latlonsel = dict(lat=slice(lat-10,lat+10),lon=slice(lon-30,lon+30))
            def dist_euclidean(ds0,ds1,field_name):
                t0 = ds0.time.to_numpy()
                t1 = ds1.time.to_numpy()
                tsel = dict(time=slice(max(t0[0],t1[0]),min(t0[-1],t1[-1])+1))
                f0 = getattr(alg.ens.dynsys, field_name)(ds0).sel(tsel).sel(latlonsel)
                f1 = getattr(alg.ens.dynsys, field_name)(ds1).sel(tsel).sel(latlonsel)
                if 'pfull' in f0.dims:
                    f0 = f0.sel(pfull=pfull,method='nearest')
                    f1 = f1.sel(pfull=pfull,method='nearest')
                # TODO add a cosine weighting 
                return np.sqrt(((f0-f1)**2).sum(dim=set(f0.dims)-{'time'})).compute().to_numpy()
            def rmsd_euclidean(ds0,ds1,field_name):
                f0 = getattr(alg.ens.dynsys, field_name)(ds0).sel(latlonsel)#.compute().to_numpy()
                f1 = getattr(alg.ens.dynsys, field_name)(ds1).sel(latlonsel)#.compute().to_numpy()
                if 'pfull' in f0.dims:
                    f0 = f0.sel(pfull=pfull,method='nearest')
                    f1 = f1.sel(pfull=pfull,method='nearest')
                f0 = f0.stack(latlon=['lat','lon']).transpose('time','latlon').to_numpy()
                f1 = f1.stack(latlon=['lat','lon']).transpose('time','latlon').to_numpy()
                D2mat = np.add.outer(np.sum(f0**2, axis=1), np.sum(f1**2, axis=1)) - 2*f0.dot(f1.T)
                return np.sqrt(np.mean(D2mat))
            dist_funs = dict({'tdep': dict(), 'rmsd': dict()})
            for field_name in ['temperature','total_rain','column_water_vapor','surface_pressure']:
                dist_funs['tdep'][field_name] = lambda ds0,ds1,field_name=field_name: dist_euclidean(ds0,ds1,field_name)
                dist_funs['rmsd'][field_name] = lambda ds0,ds1,field_name=field_name: rmsd_euclidean(ds0,ds1,field_name)
            pert_growth = alg.measure_pert_growth(dist_funs)
            pickle.dump(pert_growth, open(fndict['analysis']['pert_growth'], 'wb'))
        else:
            pert_growth = pickle.load(open(fndict['analysis']['pert_growth'], 'rb'))
        if tododict['analyze_pebr']['analyze_pert_growth']:
            lyapunov_exponents = alg.analyze_pert_growth(pert_growth)
            pickle.dump(lyapunov_exponents, open(fndict['analysis']['lyap_exp'], 'wb'))
    if utils.find_true_in_dict(tododict['plot_pebr']):
        plotdir = join(dirdict['alg'],'plots')
        makedirs(plotdir,exist_ok=True)

        alg = pickle.load(open(join(dirdict['alg'],'alg.pickle'),'rb'))
        if tododict['plot_pebr']['observables']:
            obsprop = alg.ens.dynsys.observable_props()
            lat = 45.0
            lon = 180.0
            pfull = 500.0
            obs_funs = dict()
            for obs_name in ['temperature']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(alg.ens.dynsys, obs_name)(dsmem).sel(lat=lat,lon=lon,pfull=pfull,method='nearest')
            for obs_name in ['total_rain','column_water_vapor','surface_pressure']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(alg.ens.dynsys, obs_name)(dsmem).sel(lat=lat,lon=lon,method='nearest')
            obs_names = list(obs_funs.keys())
            obs_abbrvs = dict()
            obs_labels = dict()
            obs_units = dict()
            for obs_name in obs_names:
                obs_abbrvs[obs_name] = obsprop[obs_name]['abbrv']
                if obs_name == 'temperature':
                    locstr = r'$(\lambda,\phi,p)=(%g^\circ,%g^\circ,%g hPa)$'%(lon,lat,pfull)
                else:
                    locstr = r'$(\lambda,\phi)=(%g^\circ,%g^\circ)$'%(lon,lat)
                obs_labels[obs_name] = r'%s at %s'%(obsprop[obs_name]['label'],locstr)
                obs_abbrvs[obs_name] = obsprop[obs_name]['abbrv']
                obs_units[obs_name] = r'[%s]'%(obsprop[obs_name]['unit_symbol'])
            for branch_group in range(alg.num_branch_groups): #range(alg.branching_state['next_branch_group']):
                alg.plot_obs_spaghetti(obs_funs, branch_group, plotdir, ylabels=obs_units, titles=obs_labels, abbrvs=obs_abbrvs)


        if tododict['plot_pebr']['pert_growth']:
            pert_growth_dict = pickle.load(open(fndict['analysis']['pert_growth'],'rb'))
            lyap_dict = pickle.load(open(fndict['analysis']['lyap_exp'],'rb'))
            alg.plot_pert_growth(pert_growth_dict, lyap_dict, fndict['plots'], logscale=True)
        if False:
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
    pert_type = ['IMP','SPPT'][int(sys.argv[2])]
    print(f'{nproc = }')
    test_periodic_branching(nproc,pert_type)
