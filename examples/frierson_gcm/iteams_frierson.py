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
import algorithms; reload(algorithms)
print(f'{i = }'); i += 1
import frierson_gcm; reload(frierson_gcm)
from frierson_gcm import FriersonGCM
print(f'{i = }'); i += 1
import algorithms_frierson


def iteams(nproc,recompile,i_param,seed_inc):

    tododict = dict({
        'run_iteams':             1,
        'plot_spaghetti':         1,
        })
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-05"
    sub_date_str = "0/ITEAMS"
    config_gcm = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)

    # Parameters to loop over
    pert_type_list = ['IMP']        + ['SPPT']*20
    std_sppt_list = [0.5]           + [0.5,0.3,0.1,0.05,0.01]*4
    tau_sppt_list = [6.0*3600]      + [6.0*3600]*5   + [6.0*3600]*5    + [24.0*3600]*5     + [96.0*3600]*5 
    L_sppt_list = [500.0*1000]      + [500.0*1000]*5 + [2000.0*1000]*5 + [500.0*1000]*5    + [500.0*1000]*5

    config_gcm['pert_type'] = pert_type_list[i_param]
    if config_gcm['pert_type'] == 'SPPT':
        config_gcm['SPPT']['tau_sppt'] = tau_sppt_list[i_param]
        config_gcm['SPPT']['std_sppt'] = std_sppt_list[i_param]
        config_gcm['SPPT']['L_sppt'] = L_sppt_list[i_param]
    config_gcm['remove_temp'] = 1
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    config_algo = dict({
        'autonomy': True,
        'num_levels_max': 5,
        'seed_min': 1000,
        'seed_max': 10000,
        'population_size': 4,
        'time_horizon_phys': 15,
        'buffer_time_phys': 0,
        'advance_split_time_phys': 3,
        'num2drop': 1,
        'score_components': dict({
            'rainrate': dict({
                'observable': 'total_rain',
                'roi': dict({
                    'lat': 45,
                    'lon': 180,
                    }),
                'tavg': 1,
                'weight': 1.0,
                }),
            }),
        })
    param_abbrv_algo,param_label_algo = algorithms_frierson.FriersonGCMITEAMS.label_from_config(config_algo)

    dirdict = dict({
        'alg': join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
        })
    dirdict['plots'] = join(dirdict['alg'], 'plots')
    for dirname in list(dirdict.values()):
        makedirs(dirname, exist_ok=True)
    root_dir = dirdict['alg']
                
    alg_filename = join(dirdict['alg'],'alg.pickle')
    # TODO write config to file, too 
    init_cond_dir = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-03-05/0/DNS/',
            param_abbrv_gcm)
    init_time = int(xr.open_mfdataset(join(init_cond_dir,'mem20.nc'),decode_times=False)['time'].load()[-1].item())
    init_cond = relpath(join(init_cond_dir,'restart_mem20.cpio'), root_dir)
        
    if tododict['run_iteams']:
        if exists(alg_filename):
            alg = pickle.load(open(alg_filename, 'rb'))
        else:
            gcm = FriersonGCM(config_gcm, recompile=recompile)
            ens = Ensemble(gcm, root_dir=root_dir)
            alg = algorithms_frierson.FriersonGCMITEAMS(init_time, init_cond, config_algo, ens, gcm.seed_min+seed_inc)
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
    alg = pickle.load(open(alg_filename, 'rb'))
    if tododict['plot_spaghetti']:
        obsprop = alg.ens.dynsys.observable_props()
        obs_rois = [
                ('total_rain',dict(lat=45,lon=180)),
                ('column_water_vapor',dict(lat=45,lon=180)),
                ('temperature',dict(lat=45,lon=180,pfull=750)),
                ]
        for (obs_name,roi) in obs_rois:
            is_score = (obs_name in list(alg.score_params['components'].keys()))
            def obs_fun(dsmem):
                da = alg.ens.dynsys.sel_from_roi(getattr(alg.ens.dynsys, obs_name)(dsmem), roi)
                return da
            abbrv_obs = obsprop[obs_name]['abbrv']
            abbrv_roi,label_roi = alg.ens.dynsys.label_from_roi(roi)
            title = r'%s at %s'%(obsprop[obs_name]['label'],label_roi)
            abbrv = r'%s_%s'%(abbrv_obs,abbrv_roi)
            alg.plot_obs_spaghetti(obs_fun, dirdict['plots'], title=title, abbrv=abbrv, is_score=is_score)
    return

if __name__ == "__main__":
    procedure = 'run'
    print(f'Got into Main')
    if procedure == 'run':
        nproc = 4 
        recompile = 0 
        i_param = 2 #int(sys.argv[1])
        seed_inc = 0 #int(sys.argv[1])
        iteams(nproc,recompile,i_param,seed_inc)



