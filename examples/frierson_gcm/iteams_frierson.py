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
import pprint
print(f'{i = }'); i += 1
from importlib import reload

sys.path.append("../..")
print(f'Now starting to import my own modules')
import utils; reload(utils)
print(f'{i = }'); i += 1
import ensemble; reload(ensemble)
print(f'{i = }'); i += 1
import forcing; reload(forcing)
print(f'{i = }'); i += 1
import algorithms; reload(algorithms)
print(f'{i = }'); i += 1
import frierson_gcm; reload(frierson_gcm)
print(f'{i = }'); i += 1
import algorithms_frierson; reload(algorithms_frierson)


def iteams_paramset(i_param):
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    config_gcm = frierson_gcm.FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)

    # Parameters to loop over
    pert_types = ['IMP']        + ['SPPT']*20
    std_sppts = [0.5]           + [0.5,0.3,0.1,0.05,0.01]*4
    tau_sppts = [6.0*3600]      + [6.0*3600]*5   + [6.0*3600]*5    + [24.0*3600]*5     + [96.0*3600]*5 
    L_sppts = [500.0*1000]      + [500.0*1000]*5 + [2000.0*1000]*5 + [500.0*1000]*5    + [500.0*1000]*5
    outputs_per_days = [4]*21
    seed_incs = [0]*21

    if pert_types[i_param] == 'IMP':
        expt_label = 'Impulsive'
        expt_abbrv = 'IMP'
    else:
        expt_label = r'SPPT, $\sigma=%g$, $\tau=%g$ h, $L=%g$ km'%(std_sppts[i_param],tau_sppts[i_param]/3600,L_sppts[i_param]/1000)
        expt_abbrv = r'SPPT_std%g_tau%gh_L%gkm'%(std_sppts[i_param],tau_sppts[i_param]/3600,L_sppts[i_param]/1000)

    config_gcm['outputs_per_day'] = outputs_per_days[i_param]
    config_gcm['pert_type'] = pert_types[i_param]
    if config_gcm['pert_type'] == 'SPPT':
        config_gcm['SPPT']['tau_sppt'] = tau_sppts[i_param]
        config_gcm['SPPT']['std_sppt'] = std_sppts[i_param]
        config_gcm['SPPT']['L_sppt'] = L_sppts[i_param]
    config_gcm['remove_temp'] = 1
    print(f'{i_param = }')
    pprint.pprint(config_gcm)

    config_algo = dict({
        'autonomy': True,
        'num_levels_max': 15,
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_param], 
        'population_size': 4,
        'time_horizon_phys': 15,
        'buffer_time_phys': 0,
        'advance_split_time_phys': 2,
        'num2drop': 1,
        'score_components': dict({
            'rainrate': dict({
                'observable': 'total_rain',
                'roi': dict({
                    'lat': 45,
                    'lon': 180,
                    }),
                'tavg': 1 * config_gcm['outputs_per_day'],
                'weight': 1.0,
                }),
            }),
        })
    return config_gcm,config_algo,expt_label,expt_abbrv

def iteams_single_workflow(i_param):
    config_gcm,config_algo,expt_label,expt_abbrv = iteams_paramset(i_param)
    param_abbrv_gcm,param_label_gcm = frierson_gcm.FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = algorithms_frierson.FriersonGCMITEAMS.label_from_config(config_algo)
    config_analysis = dict()
    config_analysis['target_location'] = dict(lat=45, lon=180)
    # observables (scalar quantities)
    observables = dict({
        'local_rain': dict({
            'fun': frierson_gcm.FriersonGCM.regional_rain,
            'kwargs': dict({
                'roi': config_analysis['target_location'],
                }),
            'abbrv': 'Rloc',
            'label': r'Rain rate $(\phi,\lambda)=(45,180)$',
            }),
        'local_dayavg_rain': dict({
            'fun': lambda ds,num_steps=1,roi=None: frierson_gcm.FriersonGCM.rolling_time_mean(
                frierson_gcm.FriersonGCM.regional_rain(ds,roi), num_steps),
            'kwargs': dict({
                'roi': config_analysis['target_location'],
                'num_steps': config_gcm['outputs_per_day'],
                }),
            'abbrv': 'Rloc1day',
            'label': r'Rain rate (day avg) $(\phi,\lambda)=(45,180)$',
            }),
        'area_rain_60x20': dict({
            'fun': frierson_gcm.FriersonGCM.regional_rain,
            'kwargs': dict(
                roi = dict({
                    'lat': slice(config_analysis['target_location']['lat']-10,config_analysis['target_location']['lat']+10),
                    'lon': slice(config_analysis['target_location']['lon']-30,config_analysis['target_location']['lon']+30),
                    }),
                ),
            'abbrv': 'R60x20',
            'label': r'Rain rate $(\phi,\lambda)=(45\pm10,180\pm30)$',
            }),
        'area_rain_90x30': dict({
            'fun': frierson_gcm.FriersonGCM.regional_rain,
            'kwargs': dict(
                roi = dict({
                    'lat': slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                    'lon': slice(config_analysis['target_location']['lon']-45,config_analysis['target_location']['lon']+45),
                    }),
                ),
            'abbrv': 'R90x30',
            'label': r'Rain rate $(\phi,\lambda)=(45\pm15,180\pm45)$',
            }),
        'local_cwv': dict({
            'fun': frierson_gcm.FriersonGCM.regional_cwv,
            'kwargs': dict(
                roi = config_analysis['target_location'],
                ),
            'abbrv': 'CWVloc',
            'label': r'Column water vapor $(\phi,\lambda)=(45,180)$',
            }),
        'area_cwv_60x20': dict({
            'fun': frierson_gcm.FriersonGCM.regional_cwv,
            'kwargs': dict(
                roi = dict(
                    lat=slice(config_analysis['target_location']['lat']-10,config_analysis['target_location']['lat']+10),
                    lon=slice(config_analysis['target_location']['lon']-30,config_analysis['target_location']['lon']+30),
                    ),
                ),
            'abbrv': 'CWV60x20',
            'label': r'Column water vapor $(\phi,\lambda)=(45\pm10,180\pm30)$',
            }),
        'area_cwv_90x30': dict({
            'fun': frierson_gcm.FriersonGCM.regional_cwv,
            'kwargs': dict({
                'roi': dict(
                    lat=slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                    lon=slice(config_analysis['target_location']['lon']-45,config_analysis['target_location']['lon']+45),
                    ),
                }),
            'abbrv': 'CWV90x30',
            'label': r'Column water vapor $(\phi,\lambda)=(45\pm15,180\pm45)$',
            }),
        })
    config_analysis['observables'] = observables
    obs_names = list(observables.keys())
    # Set up directories
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-26"
    sub_date_str = "0"
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    dirdict['init_cond'] = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-03-26/0/',
            param_abbrv_gcm, 'DNS_si0', 'data')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    filedict = dict()
    # Initial conditions
    filedict['init_cond'] = dict()
    filedict['init_cond']['restart'] = join(dirdict['init_cond'],'restart_mem20.cpio')
    filedict['init_cond']['trajectory'] = join(dirdict['init_cond'],'mem20.nc')
    print(f'{filedict["init_cond"] = }')
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')

    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def plot_observable_spaghetti(config_analysis, alg, dirdict):
    for (obs_name,obs_props) in config_analysis['observables'].items():
        is_score = False #(obs_name in list(alg.score_params['components'].keys()))
        obs_fun = lambda ds: obs_props['fun'](ds, **obs_props['kwargs'])
        outfile = join(dirdict['plots'], r'spaghetti_%s.png'%(obs_props['abbrv']))
        alg.plot_observable_spaghetti(obs_fun, outfile, title=obs_props['label'], is_score=is_score)
    return

def run_iteams(dirdict,filedict,config_gcm,config_algo):
    nproc = 4
    recompile = False
    root_dir = dirdict['data']
    init_time = int(round(
        xr.open_mfdataset(filedict['init_cond']['trajectory'], decode_times=False)['time'].load()[-1].item() 
        * config_gcm['outputs_per_day']))
    init_cond = relpath(filedict['init_cond']['restart'], root_dir)
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
    else:
        gcm = frierson_gcm.FriersonGCM(config_gcm, recompile=recompile)
        ens = ensemble.Ensemble(gcm, root_dir=root_dir)
        alg = algorithms_frierson.FriersonGCMITEAMS(init_time, init_cond, config_algo, ens)

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
        if exists(filedict['alg']):
            os.rename(filedict['alg'], filedict['alg_backup'])
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def iteams_single_procedure(i_param):

    tododict = dict({
        'run':             1,
        'analysis': dict({
            'observable_spaghetti':     1,
            }),
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = iteams_single_workflow(i_param)
    if tododict['run']:
        run_iteams(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, alg, dirdict)
    return

if __name__ == "__main__":
    print(f'Got into Main')
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_param = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'single'
        idx_param = [5] #list(range(1,21))
    print(f'Got into Main')
    if procedure == 'single':
        for i_param in idx_param:
            iteams_single_procedure(i_param)
    elif procedure == 'meta':
        iteams_meta_procedure(idx_param)



