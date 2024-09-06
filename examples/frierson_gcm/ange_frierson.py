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


def ange_paramset(i_param):
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    config_gcm = frierson_gcm.FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)

    # Parameters to loop over
    pert_types = ['SPPT'] * 5
    std_sppts = [0.01,0.1,0.3,0.4,0.5]
    tau_sppts = [6.0*3600] * 5
    L_sppts = [500.0*1000] * 5
    seed_incs = [0] * 5

    if pert_types[i_param] == 'IMP':
        expt_label = 'Impulsive'
        expt_abbrv = 'IMP'
    else:
        expt_label = r'SPPT, $\sigma=%g$, $\tau=%g$ h, $L=%g$ km'%(std_sppts[i_param],tau_sppts[i_param]/3600,L_sppts[i_param]/1000)
        expt_abbrv = r'SPPT_std%g_tau%gh_L%gkm'%(std_sppts[i_param],tau_sppts[i_param]/3600,L_sppts[i_param]/1000)

    config_gcm['outputs_per_day'] = 4
    config_gcm['pert_type'] = pert_types[i_param]
    if config_gcm['pert_type'] == 'SPPT':
        config_gcm['SPPT']['tau_sppt'] = tau_sppts[i_param]
        config_gcm['SPPT']['std_sppt'] = std_sppts[i_param]
        config_gcm['SPPT']['L_sppt'] = L_sppts[i_param]
    config_gcm['remove_temp'] = 1
    print(f'{i_param = }')
    pprint.pprint(config_gcm)

    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_param], 
        'burnin_time_phys': 50, # should be about 100; start small for testing 
        'time_horizon_phys': 30,
        # mutable parameters below 
        'num_buicks': 128,
        'branches_per_buick': 3, 
        })
    return config_gcm,config_algo,expt_label,expt_abbrv

def ange_single_workflow(i_param):
    config_gcm,config_algo,expt_label,expt_abbrv = ange_paramset(i_param)
    param_abbrv_gcm,param_label_gcm = frierson_gcm.FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = algorithms_frierson.FriersonGCMAncestorGenerator.label_from_config(config_algo)
    config_analysis = dict()
    config_analysis['target_location'] = dict(lat=45, lon=180)
    # observables (scalar quantities)
    config_analysis['observables'] = dict({
        'local_dayavg_rain': dict({
            'fun': lambda ds,num_steps=1,roi=None: frierson_gcm.FriersonGCM.rolling_time_mean(
                frierson_gcm.FriersonGCM.regional_rain(ds,roi), num_steps),
            'kwargs': dict({
                'roi': config_analysis['target_location'],
                'num_steps': config_gcm['outputs_per_day'],
                }),
            'abbrv': 'Rloc1day',
            'label': r'Rain rate (day avg) $(\phi,\lambda)=(45,180)$',
            'unit_symbol': 'mm/day',
            }),
        'local_rain': dict({
            'fun': frierson_gcm.FriersonGCM.regional_rain,
            'kwargs': dict({
                'roi': config_analysis['target_location'],
                }),
            'abbrv': 'Rloc',
            'label': r'Rain rate $(\phi,\lambda)=(45,180)$',
            'unit_symbol': 'mm/day',
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
            'unit_symbol': 'mm/day',
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
            'unit_symbol': 'mm/day',
            }),
        'local_cwv': dict({
            'fun': frierson_gcm.FriersonGCM.regional_cwv,
            'kwargs': dict(
                roi = config_analysis['target_location'],
                ),
            'abbrv': 'CWVloc',
            'label': r'Column water vapor $(\phi,\lambda)=(45,180)$',
            "unit_symbol": r"kg m$^{-2}$",
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
            "unit_symbol": r"kg m$^{-2}$",
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
            "unit_symbol": r"kg m$^{-2}$",
            }),
        })

    # Distance metrics to quantify dispersion rates
    config_analysis['distance_metrics'] = dict()
    cwv_dist_fun = lambda ds0,ds1,dlon,dlat: frierson_gcm.FriersonGCM.dist_euc_cwv(ds0,ds1,{'lon': slice(180-dlon/2,180+dlon/2), 'lat': slice(45-dlat/2,45+dlat/2)})
    for (dlon,dlat) in ((15,5),(30,10),(60,20),(120,40)):
        key = r'area_cwv_%dx%d'%(dlon,dlat)
        config_analysis['distance_metrics'][key] = dict(
                fun = cwv_dist_fun,
                args = (dlon,dlat),
                abbrv = r'dCWV%dx%d'%(dlon,dlat),
                label = r'$L^2$(CWV), $%d^\circ\mathrm{lon}\times%d^\circ\mathrm{lat}$'%(dlon,dlat),
                unit_symbol = r'kg m$^{-2}$',
                )

    

    obs_names = list(config_analysis['observables'].keys())
    # Set up directories
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-07-07"
    sub_date_str = "0"
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    dirdict['init_cond'] = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-07-07/0',
            param_abbrv_gcm, 'DNS_si0', 'data', 'mem255')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    filedict = dict()
    # Initial conditions
    filedict['init_cond'] = dict()
    filedict['init_cond']['restart'] = join(dirdict['init_cond'],'restart_mem255.cpio')
    filedict['init_cond']['trajectory'] = join(dirdict['init_cond'],'history_mem255.nc')
    print(f'{filedict["init_cond"] = }')
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')

    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def plot_observable_spaghetti(config_analysis, alg, dirdict):
    for (obs_name,obs_props) in config_analysis['observables'].items():
        obs_fun = lambda ds: obs_props['fun'](ds, **obs_props['kwargs'])
        outfile = join(dirdict['plots'], r'spaghetti_burnin_%s.png'%(obs_props['abbrv']))
        alg.plot_observable_spaghetti_burnin(obs_fun, outfile, title=obs_props['label'], ylabel=r'[%s]'%(obs_props['unit_symbol']))
        for family in range(alg.num_buicks):
            outfile = join(dirdict['plots'], r'spaghetti_branching_%s_fam%d.png'%(obs_props['abbrv'],family))
            alg.plot_observable_spaghetti_branching(obs_fun, family, outfile, title=obs_props['label'], ylabel=r'[%s]'%(obs_props['unit_symbol']))
    return

def plot_observable_distribution(config_analysis, alg, dirdict):
    # Show the probability distribution of some (scalar) observable, like a score, of the path functional, over all paths 
    for obs_name in ['local_dayavg_rain']:
        obs_props = config_analysis['observables'][obs_name]
        print(f'{obs_props = }')
        score_fun = lambda ds: obs_props['fun'](ds, **obs_props['kwargs'])
        for buick in range(alg.branching_state['num_buicks_generated']):
            print(f'Starting {buick = }')
            outfile = join(dirdict['plots'], r'score_distn_%s_buick%d.png'%(obs_props['abbrv'],buick))
            alg.plot_score_distribution_branching(score_fun, buick, outfile, label=obs_props['label'])
    return

def quantify_dispersion_rate(config_analysis, alg, dirdict):
    tu = alg.ens.dynsys.dt_save 
    for (metric_name,metric) in config_analysis['distance_metrics'].items():
        dist_fun = lambda ds0,ds1: metric['fun'](ds0,ds1,*metric['args'])
        buicks2measure = np.arange(5, dtype=int)
        dists = alg.measure_dispersion(dist_fun,buicks=buicks2measure)
        # TODO fill in using the generic method from algorithms 
        fig_glob,axes_glob = plt.subplots(figsize=(8,5),nrows=2,ncols=1)
        for i_buick,buck in enumerate(buicks2measure):
            fig,ax = plt.subplots(figsize=(8,3))
            for dist in dists[i_buick]:
                ax.plot(tu*np.arange(1,alg.time_horizon+1), dist)
                for ax_glob in axes_glob:
                    ax_glob.plot(tu*np.arange(1,alg.time_horizon+1), dist,color=plt.cm.rainbow(i_buick/len(buicks2measure)))

            ax.set_xlabel('time [days]')
            ax.set_title(r'Buick %d, metric %s'%(i_buick,metric['label']))
            ax.set_ylabel(r'[%s]'%(metric['unit_symbol']))
            ax.set_yscale('log')
            fig.savefig(join(dirdict['plots'], r'dispersion_metric%s_buick%d.png'%(metric['abbrv'],i_buick)), **pltkwargs)
            plt.close(fig)
        for ax_glob in axes_glob:
            ax_glob.set_xlabel('time [days]')
            ax_glob.set_title(r'Metric %s'%(metric['label']))
            ax_glob.set_ylabel(r'[%s]'%(metric['unit_symbol']))
        axes_glob[1].set_yscale('log')
        fig_glob.savefig(join(dirdict['plots'], r'dispersion_metric%s.png'%(metric['abbrv'])), **pltkwargs)
        plt.close(fig_glob)
    return



def run_ange(dirdict,filedict,config_gcm,config_algo):
    nproc = 4
    recompile = False
    root_dir = dirdict['data']
    uic_time = int(round(
        xr.open_mfdataset(filedict['init_cond']['trajectory'], decode_times=False)['time'].load()[-1].item() 
        * config_gcm['outputs_per_day']))
    uic = relpath(filedict['init_cond']['restart'], root_dir)
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
        alg.set_capacity(config_algo['num_buicks'], config_algo['branches_per_buick'])
    else:
        gcm = frierson_gcm.FriersonGCM(config_gcm, recompile=recompile)
        ens = ensemble.Ensemble(gcm, root_dir=root_dir)
        alg = algorithms_frierson.FriersonGCMAncestorGenerator(uic_time, uic, config_algo, ens)

    alg.ens.dynsys.set_nproc(nproc)
    alg.ens.set_root_dir(root_dir)
    while not alg.terminate:
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict({
            'temp_dir': f'mem{mem}_temp',
            'final_dir': f'mem{mem}',
            })
        saveinfo.update(dict({
            'filename_traj': join(saveinfo['final_dir'],f'history_mem{mem}.nc'),
            'filename_restart': join(saveinfo['final_dir'],f'restart_mem{mem}.cpio'),
            }))
        alg.take_next_step(saveinfo)
        if exists(filedict['alg']):
            os.rename(filedict['alg'], filedict['alg_backup'])
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def measure_running_max(config_analysis, alg, dirdict):
    for (obs_name,obs_props) in config_analysis['observables'].items():
        print(f'Beginning running max analysis of {obs_name}')
        runmax_file = join(dirdict['analysis'], r'running_max_%s.npz'%(obs_props['abbrv']))
        figfile_prefix = join(dirdict['plots'], r'running_max_%s'%(obs_props['abbrv']))
        score_fun = lambda ds: obs_props['fun'](ds, **obs_props['kwargs'])
        alg.measure_running_max(score_fun, runmax_file, figfile_prefix, label=obs_props['label'], abbrv=obs_props['abbrv'], precomputed=False)
    return

def ange_single_procedure(i_expt):
    tododict = dict({
        'run':             1,
        'analysis': dict({
            'observable_spaghetti':     1,
            'observable_distribution':  1,
            'observable_running_max':   1,
            'dispersion':               0,
            }),
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = ange_single_workflow(i_expt)
    if tododict['run']:
        run_ange(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, alg, dirdict)
        # TODO have another ancestor-wise version, and another that shows family lines improving in parallel and dropping out
    if tododict['analysis']['observable_distribution']:
        plot_observable_distribution(config_analysis, alg, dirdict)
    if tododict['analysis']['observable_running_max']:
        measure_running_max(config_analysis, alg, dirdict)
    if tododict['analysis']['dispersion']:
        quantify_dispersion_rate(config_analysis, alg, dirdict)
    return

if __name__ == "__main__":
    print(f'Got into Main')
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_param = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'single'
        idx_param = [3] #list(range(1,21))
    print(f'Got into Main')
    if procedure == 'single':
        for i_param in idx_param:
            ange_single_procedure(i_param)
    elif procedure == 'meta':
        ange_meta_procedure(idx_param)



