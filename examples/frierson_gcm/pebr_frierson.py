
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
#from importlib import reload

sys.path.append("../..")
print(f'Now starting to import my own modules')
import utils; #reload(utils)
print(f'{i = }'); i += 1
from ensemble import Ensemble
print(f'{i = }'); i += 1
import forcing; #reload(forcing)
print(f'{i = }'); i += 1
import algorithms; #reload(forcing)
print(f'{i = }'); i += 1
#import frierson_gcm; #reload(frierson_gcm)
from frierson_gcm import FriersonGCM
print(f'{i = }'); i += 1
from algorithms_frierson import FriersonGCMPeriodicBranching
print(f'{i = }'); i += 1

def pebr_paramset(i_param):
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    config_gcm = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)

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
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_param], 
        'branches_per_group': 2, 
        'interbranch_interval_phys': 2.0, # small interval helps to see continuity in stability
        'branch_duration_phys': 5.0,
        'num_branch_groups': 2,
        'max_member_duration_phys': 40.0,
        })
    return config_gcm,config_algo,expt_label,expt_abbrv

# -------------- Define the observable functions of interest (to be further parameterized) ---------
# ----------- Define the distance metrics of interest -------



def pebr_workflow(i_param):
    print(f'About to generate default config; {i_param = }')
    config_gcm,config_algo,expt_label,expt_abbrv = pebr_paramset(i_param)
    ngroups = config_algo['num_branch_groups']
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = FriersonGCMPeriodicBranching.label_from_config(config_algo)
    # Configure post-analysis
    # List the quantities of interest
    config_analysis = dict()
    config_analysis['target_location'] = dict(lat=45, lon=180)
    # observables (scalar quantities)
    obsprop = FriersonGCM.observable_props()
    observables = dict({
        'local_rain': dict({
            'fun': FriersonGCM.regional_rain,
            'kwargs': dict({
                'roi': config_analysis['target_location'],
                }),
            'abbrv': 'Rloc',
            'label': r'Rain rate $(\phi,\lambda)=(45,180)$',
            }),
        'area_rain_60x20': dict({
            'fun': FriersonGCM.regional_rain,
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
            'fun': FriersonGCM.regional_rain,
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
            'fun': FriersonGCM.regional_cwv,
            'kwargs': dict(
                roi = config_analysis['target_location'],
                ),
            'abbrv': 'CWVloc',
            'label': r'Column water vapor $(\phi,\lambda)=(45,180)$',
            }),
        'area_cwv_60x20': dict({
            'fun': FriersonGCM.regional_cwv,
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
            'fun': FriersonGCM.regional_cwv,
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
    # distance metrics
    dist_metrics = dict({
        'euc_area_horzvel_30x10': dict({
            'fun': FriersonGCM.dist_euc_horzvel,
            'abbrv': 'UVEuc30x10',
            'label': 'Horz. Vel. Eucl. dist. (30x10)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-5,config_analysis['target_location']['lat']+5),
                    'lon': slice(config_analysis['target_location']['lon']-15,config_analysis['target_location']['lon']+15),
                    'pfull': 500,
                    }),
                }),
            }),
        'euc_area_horzvel_60x20': dict({
            'fun': FriersonGCM.dist_euc_horzvel,
            'abbrv': 'UVEuc60x20',
            'label': 'Horz. Vel. Eucl. dist. (60x20)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-10,config_analysis['target_location']['lat']+10),
                    'lon': slice(config_analysis['target_location']['lon']-30,config_analysis['target_location']['lon']+30),
                    'pfull': 500,
                    }),
                }),
            }),
        'euc_area_horzvel_90x30': dict({
            'fun': FriersonGCM.dist_euc_horzvel,
            'abbrv': 'UVEuc90x30',
            'label': 'Horz. Vel. Eucl. dist. (90x30)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                    'lon': slice(config_analysis['target_location']['lon']-45,config_analysis['target_location']['lon']+45),
                    'pfull': 500,
                    }),
                }),
            }),
        'euc_area_ps_30x10': dict({
            'fun': FriersonGCM.dist_euc_ps,
            'abbrv': 'PsurfEuc30x10',
            'label': 'Surf. Pres. Eucl. dist. (30x10)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-5,config_analysis['target_location']['lat']+5),
                    'lon': slice(config_analysis['target_location']['lon']-15,config_analysis['target_location']['lon']+15),
                    }),
                }),
            }),
        'euc_area_ps_60x20': dict({
            'fun': FriersonGCM.dist_euc_ps,
            'abbrv': 'PsurfEuc60x20',
            'label': 'Surf. Pres. Eucl. dist. (60x20)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-10,config_analysis['target_location']['lat']+10),
                    'lon': slice(config_analysis['target_location']['lon']-30,config_analysis['target_location']['lon']+30),
                    }),
                }),
            }),
        'euc_area_ps_90x30': dict({
            'fun': FriersonGCM.dist_euc_ps,
            'abbrv': 'PsurfEuc90x30',
            'label': 'Surf. Pres. Eucl. dist. (90x30)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                    'lon': slice(config_analysis['target_location']['lon']-45,config_analysis['target_location']['lon']+45),
                    }),
                }),
            }),
        'euc_area_rain_30x10': dict({
            'fun': FriersonGCM.dist_euc_rain,
            'abbrv': 'RainEuc30x10',
            'label': 'Rain-Euclidean distance (30x10)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-5,config_analysis['target_location']['lat']+5),
                    'lon': slice(config_analysis['target_location']['lon']-15,config_analysis['target_location']['lon']+15),
                    }),
                }),
            }),
        'euc_area_rain_60x20': dict({
            'fun': FriersonGCM.dist_euc_rain,
            'abbrv': 'RainEuc60x20',
            'label': 'Rain-Euclidean distance (60x20)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-10,config_analysis['target_location']['lat']+10),
                    'lon': slice(config_analysis['target_location']['lon']-30,config_analysis['target_location']['lon']+30),
                    }),
                }),
            }),
        'euc_area_rain_90x30': dict({
            'fun': FriersonGCM.dist_euc_rain,
            'abbrv': 'RainEuc90x30',
            'label': 'Rain-Euclidean distance (90x30)',
            'kwargs': dict({
                'roi': dict({
                    'lat': slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                    'lon': slice(config_analysis['target_location']['lon']-45,config_analysis['target_location']['lon']+45),
                    }),
                }),
            }),
        })
    dist_names = list(dist_metrics.keys())
    config_analysis['dist_metrics'] = dist_metrics

    # How to quantitatively measure perturbation growth, and also perhaps the Lyapunov exponents/power laws between them 
    config_analysis['satfracs'] = np.array([1/8,1/4,3/8,1/2])


    # Set up directories
    dirdict = dict()
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-26"
    sub_date_str = "DEBUG/1"
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    dirdict['init_cond'] = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-03-26/0/',
            param_abbrv_gcm, 'DNS_si0', 'data')

    for dirname in ['data','analysis','plots']:
        makedirs(dirdict[dirname], exist_ok=True)

    filedict = dict()
    # Initial conditions
    filedict['init_cond'] = dict()
    filedict['init_cond']['restart'] = join(dirdict['init_cond'],'restart_mem20.cpio')
    filedict['init_cond']['trajectory'] = join(dirdict['init_cond'],'mem20.nc')
    print(f'{filedict["init_cond"] = }')
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')

    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def run_pebr(dirdict,filedict,config_gcm,config_algo):
    nproc = 4
    recompile = False
    root_dir = dirdict['data']
    init_time = int(round(xr.open_mfdataset(filedict['init_cond']['trajectory'], decode_times=False)['time'].load()[-1].item() * config_gcm['outputs_per_day']))
    init_cond = relpath(filedict['init_cond']['restart'], root_dir)
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
    else:
        gcm = FriersonGCM(config_gcm, recompile=recompile)
        ens = Ensemble(gcm, root_dir=root_dir)
        alg = FriersonGCMPeriodicBranching(config_algo, ens)
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
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def quantify_dispersion_rates(config_analysis, alg, dirdict):
    for dist_name,dist_props in config_analysis['dist_metrics'].items():
        print(f'{dist_name = }')
        print(f'{dist_props = }')
        def dist_fun(ds0,ds1):
            # TODO anticipate future when time samples are subdaily
            t0 = (ds0['time'].to_numpy() / alg.ens.dynsys.dt_save).astype(int)
            t1 = (ds1['time'].to_numpy() / alg.ens.dynsys.dt_save).astype(int)
            print(f'{t0 = }\n{t1 = }')
            trange_full = np.arange(min(t0[0],t1[0]),max(t0[-1],t1[-1])+1)
            trange_valid = np.arange(max(t0[0],t1[0]),min(t0[-1],t1[-1])+1)
            tidx0 = trange_valid - t0[0]
            tidx1 = trange_valid - t1[0]
            dist = np.nan*np.ones_like(trange_full)
            dist = dist_props['fun'](ds0.isel(time=tidx0), ds1.isel(time=tidx1), **dist_props['kwargs'])
            return dist
        dispersion_file = join(dirdict['analysis'],r'dispersion_%s.npz'%(dist_props['abbrv']))
        dispersion_stats = alg.measure_dispersion(dist_fun, config_analysis['satfracs'], dispersion_file)
        # Should we also plot right here? Yes, why not 
        figfile_prefix = join(dirdict['plots'],r'dispersion_%s'%(dist_props['abbrv']))
        groups2plot = np.arange(min(dispersion_stats['dists'].shape[0],10), dtype=int)
        alg.plot_dispersion(
                dispersion_stats, figfile_prefix, groups2plot=groups2plot,  
                title=dist_props['label'], logscale=False
                )
    return 

def plot_observable_spaghetti(config_analysis, alg, dirdict):
    for obs_name,obs_props in config_analysis['observables'].items():
        print(f'{obs_name = }')
        print(f'{obs_props = }')
        obs_fun = lambda ds: obs_props['fun'](ds,**obs_props['kwargs'])
        ylabel = obs_props['label']
        for group in range(min(4,alg.branching_state['next_branch_group']+1)):
            title = r'Group %d'%(group)
            outfile = join(dirdict['plots'], r'spaghetti_obs%s_bg%d.png'%(obs_props['abbrv'],group))
            alg.plot_observable_spaghetti(obs_fun,group,outfile,ylabel=ylabel,title=title)
            # TODO maybe precompute all the observables in advance, in case they're used for multiple purposes
    return

def quantify_running_max_convergence(config_analysis, alg, dirdict):
    for obs_name,obs_props in config_analysis['observables'].items():
        print(f'{obs_name = }')
        print(f'{obs_props = }')
        obs_fun = lambda ds: obs_props['fun'](ds,**obs_props['kwargs'])
        # Collect running maxima 
        runmax_file = join(dirdict['analysis'], r'running_max_%s.npz'%(obs_props['abbrv']))
        figfile_prefix = join(dirdict['plots'], r'running_max_%s'%(obs_props['abbrv']))
        alg.measure_running_max(obs_fun, runmax_file, figfile_prefix, label=obs_props['label'])
    return

def old_thing():
    if 0 and utils.find_true_in_dict(tododict['plot_pebr']):
        alg = pickle.load(open(join(dirdict['alg'],'alg.pickle'),'rb'))
        # ----------------- Perturbation growth ---------------------------
        if 0 and tododict['plot_pebr']['pert_growth']:
            split_times = np.load(join(dirdict['analysis'],'split_times.npy'))
            for field_name in ['temperature','total_rain']:
                location_abbrv,location_label = alg.ens.dynsys.label_from_roi(dist_roi[field_name]) 
                pgs = np.load(join(dirdict['analysis'],r'pert_growth_summary_%s_%s.npz'%(obsprop[field_name]['abbrv'],location_abbrv)))
                dist_filename = (r'dist_%s_%s'%(obsprop[field_name]['abbrv'],location_abbrv)).replace('.','p')
                dists = np.load(join(dirdict['analysis'],f'{dist_filename}.npy'))
                plot_suffix = r'%s_%s'%(obsprop[field_name]['abbrv'],location_abbrv)
                alg.plot_pert_growth(split_times, dists, pgs['thalfsat'], pgs['diff_expons'], pgs['lyap_expons'], pgs['rmses'], pgs['rmsd'].item(), dirdict['plots'], plot_suffix, logscale=True)
        # ---------------- Observables -------------------
        if 0 and tododict['plot_pebr']['observables']:
            for obs_name in ['temperature','total_rain']:
                obs_fun = lambda dsmem: alg.ens.dynsys.sel_from_roi(getattr(alg.ens.dynsys, obs_name)(dsmem), obs_roi[obs_name])
                roi_abbrv,roi_label = alg.ens.dynsys.label_from_roi(obs_roi[obs_name])
                obs_abbrv = r'%s_%s'%(obsprop[obs_name]['abbrv'],roi_abbrv)
                obs_label = r'%s at %s'%(obsprop[obs_name]['label'],roi_label)
                obs_unit = r'[%s]'%(obsprop[obs_name]['unit_symbol'])
                for branch_group in range(min(3,alg.num_branch_groups)): #range(alg.branching_state['next_branch_group']):
                    alg.plot_obs_spaghetti(obs_fun, branch_group, dirdict['plots'], ylabel=obs_unit, title=obs_label, abbrv=obs_abbrv)
        if 0 and tododict['plot_pebr']['fields']:
            # Plot a panel of ensemble members each day 
            obs_funs = dict()
            for obs_name in ['temperature']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(alg.ens.dynsys, obs_name)(dsmem).sel(lat=slice(lat-20,lat+20),lon=slice(lon-45,lon+45)).sel(pfull=pfull,method='nearest')
            for obs_name in ['r_sppt_g','total_rain','column_water_vapor','surface_pressure']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(alg.ens.dynsys, obs_name)(dsmem).sel(lat=slice(lat-20,lat+20),lon=slice(lon-45,lon+45))
            obs_names = list(obs_funs.keys())
            tu = alg.ens.dynsys.dt_save
            for branch_group in range(min(5,alg.num_branch_groups)): #range(alg.branching_state['next_branch_group']):
                time,mems_trunk,tidx_trunk,mems_branch,tidxs_branch = alg.get_tree_subset(branch_group)
                print(f'{time = }')
                obs_dict_branch = alg.ens.compute_observables(obs_funs, mems_branch)
                obs_dict_trunk = alg.ens.compute_observables(obs_funs, mems_trunk)
                for obs_name in obs_names:
                    obs_dict_trunk[obs_name] = xr.concat(obs_dict_trunk[obs_name],dim='time')
                    print(f'{obs_dict_trunk[obs_name].time.values = }')
                print(f'{mems_branch = }')
                for obs_name in obs_names:
                    print(f'----------Plotting {obs_name}-----------')
                    nmem2plot = min(4,len(mems_branch))
                    for i_time,t in enumerate(time):
                        fig,axes = plt.subplots(nrows=1+nmem2plot,ncols=2,figsize=(12,(1+nmem2plot)*4))
                        axes[0,1].axis('off')
                        ax = axes[0,0]
                        xr.plot.pcolormesh(obs_dict_trunk[obs_name].sel(time=t), x='lon', y='lat', ax=ax, cmap=obsprop[obs_name]['cmap'])
                        ax.set_title('CTRL')
                        for i_mem,mem in enumerate(mems_branch[:nmem2plot]):
                            print(f'{i_mem = }, {mem = }')
                            ax = axes[i_mem+1,0]
                            field2plot = obs_dict_branch[obs_name][i_mem].sel(time=t)
                            print(f'{field2plot.min().item() = }, {field2plot.max().item() = }')
                            xr.plot.pcolormesh(obs_dict_branch[obs_name][i_mem].sel(time=t), x='lon', y='lat', ax=ax, cmap=obsprop[obs_name]['cmap'])
                            ax.set_title(r'PERT %d'%(i_mem))
                            ax = axes[i_mem+1,1]
                            xr.plot.pcolormesh(obs_dict_branch[obs_name][i_mem].sel(time=t)-obs_dict_trunk[obs_name].sel(time=t), x='lon', y='lat', ax=ax, cmap='PiYG')
                            ax.set_title(r'(PERT %d) $-$ CTRL'%(i_mem))
                        for i_row in range(axes.shape[0]):
                            for i_col in range(axes.shape[1]):
                                axes[i_row,i_col].set_ylabel('Latitude' if i_col==0 else '')
                                axes[i_row,i_col].set_xlabel('Longitude' if i_row==axes.shape[0]-1 else '')
                        fig.suptitle(r'%s at $t=%g$'%(obs_name,t*tu))
                        filename = join(dirdict['plots'],r'field%s_bg%d_mem%d_t%d'%(obs_name,branch_group,i_mem,i_time))
                        if exists(filename): os.remove(filename)
                        fig.savefig(filename, **pltkwargs)
                        plt.close(fig)
    return


def pebr_meta_analysis_workflow(idx_param):
    num_expts = len(idx_param)
    workflows = []
    for i_param in idx_param:
        workflows.append(pebr_workflow(i_param))
    configs_gcm,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts = tuple(
            [workflows[i][j] for i in range(len(idx_param))]
            for j in range(7))

    config_meta_analysis = dict()
    # Divergence timescales

    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-21"
    sub_date_str = "0"
    meta_dirdict = dict()
    meta_dirdict['expts'] = join(scratch_dir, date_str, sub_date_str)
    meta_dirdict['analysis'] = join(meta_dirdict['expts'], 'meta_analysis')
    meta_dirdict['plots'] = join(meta_dirdict['expts'], 'meta_plots')
    for meta_dir in ['analysis','plots']:
        makedirs(meta_dirdict[meta_dir], exist_ok=True)

    # Configure the meta-analysis (e.g. which observables and metrics to plot) based on the config in one of the list
    config_meta_analysis = dict()
    config_meta_analysis['dist_metrics'] = configs_analysis[0]['dist_metrics']
    config_meta_analysis['satfracs'] = configs_analysis[0]['satfracs']
    config_meta_analysis['observables'] = configs_analysis[0]['observables'] # TODO decide on some extreme value analysis for selected observables, especially over finite-time horizons etc. For example, how do block maxima statistics converge as the block size gets longer? 



    meta_filedict = dict()
    meta_filedict['analysis'] = dict()
    meta_filedict['plots'] = dict()
    meta_filedict['plots']['elfs_vs_sigma'] = dict()
    dist_names = list(config_meta_analysis['dist_metrics'].keys())
    for dist_name in dist_names:
        meta_filedict['plots']['elfs_vs_sigma'][dist_name] = join(meta_dirdict['plots'], r'elfs_vs_sigma_dist%s.png'%(config_meta_analysis['dist_metrics'][dist_name]['abbrv']))

    return config_meta_analysis,meta_dirdict,meta_filedict,workflows

def pebr_meta_analysis_procedure(idx_param):
    tododict = dict({
        'analysis': 0,
        'plots': dict({
            'elfs_vs_sigma':        1,
            'fsle_vs_sigma':        0,
            }),
        })
    config_meta_analysis,meta_dirdict,meta_filedict,workflows = pebr_meta_analysis_workflow(idx_param)
    num_expts = len(idx_param)
    configs_gcm,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts = tuple(
            [workflows[i][j] for i in range(len(idx_param))]
            for j in range(7))


    # Plot fractional saturation time 
    if tododict['plots']['elfs_vs_sigma']:
        for dist_name,dist_metric in config_meta_analysis['dist_metrics'].items():
            print(f'{filedicts[0].keys() = }')
            elfs_files = [filedicts[i]['dispersion']['satfractime'][dist_name] for i in range(len(idx_param))]
            # Split these files into groups based on non-sigma parameters
            plot_meta_elfs_vs_sigma(config_meta_analysis, configs_gcm, expt_labels, elfs_files, meta_filedict['plots']['elfs_vs_sigma'][dist_name], title=dist_metric['label'])
    return

            

def plot_meta_elfs_vs_sigma(config_meta_analysis, configs_gcm, expt_labels, elfs_files, outfile, title=''):
    # Group experiments by non-sigma parameter
    num_expts = len(configs_gcm)
    Ls = np.array([configs_gcm[i]['SPPT']['L_sppt'] for i in range(num_expts)])
    taus = np.array([configs_gcm[i]['SPPT']['tau_sppt'] for i in range(num_expts)])
    sigmas = np.array([configs_gcm[i]['SPPT']['std_sppt'] for i in range(num_expts)])
    unique_Ls_taus,unique_inverse = np.unique(np.array([Ls,taus]).T, axis=0, return_inverse=True)
    satfracs = config_meta_analysis['satfracs']
    fig,axes = plt.subplots(ncols=len(satfracs),figsize=(6*len(satfracs),6),sharey=True)
    handles = []
    for i_Ltau in range(len(unique_Ls_taus)):
        color = plt.cm.Set1(i_Ltau/len(unique_Ls_taus))
        L,tau = unique_Ls_taus[i_Ltau]
        idx = np.where(unique_inverse == i_Ltau)[0]
        elfs_mean = np.nan*np.ones((len(idx),len(satfracs)))
        for ii,i in enumerate(idx):
            elfs = np.load(elfs_files[i])
            elfs_mean[ii,:] = np.mean(elfs['elfs'], axis=0) # mean across branch groups 
        for i_sf,sf in enumerate(satfracs):
            ax = axes[i_sf]
            h, = ax.plot(sigmas[idx], elfs_mean[:,i_sf], label=r'$L=%g$km, $\tau=%g$h'%(L/1000,tau/3600),color=color,marker='.')
            if i_sf == 0: handles.append(h)
            ax.set_title(f'$f=%g$'%(sf))
            ax.set_xlabel(r'$\sigma_{\mathrm{SPPT}}$')
            ax.set_ylabel(r'$\tau(f)$')
    axes[0].legend(handles=handles)
    fig.suptitle(title)
    fig.savefig(outfile, **pltkwargs)
    plt.close(fig)
    return


def old_thing():
    expt_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-03-05/0/PeBr"
    meta_dir = join(expt_dir,'meta_analysis')
    makedirs(meta_dir,exist_ok=True)
    # -------- Specify which variables to fix and which to vary ---------
    params = dict()
    params['L_sppt'] = dict({
        'fun': lambda config: config['SPPT']['L_sppt'],
        'scale': 1000, # for display purposes
        'symbol': r'$L_{\mathrm{SPPT}}$',
        'unit_symbol': 'km',
        })
    params['tau_sppt'] = dict({
        'fun': lambda config: config['SPPT']['tau_sppt'],
        'scale': 3600, 
        'symbol': r'$\tau_{\mathrm{SPPT}}$',
        'unit_symbol': 'h',
        })
    params['std_sppt'] = dict({
        'fun': lambda config: config['SPPT']['std_sppt'],
        'scale': 1.0,
        'symbol': r'$\sigma_{\mathrm{SPPT}}$',
        })
    params2fix = ['L_sppt','tau_sppt']
    param2vary = 'std_sppt'
    # --------- Analysis for nosppt ---------
    algdir_nosppt = glob.glob(join(expt_dir,f'abs1_resT21_pertIMP*/PeBr*/'))[0]
    pert_growth_nosppt = pickle.load(open(join(algdir_nosppt,'analysis/pert_growth.pickle'),'rb'))
    dist_names = list(pert_growth_nosppt.keys())
    fracsat_nosppt,t2fracsat_nosppt = FriersonGCMPeriodicBranching.analyze_pert_growth_meta([pert_growth_nosppt])
    print(f'{t2fracsat_nosppt = }')
    # --------- Analysis for sppt -----------
    param_vals = dict({p: [] for p in params.keys()})
    algdir_pattern = join(expt_dir,f"abs1_resT21_pertSPPT_std0p*_clip2_tau*h_L*km/PeBr*/")
    print(f'{algdir_pattern = }')
    algdirs = glob.glob(algdir_pattern)
    algdirs2include = []
    print(f'{len(algdirs) = }')
    pert_growth_list = []
    for i_algdir,algdir in enumerate(algdirs):
        pg_filename = join(algdir,'analysis/pert_growth.pickle')
        if exists(pg_filename):
            algdirs2include.append(algdir)
            pert_growth_list.append(pickle.load(open(pg_filename,'rb')))
            alg = pickle.load(open(join(algdir,'alg.pickle'),'rb'))
            if i_algdir == 0:
                obsprop = alg.ens.dynsys.observable_props()
                tu = alg.ens.dynsys.dt_save
            for p in params.keys():
                param_vals[p].append(params[p]['fun'](alg.ens.dynsys.config))

    algdirs = algdirs2include
    # For each fixed set of independent variables, analyze them as a group
    param_vals_fixed = list(zip(*(param_vals[p] for p in params2fix)))
    print(f'{param_vals_fixed = }')
    unique_param_vals_fixed = set(param_vals_fixed)
    for pvf in unique_param_vals_fixed:
        print(f'{pvf = }')
        fixed_param_abbrv = ('_'.join([
            r'%s%g%s'%(params2fix[i],pvf[i]/params[params2fix[i]]['scale'],params[params2fix[i]]['unit_symbol']) 
            for i in range(len(params2fix))
            ])
            ).replace('.','p')
        print(f'{fixed_param_abbrv = }')
        fixed_param_label = ', '.join([
            r'%s $=%g$ %s'%(params[params2fix[i]]['symbol'],pvf[i]/params[params2fix[i]]['scale'],params[params2fix[i]]['unit_symbol']) 
            for i in range(len(params2fix))
            ])
        idx = np.array([i for i in range(len(algdirs)) if (param_vals_fixed[i] == pvf)])
        param2vary_vals = np.array([param_vals[param2vary][i] for i in idx])
        order = np.argsort(param2vary_vals)
        param2vary_vals = param2vary_vals[order]
        idx = idx[order] 
        pert_growth_sublist = [pert_growth_list[i] for i in idx]
        fracsat,t2fracsat = FriersonGCMPeriodicBranching.analyze_pert_growth_meta(pert_growth_sublist)
        FriersonGCMPeriodicBranching.plot_pert_growth_meta(param2vary_vals, fracsat, t2fracsat, join(meta_dir,f't2fracsat_{fixed_param_abbrv}.png'), r'$\sigma_{\mathrm{SPPT}}$', tu=tu, fracsat_ref=fracsat_nosppt, t2fracsat_ref=t2fracsat_nosppt)
    return

def pebr_single(i_param):
    tododict = dict({
        'run':                           1,
        'analysis': dict({
            'observable_spaghetti':      1,
            'dispersion_rate':           0, # including both Lyapunov analysis (FSLE) and expected leadtime until fractional saturation (ELFS)
            'running_max':               0, # watch extreme value statistics (curves and parameters) converge to the true values with longer time blocks
            }),
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = pebr_workflow(i_param)
    if tododict['run']:
        run_pebr(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, alg, dirdict)
    if tododict['analysis']['dispersion_rate']:
        quantify_dispersion_rates(config_analysis, alg, dirdict)
    if tododict['analysis']['running_max']:
        quantify_running_max_convergence(config_analysis, alg, dirdict)
    return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_param = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'meta'
        idx_param = list(range(1,21))
    print(f'Got into Main')
    if procedure == 'single':
        for i_param in idx_param:
            pebr_single(i_param)
    elif procedure == 'meta':
        pebr_meta_analysis_procedure(idx_param)

