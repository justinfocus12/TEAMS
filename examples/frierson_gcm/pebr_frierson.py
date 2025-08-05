
i = 0
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
import pdb
import subprocess
import resource
import pickle
import copy as copylib
import pprint
#from importlib import reload

sys.path.append("../..")
import utils; #reload(utils)
from ensemble import Ensemble
import forcing; #reload(forcing)
import algorithms; #reload(forcing)
#import frierson_gcm; #reload(frierson_gcm)
from frierson_gcm import FriersonGCM
from algorithms_frierson import FriersonGCMPeriodicBranching

def pebr_multiparams():
    seed_incs = [0]
    sigmas = [0.0,0.05,0.1,0.2,0.3,0.4,0.5]
    taus = [tau_hrs * 3600 for tau_hrs in [6]]
    Ls = [L_km * 1000 for L_km in [500]]
    return seed_incs,sigmas,taus,Ls

def pebr_paramset(i_expt):
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    config_gcm = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)

    multiparams = pebr_multiparams()
    idx_multiparam = np.unravel_index(i_expt, tuple(len(mp) for mp in multiparams))
    seed_inc,std_sppt,tau_sppt,L_sppt = (multiparams[i][i_param] for (i,i_param) in enumerate(idx_multiparam))

    config_gcm['resolution'] = 'T21'
    config_gcm['outputs_per_day'] = 4
    config_gcm['pert_type'] = 'SPPT'
    config_gcm['SPPT']['tau_sppt'] = tau_sppt
    config_gcm['SPPT']['std_sppt'] = std_sppt
    config_gcm['SPPT']['L_sppt'] = L_sppt
    config_gcm['remove_temp'] = 1
    pprint.pprint(config_gcm)


    expt_label = r'SPPT, $\sigma=%g$, $\tau=%g$ h, $L=%g$ km'%(std_sppt,tau_sppt/3600,L_sppt/1000)
    expt_abbrv = r'SPPT_std%g_tau%gh_L%gkm'%(std_sppt,tau_sppt/3600,L_sppt/1000)


    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_inc, 
        'branches_per_group': 12, 
        'interbranch_interval_phys': 6.0, # small interval helps to see continuity in stability
        'branch_duration_phys': 50.0,
        'num_branch_groups': 20,
        'max_member_duration_phys': 50.0,
        'bole_duration_phys': 60.0, # using "bole" from forestry to indicate the height of the trunk below the first branch. How far the initial simulation should diverge from the DNS-generated source it came from, before the branching begins  
        })
    return config_gcm,config_algo,expt_label,expt_abbrv

# -------------- Define the observable functions of interest (to be further parameterized) ---------
# ----------- Define the distance metrics of interest -------



def pebr_single_workflow(i_expt):
    config_gcm,config_algo,expt_label,expt_abbrv = pebr_paramset(i_expt)
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = FriersonGCMPeriodicBranching.label_from_config(config_algo)
    # Configure post-analysis
    # List the quantities of interest
    config_analysis = dict()
    config_analysis['target_location'] = dict(lat=45, lon=180)
    # observables (scalar quantities)
    observables_scalar = dict({
        'rainrate': dict({
            'fun': lambda ds,num_steps=1,roi=None: FriersonGCM.rolling_time_mean(
                FriersonGCM.regional_rain(ds,roi), num_steps),
            'kwargs': dict({
                'roi': config_analysis['target_location'],
                'num_steps': config_gcm['outputs_per_day'],
                }),
            'abbrv': 'Rtot',
            'label': r'1-day precip. (target)',
            'unit_symbol': 'mm/day',
            }),
        'temp': dict({
            'fun': lambda ds: FriersonGCM.sel_from_roi(
                FriersonGCM.temperature(ds), 
                dict(**config_analysis['target_location'], pfull=1000)),
            'kwargs': dict(),
            'abbrv': 'T',
            'label': 'Temperature (target)',
            'unit_symbol': 'K',
            }),
        'surf_horz_wind': dict({
            'fun': lambda ds: FriersonGCM.sel_from_roi(
                FriersonGCM.horizontal_wind_speed(ds), 
                dict(**config_analysis['target_location'], pfull=1000)),
            'kwargs': dict(),
            'abbrv': 'UV',
            'label': 'Horizontal wind speed $(\phi,\lambda,\sigma)=(%d,%d,1.0)$'%(config_analysis['target_location']['lat'],config_analysis['target_location']['lon']),
            'unit_symbol': 'm/s',
            }),
        })
    config_analysis['observables_scalar'] = observables_scalar
    obs_names = list(observables_scalar.keys())
    # distance metrics
    dist_metrics = dict() 
    for (lonrange,latrange) in ((40,10),(120,30),(360,90))[2:]:
        areastr = r'%dx%d'%(lonrange,latrange) 
        if (lonrange == 360) and (latrange == 90):
            area_label = "NH"
        else:
            area_label = areastr
        roi = dict({
            'lat': slice(config_analysis['target_location']['lat']-latrange/2,config_analysis['target_location']['lat']+latrange/2),
            'lon': slice(config_analysis['target_location']['lon']-lonrange/2,config_analysis['target_location']['lon']+lonrange/2),

            })
        dist_metrics[r'rain_%s'%(areastr)] = dict({
            'fun': FriersonGCM.dist_euc_rain,
            'abbrv': r'RainEuc%s'%(areastr),
            'label': r'1-day Precip. Eucl. dist. (%s)'%(area_label),
            'field_name': r'Precip',
            'kwargs': dict(roi=roi,outputs_per_day=config_gcm['outputs_per_day']),
            'unit_symbol': 'mm/day',
            })
        dist_metrics[r'temp_%s'%(areastr)] = dict({
            'fun': FriersonGCM.dist_euc_temp,
            'abbrv': r'TempEuc%s'%(areastr),
            'label': r'Surf. Temp.  Eucl. dist. (%s)'%(area_label),
            'field_name': r'Temp.',
            'kwargs': dict({'roi': dict(pfull=1000, **roi)}),
            'unit_symbol': 'K',
            })
        dist_metrics[r'horzvel_%s'%(areastr)] = dict({
            'fun': FriersonGCM.dist_euc_horzvel,
            'kwargs': dict({'roi': dict(pfull=1000, **roi)}),
            'abbrv': r'UVEuc%s'%(areastr),
            'field_name': "Windspeed",
            'label': r'Surf. Horz. Vel. Eucl. dist. (%s)'%(areastr),
            'unit_symbol': 'm/s',
            })
    dist_names = list(dist_metrics.keys())
    config_analysis['dist_metrics'] = dist_metrics

    # How to quantitatively measure perturbation growth, and also perhaps the Lyapunov exponents/power laws between them 
    config_analysis['satfracs'] = np.array([1/8,1/4,3/8,1/2])
    config_analysis['satfrac_symbols'] = np.array(["1/8","1/4","3/8","1/2"])


    # Set up directories
    dirdict = dict()
    scratch_dir = "/orcd/archive/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2025-05-16"
    sub_date_str = "1"
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    dirdict['init_cond'] = join(dirdict['data'], 'bole')

    for dirname in ['data','analysis','plots','init_cond']:
        makedirs(dirdict[dirname], exist_ok=True)

    filedict = dict()
    # Initial conditions
    filedict['angel'] = join(
            f'/orcd/archive/pog/001/ju26596/TEAMS/examples/frierson_gcm/2025-05-16/1',
            param_abbrv_gcm, 'DNS_si0', 'data',
            'alg.pickle') 
    
    #filedict['init_cond'] = dict()
    #filedict['init_cond']['restart'] = join(dirdict['init_cond'],'restart_mem20.cpio')
    #filedict['init_cond']['trajectory'] = join(dirdict['init_cond'],'mem20.nc')
    #print(f'{filedict["init_cond"] = }')
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')

    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def run_pebr(dirdict,filedict,config_gcm,config_algo):
    nproc = 4
    recompile = False
    root_dir = dirdict['data']
    angel = pickle.load(open(filedict['angel'], 'rb'))
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
    else:
        gcm = FriersonGCM(config_gcm, recompile=recompile)
        ens = Ensemble(gcm, root_dir=root_dir)
        alg = FriersonGCMPeriodicBranching.initialize_from_dns_appendage(angel, config_algo, ens, dirdict['init_cond'], root_dir)

    alg.ens.dynsys.set_nproc(nproc)
    alg.ens.set_root_dir(root_dir)
    while not alg.terminate:
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict({
            # Temporary folder
            'temp_dir': f'mem{mem}_temp',
            'final_dir': f'mem{mem}',
            })
        saveinfo.update(dict({
            # Ultimate resulting filenames
            'filename_traj': join(saveinfo['final_dir'],f'history_mem{mem}.nc'),
            'filename_restart': join(saveinfo['final_dir'],f'restart_mem{mem}.cpio'),
            }))
        alg.take_next_step(saveinfo)
        if exists(filedict['alg']):
            os.rename(filedict['alg'], filedict['alg_backup'])
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def quantify_dispersion_rates(config_analysis, alg, dirdict, overwrite_dispersion_stats=False):
    for dist_name,dist_props in config_analysis['dist_metrics'].items():
        print(f'{dist_name = }')
        print(f'{dist_props = }')
        def dist_fun(ds0,ds1):
            t0 = np.round(ds0['time'].to_numpy() / alg.ens.dynsys.dt_save).astype(int)
            t1 = np.round(ds1['time'].to_numpy() / alg.ens.dynsys.dt_save).astype(int)
            trange_full = np.arange(min(t0[0],t1[0]),max(t0[-1],t1[-1])+1)
            trange_valid = np.arange(max(t0[0],t1[0]),min(t0[-1],t1[-1])+1)
            tidx0 = trange_valid - t0[0]
            tidx1 = trange_valid - t1[0]
            dist = np.nan*np.ones_like(trange_full)
            dist = dist_props['fun'](ds0.isel(time=tidx0), ds1.isel(time=tidx1), **dist_props['kwargs'])
            return dist
        dispersion_file = join(dirdict['analysis'],r'dispersion_%s.npz'%(dist_props['abbrv']))
        if (not exists(dispersion_file)) or overwrite_dispersion_stats:
            dispersion_stats = alg.measure_dispersion(dist_fun, config_analysis['satfracs'], dispersion_file)
        dispersion_stats = np.load(dispersion_file)
        # Plot 
        figfile_prefix = join(dirdict['plots'],r'dispersion_%s'%(dist_props['abbrv']))
        groups2plot = np.arange(min(dispersion_stats['dists'].shape[0],10), dtype=int)
        alg.plot_dispersion(
                dispersion_stats, figfile_prefix, groups2plot=groups2plot,  
                title=dist_props['label'], logscale=False, time_unit_symbol='days', ylabel=r'%s [%s]'%(dist_props['field_name'],dist_props['unit_symbol']), satfrac_symbols=config_analysis['satfrac_symbols'],
                )
    return 

def plot_observable_spaghetti(config_analysis, alg, dirdict):
    for obs_name,obs_props in config_analysis['observables_scalar'].items():
        print(f'{obs_name = }')
        print(f'{obs_props = }')
        obs_fun = lambda ds: obs_props['fun'](ds,**obs_props['kwargs'])
        ylabel = r'[%s]'%(obs_props['unit_symbol'])
        title = obs_props['label']
        for group in range(min(4,alg.branching_state['next_branch_group']+1)):
            outfile = join(dirdict['plots'], r'spaghetti_obs%s_bg%d.png'%(obs_props['abbrv'],group))
            alg.plot_observable_spaghetti(obs_fun,group,outfile,ylabel=ylabel,title=title, time_unit_symbol='days')
            # TODO maybe precompute all the observables in advance, in case they're used for multiple purposes
    return

def quantify_running_max_convergence(config_analysis, alg, dirdict):
    for obs_name,obs_props in config_analysis['observables_scalar'].items():
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
        if 0 and tododict['plot_pebr']['observables_scalar']:
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


def pebr_meta_workflow(idx_param):
    num_expts = len(idx_param)
    num_expt = len(idx_param)
    workflow_tuple = tuple(pebr_single_workflow(i_param) for i_param in idx_param)
    workflows = dict()
    for i_key,key in enumerate(('configs_gcm,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts').split(',')):
        workflows[key] = tuple(workflow_tuple[j][i_key] for j in range(len(workflow_tuple)))

    config_meta_analysis = dict()
    for key in ['target_location','observables_scalar','dist_metrics','satfracs']:
        config_meta_analysis[key] = workflows['configs_analysis'][0][key]

    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-26"
    sub_date_str = "0"
    meta_dirdict = dict()
    meta_dirdict['meta'] = join(scratch_dir, date_str, sub_date_str, 'meta_pebr')
    meta_dirdict['analysis'] = join(meta_dirdict['meta'], 'meta_analysis')
    meta_dirdict['plots'] = join(meta_dirdict['meta'], 'meta_plots')
    for meta_dir in ['analysis','plots']:
        makedirs(meta_dirdict[meta_dir], exist_ok=True)

    return workflows,config_meta_analysis,meta_dirdict

def pebr_meta_procedure(idx_param):
    tododict = dict({
        'compare_elfs':        1,
        })
    workflows,config_meta_analysis,meta_dirdict = pebr_meta_workflow(idx_param)
    # Plot fractional saturation time 
    if tododict['compare_elfs']:
        compare_elfs(workflows, config_meta_analysis, meta_dirdict)
    return

            

def compare_elfs(workflows, config_meta_analysis, meta_dirdict):
    # group the different experiments by L_sppt and tau_sppt 
    num_expt = len(workflows['configs_gcm'])
    tu = 1/workflows['configs_gcm'][0]['outputs_per_day']
    Ls = tuple(workflows['configs_gcm'][i]['SPPT']['L_sppt'] for i in range(num_expt))
    taus = tuple(workflows['configs_gcm'][i]['SPPT']['tau_sppt'] for i in range(num_expt))
    sigmas = tuple(workflows['configs_gcm'][i]['SPPT']['std_sppt'] for i in range(num_expt))
    Ltau = tuple(zip(Ls,taus))
    Ltau_unique = list(set(Ltau))
    Ltau_idx_groups = tuple(
            tuple(i for i in range(num_expt) if Ltau[i] == Ltau_val)
            for Ltau_val in Ltau_unique
            )
    satfracs = config_meta_analysis['satfracs']
    for dist_name,dist_props in config_meta_analysis['dist_metrics'].items():
        fig,axes = plt.subplots(ncols=len(satfracs), figsize=(6*len(satfracs),4), sharey='row')
        handles = []
        for i_group,Ltau in enumerate(Ltau_unique):
            idx = Ltau_idx_groups[i_group]
            elfs_mean = np.nan*np.ones((len(satfracs),len(idx)))
            elfs_std = np.nan*np.ones((len(satfracs),len(idx)))
            for ii,i in enumerate(idx):
                disp_file = join(workflows['dirdicts'][i]['analysis'], r'dispersion_%s.npz'%(dist_props['abbrv']))
                if exists(disp_file):
                    dispersion_stats = np.load(disp_file)
                    elfs_mean[:,ii] = np.mean(dispersion_stats['elfs'], axis=0)
                    elfs_std[:,ii] = np.std(dispersion_stats['elfs'], axis=0)
                else:
                    print(f'WARNING missing the dispersion file {disp_file}')

            sigmas_idx = np.array([sigmas[i] for i in idx])
            group_color = plt.cm.Set1(i_group)
            group_label = r'$L=%g$km, $\tau=%g$h'%(Ltau_unique[i_group][0]/1000,Ltau_unique[i_group][1]/3600)
            for i_sf,sf in enumerate(satfracs):
                ax = axes[i_sf]
                h, = ax.plot(sigmas_idx,elfs_mean[i_sf,:]*tu,color=group_color,label=group_label)
                lo,hi = ((elfs_mean[i_sf,:] + sgn*elfs_std[i_sf,:])*tu for sgn in (-1,1))
                ax.fill_between(sigmas_idx, lo, hi, facecolor=group_color, zorder=-1, edgecolor='none', alpha=0.25)
            handles.append(h)
        for i_sf,sf in enumerate(satfracs):
            axes[i_sf].set(title=r'$\varepsilon=%g$'%(sf), xlabel=r'$\sigma_{\mathrm{SPPT}}$', ylabel='')
            axes[i_sf].yaxis.set_tick_params(which='both', labelbottom=True)
        fig.suptitle(dist_props['label'], y=1.0)
        axes[0].set_ylabel(r'$\overline{t_\varepsilon}\pm\mathrm{SD}(t_\varepsilon)$')
        axes[-1].legend(handles=handles, bbox_to_anchor=(1.05,1), loc='upper left')
        fig.savefig(join(meta_dirdict['plots'], r'elfs_%s.png'%(dist_props['abbrv'])), **pltkwargs)
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

def pebr_single_procedure(i_param):
    tododict = dict({
        'run':                           0,
        'analysis': dict({
            'observable_spaghetti':      0,
            'dispersion_rate':           1, # including both Lyapunov analysis (FSLE) and expected leadtime until fractional saturation (ELFS)
            'running_max':               0, # watch extreme value statistics (curves and parameters) converge to the true values with longer time blocks
            }),
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = pebr_single_workflow(i_param)
    if tododict['run']:
        run_pebr(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, alg, dirdict)
    if tododict['analysis']['dispersion_rate']:
        quantify_dispersion_rates(config_analysis, alg, dirdict, overwrite_dispersion_stats=False)
    if tododict['analysis']['running_max']:
        quantify_running_max_convergence(config_analysis, alg, dirdict)
    return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_expt = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'meta'
        idx_expt = list(range(1,21))
    print(f'Got into Main')
    if procedure == 'single':
        for i_expt in idx_expt:
            pebr_single_procedure(i_expt)
    elif procedure == 'meta':
        pebr_meta_procedure(idx_expt)

