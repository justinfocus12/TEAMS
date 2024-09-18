i = 0
print(f'--------------Beginning imports-------------')
import numpy as np
print(f'{i = }'); i += 1
from numpy.random import default_rng
print(f'{i = }'); i += 1
import networkx as nx
print(f'{i = }'); i += 1
import xarray as xr
print(f'{i = }'); i += 1
from matplotlib import pyplot as plt, rcParams, gridspec
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

def teams_multiparams():
    sigmas = [0.3,0.4]
    seed_incs = list(range(32))
    deltas_phys = [0.0,4.0,5.0,6.0,7.0,8.0,10.0]
    split_landmarks = ['thx']
    return sigmas,seed_incs,deltas_phys,split_landmarks

def teams_paramset(i_expt):
    sigmas,seed_incs,deltas_phys,split_landmarks = teams_multiparams()
    # TODO switch i_seed_inc with i_sigma below for the next round of runs, so as to avoid interference 
    i_seed_inc,i_sigma,i_delta,i_slm = np.unravel_index(i_expt, (len(seed_incs),len(sigmas),len(deltas_phys),len(split_landmarks)))
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    config_gcm = frierson_gcm.FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)
    config_gcm['outputs_per_day'] = 4
    config_gcm['pert_type'] = 'SPPT'
    config_gcm['SPPT']['tau_sppt'] = 6.0 * 3600
    config_gcm['SPPT']['std_sppt'] = sigmas[i_sigma]
    config_gcm['SPPT']['L_sppt'] = 500.0 * 1000
    config_gcm['remove_temp'] = 1
    pprint.pprint(config_gcm)


    expt_label = r'$\sigma=%g$, $\delta=%g$'%(sigmas[i_sigma],deltas_phys[i_delta])
    expt_abbrv = (r'std%g_ast%g'%(sigmas[i_sigma],deltas_phys[i_delta])).replace('.','p')


    config_algo = dict({
        'num_levels_max': 512-64, # This parameter shouldn't affect the filenaming or anything like that 
        'num_members_max': 512,
        'num_active_families_min': 2,
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_seed_inc],
        'population_size': 64,
        'time_horizon_phys': 30, #+ deltas_phys[i_delta],
        'buffer_time_phys': 0,
        'advance_split_time_phys': deltas_phys[i_delta], # TODO put this into a parameter
        'advance_split_time_max_phys': 10, # TODO put this into a parameter
        'split_landmark': split_landmarks[i_slm],
        'inherit_perts_after_split': False,
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

def teams_single_workflow(i_expt):
    # i_expt is a flat index, from which both i_param and i_buick are derived
    # Cluge; rely on knowing the menu of options from the Buick dealership and from the parameter sets 
    config_gcm,config_algo,expt_label,expt_abbrv = teams_paramset(i_expt)
    param_abbrv_gcm,param_label_gcm = frierson_gcm.FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = algorithms_frierson.FriersonGCMTEAMS.label_from_config(config_algo)
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
    config_analysis['fields_2d'] = dict({
        'area_ps_360x30': dict({
            'fun': lambda ds: frierson_gcm.FriersonGCM.sel_from_roi(
                frierson_gcm.FriersonGCM.surface_pressure(ds)/1000,
                dict(
                    lat=slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                    lon=slice(config_analysis['target_location']['lon']-180,config_analysis['target_location']['lon']+180),
                    ),
                ),
            'abbrv': 'PS360x30',
            'label': r'Surf. Pres. [kPa] $(\phi,\lambda)=(45\pm15,180\pm180)$',
            'cmap': 'coolwarm',
            }),
        'area_w500_360x30': dict({
            'fun': lambda ds: frierson_gcm.FriersonGCM.sel_from_roi(
                frierson_gcm.FriersonGCM.vertical_velocity(ds),
                dict(
                    lat=slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                    lon=slice(config_analysis['target_location']['lon']-180,config_analysis['target_location']['lon']+180),
                    pfull=500,
                    ),
                ),
            'abbrv': 'W500_360x30',
            'label': r'Vert. Vel. [Pa/s] $(\phi,\lambda)=(45\pm15,180\pm180)$',
            'cmap': 'coolwarm',
            }),
        'area_rain_360x30': dict({
            'fun': lambda ds: frierson_gcm.FriersonGCM.rolling_time_mean(
                frierson_gcm.FriersonGCM.sel_from_roi(
                    frierson_gcm.FriersonGCM.total_rain(ds),
                    dict({
                        'lat': slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                        }),
                    ),
                config_gcm['outputs_per_day'],
                ),
            'abbrv': 'R360x30',
            'label': r'1-day avg rain $(\phi,\lambda)=(45\pm15,180\pm180)$',
            'cmap': 'Blues',
            }),
        'area_cwv_360x30': dict({
            'fun': lambda ds: frierson_gcm.FriersonGCM.sel_from_roi(
                frierson_gcm.FriersonGCM.column_water_vapor(ds),
                dict(
                    lat=slice(config_analysis['target_location']['lat']-15,config_analysis['target_location']['lat']+15),
                    lon=slice(config_analysis['target_location']['lon']-180,config_analysis['target_location']['lon']+180),
                    ),
                ),
            'abbrv': 'CWV360x30',
            'label': r'Column water vapor [kg/m$^2$] $(\phi,\lambda)=(45\pm15,180\pm180)$',
            'cmap': 'Blues',
            }),
        })
    config_analysis['composites'] = dict({
        'anc_scores': [20,30,40,50],
        'boost_sizes': [15],
        'score_tolerance': 5,
        })
    


    # Set up directories
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-09-10"
    sub_date_str = "2"
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo, r'seedinc%d'%(config_algo['seed_inc_init']))
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    filedict = dict()
    # Initial conditions
    filedict['angel'] = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-09-10/0',
            param_abbrv_gcm, 'DNS_si0', 'data',
            'alg.pickle') 
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')

    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def plot_fields_2d(config_analysis, alg, dirdict, filedict, expt_label, remove_old_plots=False):
    tu = alg.ens.dynsys.dt_save
    N = alg.population_size
    all_scores = np.array(alg.branching_state['scores_max'])
    B = alg.ens.construct_descent_matrix()[:N,:].toarray()
    desc_per_anc = np.sum(B,axis=1)
    anc_scores = all_scores[:N]
    order_ancscores = np.argsort(anc_scores)[::-1]
    descendants = [np.where(B[anc,:])[0] for anc in range(N)]
    desc_scores = np.where(B==0, -np.inf, B*all_scores)
    max_desc_scores = np.max(desc_scores, axis=1)
    # Get the best score of the ultimate descendant
    order_descscores = np.argsort(max_desc_scores)[::-1]
    # Get the most-split ancestors
    order_famsize = np.argsort(desc_per_anc)[::-1]
    # Select some ancestors to plot based on two criteria: (1) largest ancestral scores, (2) largest child scores
    ancs2plot = np.unique(np.concatenate((order_ancscores[:4], order_descscores[:4])))
    print(f'{ancs2plot = }')
    if remove_old_plots:
        old_plot_filenames = glob.glob(join(dirdict['plots'],'fields*.png'))
        for f in old_plot_filenames:
            os.remove(f)
    for i_ancestor,ancestor in enumerate(ancs2plot):
        if len(descendants[ancestor]) == 0:
            continue
        best_desc = np.argmax(desc_scores[ancestor,:])
        lineage = list(sorted(nx.ancestors(alg.ens.memgraph, best_desc) | {best_desc}))
        print(f'{lineage = }')
        obs_funs = [obs_props['fun'] for obs_props in config_analysis['fields_2d'].values()]
        f_anc_multiobs,f_desc_multiobs = tuple(alg.ens.compute_observables(obs_funs, mem, compute=True) for mem in [lineage[0],lineage[-1]])
        print(f'Computed multiobs')
        for i_obs,(obs_name,obs_props) in enumerate(config_analysis['fields_2d'].items()):
            print(f'Starting to plot field {obs_name = }')
            f_anc,f_desc = f_anc_multiobs[i_obs],f_desc_multiobs[i_obs]
            # Interpoilate to 1-degree grid 
            lon_hires = np.arange(f_anc.lon.min().item(), f_anc.lon.max().item(), step=1.0)
            lat_hires = np.arange(f_anc.lat.min().item(), f_anc.lat.max().item(), step=1.0)
            f_anc = f_anc.interp(lon=lon_hires, lat=lat_hires)
            f_desc = f_desc.interp(lon=lon_hires, lat=lat_hires)

            vmin,vmax = min((f.min().item() for f in (f_anc,f_desc))),max((f.max().item() for f in (f_anc,f_desc)))
            vmax_diff = np.abs(f_desc - f_anc).max().item() 
            score_lineage = tuple(alg.branching_state['scores_tdep'][mem] for mem in lineage)
            tmx = alg.branching_state['scores_max_timing'][ancestor]
            tbr = alg.branching_state['branch_times'][best_desc]
            tinit,tfin = alg.ens.get_member_timespan(ancestor)

            for time2plot in range(max(tmx-int(4/tu),tinit+1),tmx+int(3/tu)):
                fig = plt.figure(tight_layout=True, figsize=(12,6))
                gs = gridspec.GridSpec(3,2)
                ax0 = fig.add_subplot(gs[0,0]) # Ancestor 2D field
                ax1 = fig.add_subplot(gs[1,0]) # Descendant 2D field
                ax2 = fig.add_subplot(gs[1,1]) # Descendant - Ancestor
                # Bottom row: the two timeseries, with a vertical line indicating the time of the snapshot
                ax3 = fig.add_subplot(gs[2,:]) # Timeseries of precip
                # Ancestor
                ax = ax0
                xr.plot.pcolormesh(f_anc.isel(time=time2plot-tinit-1), x='lon', y='lat', cmap=obs_props['cmap'], ax=ax, vmin=vmin, vmax=vmax, cbar_kwargs={'label': None})
                ax.set_xlabel('Lon')
                ax.set_ylabel('Lat')
                ax.set_title('Ancestor')

                # Descendant
                ax = ax1
                xr.plot.pcolormesh(f_desc.isel(time=time2plot-tinit-1), x='lon', y='lat', cmap=obs_props['cmap'], ax=ax, vmin=vmin, vmax=vmax, cbar_kwargs={'label': None})
                ax.set_xlabel('Lon')
                ax.set_ylabel('Lat')
                ax.set_title('Descendant.')

                # Difference
                ax = ax2
                xr.plot.pcolormesh(f_desc.isel(time=time2plot-tinit-1)-f_anc.isel(time=time2plot-tinit-1), x='lon', y='lat', cmap=obs_props['cmap'], cbar_kwargs={'label': None}, ax=ax, vmin=-vmax_diff, vmax=vmax_diff)
                ax.set_xlabel('Lon')
                ax.set_ylabel('Lat')
                ax.set_title('Descendant $-$ Ancestor')

                # Timeseries
                ax = ax3
                linespecs_anc = dict(color='black',linewidth=2,linestyle='--',label='Anc.')
                linespecs_desc = dict(color='red',linewidth=1,linestyle='-')
                for (mem,f) in [(ancestor,f_anc),(best_desc,f_desc)]:
                    linespecs = linespecs_anc if mem==ancestor else linespecs_desc
                    floc = f.sel(config_analysis['target_location'],method='nearest').to_numpy()
                    h, = ax.plot(np.arange(tinit+1,tfin+1)*tu, floc, **linespecs)
                    tmx_mem = alg.branching_state['scores_max_timing'][mem]
                    ax.scatter([tmx_mem*tu], floc[tmx_mem-tinit-1], marker='o', color=linespecs['color'])
                    #ax.scatter([tmx_mem*tu], score_lineage[i_mem][tmx_mem-tinit-1], marker='o', color=linespecs['color'])
                ax.axvline(tbr*tu, color='gray', linestyle='--')
                ax.axvline(time2plot*tu, color='gray')
                ax.set_ylabel(r'')

                fig.suptitle(obs_props['label'])
                filename = join(dirdict['plots'],r'fields_anc%d_%s_t%d'%(ancestor,obs_props['abbrv'],time2plot-tinit))
                fig.savefig(filename, **pltkwargs)

                plt.close(fig)
                print(f'Just saved to {filename}')
    return

def plot_observable_spaghetti(config_analysis, alg, dirdict, filedict, remove_old_plots=False):
    tu = alg.ens.dynsys.dt_save
    N = alg.population_size
    all_scores = np.array(alg.branching_state['scores_max'])
    B = alg.ens.construct_descent_matrix()[:N,:].toarray()
    desc_per_anc = np.sum(B,axis=1)
    anc_scores = all_scores[:N]
    # Select some ancestors to plot based on two criteria: (1) largest ancestral scores, (2) largest child scores
    order_ancscores = np.argsort(anc_scores)[::-1]
    print(f'{order_ancscores[:3] = }')
    print(f'{anc_scores[order_ancscores[:3]] = }')
    descendants = [np.where(B[anc,:])[0] for anc in range(N)]
    desc_scores = np.where(B==0, -np.inf, B*all_scores)
    max_desc_scores = np.max(desc_scores, axis=1)
    # Get the best score of the ultimate descendant
    order_descscores = np.argsort(max_desc_scores)[::-1]
    print(f'{order_descscores[:3] = }')
    print(f'{max_desc_scores[order_descscores[:3]] = }')
    # Get the most-split ancestors
    order_famsize = np.argsort(desc_per_anc)[::-1]
    print(f'{order_famsize[:3] = }')
    print(f'{desc_per_anc[order_famsize[:3]] = }')
    maxrank = 3
    ancs2plot = np.concatenate(tuple(order[:maxrank] for order in (order_ancscores,order_descscores,order_famsize,)))
    rank_labels = []
    for ordername in ['ancscore','descscore','famsize']:
        for r in range(maxrank):
            rank_labels.append(f'{ordername}rank{r}')
    if remove_old_plots:
        old_spaghetti_plots = glob.glob(join(dirdict['plots'],'spaghetti*.png'))
        for fig in old_spaghetti_plots:
            os.remove(fig)
    angel = pickle.load(open(filedict['angel'],'rb'))
    for (obs_name,obs_props) in config_analysis['observables'].items():
        is_score = (obs_name == 'local_dayavg_rain')
        if is_score:
            obs_fun = lambda ds: obs_props['fun'](ds, **obs_props['kwargs'])
            for i_ancestor,ancestor in enumerate(ancs2plot):
                special_descendant = np.argmax(desc_scores[ancestor,:])
                outfile = join(dirdict['plots'], r'spaghetti_%s_%s_anc%d.png'%(obs_props['abbrv'],rank_labels[i_ancestor],ancestor))
                landmark_label = {'lmx': 'local max', 'gmx': 'global max', 'thx': 'threshold crossing'}[alg.split_landmark]
                title = r'%s ($\delta=%g$ before %s)'%(obs_props['label'],alg.advance_split_time*tu,landmark_label)
                fig,axes = alg.plot_observable_spaghetti(obs_fun, ancestor, special_descendant=special_descendant, obs_label='', title=obs_props['label'], is_score=is_score, outfile=None)
                if False:
                    # Add a line for the Buick
                    mem_buick = next(angel.ens.memgraph.successors(angel.branching_state['generation_0'][ancestor]))
                    obs_buick = angel.ens.compute_observables([obs_fun], mem_buick)[0][:alg.time_horizon]
                    hbuick, = axes[0].plot((np.arange(len(obs_buick))+1)*tu, obs_buick, color='gray', linewidth=3, linestyle='--', zorder=-1, label='Buick')
                    if is_score: axes[1].axhline(np.nanmax(obs_buick), color='gray')
                fig.savefig(outfile, **pltkwargs)
                plt.close(fig)
                print(f'{outfile = }')
    return

def plot_score_spaghetti(config_analysis, alg, dirdict):
    pass

def plot_scorrelations(config_analysis, alg, dirdict, filedict, expt_label):
    # As a function of ancestor score (and also buick score), plot distribution of descendant scores (weighted and unweighted)
    bs = alg.branching_state
    scmax = np.array(bs['scores_max'])
    order = np.argsort(scmax[:alg.population_size])[::-1]
    score_fun = lambda ds: alg.score_combined(alg.score_components(ds['time'].to_numpy(),ds))
    angel = pickle.load(open(filedict['angel'],'rb'))
    scmax_buick = np.zeros(alg.population_size)
    for i in range(alg.population_size):
        if angel.branching_state['num_branches_generated'][i] > 0:
            # TODO fix this so it works on DNS sources too 
            mem_buick = next(angel.ens.memgraph.successors(angel.branching_state['generation_0'][i]))
            scmax_buick[i] = np.nanmax(angel.ens.compute_observables([score_fun], mem_buick)[0][:alg.time_horizon])
    fig,axes = plt.subplots(nrows=2, figsize=(6,12))
    ax = axes[0]
    hanc, = ax.plot(np.arange(alg.population_size), scmax[order], color='black', marker='o', label='Ancestors', zorder=0)
    desc_means = np.nan*np.ones(alg.population_size)
    for i in range(alg.population_size):
        ancestor = order[i]
        desc = list(nx.descendants(alg.ens.memgraph, ancestor))
        print(f'{desc = }')
        if len(desc) > 0: desc_means[ancestor] = np.mean(scmax[desc])
        ax.scatter(i*np.ones(len(desc)), scmax[desc], marker='.', color='red', s=8, zorder=2)
        hbuick = ax.scatter([i], [scmax_buick[ancestor]], color='gray', marker='*', s=16, label='Buicks', zorder=1)
        ax.plot([i,i], [scmax[ancestor], scmax_buick[ancestor]], color='gray')
    nnidx = np.where(np.isfinite(desc_means[order]))[0]
    hdescmean, = ax.plot(nnidx, desc_means[order][nnidx], color='red', label='Descendants')
    ax.legend(handles=[hanc,hdescmean,hbuick])
    ax.set_xlabel('Ancestor rank')
    ax.set_ylabel('Score distribution')
    ax.set_title(expt_label)

    ax = axes[1]
    # Calculate R^2
    p_descmean = np.polyfit(scmax[:alg.population_size][order][nnidx], desc_means[order][nnidx], 1)
    p_buick = np.polyfit(scmax[:alg.population_size], scmax_buick, 1)
    print(f'{p_descmean = }')
    print(f'{p_buick = }')
    R2_descmean = 1 - np.nansum((desc_means - p_descmean[1] - p_descmean[0]*scmax[:alg.population_size])**2) / np.nansum((desc_means - np.nanmean(desc_means))**2) 
    R2_buick = 1 - np.nansum((desc_means - p_buick[1] - p_buick[0]*scmax[:alg.population_size])**2) / np.nansum((scmax_buick - np.nanmean(scmax_buick))**2) 
    hdesc = ax.scatter(scmax[:alg.population_size], desc_means, color='red', marker='.', label=r'$R^2=%.2f$'%(R2_descmean))
    hbuick = ax.scatter(scmax[:alg.population_size], scmax_buick, color='gray', marker='*', label=r'$R^2=%.2f$'%(R2_buick))
    ax.set_xlabel('Ancestor score')
    ax.set_ylabel('Buick and descendant scores')
    ax.legend(handles=[hdesc,hbuick])
    fig.savefig(join(dirdict['plots'], 'scorrelation.png'), **pltkwargs)
    plt.close(fig)
    return


def measure_plot_score_distribution(config_algo, algs, dirdict, filedict, reference='dns', param_suffix='', overwrite_reference=False):
    tu = algs[0].ens.dynsys.dt_save
    if reference == 'buick':
        scmax_buick_file = join(dirdict['analysis'],'scmax_buick.npz')
        if (not exists(scmax_buick_file)) or overwrite_reference:
            print(f'{filedict["angel"] = }')
            angel = pickle.load(open(filedict['angel'], 'rb'))
            print(f'{angel.branching_state = }')
            mems_buick = []
            for i in range(angel.branching_state['num_buicks_generated']-1):
                if angel.branching_state['num_branches_generated'][i] > 0:
                    mems_buick.append(next(angel.ens.memgraph.successors(angel.branching_state['generation_0'][i])))
            score_fun = lambda ds: algs[0].score_combined(algs[0].score_components(ds['time'].to_numpy(),ds))
            lonroll = lambda ds,dlon: ds.roll(lon=int(round(dlon/ds['lon'][:2].diff('lon').item())))
            score_funs_rolled = [lambda ds: score_fun(lonroll(ds,dlon)) for dlon in [0,30,60,90,120,150,180,210,240,270,300,330]][:1]

            scbuick = np.concatenate(tuple(
                angel.ens.compute_observables(score_funs_rolled, mem) 
                for mem in mems_buick), axis=0) # TODO augment with zonal symmetry
            print(f'{scbuick = }')
            print(f'{scbuick.shape = }')
            print(f'{len(mems_buick) = }')
            scmax_ref = np.nanmax(scbuick[:,:algs[0].time_horizon], axis=1)
            np.savez(scmax_buick_file, scmax_ref=scmax_buick)
        else:
            scmax_ref = np.load(scmax_buick_file)['scmax_buick']
    elif reference == 'dns':
        spinup_phys = 700
        scmax_dns_file = join(dirdict['analysis'],'scmax_dns.npz')
        if (not exists(scmax_dns_file)) or overwrite_reference:
            dns = pickle.load(open(filedict['dns'], 'rb'))
            mems_dns = list(range(dns.ens.get_nmem()))
            print(f'{mems_dns = }')
            # TODO conctenate before taking score_dombined
            score_comp = lambda ds: algs[0].score_components(ds['time'].to_numpy(),ds)
            lonroll = lambda ds,dlon: ds.roll(lon=int(round(dlon/ds['lon'][:2].diff('lon').item())))
            dlons = range(0,360,20) #[0,30,60,90,120,150,180,210,240,270,300,330]
            score_comp_rolled = [lambda ds,dlon=dlon: score_comp(lonroll(ds,dlon)) for dlon in dlons]
            sccomp = []
            for mem in mems_dns:
                sccomp.append(dns.ens.compute_observables(score_comp_rolled,mem))
            scmax_alllon = []
            for i_lon in range(len(dlons)): 
                sccomp_ilon = xr.concat((sccomp[mem][i_lon] for mem in mems_dns), dim='time') 
                score_ilon = algs[0].score_combined(sccomp_ilon)[int(spinup_phys/tu):]
                scmax_ilon = utils.compute_block_maxima(score_ilon, algs[0].time_horizon-max(algs[0].advance_split_time_max, (algs[0].score_params['components']['rainrate']['tavg']-1)))
                print(f'At lonroll {dlons[i_lon]}: {scmax_ilon[:5] = }')
                scmax_alllon.append(scmax_ilon)
            scmax_ref = np.concatenate(scmax_alllon)

            print(f'{scmax_ref[:10] = }')
            print(f'{scmax_ref.shape = }')
            np.savez(scmax_dns_file, scmax=scmax_ref)
        else:
            scmax_ref = np.load(scmax_dns_file)['scmax']

    returnstats_file = join(dirdict['analysis'],'returnstats_%s.npz'%(param_suffix))
    figfile = join(dirdict['plots'],r'returnstats_%s.png'%(param_suffix))
    param_display = '\n'.join([
        r'$\sigma=%g$'%(algs[0].ens.dynsys.config['SPPT']['std_sppt']),
        r'$\delta=%g$'%(config_algo['advance_split_time_phys']),
        ])
    algorithms_frierson.FriersonGCMTEAMS.measure_plot_score_distribution(config_algo, algs, scmax_ref, returnstats_file, figfile, param_display=param_display)

    return




def run_teams(dirdict,filedict,config_gcm,config_algo):
    nproc = 4
    recompile = False
    root_dir = dirdict['data']
    angel = pickle.load(open(filedict['angel'], 'rb'))
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
        alg.set_capacity(config_algo['num_levels_max'], config_algo['num_members_max'])
    else:
        gcm = frierson_gcm.FriersonGCM(config_gcm, recompile=recompile)
        ens = ensemble.Ensemble(gcm, root_dir=root_dir)
        #alg = algorithms_frierson.FriersonGCMTEAMS.initialize_from_ancestorgenerator(angel, config_algo, ens)
        alg = algorithms_frierson.FriersonGCMTEAMS.initialize_from_dns(angel, config_algo, ens)

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


def teams_multiseed_procedure(i_sigma,i_delta,i_slm,idx_seed,overwrite_reference=False): # Just different seeds for now
    tododict = dict({
        'score_distribution': 1,
        'boost_distribution': 1,
        'boost_composites':   0,
        })
    # Figure out which flat indices corresond to this set of seeds
    multiparams = teams_multiparams()
    idx_multiparam = [(i_sigma,i_seed,i_delta,i_slm) for i_seed in idx_seed]
    print(f'{len(idx_multiparam) = }')
    idx_expt = []
    for i_multiparam in idx_multiparam:
        i_expt = np.ravel_multi_index(i_multiparam,tuple(len(mp) for mp in multiparams))
        idx_expt.append(i_expt) #list(range(1,21))
    workflows = []
    configs_gcm,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts = [[] for _ in range(7)]
    idx_expt_incomplete = []
    for i_expt in idx_expt:
        workflow = teams_single_workflow(i_expt)
        alg_file = workflow[6]['alg']
        if exists(alg_file):
            workflows.append(workflow)
            configs_gcm.append(workflow[0])
            configs_algo.append(workflow[1])
            configs_analysis.append(workflow[2])
            expt_labels.append(workflow[3])
            expt_abbrvs.append(workflow[4])
            dirdicts.append(workflow[5])
            filedicts.append(workflow[6])
        else:
            print(f'WARNING alg file for {i_expt = } does not exist. The location should be {alg_file}')
            idx_expt_incomplete.append(i_expt)
    for i_expt in idx_expt_incomplete:
        idx_expt.remove(i_expt)
    config_gcm = configs_gcm[0]
    config_algo = configs_algo[0]
    print(f'{idx_expt = }')
    sys.exit()
    
    filedict = dict({
        'angel': filedicts[0]['angel'],
        'dns': filedicts[0]['angel'], #'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-07-07/0/abs1_resT21_pertSPPT_std0p3_clip2_tau6h_L500km/DNS_si0/data/alg.pickle',
        })
    config_analysis = configs_analysis[0]
    param_abbrv_gcm,param_label_gcm = frierson_gcm.FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = algorithms_frierson.FriersonGCMTEAMS.label_from_config(config_algo)
    # Set up a meta-dirdict 
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-09-10"
    sub_date_str = "0"
    dirdict = dict()
    dirdict['meta'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo) 
    dirdict['data'] = join(dirdict['meta'], 'data')
    dirdict['analysis'] = join(dirdict['meta'], 'analysis')
    dirdict['plots'] = join(dirdict['meta'], 'plots')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    algs = []
    for i_alg in range(len(workflows)):
        algs.append(pickle.load(open(filedicts[i_alg]['alg'],'rb')))
    param_suffix = (r'std%g_ast%g'%(config_gcm['SPPT']['std_sppt'],config_algo['advance_split_time_phys'])).replace('.','p')
    if tododict['score_distribution']:
        measure_plot_score_distribution(config_algo, algs, dirdict, filedict, reference='dns', param_suffix=param_suffix, overwrite_reference=overwrite_reference)
    if tododict['boost_distribution']:
        figfile = join(dirdict['plots'], r'boost_distn_%s.png'%(param_suffix))
        algorithms_frierson.FriersonGCMTEAMS.measure_plot_boost_distribution(config_algo, algs, figfile)
    if tododict['boost_composites']:
        algorithms_frierson.FriersonGCMTEAMS.plot_boost_composites(algs, config_analysis, dirdict['plots'], param_suffix)
    return 

def teams_single_procedure(i_expt):

    tododict = dict({
        'run':             1,
        'analysis': dict({
            'observable_spaghetti':     1,
            'scorrelation':             1,
            'fields_2d':                1,
            }),
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = teams_single_workflow(i_expt)
    if tododict['run']:
        run_teams(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, alg, dirdict, filedict, remove_old_plots=True)
        # TODO have another ancestor-wise version, and another that shows family lines improving in parallel and dropping out
    if tododict['analysis']['scorrelation']:
        plot_scorrelations(config_analysis, alg, dirdict, filedict, expt_label)
    if tododict['analysis']['fields_2d']:
        plot_fields_2d(config_analysis, alg, dirdict, filedict, expt_label, remove_old_plots=True)
    return

def teams_multidelta_procedure(i_sigma,idx_delta,idx_seed):
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-04-04"
    sub_date_str = "1"
    multiparams = teams_multiparams()
    seed_incs,sigmas,deltas_phys,split_landmarks = multiparams
    kldiv_pooled = np.zeros(len(idx_delta))
    x2div_pooled = np.zeros(len(idx_delta))
    kldiv_sep = np.zeros((len(idx_seed),len(idx_delta)))
    x2div_sep = np.zeros((len(idx_seed),len(idx_delta)))
    boost_family_mean = np.zeros((len(idx_seed),len(idx_delta)))
    for i_delta in idx_delta:
        param_suffix = (r'std%g_ast%g'%(sigmas[i_sigma],deltas_phys[i_delta])).replace('.','p')
        idx_multiparam = [(i_seed,i_sigma,i_delta) for i_seed in idx_seed]
        idx_expt = []
        for i_multiparam in idx_multiparam:
            i_expt = np.ravel_multi_index(i_multiparam,tuple(len(mp) for mp in multiparams))
            idx_expt.append(i_expt) #list(range(1,21))
        workflows = tuple(teams_single_workflow(i_expt) for i_expt in idx_expt)
        configs_gcm,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts = tuple(
            tuple(workflows[i][j] for i in range(len(workflows)))
            for j in range(len(workflows[0])))
        config_gcm = configs_gcm[0]
        config_algo = configs_algo[0]
        param_abbrv_gcm,param_label_gcm = frierson_gcm.FriersonGCM.label_from_config(config_gcm)
        returnstats_file = join(scratch_dir,date_str,sub_date_str,param_abbrv_gcm,'meta','analysis','returnstats_%s.npz'%(param_suffix))
        print(f'{returnstats_file = }')
        returnstats = np.load(returnstats_file)
        print(f'{returnstats["hist_fin_wted"] = }')
        # Calculate the integrated error metrics 
        kldiv_pooled[i_delta],kldiv_sep[:,i_delta],x2div_pooled[i_delta],x2div_sep[:,i_delta] = compute_integrated_returnstats_error_metrics(returnstats)
        # Load the max-gains metrics
        boost_family_mean[:,i_delta] = returnstats['boost_family_mean']

    plot_dir = join(
            '/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/2024-04-12/2',
            param_abbrv_sde,
            'meta',
            'plots')

    # Plot 
    alphas = [0.5] # Sets the width of the band for KL divergence
    transparencies = [0.5,0.2]
    deltas = np.array([multiparams[2][i] for i in idx_delta])
    print(f'{x2div_sep[:,2] = }')

    # ------------ Plot X2-divergence ------------
    fig,ax = plt.subplots(figsize=(6,2))
    handles = []
    h, = ax.plot(deltas,np.median(x2div_sep,axis=0),color='red',marker='.',label='Runwise median')
    handles.append(h)
    h, = ax.plot(deltas,np.mean(x2div_sep,axis=0),color='black',marker='.',label='Runwise mean')
    handles.append(h)
    print(f'{np.mean(x2div_sep,axis=0) = }')
    for i_alpha,alpha in enumerate(alphas):
        lo,hi = np.quantile(x2div_sep, [alpha/2,1-alpha/2], axis=0)
        print(f'{alpha = }')
        print(f'{lo = }')
        print(f'{hi = }')
        h = ax.fill_between(deltas, lo, hi, fc='red', ec='none', alpha=transparencies[i_alpha], zorder=-i_alpha-1, label=r'{:d}% CI'.format(int(round((1-alpha)*100))))
        handles.append(h)
    ax.set_xlabel(r'$\delta$')
    ax.legend(handles=handles, bbox_to_anchor=(0.0,1.01), loc='lower left', title=r'$\chi^2$-divergence')
    ax.text(-0.15,0.5,r'$F_4=%g$'%(F4s[i_F4]),ha='right',va='center',transform=ax.transAxes)
    fig.savefig(join(plot_dir,'x2div.png'),**pltkwargs)
    plt.close(fig)

    # -------- Plot family gains ------------
    fig,ax = plt.subplots(figsize=(6,2))
    handles = []
    h, = ax.plot(deltas,np.median(boost_family_mean,axis=0),color='red',marker='.',label='Runwise median')
    handles.append(h)
    h, = ax.plot(deltas,np.mean(boost_family_mean,axis=0),color='black',marker='.',label='Runwise mean')
    handles.append(h)
    print(f'{np.mean(boost_family_mean,axis=0) = }')
    for i_alpha,alpha in enumerate(alphas):
        lo,hi = np.quantile(boost_family_mean, [alpha/2,1-alpha/2], axis=0)
        print(f'{alpha = }')
        print(f'{lo = }')
        print(f'{hi = }')
        h = ax.fill_between(deltas, lo, hi, fc='red', ec='none', alpha=transparencies[i_alpha], zorder=-i_alpha-1, label=r'{:d}% CI'.format(int(round((1-alpha)*100))))
        handles.append(h)
    ax.set_xlabel(r'$\delta$')
    ax.legend(handles=handles, bbox_to_anchor=(0.0,1.01), loc='lower left', title=r'Mean family boost')
    ax.text(-0.15,0.5,r'$F_4=%g$'%(F4s[i_F4]),ha='right',va='center',transform=ax.transAxes)
    fig.savefig(join(plot_dir,'mean_family_boost.png'),**pltkwargs)

    return

def compute_integrated_returnstats_error_metrics(returnstats):
    # -------- F-divergences --------
    hist_dns = returnstats['hist_dns'] / np.sum(returnstats['hist_dns'])
    hist_teams = returnstats['hist_fin_wted'] / np.sum(returnstats['hist_fin_wted'])
    hists_teams = np.diag(1/np.sum(returnstats['hists_fin_wted'], axis=1)) @ returnstats['hists_fin_wted'] 
    nalgs = len(hists_teams)
    nzidx_dns = np.where(hist_dns > 0)[0]
    # Pooled
    nzidx_teams = np.where(hist_teams > 0)[0]
    nzidx_both = np.intersect1d(nzidx_dns, nzidx_teams)
    kldiv_pooled = np.sum(hist_dns[nzidx_both] * np.log(hist_dns[nzidx_both] / hist_teams[nzidx_both]))
    x2div_pooled = np.sum((hist_dns[nzidx_dns] - hist_teams[nzidx_dns])**2 / hist_dns[nzidx_dns])
    # Separate
    kldiv_sep = np.zeros(nalgs)
    x2div_sep = np.zeros(nalgs)
    for i_alg in range(nalgs):
        nzidx_teams = np.where(hists_teams[i_alg] > 0)[0]
        nzidx_both = np.intersect1d(nzidx_dns, nzidx_teams)
        kldiv_sep[i_alg] = np.sum(hist_dns[nzidx_both] * np.log(hist_dns[nzidx_both] / hists_teams[i_alg,nzidx_both]))
        x2div_sep[i_alg] = np.sum((hist_dns[nzidx_dns] - hists_teams[i_alg,nzidx_dns])**2 / hist_dns[nzidx_dns])
    return kldiv_pooled,kldiv_sep,x2div_pooled,x2div_sep

if __name__ == "__main__":
    print(f'Got into Main')
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_expt = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'meta'
        sigmas,seed_incs,deltas_phys,split_landmarks = teams_multiparams()
        iseed_isigma_idelta_islm = [
                (i_seed,i_sigma,i_delta,0)
                for i_seed in range(8)
                for i_sigma in range(1)
                for i_delta in [3] #np.arange(4)[::-1]
                ]
        shp = (len(seed_incs),len(sigmas),len(deltas_phys),len(split_landmarks))
        idx_expt = []
        for i_multiparam in iseed_isigma_idelta_islm:
            i_expt = np.ravel_multi_index(i_multiparam,shp)
            idx_expt.append(i_expt)
    if procedure == 'single':
        for i_expt in idx_expt:
            teams_single_procedure(i_expt)
    elif procedure == 'multiseed':
        idx_seed = list(range(32))
        i_sigma = 0
        i_slm = 0
        i_delta = int(sys.argv[2])
        teams_multiseed_procedure(i_sigma,i_delta,i_slm,idx_seed,overwrite_reference=False)



