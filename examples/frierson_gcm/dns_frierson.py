# Instantiation of EnsembleMember class on Frierson GCM

import numpy as np
import xarray as xr
import f90nml
from matplotlib import pyplot as plt, rcParams 
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
from os.path import join, exists, basename
from os import mkdir, makedirs
import sys
import shutil
import glob
import subprocess
import resource
import pickle
import copy as copylib
import pprint

import sys
sys.path.append("../..")
import utils 
from ensemble import Ensemble
from dynamicalsystem import DynamicalSystem
import forcing
import precip_extremes_scaling
import algorithms_frierson
import frierson_gcm

def boolstr(b):
    if b:
        bstr = ".true."
    else:
        bstr = ".false."
    return bstr

def print_comp_proc(compproc):
    # String representation of a completed process
    print("args: \n")
    print(compproc.args)
    print(f"returncode: {compproc.returncode}\n")
    print(f"stdout: \n")
    print(compproc.stdout.decode('utf-8'))
    #for line in compproc.stdout:
    #    if isinstance(line, bytes):
    #        print(line.decode('utf-8'))
    print(f"stderr: \n")
    print(compproc.stderr.decode('utf-8'))
    #for line in compproc.stderr:
    #    if isinstance(line, bytes):
    #        print(line.decode('utf-8'))
    print("\n")
    return 

def dns_multiparams():
    seed_incs = [0]
    sigmas = [0.0,0.01,0.1,0.3,0.5]
    taus = [tau_hrs * 3600 for tau_hrs in [6,24,96]]
    Ls = [L_km * 1000 for L_km in [500,2000]]
    return seed_incs,sigmas,taus,Ls

def dns_paramset(i_expt):
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    config_gcm = frierson_gcm.FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)
    
    multiparams = dns_multiparams()
    idx_multiparam = np.unravel_index(i_expt, tuple(len(mp) for mp in multiparams))
    seed_inc,std_sppt,tau_sppt,L_sppt = (multiparams[i][i_param] for (i,i_param) in enumerate(idx_multiparam))

    expt_label = r'SPPT, $\sigma=%g$, $\tau=%g$ h, $L=%g$ km'%(std_sppt,tau_sppt/3600,L_sppt/1000)
    expt_abbrv = r'SPPT_std%g_tau%gh_L%gkm'%(std_sppt,tau_sppt/3600,L_sppt/1000)

    config_gcm['outputs_per_day'] = 4
    config_gcm['pert_type'] = 'SPPT'
    config_gcm['SPPT']['tau_sppt'] = tau_sppt
    config_gcm['SPPT']['std_sppt'] = std_sppt
    config_gcm['SPPT']['L_sppt'] = L_sppt
    config_gcm['remove_temp'] = 1
    pprint.pprint(config_gcm)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_inc, # will be added to seed_min
        'max_member_duration_phys': 100.0,
        'num_chunks_max': 70,
        })

    return config_gcm,config_algo,expt_label,expt_abbrv

def dns_single_workflow(i_expt):
    config_gcm,config_algo,expt_label,expt_abbrv = dns_paramset(i_expt)
    param_abbrv_gcm,param_label_gcm = frierson_gcm.FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = algorithms_frierson.FriersonGCMDirectNumericalSimulation.label_from_config(config_algo)
    config_analysis = dict()
    config_analysis['spinup_phys'] = 700 # at what time to start computing statistics
    config_analysis['time_block_size_phys'] = 30 # size of block for method of block maxima
    config_analysis['lon_roll_step'] = 30 # size of longitudinal shift for purposes of zonal symmetry augmentation in method of block maxima
    # Basic statistics to compute
    config_analysis['basic_stats'] = dict({
        'moments': [1,2,3],
        'quantiles': [0.5,0.9,0.99],
        })
    # Primary target location
    config_analysis['target_location'] = dict(lat=45,lon=180)
    # Fields to visualize 
    # 2D snapshots
    config_analysis['fields_lonlatdep'] = dict({
        'rain': dict({
            'fun': frierson_gcm.FriersonGCM.total_rain,
            'roi': None,
            'cmap': 'Blues',
            'label': 'Rain',
            'abbrv': 'R',
            }),
        'rain_extratropical': dict({
            'fun': frierson_gcm.FriersonGCM.total_rain,
            'roi': dict(lat=slice(30,None)),
            'cmap': 'Blues',
            'label': 'Rain',
            'abbrv': 'Rlat30-90',
            }),
        'temp_700': dict({
            'fun': frierson_gcm.FriersonGCM.temperature,
            'roi': dict(pfull=700),
            'cmap': 'Reds',
            'label': 'Temp ($p/p_s=0.7$)',
            'abbrv': 'T700',
            }),
        'u_500': dict({
            'fun': frierson_gcm.FriersonGCM.zonal_velocity,
            'roi': dict(pfull=500),
            'cmap': 'coolwarm',
            'label': 'Zon. Vel. ($p/p_s=0.5$)',
            'abbrv': 'U500',
            }),
        'surface_pressure': dict({
            'fun': frierson_gcm.FriersonGCM.surface_pressure,
            'roi': None,
            'cmap': 'coolwarm',
            'label': 'Surf. Pres.',
            'abbrv': 'PS',
            }),
        })
    # Latitude-dependent fields, zonally symmetric
    config_analysis['fields_latdep'] = dict({
        'rain': dict({
            'fun': frierson_gcm.FriersonGCM.total_rain,
            'roi': None,
            'abbrv': 'R',
            'label': 'Rain (6h avg) [mm/day]',
            }),
        'rain_lat30-90': dict({
            'fun': frierson_gcm.FriersonGCM.total_rain,
            'roi': dict(lat=slice(30,None)),
            'abbrv': 'Rlat30-90',
            'label': 'Rain (6h avg) [mm/day]',
            }),
        'u_500': dict({
            'fun': frierson_gcm.FriersonGCM.zonal_velocity,
            'roi': dict(pfull=500),
            'abbrv': 'U500',
            'label': r'Zon. Vel. ($p/p_s=0.5$) [m/s]',
            }),
        'temp_700': dict({
            'fun': frierson_gcm.FriersonGCM.temperature,
            'roi': dict(pfull=700),
            'abbrv': 'T700',
            'label': r'Temp. ($p/p_s=0.7$) [K]',
            }),
        })
    # Observables to get extreme statistics of using zonal symmetry
    config_analysis['observables_onelat_zonsym'] = dict({
        'rain_lat45': dict({
            'fun': frierson_gcm.FriersonGCM.latband_rain,
            'kwargs': dict({
                'roi': dict({'lat': 45}),
                }),
            'abbrv': 'R45',
            'label': 'Rain (6h avg, $\phi=45$) [mm/day]',
            }),
        'rain_lat35-55': dict({
            'fun': frierson_gcm.FriersonGCM.latband_rain,
            'kwargs': dict({
                'roi': dict({'lat': slice(35,55)}),
                }),
            'abbrv': 'R35-55',
            'label': 'Mean rain ($\phi\in(35,55)$) [mm/day]',
            }),
        'temp_lat45_pfull700': dict({
            'fun': frierson_gcm.FriersonGCM.latband_temp,
            'kwargs': dict({
                'roi': dict({'lat': 45, 'pfull': 700}),
                }),
            'abbrv': 'Tlat45pfull700',
            'label': 'Temperature ($\phi=45,p/p_s=0.7$)',
            }),
        'temp_lat35-55_pfull700': dict({
            'fun': frierson_gcm.FriersonGCM.latband_temp,
            'kwargs': dict({
                'roi': dict({'lat': slice(35,55), 'pfull': 700}),
                }),
            'abbrv': 'Tlat35-55pfull700',
            'label': 'Temperature ($\phi\in(35,55),p/p_s=0.7$)',
            })
        })

    # Scalar observables to plot timeseries of 
    config_analysis['observables'] = dict({
        'local_rain': dict({
            'fun': frierson_gcm.FriersonGCM.regional_rain,
            'kwargs': dict({
                'roi': config_analysis['target_location'],
                }),
            'abbrv': 'Rloc',
            'label': r'Rain rate $(\phi,\lambda)=(45,180)$',
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
        })
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-04-04"
    sub_date_str = "0"
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir,date_str,sub_date_str,param_abbrv_gcm,param_abbrv_algo)
    for subdir in ['data','analysis','plots']:
        dirdict[subdir] = join(dirdict['expt'],subdir)
        makedirs(dirdict[subdir], exist_ok=True)
    print(f'About to generate default config')
    filedict = dict({
        'alg': join(dirdict['data'],'alg.pickle'),
        'alg_backup': join(dirdict['data'],'alg_backup.pickle')
        })
    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict


def run_dns(dirdict,filedict,config_gcm,config_algo):
    nproc = 4
    recompile = False
    root_dir = dirdict['data']
    obs_fun = lambda t,x: None

    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
        alg.ens.set_root_dir(root_dir)
        alg.set_capacity(config_algo['num_chunks_max'], config_algo['max_member_duration_phys'])
    else:
        gcm = frierson_gcm.FriersonGCM(config_gcm, recompile=recompile)
        ens = Ensemble(gcm, root_dir=root_dir)
        alg = algorithms_frierson.FriersonGCMDirectNumericalSimulation(config_algo, ens)
    alg.ens.dynsys.set_nproc(nproc)
    nmem = alg.ens.get_nmem()
    while not (alg.terminate):
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict({
            'temp_dir': f'mem{mem}',
            'filename_traj': f'mem{mem}.nc',
            'filename_restart': f'restart_mem{mem}.cpio',
            })
        alg.take_next_step(saveinfo)
        if exists(filedict['alg']):
            os.rename(filedict['alg'], filedict['alg_backup'])
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def plot_snapshots(config_analysis, alg, dirdict):
    tu = alg.ens.dynsys.dt_save
    spinup = int(config_analysis['spinup_phys']/tu)
    time_block_size = int(config_analysis['time_block_size_phys']/tu)
    all_starts,all_ends = alg.ens.get_all_timespans()
    mem = np.where(all_starts >= spinup)[0][0]
    print(f'{spinup = }, {mem = }')
    for (field_name,field_props) in config_analysis['fields_lonlatdep'].items():
        fig,ax = plt.subplots()
        field = frierson_gcm.FriersonGCM.sel_from_roi(alg.ens.compute_observables([field_props['fun']], mem)[0], field_props['roi'])
        xr.plot.pcolormesh(field.isel(time=0), x='lon', y='lat', cmap=field_props['cmap'], ax=ax)
        ax.set_title(field_props['label'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.savefig(join(dirdict['plots'],r'%s_t%g.png'%(field_props['abbrv'],all_starts[mem]*tu)), **pltkwargs)
        plt.close(fig)
    return

def plot_timeseries(config_analysis, alg, dirdict):
    tu = alg.ens.dynsys.dt_save
    spinup = int(config_analysis['spinup_phys']/tu)
    time_block_size = int(config_analysis['time_block_size_phys']/tu)
    duration = 3*time_block_size
    all_starts,all_ends = alg.ens.get_all_timespans()
    mem_first = np.where(all_starts >= spinup)[0][0]
    mem_last = np.where(all_ends >= all_starts[mem_first] + duration)[0][0]
    for (obs_name,obs_props) in config_analysis['observables'].items():
        fun = lambda ds: obs_props['fun'](ds, **obs_props['kwargs'])
        obs_val = xr.concat(tuple(
            alg.ens.compute_observables([fun], mem)[0] for mem in range(mem_first,mem_last+1)), dim='time')
        obs_val = obs_val.isel(time=slice(spinup-all_starts[mem_first], spinup+duration-all_starts[mem_first]))
        fig,ax = plt.subplots(figsize=(18,3))
        xr.plot.plot(obs_val, x='time', ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('')
        ax.set_title(obs_props['label'])
        figfilename = (r'%s_t%g-%g'%(obs_props['abbrv'],spinup,spinup+duration)).replace('.','p')
        fig.savefig(join(dirdict['plots'], r'%s.png'%(figfilename)), **pltkwargs)
        plt.close(fig)
    return

def compute_basic_stats(config_analysis, alg, dirdict):
    obsprop = alg.ens.dynsys.observable_props()
    nmem = alg.ens.get_nmem()
    tu = alg.ens.dynsys.dt_save
    spinup = int(config_analysis['spinup_phys']/tu)
    time_block_size = int(config_analysis['time_block_size_phys']/tu)
    all_starts,all_ends = alg.ens.get_all_timespans()
    mems2summarize = np.where(all_starts >= spinup)[0]
    print(f'{mems2summarize = }')
    # Visualize zonal mean fields
    for (field_name,field_props) in config_analysis['fields_latdep'].items():
        print(f'Computing stats of {field_name}')
        fun = lambda ds: frierson_gcm.FriersonGCM.sel_from_roi(field_props['fun'](ds), field_props['roi'])
        f = xr.concat(tuple(alg.ens.compute_observables([fun], mem)[0] for mem in mems2summarize), dim='time')
        print(f'{f.dims = }, {f.coords = }')
        moments = config_analysis['basic_stats']['moments']
        f_stats = dict()
        for moment in moments:
            f_stats[r'moment%d'%(moment)] = np.power(f,moment).mean(dim=['time','lon'])
        quantiles = config_analysis['basic_stats']['quantiles']
        f_stats['quantiles'] = f.quantile(quantiles, dim=['time','lon']) # TODO instead stack the longitudes and then take quantiles! 
        f_stats = xr.Dataset(data_vars = f_stats)
        f_stats.to_netcdf(join(dirdict['analysis'], r'%s.nc'%(field_props['abbrv'])))
        # Plot 
        fig,ax = plt.subplots()
        hmean, = xr.plot.plot(f_stats['moment1'], x='lat', color='black', linestyle='--', linewidth=2, label=r'Mean')
        for quantile in quantiles:
            hquant, = xr.plot.plot(f_stats['quantiles'].sel(quantile=quantile), x='lat', color='dodgerblue', label=r'Quantiles')
        ax.legend(handles=[hmean,hquant])
        ax.set_xlabel("Latitude")
        ax.set_ylabel(field_props['label'])
        ax.set_title('')
        fig.savefig(join(dirdict['plots'], r'%s.png'%(field_props['abbrv'])), **pltkwargs)
        plt.close(fig)
    return


def compare_basic_stats(workflows, config_meta_analysis, meta_dirdict):
    # group the different experiments by L_sppt and tau_sppt 
    num_expt = len(workflows['configs_gcm'])
    Ls = tuple(workflows['configs_gcm'][i]['SPPT']['L_sppt'] for i in range(num_expt))
    taus = tuple(workflows['configs_gcm'][i]['SPPT']['tau_sppt'] for i in range(num_expt))
    sigmas = tuple(workflows['configs_gcm'][i]['SPPT']['std_sppt'] for i in range(num_expt))
    Ltau = list(set(Ltau))
    Ltau_unique = np.unique(np.array(Ltau),axis=0)
    Ltau_idx_groups = tuple(
            tuple(i for i in range(num_expt) if Ltau[i] == Ltau_val)
            for Ltau_val in Ltau_unique
            )
    print(f'{Ltau_idx_groups = }')
    # TODO put the above construction into a separate 'workflow'-style function
    quantiles = config_meta_analysis['basic_stats']['quantiles']
    for (field_name,field_props) in config_meta_analysis['fields_latdep'].items():
        fig,axes = plt.subplots(ncols=len(Ltau_idx_groups),nrows=len(quantiles)+1, figsize=(6*len(Ltau_idx_groups),6*(1+len(quantiles))),sharey='row',sharex=True)
        # ----------- Mean ---------
        log_sig_scaled = np.log(np.array(sigmas))
        log_sig_scaled = (log_sig_scaled - np.min(log_sig_scaled))/np.ptp(log_sig_scaled)
        colors = plt.cm.coolwarm(log_sig_scaled)
        for i_group,idx in enumerate(Ltau_idx_groups):
            print(f'{idx = }')
            f_mean = []
            f_quantiles = []
            for i in idx:
                fstats_i = xr.open_dataset(join(workflows['dirdicts'][i]['analysis'],r'%s.nc'%(field_props['abbrv'])))
                f_mean.append(fstats_i['moment1'])
                f_quantiles.append(fstats_i['quantiles'])
            print(f'{f_mean[0] = }')
            print(f'{f_quantiles[0] = }')
            f_mean = xr.concat(f_mean, dim='expt').assign_coords(expt=list(idx))
            f_quantiles = xr.concat(f_quantiles, dim='expt').assign_coords(expt=list(idx))
            # Means
            handles = []
            ax = axes[0,i_group]
            for i in idx:
                h, = xr.plot.plot(f_mean.sel(expt=i), x='lat', label=r'$\sigma_{\mathrm{SPPT}}=%g$'%(sigmas[i]), ax=ax, color=colors[i])
                handles.append(h)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            ax.xaxis.set_tick_params(which='both',labelbottom=True)
            ax.yaxis.set_tick_params(which='both',labelbottom=True)
            # Quantiles
            for i_quant,quant in enumerate(config_meta_analysis['basic_stats']['quantiles']):
                ax = axes[1+i_quant,i_group]
                for i in idx:
                    xr.plot.plot(f_quantiles.sel(expt=i,quantile=quant), x='lat', label=r'$\sigma_{\mathrm{SPPT}}=%g$'%(sigmas[i]), ax=ax, color=colors[i])
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('')
                ax.xaxis.set_tick_params(which='both',labelbottom=True)
                ax.yaxis.set_tick_params(which='both',labelbottom=True)
            group_label = r'$L=%g$km, $\tau=%g$h'%(Ltau_unique[i_group][0]/1000,Ltau_unique[i_group][1]/3600)
            axes[0,i_group].set_title(group_label)
            axes[-1,i_group].set_xlabel('Latitude')
        axes[0,0].legend(handles=handles,ncol=len(handles),loc=(0,1.1))
        axes[0,0].set_ylabel(r'Mean %s'%(field_props['label']))
        for i_quant,quant in enumerate(quantiles):
            axes[1+i_quant,0].set_ylabel(r'Quantile %g %s'%(quant,field_props['label']))
        fig.savefig(join(meta_dirdict['plots'], r'mean_latdep_%s.png'%(field_props['abbrv'])), **pltkwargs)
        plt.close(fig)
    return

def compare_extreme_stats(workflows,config_meta_analysis,meta_dirdict):
    num_expt = len(workflows['configs_gcm'])
    Ls = tuple(workflows['configs_gcm'][i]['SPPT']['L_sppt'] for i in range(num_expt))
    taus = tuple(workflows['configs_gcm'][i]['SPPT']['tau_sppt'] for i in range(num_expt))
    sigmas = tuple(workflows['configs_gcm'][i]['SPPT']['std_sppt'] for i in range(num_expt))
    tu = 1/workflows['configs_gcm'][0]['outputs_per_day']
    Ltau = tuple(zip(Ls,taus))
    Ltau_unique = list(set(Ltau))
    Ltau_idx_groups = tuple(
            tuple(i for i in range(num_expt) if Ltau[i] == Ltau_val)
            for Ltau_val in Ltau_unique
            )
    print(f'{Ltau_idx_groups = }')
    for (obs_name,obs_props) in config_meta_analysis['observables_onelat_zonsym'].items():
        # Return period curves (together with GEV fit in dashed lines)
        fig_curves,axes_curves = plt.subplots(ncols=len(Ltau_unique), figsize=(6*len(Ltau_unique),4), sharey='row', sharex=True)
        # GEV parameters as a function of sigma
        fig_gevpar,axes_gevpar = plt.subplots(nrows=3,figsize=(4,6),sharex=True)
        log_sig_scaled = np.log(np.array(sigmas))
        log_sig_scaled = (log_sig_scaled - np.min(log_sig_scaled))/np.ptp(log_sig_scaled)
        colors = plt.cm.coolwarm(log_sig_scaled)
        handles_gevpar = []
        for i_group,idx in enumerate(Ltau_idx_groups):
            sigmas_idx = np.array([sigmas[i] for i in idx])
            handles_curves = []
            shapes,locs,scales = (np.zeros(len(idx)) for _ in range(3))
            ax = axes_curves[i_group]
            for ii,i in enumerate(idx):
                extstats = np.load(join(workflows['dirdicts'][i]['analysis'],r'extstats_zonsym_%s.npz'%(obs_props['abbrv'])))
                shapes[ii],locs[ii],scales[ii] = extstats['shape'],extstats['loc'],extstats['scale']
                h, = ax.plot(extstats['rtime']*tu, extstats['bin_lows'], color=colors[i], linestyle='-', label=r'$\sigma_{\mathrm{SPPT}}=%g$'%(sigmas[i]))
                ax.plot(extstats['rtime_gev']*tu, extstats['bin_lows'], color=colors[i], linestyle='--')
                handles_curves.append(h)
            group_label = r'$L=%g$km, $\tau=%g$h'%(Ltau_unique[i_group][0]/1000,Ltau_unique[i_group][1]/3600)
            group_color = plt.cm.Set1(i_group)
            ax.set_title(group_label)
            ax.set_xlabel('Return time [days]')
            ax.set_xlim([30,20000])
            ax.set_xscale('log')
            axes_curves[-1].legend(handles=handles_curves,loc=(1,0))

            # GEV parameters
            ax = axes_gevpar[0]
            h, = ax.plot(sigmas_idx,shapes,color=group_color,marker='.',linestyle='-',label=group_label)
            handles_gevpar.append(h)
            ax = axes_gevpar[1]
            ax.plot(sigmas_idx,locs,color=group_color,marker='.',linestyle='-')
            ax = axes_gevpar[2]
            ax.plot(sigmas_idx,scales,color=group_color,marker='.',linestyle='-')
        fig_curves.suptitle(obs_props['label'], y=1.0)
        fig_curves.savefig(join(meta_dirdict['plots'], 'extstats_returncurves_%s.png'%(obs_props['abbrv'])), **pltkwargs)
        plt.close(fig_curves)

        axes_gevpar[0].set(ylabel='Shape',xlabel='')
        axes_gevpar[1].set(ylabel='Location',xlabel='')
        axes_gevpar[2].set(ylabel='Scale',xlabel=r'$\sigma_{\mathrm{SPPT}}$')
        axes_gevpar[-1].legend(handles=handles_gevpar, loc='upper left', bbox_to_anchor=(0,-0.5))
        fig_gevpar.suptitle(obs_props['label'])
        fig_gevpar.savefig(join(meta_dirdict['plots'], 'exstats_gevpar_%s.png'%(obs_props['abbrv'])), **pltkwargs)
        plt.close(fig_gevpar)
    return


def compute_extreme_stats(config_analysis, alg, dirdict):
    nmem = alg.ens.get_nmem()
    tu = alg.ens.dynsys.dt_save
    spinup = int(config_analysis['spinup_phys']/tu)
    time_block_size = int(config_analysis['time_block_size_phys']/tu)
    all_starts,all_ends = alg.ens.get_all_timespans()
    mems2summarize = np.where(all_starts >= spinup)[0]
    for obs_name,obs_props in config_analysis['observables_onelat_zonsym'].items():
        fun = lambda ds: obs_props['fun'](ds, **obs_props['kwargs'])
        fxt = xr.concat(tuple(alg.ens.compute_observables([fun], mem)[0] for mem in mems2summarize), dim='time')
        lon_roll_step_requested = config_analysis['lon_roll_step']
        bin_lows,hist,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale = alg.ens.dynsys.compute_stats_dns_rotsym(fxt, lon_roll_step_requested, time_block_size)
        extstats = dict({'bin_lows': bin_lows, 'hist': hist, 'rtime': rtime, 'logsf': logsf, 'rtime_gev': rtime_gev, 'logsf_gev': logsf_gev, 'shape': shape, 'loc': loc, 'scale': scale})
        np.savez(join(dirdict['analysis'],r'extstats_zonsym_%s.npz'%(obs_props['abbrv'])), **extstats)
        # Plot 
        bin_mids = bin_lows + 0.5*(bin_lows[1]-bin_lows[0])
        fig,axes = plt.subplots(ncols=2,figsize=(12,4),gridspec_kw={'wspace': 0.25})
        ax = axes[0]
        ax.plot(bin_lows,hist,color='black',marker='.')
        ax.set_xlabel(obs_props['label'])
        ax.set_ylabel('Prob. density')
        ax.set_yscale('log')
        ax = axes[1]
        hemp, = ax.plot(rtime,bin_lows,color='black',marker='.',label='Empirical')
        hgev, = ax.plot(rtime_gev,bin_lows,color='cyan',marker='.',label='GEV fit')
        ax.legend(handles=[hemp,hgev])
        print(f'{rtime_gev = }')
        ax.set_ylim([bin_lows[np.argmax(rtime>0)],2*bin_lows[-1]-bin_lows[-2]])
        ax.set_xlabel('Return time')
        ax.set_ylabel('Return level')
        ax.set_xscale('log')
        fig.suptitle(obs_props['label'])
        fig.savefig(join(dirdict['plots'],r'extstats_%s.png'%(obs_props['abbrv'])),**pltkwargs)
        plt.close(fig)


        if 0 and tododict['plot']['snapshots']:
            lat = 45.0
            lon = 180.0
            pfull = 1000.0
            obs_funs = dict()
            for obs_name in ['temperature']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(ens.dynsys, obs_name)(dsmem).sel(pfull=pfull,method='nearest')
            for obs_name in ['r_sppt_g','total_rain','column_water_vapor','surface_pressure']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(ens.dynsys, obs_name)(dsmem)
            mems2plot = [ens.get_nmem()-1]
            obs_vals = ens.compute_observables(obs_funs, mems2plot)

            for i_mem,mem in enumerate(mems2plot):
                for obs_name in list(obs_funs.keys()):
                    memobs = obs_vals[obs_name][i_mem].compute()
                    # Plot a few daily snapshots
                    for day in memobs.time.to_numpy()[:2]: #.astype(int):
                        fig,axes = plt.subplots(figsize=(12,5),ncols=2,sharey=True)
                        ax = axes[0]
                        xr.plot.pcolormesh(memobs.sel(time=day), x='lon', y='lat', cmap=obsprop[obs_name]['cmap'], ax=ax)
                        ax.set_title(r'%s [%s], mem. %d, day %d'%(obsprop[obs_name]['label'], obsprop[obs_name]['unit_symbol'], mem, day))
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        ax = axes[1]
                        hday, = xr.plot.plot(memobs.mean(dim=['time','lon']),y='lat',color='black',ax=ax,label=r'(zonal,time) avg')
                        havg, = xr.plot.plot(memobs.sel(time=day).mean(dim='lon'),y='lat',color='red',ax=ax,label=r'zonal avg')
                        ax.set_title("")
                        ax.set_xlabel(r'%s [%s]'%(obsprop[obs_name]['label'],obsprop[obs_name]['unit_symbol']))
                        ax.set_ylabel('')
                        ax.legend(handles=[hday,havg])

                        fig.savefig(join(plot_dir,r'%s_mem%d_day%d'%(obsprop[obs_name]['abbrv'],mem,day)),**pltkwargs)
                        plt.close(fig)
                    # Plot timeseries
                    fig,ax = plt.subplots()
                    xr.plot.plot(memobs.sel(lat=lat,lon=lon,method='nearest'), x='time', color='black')
                    ax.set_xlabel("time")
                    ax.set_ylabel(r'%s [%s]'%(obsprop[obs_name]['label'],obsprop[obs_name]['unit_symbol']))
                    ax.set_title(r'$(\lambda,\phi)=(%g,%g)$'%(lon,lat))
                    fig.savefig(join(plot_dir,r'%s_mem%d'%(obsprop[obs_name]['abbrv'],mem)),**pltkwargs)
                    plt.close(fig)
    return

def dns_meta_workflow(idx_param):
    num_expt = len(idx_param)
    workflow_tuple = tuple(dns_single_workflow(i_param) for i_param in idx_param)
    workflows = dict()
    for i_key,key in enumerate(('configs_gcm,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts').split(',')):
        workflows[key] = tuple(workflow_tuple[j][i_key] for j in range(len(workflow_tuple)))
    print(f'{workflows.keys() = }')
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-26"
    sub_date_str = "0"
    meta_dirdict = dict()
    meta_dirdict['meta'] = join(scratch_dir,date_str,sub_date_str,'meta')
    for subdir in ['data','analysis','plots']:
        meta_dirdict[subdir] = join(meta_dirdict['meta'],subdir)
        makedirs(meta_dirdict[subdir], exist_ok=True)
    config_meta_analysis = dict()
    for key in ['basic_stats','target_location','fields_latdep','observables_onelat_zonsym']:
        config_meta_analysis[key] = workflows['configs_analysis'][0][key]
    # Group together fixed and variable parameters
    return workflows,config_meta_analysis,meta_dirdict



def dns_meta_procedure(idx_expt):
    tododict = dict({
        'compare_basic_stats':            0,
        'compare_extreme_stats':          1,
        })
    workflows,config_meta_analysis,meta_dirdict = dns_meta_workflow(idx_expt)
    if tododict['compare_basic_stats']:
        compare_basic_stats(workflows,config_meta_analysis,meta_dirdict)
    if tododict['compare_extreme_stats']:
        compare_extreme_stats(workflows,config_meta_analysis,meta_dirdict)
    return

def dns_single_procedure(i_expt):
    tododict = dict({
        'run':                            1,
        'plot_snapshots':                 1,
        'plot_timeseries':                1,
        'compute_basic_stats':            1,
        'compute_extreme_stats':          1,
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = dns_single_workflow(i_expt)

    if tododict['run']:
        run_dns(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'],'rb'))
    if tododict['plot_snapshots']:
        plot_snapshots(config_analysis, alg, dirdict)
    if tododict['plot_timeseries']:
        plot_timeseries(config_analysis, alg, dirdict)
    if tododict['compute_basic_stats']:
        compute_basic_stats(config_analysis, alg, dirdict)
    if tododict['compute_extreme_stats']:
        compute_extreme_stats(config_analysis, alg, dirdict)
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
            dns_single_procedure(i_expt)
    elif procedure == 'meta':
        dns_meta_procedure(idx_expt)

