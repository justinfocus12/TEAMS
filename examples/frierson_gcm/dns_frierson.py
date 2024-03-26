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

def dns_paramset(i_param):
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
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_param], # add to seed_min
        'max_member_duration_phys': 100.0,
        'num_chunks_max': 21,
        })
    config_analysis = dict()
    config_analysis['spinup_phys'] = 700 # at what time to start computing statistics
    config_analysis['time_block_size_phys'] = 30 # size of block for method of block maxima
    # Statistics to compute
    config_analysis['stats_of_interest'] = dict({
        'moments': [1,2,3],
        'quantiles': [0.5,0.9,0.99],
        })
    # Fields to visualize mean state
    # Full-field observables for mean state and quantiles
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
            'label': 'Rain (6h avg)',
            }),
        'rain_lat30-90': dict({
            'fun': frierson_gcm.FriersonGCM.total_rain,
            'roi': dict(lat=slice(30,None)),
            'abbrv': 'Rlat30-90',
            'label': 'Rain (6h avg)',
            }),
        'u_500': dict({
            'fun': frierson_gcm.FriersonGCM.zonal_velocity,
            'roi': dict(pfull=500),
            'abbrv': 'U500',
            'label': r'Zon. Vel. ($p/p_s=0.5$)',
            }),
        'temp_700': dict({
            'fun': frierson_gcm.FriersonGCM.temperature,
            'roi': dict(pfull=700),
            'abbrv': 'T700',
            'label': r'Temp. ($p/p_s=0.7$)',
            }),
        })

    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv


def run_dns(dirdict,filedict,config_gcm,config_algo):
    nproc = 4
    recompile = False
    root_dir = dirdict['data']
    obs_fun = lambda t,x: None

    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
    else:
        gcm = frierson_gcm.FriersonGCM(config_gcm, recompile=recompile)
        ens = Ensemble(gcm, root_dir=root_dir)
        alg = algorithms_frierson.FriersonGCMDirectNumericalSimulation(config_algo, ens)
    alg.ens.dynsys.set_nproc(nproc)
    alg.ens.set_root_dir(root_dir)
    alg.set_simulation_capacity(config_algo['num_chunks_max'], config_algo['max_member_duration_phys'])
    nmem = alg.ens.get_nmem()
    num_new_chunks = alg.num_chunks_max - nmem
    print(f'{num_new_chunks = }')
    if num_new_chunks > 0:
        alg.terminate = False
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
        moments = config_analysis['stats_of_interest']['moments']
        f_stats = dict()
        for moment in moments:
            f_stats[r'moment%d'%(moment)] = np.power(f,moment).mean(dim=['time','lon'])
        quantiles = config_analysis['stats_of_interest']['quantiles']
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


    for obs_name,roi in obs_roi.items():
        obs_fun = {obs_name: lambda dsmem: getattr(ens.dynsys, obs_name)(dsmem)}
        fxypt = xr.concat(ens.compute_observables(obs_fun,mems2summarize)[obs_name], dim='time')
        # ----------------- Mean and quantiles at various latitudes ----------
        roi = {dim: val for (dim,val) in obs_roi[obs_name].items() if dim not in ['lon','lat']}
        sf = np.array([0.5,0.1,0.01,0.001]) # complementary quantiles of interest
        coords_sf = dict({c: fxypt.coords[c].to_numpy() for c in set(fxypt.dims) - {'time','lon','pfull'}})
        coords_sf['sf'] = sf
        f_sf = xr.DataArray(coords=coords_sf, dims=tuple(coords_sf.keys()), data=np.nan)
        for i,sfval in enumerate(sf):
            f_sf.loc[dict(sf=sfval)] = fxypt.sel(
                    roi,method='nearest',drop=True).quantile(1-sfval, dim=['time','lon'])

        f_mean = fxypt.sel(roi,method='nearest',drop=True).mean(dim=['time','lon'])
        f_sf_mean = xr.Dataset(data_vars={'fmean': f_mean, 'fsf': f_sf})
        f_sf_mean.to_netcdf(join(analysis_dir,f'mean_sf_{obs_name}.nc'))
        f_sf_mean.close()

        # ----------------- Return period curves at a fixed latitude --------
        roi = {dim: val for (dim,val) in obs_roi[obs_name].items() if dim not in ['lon']}
        lon_roll_step_requested = 30
        bin_lows,hist,rtime,logsf = ens.dynsys.compute_stats_dns_rotsym(fxypt, lon_roll_step_requested, time_block_size, roi)
        location_suffix = '_'.join([r'%s%g'%(roikey,roival) for (roikey,roival) in roi.items()])
        np.save(join(analysis_dir,f'distn_{obs_name}_{location_suffix}.npy'),np.vstack((bin_lows,hist,logsf,rtime)))
    return


    plot_dir = join(expt_dir,'plots')
    makedirs(plot_dir,exist_ok=True)
    if utils.find_true_in_dict(tododict['plot']):

        if tododict['plot']['return_stats']:
            for obs_name in ['temperature','total_rain']:
                # ------------------- Mean and quantiles at various latitudes ------------
                roi = {dim: val for (dim,val) in obs_roi[obs_name].items() if dim not in ['lon','lat']}
                location_suffix = ('_'.join([r'%s%g'%(roikey,roival) for (roikey,roival) in roi.items()])).replace('.','p')
                f_sf_mean = xr.open_dataset(join(analysis_dir,f'mean_sf_{obs_name}.nc'))
                print(f'{f_sf_mean.coords = }')
                print(f'{f_sf_mean["sf"].coords = }')
                fig,ax = plt.subplots()
                handles = []
                for i_sfval,sfval in enumerate(f_sf_mean['fsf'].coords['sf'].to_numpy()):
                    print(f'{i_sfval = }, {sfval = }')
                    xdata = f_sf_mean.lat.values
                    ydata = f_sf_mean['fsf'].isel(sf=i_sfval).to_numpy()
                    print(f'{xdata.shape = }')
                    print(f'{ydata.shape = }')
                    h, = ax.plot(xdata, ydata, label=f'{sfval}')
                    handles.append(h)
                h, = ax.plot(f_sf_mean['fmean'].lat.values, f_sf_mean['fmean'].values, color='black', linestyle='--', linewidth=2, label='mean')
                handles.append(h)
                ax.legend(handles=handles,title='Comp. quantiles')
                # Adjust x and y axis limits
                minlat = 30
                data4range = f_sf_mean['fsf'].sel(lat=slice(minlat,None))
                ax.set_ylim([data4range.min().item(), data4range.max().item()])
                ax.set_xlim([minlat,90])
                ax.set_xlabel('Latitude')
                fig.savefig(join(plot_dir,f'mean_sf_{obsprop[obs_name]["abbrv"]}_{location_suffix}.png'),**pltkwargs)
                plt.close(fig)


                # ------------------- Return period plots at a fixed latitude ---------
                roi = {dim: val for (dim,val) in obs_roi[obs_name].items() if dim not in ['lon']}
                location_suffix = ('_'.join([r'%s%g'%(roikey,roival) for (roikey,roival) in roi.items()])).replace('.','p')
                location_label = ', '.join([r'%s=%g'%(roikey,roival) for (roikey,roival) in roi.items()])
                bin_lows,hist,logsf,rtime = np.load(join(analysis_dir,f'distn_{obs_name}_{location_suffix}.npy'))
                print(f'{bin_lows[:3] = }')
                print(f'{hist[:3] = }')
                print(f'{rtime[:3] = }')
                bin_mids = bin_lows + 0.5*(bin_lows[1]-bin_lows[0])
                fig,axes = plt.subplots(ncols=2,figsize=(12,4),gridspec_kw={'wspace': 0.25})
                ax = axes[0]
                ax.plot(bin_lows,hist,color='black',marker='.')
                ax.set_xlabel(obsprop[obs_name]['label'])
                ax.set_ylabel('Prob. density')
                ax.set_yscale('log')
                ax = axes[1]
                ax.plot(rtime,bin_lows,color='black',marker='.')
                ax.set_ylim([bin_lows[np.argmax(rtime>0)],2*bin_lows[-1]-bin_lows[-2]])
                ax.set_xlabel('Return time')
                ax.set_ylabel('Return level')
                ax.set_xscale('log')
                ax.set_title(obsprop[obs_name]['label'])
                fig.suptitle(r'%s at %s'%(obsprop[obs_name]['label'],location_label))
                print(join(plot_dir,f'rtime_{obsprop[obs_name]["abbrv"]}.png'))
                fig.savefig(join(plot_dir,f'rtime_{obsprop[obs_name]["abbrv"]}_{location_suffix}.png'),**pltkwargs)
                plt.close(fig)


        if tododict['plot']['snapshots']:
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

def dns_workflow(i_param):
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv = dns_paramset(i_param)
    param_abbrv_gcm,param_label_gcm = frierson_gcm.FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = algorithms_frierson.FriersonGCMDirectNumericalSimulation.label_from_config(config_algo)
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-26"
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

def dns_single(i_param):
    tododict = dict({
        'run':                            0,
        'plot_snapshots':                 1,
        'compute_basic_stats':            0,
        'plot_return_stats':              0,
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = dns_workflow(i_param)

    if tododict['run']:
        run_dns(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'],'rb'))
    if tododict['plot_snapshots']:
        plot_snapshots(config_analysis, alg, dirdict)
    if tododict['compute_basic_stats']:
        compute_basic_stats(config_analysis, alg, dirdict)
    if tododict['plot_return_stats']:
        plot_return_stats(config_analysis, alg, dirdict)
    return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_param = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'single'
        idx_param = [5]
    print(f'Got into Main')
    if procedure == 'single':
        for i_param in idx_param:
            dns_single(i_param)
    elif procedure == 'meta':
        dns_meta_analysis_procedure(idx_param)

