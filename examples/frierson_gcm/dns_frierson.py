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
from algorithms_frierson import 

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
        'seed_inc_init': seed_incs[i_param], # add to seed_min
        'max_member_duration_phys': 100.0,
        'num_chunks_max': 21,
        })
    return config_gcm,config_algo,config_analysis,expt_labels[i_param],expt_abbrvs[i_param]


def run_dns(i_param):
    # Create a small ensemble

    root_dir = dirdict['data']
    obs_fun = lambda t,x: None

    if tododict['run']:
        days_per_chunk = 100
        num_chunks = 21
        if exists(ens_filename):
            ens = pickle.load(open(ens_filename,'rb'))
            ens.set_root_dir(root_dir)
            n_mem = ens.memgraph.number_of_nodes()
            parent = n_mem-1
            _,init_time = ens.get_member_timespan(n_mem-1)
            init_cond = ens.traj_metadata[n_mem-1]['filename_restart']
        else:
            gcm = FriersonGCM(config,recompile=recompile)
            ens = Ensemble(gcm,root_dir=root_dir)
            n_mem = 0
            init_time = 0
            init_cond = None
            parent = None
        ens.dynsys.set_nproc(nproc)
        for mem in range(n_mem,n_mem+num_chunks):
            fin_time = init_time + days_per_chunk
            icandf = ens.dynsys.generate_default_icandf(init_time,fin_time) # For SPPT, this will restart the random seed. 
            icandf['init_cond'] = init_cond
            # saveinfo will have RELATIVE paths 
            saveinfo = dict({
                # Temporary folder
                'temp_dir': f'mem{mem}',
                # Ultimate resulting filenames
                'filename_traj': f'mem{mem}.nc',
                'filename_restart': f'restart_mem{mem}.cpio',
                })
            _ = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
            init_time = fin_time
            parent = mem
            init_cond = ens.traj_metadata[parent]['filename_restart']
            pickle.dump(ens, open(join(expt_dir,'ens.pickle'),'wb'))
    # Load the ensemble for further analysis
    ens = pickle.load(open(join(expt_dir,'ens.pickle'),'rb'))
    obsprop = ens.dynsys.observable_props()
    # Make the directory for analysis
    analysis_dir = join(expt_dir,'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    # Select regions of interest
    lat_target = 45.0
    pfull_target = 1000
    obs_roi = dict({
        'temperature': dict(lat=lat_target,pfull=pfull_target),
        'total_rain': dict(lat=lat_target),
        })
    if tododict['summarize']:
        spinup = 700
        nmem = ens.get_nmem()
        all_starts,all_ends = ens.get_all_timespans()
        mems2summarize = np.where(all_starts >= spinup)[0]
        time_block_size = 25
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

def dns_procedure(i_param):
    tododict = dict({
        'run':                            1,
        'summarize':                      1,
        'plot': dict({
            'snapshots':    1,
            'return_stats': 1,
            }),
        })
    nproc = 4
    recompile = False
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv = dns_params(i_param)
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = FriersonGCM.label_from_config(config_algo)

    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-05"
    sub_date_str = "0/DNS"
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir,date_str,sub_date_str,param_abbrv_gcm,param_abbrv_algo)
    for subdir in ['data','analysis','plots']:
        dirdict[subdir] = join(dirdict['expt'],subdir)
        makedirs(dirdict[subdir], exist_ok=True)
    print(f'About to generate default config')
    filedict = dict({'alg': join(dirdict['data'],'alg.pickle')})

    if tododict['run']:
        run_dns(dirdict,filedict,config_gcm,config_algo)
