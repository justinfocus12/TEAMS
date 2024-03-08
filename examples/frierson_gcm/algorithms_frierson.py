# Instantiation of EnsembleMember class on Frierson GCM
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
import algorithms; reload(forcing)
print(f'{i = }'); i += 1
import frierson_gcm; reload(frierson_gcm)
from frierson_gcm import FriersonGCM
print(f'{i = }'); i += 1

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

def run_periodic_branching(nproc,recompile,i_param):
    tododict = dict({
        'run_pebr':                1,
        'analyze_pebr': dict({
            'measure_pert_growth':           1,
            'analyze_pert_growth':           1,
            }),
        'plot_pebr': dict({
            'observables':    1,
            'fields':         1,
            'pert_growth':    1,
            'response':       0,
            }),
        })
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-05"
    sub_date_str = "0/PeBr"
    print(f'About to generate default config')
    config_gcm = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)

    # Parameters to loop over
    pert_type_list = ['IMP']        + ['SPPT']*16
    std_sppt_list = [0.5]           + [0.5,0.1,0.05,0.01]*4
    tau_sppt_list = [6.0*3600]      + [6.0*3600]*4   + [6.0*3600]*4    + [24.0*3600]*4     + [96.0*3600]*4 
    L_sppt_list = [500.0*1000]      + [500.0*1000]*4 + [2000.0*1000]*4 + [500.0*1000]*4    + [500.0*1000]*4

    config_gcm['pert_type'] = pert_type_list[i_param]
    if config_gcm['pert_type'] == 'SPPT':
        config_gcm['SPPT']['tau_sppt'] = tau_sppt_list[i_param]
        config_gcm['SPPT']['std_sppt'] = std_sppt_list[i_param]
        config_gcm['SPPT']['L_sppt'] = L_sppt_list[i_param]
    config_gcm['remove_temp'] = 1
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'branches_per_group': 12, 
        'interbranch_interval_phys': 10.0,
        'branch_duration_phys': 30.0,
        'num_branch_groups': 20,
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

    # ----------- Configure post-analysis ---------------------
    fndict['plots'] = dict()
    config_analysis = dict({
        'dist_names': ['total_rain_eucdist','column_water_vapor_eucdist','surface_pressure_eucdist'],


    dist_names = ['temperature','column_water_vapor','surface_pressure','total_rain',]
    for dist_name in dist_names:
        fndict['plots'][dist_name] = dict({'rmse': join(dirdict['plots'],f'rmse_dist{dist_name}')})
        fndict['plots'][dist_name]['lyap_exp'] = join(dirdict['plots'], f'lyap_exp_dist{dist_name}')
        for branch_group in range(config_algo['num_branch_groups']):
            fndict['plots'][dist_name][branch_group] = join(dirdict['plots'],f'pert_growth_bg{branch_group}_dist{dist_name}.png')

    root_dir = dirdict['alg']
    alg_filename = join(dirdict['alg'],'alg.pickle')
    # TODO write config to file, too 
    init_cond_dir = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-03-05/0/DNS/',
            param_abbrv_gcm)
    init_time = int(xr.open_mfdataset(join(init_cond_dir,'mem20.nc'),decode_times=False)['time'].load()[-1].item())
    init_cond = relpath(join(init_cond_dir,'restart_mem20.cpio'), root_dir)
        
    if tododict['run_pebr']:
        if exists(alg_filename):
            alg = pickle.load(open(alg_filename, 'rb'))
        else:
            gcm = FriersonGCM(config_gcm, recompile=recompile)
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
        if tododict['analyze_pebr']['measure_pert_growth'] and (not exists(fndict['analysis']['pert_growth'])):
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
            def dist_rolling_window(ds0,ds1,field_name,window):
                t0 = ds0.time.to_numpy()
                t1 = ds1.time.to_numpy()
                tsel = dict(time=slice(max(t0[0],t1[0]),min(t0[-1],t1[-1])+1))
                nt = tsel.stop - tsel.start + 1
                f0 = getattr(alg.ens.dynsys, field_name)(ds0).sel(tsel).sel(lat=lat,lon=lon,method='nearest')
                f1 = getattr(alg.ens.dynsys, field_name)(ds1).sel(tsel).sel(lat=lat,lon=lon,method='nearest')
                if 'pfull' in f0.dims:
                    f0 = f0.sel(pfull=pfull,method='nearest')
                    f1 = f1.sel(pfull=pfull,method='nearest')
                f0 = f0.compute().to_numpy()
                f1 = f1.compute().to_numpy()
                D2 = np.convolve((f0 - f1)**2, np.ones(window), 'full')[:nt]
                D2[:window] = np.nan
                return D2
            def rmsd_rolling_window(ds0,ds1,field_name,window):
                f0 = getattr(alg.ens.dynsys, field_name)(ds0).sel(tsel).sel(lat=lat,lon=lon,method='nearest')
                f1 = getattr(alg.ens.dynsys, field_name)(ds1).sel(tsel).sel(lat=lat,lon=lon,method='nearest')
                if 'pfull' in f0.dims:
                    f0 = f0.sel(pfull=pfull,method='nearest')
                    f1 = f1.sel(pfull=pfull,method='nearest')
                f0 = f0.compute().to_numpy()
                f1 = f1.compute().to_numpy()
                n0 = len(f0)
                n1 = len(f1)
                F0 = np.vstack(tuple(np.roll(f0,lag) for lag in range(window))).T[window:]
                F1 = np.vstack(tuple(np.roll(f1,lag) for lag in range(window))).T[window:]
                D2mat = np.add.outer(np.sum(F0**2, axis=1), np.sum(F1**2, axis=1)) - 2*F0.dot(F1.T)
                return np.sqrt(np.mean(D2mat))



            dist_funs = dict({'tdep': dict(), 'rmsd': dict()})
            for field_name in ['temperature','total_rain','column_water_vapor','surface_pressure']:
                dist_funs['tdep'][field_name] = lambda ds0,ds1,field_name=field_name: dist_euclidean(ds0,ds1,field_name)
                dist_funs['rmsd'][field_name] = lambda ds0,ds1,field_name=field_name: rmsd_euclidean(ds0,ds1,field_name)
            pert_growth = alg.measure_pert_growth(dist_funs)
            pickle.dump(pert_growth, open(fndict['analysis']['pert_growth'], 'wb'))
        else:
            pert_growth = pickle.load(open(fndict['analysis']['pert_growth'], 'rb'))
        if tododict['analyze_pebr']['analyze_pert_growth'] and (not exists(fndict['analysis']['lyap_exp'])):
            lyapunov_exponents = alg.analyze_pert_growth(pert_growth)
            pickle.dump(lyapunov_exponents, open(fndict['analysis']['lyap_exp'], 'wb'))
    if utils.find_true_in_dict(tododict['plot_pebr']):
        lat = 45.0
        lon = 180.0
        pfull = 500.0
        alg = pickle.load(open(join(dirdict['alg'],'alg.pickle'),'rb'))
        if tododict['plot_pebr']['fields']:
            # Plot a panel of ensemble members each day 
            obsprop = alg.ens.dynsys.observable_props()
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
        if tododict['plot_pebr']['observables']:
            obsprop = alg.ens.dynsys.observable_props()
            obs_funs = dict()
            for obs_name in ['temperature']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(alg.ens.dynsys, obs_name)(dsmem).sel(lat=lat,lon=lon,pfull=pfull,method='nearest')
            for obs_name in ['r_sppt_g','total_rain','column_water_vapor','surface_pressure']:
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
            for branch_group in range(min(3,alg.num_branch_groups)): #range(alg.branching_state['next_branch_group']):
                alg.plot_obs_spaghetti(obs_funs, branch_group, dirdict['plots'], ylabels=obs_units, titles=obs_labels, abbrvs=obs_abbrvs)
        if tododict['plot_pebr']['pert_growth']:
            pert_growth_dict = pickle.load(open(fndict['analysis']['pert_growth'],'rb'))
            lyap_dict = pickle.load(open(fndict['analysis']['lyap_exp'],'rb'))
            alg.plot_pert_growth(pert_growth_dict, lyap_dict, fndict['plots'], logscale=True)
    return

def meta_analyze_periodic_branching():
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

if __name__ == "__main__":
    procedure = 'meta'
    print(f'Got into Main')
    if procedure == 'run':
        nproc = 4 
        recompile = 0 
        i_param = int(sys.argv[1])
        run_periodic_branching(nproc,recompile,i_param)
    elif procedure == 'meta':
        meta_analyze_periodic_branching()
