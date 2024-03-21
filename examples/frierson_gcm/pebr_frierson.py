
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

def pebr_paramset(i_param):
    config_gcm = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)

    # Parameters to loop over
    pert_types = ['IMP']        + ['SPPT']*20
    std_sppts = [0.5]           + [0.5,0.3,0.1,0.05,0.01]*4
    tau_sppts = [6.0*3600]      + [6.0*3600]*5   + [6.0*3600]*5    + [24.0*3600]*5     + [96.0*3600]*5 
    L_sppts = [500.0*1000]      + [500.0*1000]*5 + [2000.0*1000]*5 + [500.0*1000]*5    + [500.0*1000]*5
    seed_incs = [0]*21

    expt_labels = []
    expt_abbrvs = []
    for i_param in range(len(pert_types)):
        if pert_types[i_param] == 'IMP':
            label = 'Impulsive'
            abbrv = 'IMP'
        else:
            label = r'SPPT, $\sigma=%g$, $\tau=%g$ h, $L=%g$ km'%(std_sppts[i_param],tau_sppts[i_param]/3600,L_sppts[i_param]/1000)
            abbrv = r'SPPT_std%g_tau%gh_L%gkm'%(std_sppts[i_param],tau_sppts[i_param]/3600,L_sppts[i_param]/1000)
        expt_labels.append(label)
        expt_abbrvs.append(abbrv)

    config_gcm['pert_type'] = pert_types[i_param]
    if config_gcm['pert_type'] == 'SPPT':
        config_gcm['SPPT']['tau_sppt'] = tau_sppts[i_param]
        config_gcm['SPPT']['std_sppt'] = std_sppts[i_param]
        config_gcm['SPPT']['L_sppt'] = L_sppts[i_param]
    config_gcm['remove_temp'] = 1

    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_param], 
        'branches_per_group': 12, 
        'interbranch_interval_phys': 10.0,
        'branch_duration_phys': 30.0,
        'num_branch_groups': 20,
        'max_member_duration_phys': 30.0,
        })
    return config_gcm,config_algo,expt_labels[i_param],expt_abbrvs[i_param]

def pebr_workflow(i_param):
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-05"
    sub_date_str = "0/PeBr"
    print(f'About to generate default config')
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    param_abbrv_algo,param_label_algo = FriersonGCMPeriodicBranching.label_from_config(config_algo)
    # Configure post-analysis
    config_analysis = dict() 

    # Set up directories
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    dirdict['init_cond'] = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-03-05/0/DNS/',
            param_abbrv_gcm)

    for dirname in ['data','analysis','plots']:
        makedirs(dirname, exist_ok=True)

    filedict = dict()
    # Initial conditions
    filedict['init_cond'] = dict()
    filedict['init_cond']['restart'] = join(dirdict['init_cond'],'restart_mem20.cpio')
    filedict['init_cond']['trajectory'] = join(dirdict['init_cond'],'mem20.cpio')
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    # Quantitative analysis
    filedict['dispersion'] = dict()
    filedict['dispersion']['distance'] = dict()
    filedict['dispersion']['satfractime'] = dict()

    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def run_pebr(dirdict,filedict,config_gcm,config_algo):
    nproc = 4
    recompile = False
    root_dir = dirdict['data']
    init_time = int(xr.open_mfdataset(filedict['init_cond']['trajectory'])['time'].load()[-1].item())
    init_cond = relpath(filedict['init_cond']['restart'], root_dir)
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
    return

def analyze_pebr():
    # -------------------- Post-analysis -----------------------------
    dist_roi = dict({
        'temperature': dict(lat=slice(35,55),lon=slice(150,210),pfull=500),
        'total_rain': dict(lat=slice(35,55),lon=slice(150,210))
        })
    obs_roi = dict({
        'temperature': dict(lat=45,lon=180,pfull=500),
        'total_rain': dict(lat=45,lon=180),
        })
    alg = pickle.load(open(alg_filename, 'rb'))
    obsprop = alg.ens.dynsys.observable_props()
    if utils.find_true_in_dict(tododict['analyze_pebr']):
        if tododict['analyze_pebr']['measure_pert_growth']:
            for field_name in ['temperature','total_rain']:
                dist_fun = lambda ds0,ds1: alg.ens.dynsys.dist_euclidean_tdep(
                        *(getattr(alg.ens.dynsys,field_name)(ds) for ds in [ds0,ds1]),
                        dist_roi[field_name])
                split_times,dists = alg.measure_pert_growth(dist_fun)
                location_abbrv,location_label = alg.ens.dynsys.label_from_roi(dist_roi[field_name]) 
                filename = (r'dist_%s_%s'%(obsprop[field_name]['abbrv'],location_abbrv)).replace('.','p')
                np.save(join(dirdict['analysis'],f'{filename}.npy'), dists)
            np.save(join(dirdict['analysis'],'split_times.npy'), split_times)
        if tododict['analyze_pebr']['analyze_pert_growth']:
            split_times = np.load(join(dirdict['analysis'],'split_times.npy'))
            for field_name in ['temperature','total_rain']:
                location_abbrv,location_label = alg.ens.dynsys.label_from_roi(dist_roi[field_name]) 
                dist_filename = (r'dist_%s_%s'%(obsprop[field_name]['abbrv'],location_abbrv)).replace('.','p')
                dists = np.load(join(dirdict['analysis'],f'{dist_filename}.npy'))
                thalfsat,diff_expons,lyap_expons,rmses,rmsd = alg.summarize_pert_growth(dists)
                np.savez(
                        join(dirdict['analysis'], r'pert_growth_summary_%s_%s.npz'%(obsprop[field_name]['abbrv'],location_abbrv)),
                        thalfsat=thalfsat, diff_expons=diff_expons, lyap_expons=lyap_expons,rmses=rmses,rmsd=np.array([rmsd]))
    if utils.find_true_in_dict(tododict['plot_pebr']):
        alg = pickle.load(open(join(dirdict['alg'],'alg.pickle'),'rb'))
        # ----------------- Perturbation growth ---------------------------
        if tododict['plot_pebr']['pert_growth']:
            split_times = np.load(join(dirdict['analysis'],'split_times.npy'))
            for field_name in ['temperature','total_rain']:
                location_abbrv,location_label = alg.ens.dynsys.label_from_roi(dist_roi[field_name]) 
                pgs = np.load(join(dirdict['analysis'],r'pert_growth_summary_%s_%s.npz'%(obsprop[field_name]['abbrv'],location_abbrv)))
                dist_filename = (r'dist_%s_%s'%(obsprop[field_name]['abbrv'],location_abbrv)).replace('.','p')
                dists = np.load(join(dirdict['analysis'],f'{dist_filename}.npy'))
                plot_suffix = r'%s_%s'%(obsprop[field_name]['abbrv'],location_abbrv)
                alg.plot_pert_growth(split_times, dists, pgs['thalfsat'], pgs['diff_expons'], pgs['lyap_expons'], pgs['rmses'], pgs['rmsd'].item(), dirdict['plots'], plot_suffix, logscale=True)
        # ---------------- Observables -------------------
        if tododict['plot_pebr']['observables']:
            for obs_name in ['temperature','total_rain']:
                obs_fun = lambda dsmem: alg.ens.dynsys.sel_from_roi(getattr(alg.ens.dynsys, obs_name)(dsmem), obs_roi[obs_name])
                roi_abbrv,roi_label = alg.ens.dynsys.label_from_roi(obs_roi[obs_name])
                obs_abbrv = r'%s_%s'%(obsprop[obs_name]['abbrv'],roi_abbrv)
                obs_label = r'%s at %s'%(obsprop[obs_name]['label'],roi_label)
                obs_unit = r'[%s]'%(obsprop[obs_name]['unit_symbol'])
                for branch_group in range(min(3,alg.num_branch_groups)): #range(alg.branching_state['next_branch_group']):
                    alg.plot_obs_spaghetti(obs_fun, branch_group, dirdict['plots'], ylabel=obs_unit, title=obs_label, abbrv=obs_abbrv)
        if tododict['plot_pebr']['fields']:
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

def pebr_procedure(i_param):
    tododict = dict({
        'run_pebr':                1,
        'analyze_pebr': dict({
            'measure_pert_growth':           1,
            'analyze_pert_growth':           1,
            }),
        'plot_pebr': dict({
            'observables':    1,
            'fields':         0,
            'pert_growth':    1,
            'response':       0,
            }),
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = pebr_workflow(i_param)
    if tododict['run']:
        run_pebr(dirdict,filedict,config_gcm,config_algo)


if __name__ == "__main__":
    procedure = 'run'
    print(f'Got into Main')
    if procedure == 'run':
        nproc = 4 
        recompile = 0 
        i_param = int(sys.argv[1])
        run_periodic_branching(nproc,recompile,i_param)
    elif procedure == 'meta':
        meta_analyze_periodic_branching()

