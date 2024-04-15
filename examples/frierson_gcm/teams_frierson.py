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

def teams_multiparams():
    seed_incs = list(range(8))
    sigmas = [0.3]
    deltas_phys = [0.0,6.0,8.0,10.0]
    split_landmarks = ['thx']
    return seed_incs,sigmas,deltas_phys,split_landmarks

def teams_paramset(i_expt):
    seed_incs,sigmas,deltas_phys,split_landmarks = teams_multiparams()
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
        'num_members_max': 256,
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
    obs_names = list(observables.keys())
    # Set up directories
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-04-04"
    sub_date_str = "1"
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    filedict = dict()
    # Initial conditions
    filedict['angel'] = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-04-04/0/',
            param_abbrv_gcm, 'AnGe_si0_Tbrn50_Thrz30', 'data',
            'alg.pickle') 
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')

    return config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def plot_observable_spaghetti(config_analysis, alg, dirdict, filedict):
    tu = alg.ens.dynsys.dt_save
    desc_per_anc = np.array([len(list(nx.descendants(alg.ens.memgraph,ancestor))) for ancestor in range(alg.population_size)])
    order = np.argsort(desc_per_anc)[::-1]
    print(f'{desc_per_anc[order] = }')
    angel = pickle.load(open(filedict['angel'],'rb'))
    for (obs_name,obs_props) in config_analysis['observables'].items():
        is_score = (obs_name == 'local_dayavg_rain')
        if is_score:
            obs_fun = lambda ds: obs_props['fun'](ds, **obs_props['kwargs'])
            for ancestor in order[:min(alg.population_size,12)]:
                outfile = join(dirdict['plots'], r'spaghetti_%s_anc%d.png'%(obs_props['abbrv'],ancestor))
                landmark_label = {'lmx': 'local max', 'gmx': 'global max', 'thx': 'threshold crossing'}[alg.split_landmark]
                title = r'%s ($\delta=%g$ before %s)'%(obs_props['label'],alg.advance_split_time*tu,landmark_label)
                fig,axes = alg.plot_observable_spaghetti(obs_fun, ancestor, outfile=None, title=title, is_score=is_score)
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


def measure_score_distribution(config_analysis, config_algo, alg, dirdict, filedict, expt_label, overwrite_flag=False):
    # Three histograms: initial population, weighted, and unweighted
    scmax,sclev,logw,mult,tbr,tmx = (alg.branching_state[s] for s in 'scores_max score_levels log_weights multiplicities branch_times scores_max_timing'.split(' '))
    hist_init,bin_edges_init = np.histogram(scmax[:alg.population_size], bins=15, density=True)
    hist_unif,bin_edges_unif = np.histogram(scmax, bins=15, density=True)
    hist_wted,bin_edges_wted = np.histogram(scmax, bins=15, weights=mult*np.exp(logw), density=True)
    # Measure corresponding Buick distribution
    scmax_buick_file = join(dirdict['analysis'],'scmax_buick.npz')
    if (not exists(scmax_buick_file)) or overwrite_flag:
        print(f'{filedict["angel"] = }')
        angel = pickle.load(open(filedict['angel'], 'rb'))
        print(f'{angel.branching_state = }')
        mems_buick = []
        for i in range(angel.branching_state['num_buicks_generated']-1):
            if angel.branching_state['num_branches_generated'][i] > 0:
                mems_buick.append(next(angel.ens.memgraph.successors(angel.branching_state['generation_0'][i])))
        score_fun = lambda ds: alg.score_combined(alg.score_components(ds['time'].to_numpy(),ds))
        lonroll = lambda ds,dlon: ds.roll(lon=int(round(dlon/ds['lon'][:2].diff('lon').item())))
        score_funs_rolled = [lambda ds: score_fun(lonroll(ds,dlon)) for dlon in [0,30,60,90,120,150,180,210,240,270,300,330]]

        scbuick = np.concatenate(tuple(
            angel.ens.compute_observables(score_funs_rolled, mem) 
            for mem in mems_buick), axis=0) # TODO augment with zonal symmetry
        print(f'{scbuick = }')
        print(f'{scbuick.shape = }')
        print(f'{len(mems_buick) = }')
        scmax_buick = np.nanmax(scbuick[:,:alg.time_horizon], axis=1)
        np.savez(scmax_buick_file, scmax_buick=scmax_buick)
    else:
        scmax_buick = np.load(scmax_buick_file)['scmax_buick']
    hist_buick,bin_edges_buick = np.histogram(scmax_buick, bins=15, density=True)
    cbinfunc = lambda bin_edges: (bin_edges[1:] + bin_edges[:-1])/2

    fig,axes = plt.subplots(nrows=2,figsize=(6,8))
    ax = axes[0]
    hinit, = ax.plot(cbinfunc(bin_edges_init), hist_init, marker='.', color='black', linestyle='--', linewidth=3, label=r'Init (%g)'%(alg.population_size))
    hunif, = ax.plot(cbinfunc(bin_edges_unif), hist_unif, marker='.', color='dodgerblue', label=r'Fin. unweighted (%g)'%(alg.ens.get_nmem()))
    hwted, = ax.plot(cbinfunc(bin_edges_wted), hist_wted, marker='.', color='red', label=r'Fin. weighted (%g)'%(np.sum(mult)))
    hbuick, = ax.plot(cbinfunc(bin_edges_buick), hist_buick, marker='.', color='gray', label=r'Buick (%g)'%(len(scmax_buick)))
    ax.set_yscale('log')
    ax.set_title(expt_label)
    ax.set_ylabel(r'Freq.')
    ax = axes[1]
    pmf2ccdf = lambda hist,bin_edges: np.cumsum((hist*np.diff(bin_edges))[::-1])[::-1]
    hinit, = ax.plot(bin_edges_init[:-1], pmf2ccdf(hist_init,bin_edges_init), marker='.', color='black', linestyle='--', linewidth=3, label=r'Init (%g)'%(alg.population_size))
    hunif, = ax.plot(bin_edges_unif[:-1], pmf2ccdf(hist_unif,bin_edges_unif), marker='.', color='dodgerblue', label=r'Fin. unweighted (%g)'%(alg.ens.get_nmem()))
    hwted, = ax.plot(bin_edges_wted[:-1], pmf2ccdf(hist_wted,bin_edges_wted), marker='.', color='red', label=r'Fin. weighted (%g)'%(np.sum(mult)))
    hbuick, = ax.plot(bin_edges_buick[:-1], pmf2ccdf(hist_buick,bin_edges_buick), marker='.', color='gray', label=r'Buick (%g)'%(len(scmax_buick)))
    ax.set_yscale('log')
    ax.set_ylabel(r'Exc. Prob.')
    ax.set_xlabel(r'$S(X)$')
    ax.legend(handles=[hinit,hunif,hwted,hbuick],bbox_to_anchor=(0,-0.2),loc='upper left')
    fig.savefig(join(dirdict['plots'],'score_hist.png'), **pltkwargs)
    print(f'{dirdict["plots"] = }')
    plt.close(fig)
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
        alg = algorithms_frierson.FriersonGCMTEAMS.initialize_from_ancestorgenerator(angel, config_algo, ens)

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

def teams_single_procedure(i_expt):

    tododict = dict({
        'run':             1,
        'analysis': dict({
            'observable_spaghetti':     1,
            'score_distribution':       1,
            'scorrelation':             0,
            'fields_2d':                1,
            }),
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = teams_single_workflow(i_expt)
    if tododict['run']:
        run_teams(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, alg, dirdict, filedict)
        # TODO have another ancestor-wise version, and another that shows family lines improving in parallel and dropping out
    if tododict['analysis']['score_distribution']:
        measure_score_distribution(config_analysis, config_algo, alg, dirdict, filedict, expt_label, overwrite_flag=True)
    if tododict['analysis']['scorrelation']:
        plot_scorrelations(config_analysis, alg, dirdict, filedict, expt_label)
    return

if __name__ == "__main__":
    print(f'Got into Main')
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_expt = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'single'
        seed_incs,sigmas,deltas_phys,split_landmarks = teams_multiparams()
        iseed_isigma_idelta_islm = [
                (i_seed,i_sigma,i_delta,0)
                for i_seed in range(8)
                for i_sigma in range(1)
                for i_delta in range(3)
                ]
        shp = (len(seed_incs),len(sigmas),len(deltas_phys),len(split_landmarks))
        idx_expt = []
        for i_multiparam in iseed_isigma_idelta_islm:
            i_expt = np.ravel_multi_index(i_multiparam,shp)
            idx_expt.append(i_expt)
    print(f'Got into Main')
    if procedure == 'single':
        for i_expt in idx_expt:
            teams_single_procedure(i_expt)
    elif procedure == 'meta':
        teams_meta_procedure(idx_expt)



