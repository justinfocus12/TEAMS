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
import algorithms; reload(algorithms)
print(f'{i = }'); i += 1
import lorenz96; reload(lorenz96)
print(f'{i = }'); i += 1
import algorithms_lorenz96; reload(algorithms_lorenz96)
print(f'{i = }'); i += 1

def iteams_multiparams():
    # Random seed
    seed_incs = [0,1,2,3,4,5,6]
    # Buick choices
    buicks = list(range(10))
    # Physical
    F4s = [0.25] #,0.5,1.0,3.0]
    # Algorithmic
    deltas_phys = [0.0,1.0,2.0]
    split_landmarks = ['gmx','lmx','thx']
    return seed_incs,buicks,F4s,deltas_phys,split_landmarks

def iteams_paramset(i_expt):
    seed_incs,buicks,F4s,deltas_phys,split_landmarks = iteams_multiparams()
    
    i_seed_inc,i_buick,i_F4,i_delta,i_slm = np.unravel_index(i_expt, (len(seed_incs),len(buicks),len(F4s),len(deltas_phys),len(split_landmarks)))

    config_sde = lorenz96.Lorenz96SDE.default_config()
    config_sde['frc']['white']['wavenumber_magnitudes'][0] = F4s[i_F4]
    config_algo = dict({
        'autonomy': True,
        'num_levels_max': 20,
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_seed_inc], 
        'buick': i_buick,
        'population_size': 8,
        'time_horizon_phys': 8,
        'buffer_time_phys': 0,
        'advance_split_time_phys': deltas_phys[i_delta],
        'split_landmark': split_landmarks[i_slm],
        'num2drop': 1,
        'score': dict({
            'ks': [0],
            'kweights': [1],
            'tavg_phys': 0.0,
            }),
        })
    expt_label = r'$F_4=%g$, seed %d, buick %d'%(F4s[i_F4],seed_incs[i_seed_inc],buicks[i_buick])
    expt_abbrv = (r'F%g_seed%d_buick%d'%(F4s[i_F4],seed_incs[i_seed_inc],buicks[i_buick])).replace('.','p')
    return config_sde,config_algo,expt_label,expt_abbrv

def iteams_single_workflow(i_expt):
    config_sde,config_algo,expt_label,expt_abbrv = iteams_paramset(i_expt)
    param_abbrv_sde,param_label_sde = lorenz96.Lorenz96SDE.label_from_config(config_sde)
    param_abbrv_algo,param_label_algo = algorithms_lorenz96.Lorenz96SDEITEAMS.label_from_config(config_algo)
    config_analysis = dict()
    config_analysis['observables'] = dict({
        'x0': dict({
            'fun': lambda t,x: x[:,0],
            'abbrv': 'x0',
            'label': r'$x(k=0)$',
            }),
        'E0': dict({
            'fun': lambda t,x: x[:,0]**2/2,
            'abbrv': 'E0',
            'label': r'$\frac{1}{2}x_0^2$',
            }),
        'E': dict({
            'fun': lambda t,x: np.sum(x**2, axis=1)/2,
            'abbrv': 'E',
            'label': r'$\frac{1}{2}\sum_kx_k^2$',
            }),
        })
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-04-02"
    sub_date_str = "0"
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_sde, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    print(f'Before makedirs')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    print(f'After makedirs')
    filedict = dict()
    filedict['angel'] = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/2024-04-02/0/',
            param_abbrv_sde, 'AnGe_si0_Tbrn15_Thrz20', 'data',
            'alg.pickle') 
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')

    print(f'Finished setting up workflow')
    return config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def plot_observable_spaghetti(config_analysis, alg, dirdict):
    tu = alg.ens.dynsys.dt_save
    for (obs_name,obs_props) in config_analysis['observables'].items():
        is_score = (obs_name == 'E0')
        obs_fun = lambda t,x: obs_props['fun'](t,x)
        outfile = join(dirdict['plots'], r'spaghetti_%s.png'%(obs_props['abbrv']))
        landmark_label = {'lmx': 'Local max', 'gmx': 'Global max', 'thx': 'Threshold crossing'}[alg.split_landmark]
        title = r'%s ($\delta=%g$ before %s)'%(obs_props['label'],alg.advance_split_time*tu,landmark_label)
        alg.plot_observable_spaghetti(obs_fun, outfile, title=title, is_score=is_score)
    return

def plot_score_distribution(config_analysis, alg, dirdict, filedict):
    # TODO overlay the angel distribution on top 
    # Three histograms: initial population, weighted, and unweighted
    scmax,sclev,logw,mult,tbr,tmx = (alg.branching_state[s] for s in 'scores_max score_levels log_weights multiplicities branch_times scores_max_timing'.split(' '))
    hist_init,bin_edges_init = np.histogram(scmax[:alg.population_size], bins=10, density=True)
    hist_unif,bin_edges_unif = np.histogram(scmax, bins=10, density=True)
    hist_wted,bin_edges_wted = np.histogram(scmax, bins=10, weights=mult*np.exp(logw), density=True)
    fig,ax = plt.subplots()
    cbinfunc = lambda bin_edges: (bin_edges[1:] + bin_edges[:-1])/2
    hinit, = ax.plot(cbinfunc(bin_edges_init), hist_init, marker='.', color='black', linestyle='--', linewidth=3, label=r'Init (%g)'%(alg.population_size))
    hunif, = ax.plot(cbinfunc(bin_edges_unif), hist_unif, marker='.', color='dodgerblue', label=r'Fin. unweighted (%g)'%(alg.ens.get_nmem()))
    hwted, = ax.plot(cbinfunc(bin_edges_wted), hist_wted, marker='.', color='red', label=r'Fin. weighted (%g)'%(np.sum(mult)))
    ax.set_yscale('log')
    ax.legend(handles=[hinit,hunif,hwted])
    ax.set_title('Score distribution')
    ax.set_ylabel(r'$S(X)$')
    fig.savefig(join(dirdict['plots'],'score_hist.png'), **pltkwargs)
    plt.close(fig)
    return

def run_iteams(dirdict,filedict,config_sde,config_algo):
    root_dir = dirdict['data']
    angel = pickle.load(open(filedict['angel'], 'rb'))
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
    else:
        sde = lorenz96.Lorenz96SDE(config_sde)
        ens = ensemble.Ensemble(sde, root_dir=root_dir)
        alg = algorithms_lorenz96.Lorenz96SDEITEAMS.initialize_from_ancestorgenerator(angel, config_algo, ens)

    alg.ens.set_root_dir(root_dir)
    while not alg.terminate:
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict(filename=f'mem{mem}.npz')
        alg.take_next_step(saveinfo)
        if exists(filedict['alg']):
            os.rename(filedict['alg'], filedict['alg_backup'])
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def iteams_single_procedure(i_expt):

    tododict = dict({
        'run':             1,
        'analysis': dict({
            'observable_spaghetti':     1,
            'score_distribution':       1,
            }),
        })
    config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = iteams_single_workflow(i_expt)
    if tododict['run']:
        run_iteams(dirdict,filedict,config_sde,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, alg, dirdict)
        # TODO have another ancestor-wise version, and another that shows family lines improving in parallel and dropping out
    if tododict['analysis']['score_distribution']:
        plot_score_distribution(config_analysis, alg, dirdict, filedict)
    return

if __name__ == "__main__":
    print(f'Got into Main')
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_expt = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'single'
        seed_incs,buicks,F4s,deltas_phys,split_landmarks = iteams_multiparams()
        iseed_ibuick_iF4_idelta_islm = [(0,i_buick,3,4,i_slm) for i_buick in range(1,2) for i_slm in range(3)]
        shp = (len(seed_incs),len(buicks),len(F4s),len(deltas_phys),len(split_landmarks))
        idx_expt = []
        for i_multiparam in iseed_ibuick_iF4_idelta_islm:
            i_expt = np.ravel_multi_index(i_multiparam,shp)
            idx_expt.append(i_expt) #list(range(1,21))
    if procedure == 'single':
        for i_expt in idx_expt:
            iteams_single_procedure(i_expt)
    elif procedure == 'meta':
        iteams_meta_procedure(idx_expt)



