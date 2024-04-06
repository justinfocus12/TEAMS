import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
import os
from os.path import join, exists
from os import makedirs
import sys
import copy as copylib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from importlib import reload

sys.path.append('../..')
import lorenz96; reload(lorenz96)
import ensemble; reload(ensemble)
import forcing; reload(forcing)
import algorithms; reload(algorithms)
import algorithms_lorenz96; reload(algorithms_lorenz96)
import utils; reload(utils)

def ange_paramset(i_expt):
    # Random seed
    seed_incs = [0]
    # Physical
    F4s = [0.25,0.5,1.0,3.0]
    
    i_seed_inc,i_F4 = np.unravel_index(i_expt, (len(seed_incs),len(F4s),))

    config_sde = lorenz96.Lorenz96SDE.default_config()
    config_sde['frc']['white']['wavenumber_magnitudes'][0] = F4s[i_F4]
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_seed_inc], 
        'burnin_time_phys': 15, # should be about 100; start small for testing 
        'time_horizon_phys': 20,
        # mutable parameters below 
        'num_buicks': 512,
        'branches_per_buick': 1, 
        })
    expt_label = r'$F_4=%g$, seed %d'%(F4s[i_F4],seed_incs[i_seed_inc])
    expt_abbrv = (r'F%g_seed%d'%(F4s[i_F4],seed_incs[i_seed_inc])).replace('.','p')
    return config_sde,config_algo,expt_label,expt_abbrv

def ange_single_workflow(i_expt):
    print(f'Starting to set up workflow')
    config_sde,config_algo,expt_label,expt_abbrv = ange_paramset(i_expt)
    param_abbrv_sde,param_label_sde = lorenz96.Lorenz96SDE.label_from_config(config_sde)
    param_abbrv_algo,param_label_algo = algorithms_lorenz96.Lorenz96AncestorGenerator.label_from_config(config_algo)
    obsprops = lorenz96.Lorenz96ODE.observable_props()
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
    date_str = "2024-04-04"
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
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')

    print(f'Finished setting up workflow')
    return config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def run_ange(dirdict,filedict,config_sde,config_algo):
    root_dir = dirdict['data']
    print(f'{root_dir = }')
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
        alg.set_capacity(config_algo['num_buicks'], config_algo['branches_per_buick'])
    else:
        sde = lorenz96.Lorenz96SDE(config_sde)
        ens = ensemble.Ensemble(sde,root_dir=root_dir)
        alg = algorithms_lorenz96.Lorenz96AncestorGenerator.default_init(config_algo, ens)
    alg.ens.set_root_dir(root_dir)
    print(f'{alg = }')
    while not alg.terminate:
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict(filename=f'mem{mem}.npz')
        alg.take_next_step(saveinfo)
        if exists(filedict['alg']):
            os.rename(filedict['alg'], filedict['alg_backup'])
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def plot_observable_spaghetti(config_analysis, alg, dirdict):
    for (obs_name,obs_props) in config_analysis['observables'].items():
        obs_fun = obs_props['fun']
        outfile = join(dirdict['plots'], r'spaghetti_burnin_%s.png'%(obs_props['abbrv']))
        alg.plot_observable_spaghetti_burnin(obs_fun, outfile, title=obs_props['label'], ylabel=r'')
        for family in range(alg.num_buicks):
            outfile = join(dirdict['plots'], r'spaghetti_branching_%s_fam%d.png'%(obs_props['abbrv'],family))
            alg.plot_observable_spaghetti_branching(obs_fun, family, outfile, title=obs_props['label'], ylabel='')
    return

def plot_observable_distribution(config_analysis, alg, dirdict):
    # Show the probability distribution of some (scalar) observable, like a score, of the path functional, over all paths 
    for (obs_name,obs_props) in config_analysis['observables'].items():
        print(f'{obs_props = }')
        score_fun = obs_props['fun']
        for buick in range(alg.branching_state['num_buicks_generated']):
            print(f'Starting {buick = }')
            outfile = join(dirdict['plots'], r'score_distn_%s_buick%d.png'%(obs_props['abbrv'],buick))
            alg.plot_score_distribution_branching(score_fun, buick, outfile, label=obs_props['label'])
    return

def measure_running_max(config_analysis, alg, dirdict):
    for (obs_name,obs_props) in config_analysis['observables'].items():
        print(f'Beginning running max analysis of {obs_name}')
        runmax_file = join(dirdict['analysis'], r'running_max_%s.npz'%(obs_props['abbrv']))
        figfile_prefix = join(dirdict['plots'], r'running_max_%s'%(obs_props['abbrv']))
        alg.measure_running_max(obs_props['fun'], runmax_file, figfile_prefix, label=obs_props['label'], abbrv=obs_props['abbrv'], precomputed=False)
    return

def ange_single_procedure(i_expt):
    print(f'Got into ange_single_procedure')
    tododict = dict({
        'run':             1,
        'analysis': dict({
            'observable_spaghetti':     1,
            'observable_distribution':  1,
            'observable_running_max':   1,
            }),
        })
    config_gcm,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = ange_single_workflow(i_expt)
    if tododict['run']:
        print(f'About to run')
        run_ange(dirdict,filedict,config_gcm,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, alg, dirdict)
    if tododict['analysis']['observable_distribution']:
        plot_observable_distribution(config_analysis, alg, dirdict)
    if tododict['analysis']['observable_running_max']:
        measure_running_max(config_analysis, alg, dirdict)
    return

if __name__ == "__main__":
    print(f'Got into Main')
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_expt = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'single'
        idx_expt = [3] #list(range(1,21))
    print(f'{procedure = }, {idx_expt = }')
    if procedure == 'single':
        for i_expt in idx_expt:
            ange_single_procedure(i_expt)
    elif procedure == 'meta':
        ange_meta_procedure(idx_expt)




    







