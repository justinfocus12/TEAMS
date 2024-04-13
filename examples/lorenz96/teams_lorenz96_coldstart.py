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

def teams_multiparams():
    # Random seed
    seed_incs = list(range(16)) #,1,2,3,4,5,6]
    # Physical
    F4s = [0.25,0.5,1.0,3.0]
    # Algorithmic
    deltas_phys = list(np.linspace(0.0,2.0,11))
    return seed_incs,F4s,deltas_phys

def teams_paramset(i_expt):
    multiparams = teams_multiparams()
    idx_multiparam = np.unravel_index(i_expt, tuple(len(mp) for mp in multiparams))
    seed_inc,F4,delta_phys = (multiparams[i][i_param] for (i,i_param) in enumerate(idx_multiparam))

    config_sde = lorenz96.Lorenz96SDE.default_config()
    config_sde['frc']['white']['wavenumber_magnitudes'][0] = F4
    config_algo = dict({
        'num_levels_max': 1024,
        'num_members_max': 1024,
        'num_active_families_min': 2,
        #'buick_choices': [buicks[i_buick]],
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_inc, 
        'population_size': 128,
        'time_horizon_phys': 6 + delta_phys,
        'buffer_time_phys': 0,
        'advance_split_time_phys': delta_phys,
        'split_landmark': 'thx',
        'inherit_perts_after_split': False,
        'num2drop': 1,
        'score': dict({
            'ks': [0],
            'kweights': [1],
            'tavg_phys': 0.0,
            }),
        })
    expt_label = r'$F_4=%g$, seed %d'%(F4,seed_inc)
    expt_abbrv = (r'F%g_seed%d'%(F4,seed_inc)).replace('.','p')
    return config_sde,config_algo,expt_label,expt_abbrv

def teams_single_workflow(i_expt):
    config_sde,config_algo,expt_label,expt_abbrv = teams_paramset(i_expt)
    param_abbrv_sde,param_label_sde = lorenz96.Lorenz96SDE.label_from_config(config_sde)
    param_abbrv_algo,param_label_algo = algorithms_lorenz96.Lorenz96SDETEAMS.label_from_config(config_algo)
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
    date_str = "2024-04-12"
    sub_date_str = "DEBUG_RIGHT"
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
    filedict['dns'] = join(scratch_dir,'2024-04-12/0',param_abbrv_sde,'DNS_si0','data','alg.pickle')

    print(f'Finished setting up workflow')
    return config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def plot_observable_spaghetti(config_analysis, config_algo, alg, dirdict, filedict):
    tu = alg.ens.dynsys.dt_save
    desc_per_anc = np.array([len(list(nx.descendants(alg.ens.memgraph,ancestor))) for ancestor in range(alg.population_size)])
    order = np.argsort(desc_per_anc)[::-1]
    print(f'{desc_per_anc[order] = }')
    for (obs_name,obs_props) in config_analysis['observables'].items():
        is_score = (obs_name == 'E0')
        obs_fun = lambda t,x: obs_props['fun'](t,x)
        for ancestor in order[:4]:
            outfile = join(dirdict['plots'], r'spaghetti_%s_anc%d.png'%(obs_props['abbrv'],ancestor))
            title = r'%s ($\delta=%g$)'%(obs_props['label'],alg.advance_split_time*tu)
            fig,axes = alg.plot_observable_spaghetti(obs_fun, ancestor, title=title, is_score=is_score, outfile=None)
            fig.savefig(outfile, **pltkwargs)
            plt.close(fig)
    return

def measure_score_distribution(config_analysis, config_algo, alg, dirdict, filedict, expt_label, overwrite_flag=False):
    print(f'Plotting score distribution')
    # TODO overlay the angel distribution on top 
    # Three histograms: initial population, weighted, and unweighted
    scmax,sclev,logw,mult,tbr,tmx = (alg.branching_state[s] for s in 'scores_max score_levels log_weights multiplicities branch_times scores_max_timing'.split(' '))
    hist_init,bin_edges_init = np.histogram(scmax[:alg.population_size], bins=15, density=True)
    hist_unif,bin_edges_unif = np.histogram(scmax, bins=15, density=True)
    hist_wted,bin_edges_wted = np.histogram(scmax, bins=15, weights=mult*np.exp(logw), density=True)

    # Calculate DNS statistics
    scmax_dns_file = join(dirdict['analysis'], 'scmax_dns.npz')
    if (not exists(scmax_dns_file)) or overwrite_flag:
        dns = pickle.load(open(filedict['dns'], 'rb'))
        sccomp_dns = []
        for mem in range(dns.ens.get_nmem()):
            sccomp_dns.append(
                    dns.ens.compute_observables([alg.score_components],mem)[0])
        ncomp = len(sccomp_dns[0])
        sccomp_dns = [
                np.concatenate([
                    sccomp_dns[mem][i] for mem in range(dns.ens.get_nmem())
                    ])
                for i in range(ncomp)
                ]
        score_dns = alg.score_combined(sccomp_dns)[alg.ens.dynsys.t_burnin:]
        print(f'{score_dns.shape = }')
        scmax_dns = utils.compute_block_maxima(score_dns, alg.time_horizon-alg.advance_split_time-(alg.score_params['tavg']-1))
        print(f'{np.min(scmax_dns) = }, {np.max(scmax_dns) = }, {scmax_dns.shape = }')
        np.savez(scmax_dns_file, scmax_dns=scmax_dns)
    else:
        scmax_dns = np.load(scmax_dns_file)['scmax_dns']
    hist_dns,bin_edges_dns = np.histogram(scmax_dns, bins=15, density=True)

    cbinfunc = lambda bin_edges: (bin_edges[1:] + bin_edges[:-1])/2

    fig,axes = plt.subplots(nrows=2,figsize=(6,8))
    ax = axes[0]
    hinit, = ax.plot(cbinfunc(bin_edges_init), hist_init, marker='.', color='dodgerblue', linestyle='-', linewidth=1, label=r'Init (%d)'%(alg.population_size))
    hunif, = ax.plot(cbinfunc(bin_edges_unif), hist_unif, marker='.', color='red', linestyle='--', label=r'Fin. unweighted (%d)'%(alg.ens.get_nmem()))
    hwted, = ax.plot(cbinfunc(bin_edges_wted), hist_wted, marker='.', color='red', label=r'Fin. weighted (%d)'%(np.sum(mult)))
    hdns, = ax.plot(cbinfunc(bin_edges_dns), hist_dns, marker='.', color='black', label=r'DNS (%d)'%(len(scmax_dns)))
    #ax.set_yscale('log')
    ax.set_title('Score distribution')
    ax.set_xlabel(r'$S(X)$')
    ax.set_ylabel('Freq.')
    ax.set_yscale('log')
    ax = axes[1]
    pmf2ccdf = lambda hist,bin_edges: np.cumsum((hist*np.diff(bin_edges))[::-1])[::-1]
    hinit, = ax.plot(bin_edges_init[:-1], pmf2ccdf(hist_init,bin_edges_init), marker='.', color='dodgerblue', linestyle='-', linewidth=1, label=r'Init (%d)'%(alg.population_size))
    hunif, = ax.plot(bin_edges_unif[:-1], pmf2ccdf(hist_unif,bin_edges_unif), marker='.', color='red', linestyle='--', label=r'Fin. unweighted (%d)'%(alg.ens.get_nmem()))
    hwted, = ax.plot(bin_edges_wted[:-1], pmf2ccdf(hist_wted,bin_edges_wted), marker='.', color='red', label=r'Fin. weighted (%d)'%(np.sum(mult)))
    hdns, = ax.plot(bin_edges_dns[:-1], pmf2ccdf(hist_dns,bin_edges_dns), marker='.', color='black', label=r'DNS (%g)'%(len(scmax_dns)))
    ax.set_yscale('log')
    ax.set_ylabel(r'Exc. Prob.')
    ax.set_xlabel(r'$S(X)$')
    ax.legend(handles=[hinit,hunif,hwted,hdns],bbox_to_anchor=(0,-0.2),loc='upper left')
    fig.savefig(join(dirdict['plots'],'score_hist.png'), **pltkwargs)
    plt.close(fig)
    print(f'{dirdict["plots"] = }')
    return

def run_teams(dirdict,filedict,config_sde,config_algo):
    root_dir = dirdict['data']
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
        alg.set_capacity(config_algo['num_levels_max'], config_algo['num_members_max'])
    else:
        sde = lorenz96.Lorenz96SDE(config_sde)
        ens = ensemble.Ensemble(sde, root_dir=root_dir)
        alg = algorithms_lorenz96.Lorenz96SDETEAMS.initialize_from_coldstart(config_algo, ens)

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

def teams_single_procedure(i_expt):

    tododict = dict({
        'run':             1,
        'analysis': dict({
            'observable_spaghetti':     0,
            'score_distribution':       1,
            }),
        })
    config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = teams_single_workflow(i_expt)
    if tododict['run']:
        run_teams(dirdict,filedict,config_sde,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, config_algo, alg, dirdict, filedict)
        # TODO have another ancestor-wise version, and another that shows family lines improving in parallel and dropping out
    if tododict['analysis']['score_distribution']:
        measure_score_distribution(config_analysis, config_algo, alg, dirdict, filedict, expt_label, overwrite_flag=True)
    return

if __name__ == "__main__":
    print(f'Got into Main')
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
        idx_expt = [int(arg) for arg in sys.argv[2:]]
    else:
        procedure = 'single'
        multiparams = teams_multiparams()
        idx_multiparam = [
                (i_seed,i_F4,i_delta) 
                for i_seed in  range(0,1)
                for i_F4 in range(3,4) 
                for i_delta in range(1,2) 
                ]
        idx_expt = []
        for i_multiparam in idx_multiparam:
            i_expt = np.ravel_multi_index(i_multiparam,tuple(len(mp) for mp in multiparams))
            idx_expt.append(i_expt) #list(range(1,21))
    if procedure == 'single':
        for i_expt in idx_expt:
            teams_single_procedure(i_expt)
    elif procedure == 'meta':
        teams_meta_procedure(idx_expt)



