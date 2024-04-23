i = 0
print(f'--------------Beginning imports-------------')
import numpy as np
print(f'{i = }'); i += 1
from numpy.random import default_rng
print(f'{i = }'); i += 1
from scipy.special import logsumexp,softmax
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
    seed_incs = list(range(64)) 
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
        'time_horizon_phys': 8, #6 + delta_phys,
        'buffer_time_phys': 0,
        'advance_split_time_phys': delta_phys,
        'advance_split_time_max_phys': 2.0,
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
    sub_date_str = "2"
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_sde, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    filedict = dict()
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')
    filedict['dns'] = join(scratch_dir,'2024-04-12/0',param_abbrv_sde,'DNS_si0','data','alg.pickle')

    return config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def plot_observable_spaghetti(config_analysis, config_algo, alg, dirdict, filedict):
    tu = alg.ens.dynsys.dt_save
    desc_per_anc = np.array([len(list(nx.descendants(alg.ens.memgraph,ancestor))) for ancestor in range(alg.population_size)])
    anc_scores = alg.branching_state['scores_max'][:alg.population_size]
    #order = np.argsort(desc_per_anc)[::-1]
    order = np.argsort(anc_scores)[::-1]
    ancs2plot = order[np.linspace(0,len(order)-1,6).astype(int)]
    print(f'{desc_per_anc[order] = }')
    for (obs_name,obs_props) in config_analysis['observables'].items():
        is_score = (obs_name == 'E0')
        obs_fun = lambda t,x: obs_props['fun'](t,x)
        for ancestor in ancs2plot:
            outfile = join(dirdict['plots'], r'spaghetti_%s_anc%d.png'%(obs_props['abbrv'],ancestor))
            title = r'%s ($\delta=%g$)'%(obs_props['label'],alg.advance_split_time*tu)
            fig,axes = alg.plot_observable_spaghetti(obs_fun, ancestor, title=title, is_score=is_score, outfile=None)
            fig.savefig(outfile, **pltkwargs)
            plt.close(fig)
    return


def measure_plot_score_distribution(config_algo, algs, dirdict, filedict, param_suffix, alpha=0.1, overwrite_dns=False):
    print(f'Measuring score distribution')
    # TODO overlay the angel distribution on top 
    # Three histograms: initial population, weighted, and unweighted

    # ---------------------- Calculate DNS max scores ------------------------
    scmax_dns_file = join(dirdict['analysis'], 'scmax_dns.npz')
    if (not exists(scmax_dns_file)) or overwrite_dns:
        print(f'About to compute DNS scores')
        dns = pickle.load(open(filedict['dns'], 'rb'))
        sccomp_dns = []
        # TODO multi-thread this computation
        nmem_dns = dns.ens.get_nmem()
        for mem in range(nmem_dns):
            if mem % 100 == 0: print(f'Scoring member {mem} out of {nmem_dns}')
            sccomp_dns.append(
                    dns.ens.compute_observables([algs[0].score_components],mem)[0])
        ncomp = len(sccomp_dns[0])
        sccomp_dns = [
                np.concatenate([
                    sccomp_dns[mem][i] for mem in range(dns.ens.get_nmem())
                    ])
                for i in range(ncomp)
                ]
        score_dns = algs[0].score_combined(sccomp_dns)[algs[0].ens.dynsys.t_burnin:]
        print(f'{score_dns.shape = }')
        scmax_dns = utils.compute_block_maxima(score_dns, algs[0].time_horizon-max(algs[0].advance_split_time_max, (algs[0].score_params['tavg']-1)))
        print(f'{np.min(scmax_dns) = }, {np.max(scmax_dns) = }, {scmax_dns.shape = }')
        np.savez(scmax_dns_file, scmax_dns=scmax_dns)
    else:
        scmax_dns = np.load(scmax_dns_file)['scmax_dns']
    N_dns = len(scmax_dns)

    # ---------------- Calculate TEAMS statistics -------------------
    returnstats_file = join(dirdict['analysis'],r'returnstats_%s.npz'%(param_suffix))
    figfile = join(dirdict['plots'],r'returnstats_%s.png'%(param_suffix))
    delta_phys = config_algo['advance_split_time_phys']
    F4 = algs[0].ens.dynsys.config['frc']['white']['wavenumber_magnitudes'][0]
    param_display = '\n'.join([
        r'$\delta=%g$'%(delta_phys),
        r'$F_4=%g$'%(F4),
        ])
    algorithms_lorenz96.Lorenz96SDETEAMS.measure_plot_score_distribution(config_algo, algs, scmax_dns, returnstats_file, figfile, param_display=param_display)
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
        'run':             0,
        'analysis': dict({
            'observable_spaghetti':     0,
            'hovmoller':                0,
            }),
        })
    config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = teams_single_workflow(i_expt)
    if tododict['run']:
        run_teams(dirdict,filedict,config_sde,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, config_algo, alg, dirdict, filedict)
        # TODO have another ancestor-wise version, and another that shows family lines improving in parallel and dropping out
    return

def meta_workflow_multiseed(i_F4,i_delta,idx_seed):
    return


def teams_multiseed_procedure(i_F4,i_delta,idx_seed,overwrite_dns=False): # Just different seeds for now
    tododict = dict({
        'score_distribution': 1,
        })
    # Figure out which flat indices corresond to this set of seeds
    multiparams = teams_multiparams()
    idx_multiparam = [(i_seed,i_F4,i_delta) for i_seed in idx_seed]
    idx_expt = []
    for i_multiparam in idx_multiparam:
        i_expt = np.ravel_multi_index(i_multiparam,tuple(len(mp) for mp in multiparams))
        idx_expt.append(i_expt) #list(range(1,21))
    workflows = tuple(teams_single_workflow(i_expt) for i_expt in idx_expt)
    configs_sde,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts = tuple(
            tuple(workflows[i][j] for i in range(len(workflows)))
            for j in range(len(workflows[0])))
    config_sde = configs_sde[0]
    config_algo = configs_algo[0]
    config_analysis = configs_analysis[0]
    param_abbrv_sde,param_label_sde = lorenz96.Lorenz96SDE.label_from_config(config_sde)
    param_abbrv_algo,param_label_algo = algorithms_lorenz96.Lorenz96SDETEAMS.label_from_config(config_algo)
    # Set up a meta-dirdict 
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-04-12"
    sub_date_str = "2"
    dirdict = dict()
    dirdict['meta'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_sde, 'meta')
    dirdict['data'] = join(dirdict['meta'], 'data')
    dirdict['analysis'] = join(dirdict['meta'], 'analysis')
    dirdict['plots'] = join(dirdict['meta'], 'plots')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    filedict = dict({'dns': filedicts[0]['dns']})
    
    # Load all the algs
    algs = []
    for i_alg in range(len(workflows)):
        algs.append(pickle.load(open(filedicts[i_alg]['alg'],'rb')))
    # Do meta-analysis
    if tododict['score_distribution']:
        print(f'About to measure score distribution')
        param_suffix = (r'F%g_ast%g'%(multiparams[1][i_F4],multiparams[2][i_delta])).replace('.','p')
        measure_plot_score_distribution(config_algo, algs, dirdict, filedict, param_suffix, overwrite_dns=overwrite_dns)
    return

def teams_multidelta_procedure(i_F4,idx_delta,idx_seed):
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-04-12"
    sub_date_str = "2"
    multiparams = teams_multiparams()
    seed_incs,F4s,deltas = multiparams
    kldiv_pooled = np.zeros(len(idx_delta))
    x2div_pooled = np.zeros(len(idx_delta))
    kldiv_sep = np.zeros((len(idx_seed),len(idx_delta)))
    x2div_sep = np.zeros((len(idx_seed),len(idx_delta)))
    boost_family_mean = np.zeros((len(idx_seed),len(idx_delta)))
    for i_delta in idx_delta:
        param_suffix = (r'F%g_ast%g'%(F4s[i_F4],deltas[i_delta])).replace('.','p')
        idx_multiparam = [(i_seed,i_F4,i_delta) for i_seed in idx_seed]
        idx_expt = []
        for i_multiparam in idx_multiparam:
            i_expt = np.ravel_multi_index(i_multiparam,tuple(len(mp) for mp in multiparams))
            idx_expt.append(i_expt) #list(range(1,21))
        workflows = tuple(teams_single_workflow(i_expt) for i_expt in idx_expt)
        configs_sde,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts = tuple(
            tuple(workflows[i][j] for i in range(len(workflows)))
            for j in range(len(workflows[0])))
        config_sde = configs_sde[0]
        config_algo = configs_algo[0]
        param_abbrv_sde,param_label_sde = lorenz96.Lorenz96SDE.label_from_config(config_sde)
        returnstats_file = join(scratch_dir,date_str,sub_date_str,param_abbrv_sde,'meta','analysis','returnstats_%s.npz'%(param_suffix))
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
    multiparams = teams_multiparams()
    shp = tuple(len(mp) for mp in multiparams)
    seed_incs,F4s,deltas_phys = multiparams
    nseeds,nFs,ndeltas = shp
    # The "procedure" argument determines how following arguments are interpreted
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
    else:
        # This little section is for ad-hoc testing
        procedure = 'meta'
        idx_multiparam = [
                (i_seed,i_F4,i_delta) 
                for i_seed in range(0,4)
                for i_F4 in range(0,1) 
                for i_delta in range(0,1) 
                ]
        idx_expt = []
        for i_multiparam in idx_multiparam:
            i_expt = np.ravel_multi_index(i_multiparam,shp)
            idx_expt.append(i_expt) #list(range(1,21))
    if procedure == 'single':
        i_F4,i_delta = np.unravel_index(int(sys.argv[2]), (nFs,ndeltas))
        idx_expt = [
                np.ravel_multi_index((i_seed,i_F4,i_delta), (len(seed_incs),len(F4s),len(deltas_phys)))
                for i_seed in range(len(seed_incs))
                ]
        for i_expt in idx_expt:
            teams_single_procedure(i_expt)
    elif procedure == 'multiseed':
        idx_seed = list(range(nseeds))
        i_F4,i_delta = np.unravel_index(int(sys.argv[2]),(nFs,ndeltas))
        teams_multiseed_procedure(i_F4,i_delta,idx_seed,overwrite_dns=False)
    elif procedure == 'multidelta':
        i_F4 = int(sys.argv[2])
        idx_delta = list(range(len(deltas_phys)))
        idx_seed = list(range(len(seed_incs)))
        teams_multidelta_procedure(i_F4,idx_delta,idx_seed)
    elif procedure == 'multiF':
        # analyze all experiments together
        pass



