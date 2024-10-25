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
    # Physical
    F4s = [0.25] # [0.0, 0.25, 0.5, 1.0, 3.0]
    # Algorithmic
    deltas_phys = [1.4] #list(np.linspace(0.0,2.0,11))
    population_params = [
            ('num',1,'const_pop'),
            ('frac',0.1,'const_pop'),
            ('frac',0.5,'one_birth'),
            ('frac_once_then_num',(0.5,1),'cull_once_then_const_pop')
            ]
    # Random seed
    seed_incs = list(range(64)) 
    return F4s,deltas_phys,population_params,seed_incs

def teams_paramset(i_expt=None):
    multiparams = teams_multiparams()
    if i_expt is None:
        idx_multiparam = (0,0,0)
    else:
        idx_multiparam = np.unravel_index(i_expt, tuple(len(mp) for mp in multiparams))
    F4,delta_phys,population_params,seed_inc = (multiparams[i][i_param] for (i,i_param) in enumerate(idx_multiparam))

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
        'population_size': 32,
        'time_horizon_phys': 8, #6 + delta_phys,
        'buffer_time_phys': 0,
        'advance_split_time_phys': delta_phys,
        'advance_split_time_max_phys': 2.0,
        'split_landmark': 'thx',
        'inherit_perts_after_split': False,
        'score': dict({
            'ks': [0],
            'kweights': [1],
            'tavg_phys': 0.0,
            }),
        })
    """
    Set level-raising and re-birthing schedules through three parameters: 'drop_sched' sets the type of level-raising protocol; 'drop_rate' sets how fast, and is interpreted differently depending on 'drop_sched'; and 'birth_sched' sets how to choose the number of re-births at every round.

    The drop_sched options are as follows:
    1. drop_sched='num' means kill a constant number of ensemble members each iteration, given by 'drop_rate' (an integer). 
    2. drop_sched='frac' means kill a constant fraction of the surviving population, given by 'drop_rate' (a fraction). 
    3. drop_sched='frac_then_num' means kill a fixed fraction at the first round, then a fixed number thereafter. Since we need two numbers to specify this, drop_rate has to be an ordered pair also (real, int). 
    4. Feel free to design new schedules and rates by adding more parameters. There's no fixed format, but you will have to add two corresponding pieces of code in ../../algorithms.py:

    The birth_sched option sare as follows:
    1. birth_sched='const_pop' means always replenish the population to its original level.
    2. birth_sched='one_birth' means always re-birth exactly once. 
    3. birth_sched='cull_once_then_const_pop' means only re-birth once at the first round, but replenish the (reduced-size) population thereafter. 



    """
    config_algo['drop_sched'],config_algo['drop_rate'],config_algo['birth_sched'] = population_params
    expt_label = r'$F_4=%g$, seed %d'%(F4,seed_inc)
    expt_abbrv = (r'F%g_seed%d'%(F4,seed_inc)).replace('.','p')
    return config_sde,config_algo,expt_label,expt_abbrv


def teams_single_workflow(i_expt,expt_supdir_teams,expt_supdir_dns):
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
    dirdict = dict()
    dirdict['expt'] = join(expt_supdir_teams, param_abbrv_sde, param_abbrv_algo, r'si%d'%(config_algo['seed_inc_init']))
    print(f'{dirdict = }')
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    filedict = dict()
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    filedict['alg_backup'] = join(dirdict['data'], 'alg_backup.pickle')
    filedict['dns'] = join(expt_supdir_dns,param_abbrv_sde,'DNS_si0','data','alg.pickle') 

    return config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def plot_observable_spaghetti(config_analysis, config_sde, config_algo, alg, dirdict, filedict, remove_old_plots=False):
    tu = alg.ens.dynsys.dt_save
    N = alg.population_size
    all_scores = np.array(alg.branching_state['scores_max'])
    B = alg.ens.construct_descent_matrix()[:N,:].toarray()
    desc_per_anc = np.sum(B,axis=1)
    print(f'{desc_per_anc = }')
    anc_scores = all_scores[:N]
    # Select some ancestors to plot based on two criteria: (1) largest ancestral scores, (2) largest child scores
    order_ancscores = np.argsort(anc_scores)[::-1]
    print(f'{order_ancscores[:3] = }')
    print(f'{anc_scores[order_ancscores[:3]] = }')
    descendants = [np.where(B[anc,:])[0] for anc in range(N)]
    desc_scores = np.where(B==0, -np.inf, B*all_scores)
    max_desc_scores = np.max(desc_scores, axis=1)
    # Get the best score of the ultimate descendant
    order_descscores = np.argsort(max_desc_scores)[::-1]
    print(f'{order_descscores[:3] = }')
    print(f'{max_desc_scores[order_descscores[:3]] = }')
    # Get the most-split ancestors
    order_famsize = np.argsort(desc_per_anc)[::-1]
    print(f'{order_famsize[:3] = }')
    print(f'{desc_per_anc[order_famsize[:3]] = }')
    maxrank = 1
    ancs2plot = np.concatenate(tuple(order[:maxrank] for order in (order_ancscores,order_descscores,order_famsize,)))
    rank_labels = []
    for ordername in ['ancscore','descscore','famsize']:
        for r in range(maxrank):
            rank_labels.append(f'{ordername}rank{r}')
    wavenum = config_sde['frc']['white']['wavenumbers'][0]
    F_wavenum = config_sde['frc']['white']['wavenumber_magnitudes'][0]
    if remove_old_plots:
        old_spaghetti_plots = glob.glob(join(dirdict['plots'],'spaghetti*.png'))
        old_hovmoller_plots = glob.glob(join(dirdict['plots'],'hovmoller*.png'))
        for fig in old_spaghetti_plots + old_hovmoller_plots:
            os.remove(fig)
    for (obs_name,obs_props) in config_analysis['observables'].items():
        is_score = (obs_name == 'E0')
        obs_fun = lambda t,x: obs_props['fun'](t,x)
        for i_ancestor,ancestor in enumerate(ancs2plot):
            special_descendant = np.argmax(desc_scores[ancestor,:])
            outfile = join(dirdict['plots'], r'spaghetti_%s_%s_anc%d.png'%(obs_props['abbrv'],rank_labels[i_ancestor],ancestor))
            fig,axes = alg.plot_observable_spaghetti(obs_fun, ancestor, special_descendant=special_descendant, obs_label=obs_props['label'], title='', is_score=is_score, outfile=None)
            display = '\n'.join([
                r'$F_{%d}=%g$'%(wavenum,F_wavenum),
                r'$\delta=%g$'%(config_algo['advance_split_time_phys']),
                r'',
                r'Run %d'%(config_algo['seed_inc_init']),
                r'Ancestor %d'%(ancestor),
                ])
            axes[0].text(-0.15,0.5,display,transform=axes[0].transAxes,ha='right',va='center')
            fig.savefig(outfile, **pltkwargs)
            plt.close(fig)
            print(f'{outfile = }')
            if obs_name == 'x0':
                outfile = join(dirdict['plots'], r'hovmoller_%s_%s_anc%d.png'%(obs_props['abbrv'],rank_labels[i_ancestor],ancestor))
                fig,axes = alg.plot_hovmoller_lineage(ancestor, special_descendant, outfile=None)
                axes[0,1].text(0.8,0.5,display,transform=axes[0,1].transAxes,ha='right',va='center')
                fig.savefig(outfile, **pltkwargs)
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
        tu = dns.ens.dynsys.dt_save
        sccomp_dns = []
        # TODO multi-thread this computation, and build in rotational symmetry
        nmem_dns = dns.ens.get_nmem()
        all_starts,all_ends = dns.ens.get_all_timespans()
        cost_teams = sum([alg.ens.get_nmem() * alg.time_horizon for alg in algs])
        spinup_dns = int(100.0/tu) 
        dns_firstmem = np.argmax(all_starts > spinup_dns)
        if all_ends[-1] >= spinup_dns + cost_teams:
            dns_lastmem = np.argmax(all_ends >= spinup_dns + cost_teams)
        else:
            dns_lastmem = len(all_ends)-1
        print(f'{dns_firstmem = }, {dns_lastmem = }')
        dns_mems = range(dns_firstmem,dns_lastmem+1)
        for (i_mem,mem) in enumerate(dns_mems):
            if (i_mem) % 100 == 0: print(f'Scoring member {i_mem} out of {dns_mems}')
            sccomp_dns.append(
                    dns.ens.compute_observables([algs[0].score_components],mem)[0])
        ncomp = len(sccomp_dns[0])
        sccomp_dns = [
                np.concatenate([
                    sccomp_dns[i_mem][i] for (i_mem,mem) in enumerate(dns_mems)
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
    figfileh = join(dirdict['plots'],r'returnstats_%s_horz.png'%(param_suffix))
    figfilev = join(dirdict['plots'],r'returnstats_%s_vert.png'%(param_suffix))
    delta_phys = config_algo['advance_split_time_phys']
    F4 = algs[0].ens.dynsys.config['frc']['white']['wavenumber_magnitudes'][0]
    param_display = '\n'.join([
        r'$\delta=%g$'%(delta_phys),
        r'$F_4=%g$'%(F4),
        ])
    algorithms_lorenz96.Lorenz96SDETEAMS.measure_plot_score_distribution(config_algo, algs, scmax_dns, returnstats_file, figfileh, figfilev, param_display=param_display)
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

def teams_single_procedure(i_expt,expt_supdir_teams,expt_supdir_dns):

    tododict = dict({
        'run':             1,
        'analysis': dict({
            'observable_spaghetti':     1,
            }),
        })
    config_sde,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = teams_single_workflow(i_expt,expt_supdir_teams,expt_supdir_dns)
    if tododict['run']:
        run_teams(dirdict,filedict,config_sde,config_algo)
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['analysis']['observable_spaghetti']:
        plot_observable_spaghetti(config_analysis, config_sde, config_algo, alg, dirdict, filedict, remove_old_plots=True)
        # TODO have another ancestor-wise version, and another that shows family lines improving in parallel and dropping out
    return



def teams_multiseed_procedure(i_F4,i_delta,i_droppar,idx_seed,expt_supdir_teams,expt_supdir_dns,overwrite_dns=False): # Just different seeds for now
    tododict = dict({
        'score_distribution': 1,
        })
    # Figure out which flat indices corresond to this set of seeds
    multiparams = teams_multiparams()
    idx_multiparam = [(i_F4,i_delta,i_droppar,i_seed) for i_seed in idx_seed]
    idx_expt = []
    for i_multiparam in idx_multiparam:
        i_expt = np.ravel_multi_index(i_multiparam,tuple(len(mp) for mp in multiparams))
        idx_expt.append(i_expt) #list(range(1,21))
    workflows = tuple(teams_single_workflow(i_expt,expt_supdir_teams,expt_supdir_dns,) for i_expt in idx_expt)
    configs_sde,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts = tuple(
            tuple(workflows[i][j] for i in range(len(workflows)))
            for j in range(len(workflows[0])))
    config_sde = configs_sde[0]
    config_algo = configs_algo[0]
    config_analysis = configs_analysis[0]
    param_abbrv_sde,param_label_sde = lorenz96.Lorenz96SDE.label_from_config(config_sde)
    param_abbrv_algo,param_label_algo = algorithms_lorenz96.Lorenz96SDETEAMS.label_from_config(config_algo)
    # Set up a meta-dirdict 
    dirdict = dict()
    dirdict['multiseed'] = join(expt_supdir_teams, param_abbrv_sde, param_abbrv_algo, 'multiseed')
    dirdict['data'] = join(dirdict['multiseed'], 'data')
    dirdict['analysis'] = join(dirdict['multiseed'], 'analysis')
    dirdict['plots'] = join(dirdict['multiseed'], 'plots')
    for dirname in ('data','analysis','plots'):
        makedirs(dirdict[dirname], exist_ok=True)
    filedict = dict({'dns': filedicts[0]['dns']})
    
    # Load all the algs
    algs = []
    for i_alg in range(len(workflows)):
        algs.append(pickle.load(open(filedicts[i_alg]['alg'],'rb')))
        print(f'{algs[i_alg].ens.get_nmem() = }')
    # Do multiseed-analysis
    if tododict['score_distribution']:
        print(f'About to measure score distribution')
        param_suffix = ('')
        measure_plot_score_distribution(config_algo, algs, dirdict, filedict, param_suffix, overwrite_dns=overwrite_dns)
    return

def teams_multidelta_procedure(i_F4,idx_delta,idx_seed,expt_supdir_teams):
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
        workflows = tuple(teams_single_workflow(i_expt,expt_supdir_teams) for i_expt in idx_expt)
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
            expt_supdir_teams,
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
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = '2024-10-15'
    sub_date_str = '0'
    expt_supdir_teams = join(scratch_dir,date_str,sub_date_str)
    expt_supdir_dns =   join(scratch_dir,date_str,sub_date_str) # but could be different in general
    # The "procedure" argument determines how following arguments are interpreted
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
    else:
        # This little section is for ad-hoc testing
        procedure = 'meta'
    if procedure == 'single':
        if len(sys.argv) > 2:
            idx_expt = [int(sys.argv[i]) for i in range(2,len(sys.argv))]
        else:
            idx_expt = [None]
        for i_expt in idx_expt:
            teams_single_procedure(i_expt,expt_supdir_teams,expt_supdir_dns)
    elif procedure == 'multiseed':
        multiparams = teams_multiparams()
        nFs,ndeltas,ndroppars,nseeds = (len(mp) for mp in multiparams)
        print(f'{nFs = }, {ndeltas = }, {nseeds = }')
        idx_seed = list(range(nseeds))
        i_F4,i_delta,i_droppar = np.unravel_index(int(sys.argv[2]),(nFs,ndeltas,ndroppars))
        teams_multiseed_procedure(i_F4,i_delta,i_droppar,idx_seed,expt_supdir_teams,expt_supdir_dns,overwrite_dns=True)
    elif procedure == 'multidelta':
        i_F4 = int(sys.argv[2])
        idx_delta = list(range(len(deltas_phys)))
        idx_seed = list(range(len(seed_incs)))
        teams_multidelta_procedure(i_F4,idx_delta,idx_seed)
    elif procedure == 'multiF':
        # analyze all experiments together
        pass



