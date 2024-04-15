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
    seed_incs = list(range(64)) #,1,2,3,4,5,6]
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


def measure_score_distribution(config_algo, algs, dirdict, filedict, figfile_suffix, alpha=0.5, overwrite_dns=False):
    print(f'Plotting score distribution')
    # TODO overlay the angel distribution on top 
    # Three histograms: initial population, weighted, and unweighted

    # ---------------------- Calculate DNS max scores ------------------------
    scmax_dns_file = join(dirdict['analysis'], 'scmax_dns.npz')
    if (not exists(scmax_dns_file)) or overwrite_dns:
        dns = pickle.load(open(filedict['dns'], 'rb'))
        sccomp_dns = []
        for mem in range(dns.ens.get_nmem()):
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
    #Iterate through alg objects first to collect scores and define bin edges
    sclim = [np.min(scmax_dns),np.max(scmax_dns)]
    scmaxs,logws,mults = ([] for i in range(3))
    Ns_init,Ns_fin = (np.zeros(len(algs),dtype=int) for i in range(2))
    for i_alg,alg in enumerate(algs):
        scmax,logw,mult = (alg.branching_state[s] for s in 'scores_max,log_weights,multiplicities'.split(','))
        scmaxs.append(scmax)
        logws.append(logw)
        mults.append(mult)
        Ns_init[i_alg] = alg.population_size
        Ns_fin[i_alg] = alg.ens.get_nmem()
        assert int(round(np.exp(logsumexp(logw,b=mult)))) == Ns_init[i_alg]
        sclim[0],sclim[1] = min(sclim[0],np.min(scmax)),max(sclim[1],np.max(scmax))
    bin_edges = np.linspace(sclim[0]-1e-10,sclim[1]+1e-10,16)
    hist_dns,_ = np.histogram(scmax_dns, bins=bin_edges, density=False)
    # Now put the scores from separate runs into thi scommon set of bins
    hists_init,hists_fin_unif,hists_fin_wted,ccdfs_init,ccdfs_fin_unif,ccdfs_fin_wted = (np.zeros((len(algs),len(bin_edges)-1)) for i in range(6))
    for i_alg,alg in enumerate(algs):
        hists_init[i_alg],_ = np.histogram(scmaxs[i_alg][:alg.population_size], bins=bin_edges, density=False)
        hists_fin_unif[i_alg],_ = np.histogram(scmaxs[i_alg], bins=bin_edges, density=False)
        hists_fin_wted[i_alg],_ = np.histogram(scmaxs[i_alg], bins=bin_edges, weights=mults[i_alg]*np.exp(logws[i_alg]), density=False)
        ccdfs_init[i_alg] = utils.pmf2ccdf(hists_init[i_alg],bin_edges)
        ccdfs_fin_wted[i_alg] = utils.pmf2ccdf(hists_fin_wted[i_alg],bin_edges)
        ccdfs_fin_unif[i_alg] = utils.pmf2ccdf(hists_fin_unif[i_alg],bin_edges)
    hist_init = np.sum(hists_init, axis=0)
    hist_fin_unif = np.sum(hists_fin_unif, axis=0)
    hist_fin_wted = np.sum(hists_fin_wted, axis=0)
    ccdf_init,ccdf_init_lower,ccdf_init_upper = utils.pmf2ccdf(hist_init,bin_edges,return_errbars=True,alpha=alpha)
    ccdf_fin_wted = utils.pmf2ccdf(hist_fin_wted,bin_edges)
    ccdf_fin_wted_lower = np.nanquantile(ccdfs_fin_wted, alpha/2, axis=0)
    ccdf_fin_wted_upper = np.nanquantile(ccdfs_fin_wted, 1-alpha/2, axis=0)
    ccdf_fin_unif = utils.pmf2ccdf(hist_fin_unif,bin_edges)
    # TODO put error bars on TEAMS by bootstrapping
    rng_boot = default_rng(45839)
    n_boot = 1000
    idx_alg_boot = rng_boot.choice(np.arange(len(algs)), replace=True, size=(n_boot,len(algs)))
    ccdf_fin_wted_boot = np.nan*np.ones((n_boot,len(bin_edges)-1))
    for i_boot in range(n_boot):
        hist_fin_wted_boot = np.sum(hists_fin_wted[idx_alg_boot[i_boot,:]],axis=0)
        ccdf_fin_wted_boot[i_boot,:] = utils.pmf2ccdf(hist_fin_wted_boot,bin_edges)
    ccdf_fin_wted_pooled_lower = np.nanquantile(ccdf_fin_wted_boot,alpha/2,axis=0)
    ccdf_fin_wted_pooled_upper = np.nanquantile(ccdf_fin_wted_boot,1-alpha/2,axis=0)
        

    # --------------------- Tally costs ------------------------
    N_teams_init = np.sum(Ns_init)
    N_teams_fin = np.sum(Ns_fin)
    cost_teams_init = N_teams_init * (config_algo['time_horizon_phys'] - config_algo['advance_split_time_max_phys'] + config_algo['advance_split_time_phys'])
    cost_teams_fin = N_teams_fin/N_teams_init * cost_teams_init
    cost_dns = N_dns * (config_algo['time_horizon_phys'] - config_algo['advance_split_time_max_phys'])
    # Get DNS stats, comparing either to a single TEAMS run or the aggregate in cost 
    ccdf_dns,ccdf_dns_sep_lower,ccdf_dns_sep_upper = utils.pmf2ccdf(hist_dns,bin_edges,return_errbars=True,alpha=alpha,N_errbars=int(N_dns * cost_teams_fin/cost_dns * 1/len(algs)))
    _,ccdf_dns_pooled_lower,ccdf_dns_pooled_upper = utils.pmf2ccdf(hist_dns,bin_edges,return_errbars=True,alpha=alpha,N_errbars=int(N_dns * cost_teams_fin/cost_dns))

    # Collect in a dictionary and store 
    returnstats = dict({
        'bin_edges': bin_edges,
        # Separate TEAMS runs
        'hists_init': hists_init,
        'hists_fin_wted': hists_fin_wted,
        'hists_fin_unif': hists_fin_unif,
        'ccdfs_init': ccdfs_init,
        'ccdfs_fin_wted': ccdf_fin_wted,
        'ccdfs_fin_unif': ccdf_fin_unif,
        # Pooled TEAMS runs
        'hist_init': hist_init,
        'hist_fin_wted': hist_fin_wted,
        'hist_fin_unif': hist_fin_unif,
        'ccdf_init': ccdf_init,
        'ccdf_init_lower': ccdf_init_lower,
        'ccdf_init_upper': ccdf_init_upper,
        'ccdf_fin_wted': ccdf_fin_wted,
        'ccdf_fin_wted_lower': ccdf_fin_wted_lower,
        'ccdf_fin_wted_upper': ccdf_fin_wted_upper,
        'ccdf_fin_unif': ccdf_fin_unif,
        'ccdf_fin_wted_pooled_lower': ccdf_fin_wted_pooled_lower,
        'ccdf_fin_wted_pooled_upper': ccdf_fin_wted_pooled_upper,
        # DNS
        'hist_dns': hist_dns,
        'ccdf_dns': ccdf_dns,
        'ccdf_dns_sep_lower': ccdf_dns_sep_lower,
        'ccdf_dns_sep_upper': ccdf_dns_sep_upper,
        'ccdf_dns_pooled_lower': ccdf_dns_pooled_lower,
        'ccdf_dns_pooled_upper': ccdf_dns_pooled_upper,
        # Scalars
        'cost_teams_init': cost_teams_init,
        'cost_teams_fin': cost_teams_fin,
        'cost_dns': cost_dns,
        'time_horizon_effective': config_algo['time_horizon_phys'] - config_algo['advance_split_time_max_phys'],
        })
    # TODO compute skill

    np.savez(join(dirdict['analysis'],'returnstats.npz'), **returnstats)

    # ---------------------------- Plot ----------------------------
    # 3 columns: (0) all separate TEAMS results, (1) pooled TEAMS results, (2) GEV estimates
    teams_abbrv = 'TEAMS' if alg.advance_split_time>0 else 'AMS'
    fig,axes = plt.subplots(ncols=3, figsize=(18,4), sharex=False, sharey=True)

    # ++++ left-hand text label +++
    delta_phys = config_algo['advance_split_time_phys']
    F4 = algs[0].ens.dynsys.config['frc']['white']['wavenumber_magnitudes'][0]
    param_label = '\n'.join([
        r'$\delta=%g$'%(delta_phys),
        r'$F_4=%g$'%(F4),
        r' ',
        r'%s cost:'%(teams_abbrv),
        r'%.1E'%(cost_teams_fin/len(algs)),
        r'$\times$ %d runs'%(len(algs)),
        r'$=$%.1E'%(cost_teams_fin),
        r' ',
        r'DNS cost:',
        r'%.1E'%(cost_dns)
        ])
    axes[0].text(-0.3,0.5,param_label,fontsize=15,transform=axes[0].transAxes,horizontalalignment='right',verticalalignment='center')

    # ++++ Column 0: individual curves on the left ++++
    ax = axes[0]
    # DNS, with equal-cost errorbars to compare to single DNS runs
    sf2rt = lambda sf: utils.convert_sf_to_rtime(sf, returnstats['time_horizon_effective'])
    hdns, = ax.plot(sf2rt(ccdf_dns), bin_edges[:-1], marker='.', color='black', label=r'DNS (cost %.1E)'%(cost_dns))
    ax.fill_betweenx(bin_edges[:-1], sf2rt(ccdf_dns_sep_lower), sf2rt(ccdf_dns_sep_upper), fc='gray', ec='none', zorder=-1, alpha=0.5)
    for i_alg,alg in enumerate(algs):
        # Initialization
        hinit_sep, = ax.plot(sf2rt(ccdfs_init[i_alg]),bin_edges[:-1],color='dodgerblue',linestyle='-',linewidth=1,alpha=0.5,label=r'Init')
        # Final (weighted)
        hfin_wted_sep, = ax.plot(sf2rt(ccdfs_fin_wted[i_alg]),bin_edges[:-1],color='red',linestyle='-',linewidth=1,alpha=0.5,label=teams_abbrv)
    ax.fill_betweenx(bin_edges[:-1],sf2rt(ccdf_fin_wted_lower),sf2rt(ccdf_fin_wted_upper),fc='red',ec='none',zorder=-1,alpha=0.5)
    ax.set_ylabel(r'$\frac{1}{2}x_0^2$ Return level')
    ax.set_title(r'Single %s runs'%(teams_abbrv))

    # ++++ Column 1: pooled curves ++++
    ax = axes[1]
    # DNS again, this time accounting for total cost 
    hdns, = ax.plot(sf2rt(ccdf_dns), bin_edges[:-1], color='black', label=r'DNS')
    ax.fill_betweenx(bin_edges[:-1], sf2rt(ccdf_dns_pooled_lower), sf2rt(ccdf_dns_pooled_upper), fc='gray', ec='none', zorder=-1, alpha=0.5)
    # Initialization
    hinit, = ax.plot(sf2rt(ccdf_init), bin_edges[:-1], marker='.', color='dodgerblue', label=r'Init.')
    ax.fill_betweenx(bin_edges[:-1],sf2rt(ccdf_init_lower),sf2rt(ccdf_init_upper),fc='dodgerblue',ec='none',zorder=-1,alpha=0.5)
    # Final TEAMS (weighted)
    hfin_wted, = ax.plot(sf2rt(ccdf_fin_wted), bin_edges[:-1], marker='.', color='red', label=teams_abbrv)
    ax.fill_betweenx(bin_edges[:-1],sf2rt(ccdf_fin_wted_pooled_lower),sf2rt(ccdf_fin_wted_pooled_upper),fc='red',ec='none',zorder=-1,alpha=0.5)
    ax.legend(handles=[hinit,hfin_wted,hdns],bbox_to_anchor=(1,0),loc='lower right')
    ax.set_ylabel('')
    ax.yaxis.set_tick_params(which='both',labelbottom=True)
    ax.set_title('Pooled results')

    xlim = [returnstats['time_horizon_effective'],5*sf2rt(min(np.nanmin(ccdf_dns),np.nanmin(ccdf_fin_wted)))]
    ylim = [bin_edges[np.argmax(sf2rt(ccdf_dns) > xlim[0])],bin_edges[-1]]
    for ax in axes[:2]:
        ax.set_xscale('log')
        print(f'{xlim = }')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r'Return time')

    # ++++ Column 2: Histograms ++++
    ax = axes[2]
    ax.plot(hist_dns, bin_edges[:-1], color='black')
    ax.plot(hist_init, bin_edges[:-1], color='dodgerblue')
    ax.plot(hist_fin_unif, bin_edges[:-1], color='red')
    ax.yaxis.set_tick_params(which='both',labelbottom=True)
    ax.set_xscale('log')
    ax.set_ylim(ylim)
    ax.set_xlabel('Counts')
    ax.set_title('Score histograms')

    fig.savefig(join(dirdict['plots'],r'score_hist_%s.png'%(figfile_suffix)), **pltkwargs)
    plt.close(fig)
    print(f'SAvefile: ')
    print(join(dirdict['plots'],r'score_hist_%s.png'%(figfile_suffix)))
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
        measure_score_distribution(config_algo, [alg], dirdict, filedict, overwrite_dns=False, alpha=0.9)
    return

def teams_meta_procedure_1param_multiseed(i_F4,i_delta,idx_seed): # Just different seeds for now
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
    sub_date_str = "0"
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
        figfile_suffix = (r'meta_F%g_ast%g'%(multiparams[1][i_F4],multiparams[2][i_delta])).replace('.','p')
        measure_score_distribution(config_algo, algs, dirdict, filedict, figfile_suffix, overwrite_dns=False)

    return

if __name__ == "__main__":
    print(f'Got into Main')
    if len(sys.argv) > 1:
        procedure = sys.argv[1]
    else:
        # This little section is for ad-hoc testing
        procedure = 'meta'
        multiparams = teams_multiparams()
        idx_multiparam = [
                (i_seed,i_F4,i_delta) 
                for i_seed in range(0,4)
                for i_F4 in range(0,1) 
                for i_delta in range(0,1) 
                ]
        idx_expt = []
        for i_multiparam in idx_multiparam:
            i_expt = np.ravel_multi_index(i_multiparam,tuple(len(mp) for mp in multiparams))
            idx_expt.append(i_expt) #list(range(1,21))
    if procedure == 'single':
        idx_expt = [int(arg) for arg in sys.argv[2:]]
        for i_expt in idx_expt:
            teams_single_procedure(i_expt)
    elif procedure == 'meta':
        idx_seed = list(range(32))
        i_F4 = 0
        for i_delta in range(11):
            teams_meta_procedure_1param_multiseed(i_F4,i_delta,idx_seed)



