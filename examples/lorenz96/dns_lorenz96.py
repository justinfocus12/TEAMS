import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import sys
import psutil
import glob
import copy as copylib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
sys.path.append('../..')
from lorenz96 import Lorenz96ODE,Lorenz96SDE
from ensemble import Ensemble
import forcing
from algorithms_lorenz96 import Lorenz96ODEDirectNumericalSimulation as L96ODEDNS, Lorenz96SDEDirectNumericalSimulation as L96SDEDNS
import utils

def dns_multiparams():
    F4s = [0.0,0.25,0.5,1.0,3.0]
    seed_incs = [0]
    return F4s,seed_incs

def dns_paramset(i_expt):
    # Organize the array of parameters as well as the output files 
    multiparams = dns_multiparams()
    idx_multiparam = np.unravel_index(i_expt, tuple(len(mp) for mp in multiparams))
    F4,seed_inc = (multiparams[i][i_param] for (i,i_param) in enumerate(idx_multiparam))
    # Minimal labels to differentiate them 
    expt_label = r'$F_4=%g$'%(F4)
    expt_abbrv = (r'F%g'%(F4)).replace('.','p') 
    config_dynsys = Lorenz96SDE.default_config()
    config_dynsys['frc']['white']['wavenumber_magnitudes'][0] = F4


    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_inc,
        'max_member_duration_phys': 1000.0,
        'num_chunks_max': 500, #2500,
        })


    return config_dynsys,config_algo,expt_label,expt_abbrv

def dns_single_workflow(i_expt,expt_supdir):
    config_dynsys,config_algo,expt_label,expt_abbrv = dns_paramset(i_expt)
    # Organize output directories
    param_abbrv_dynsys,param_label_dynsys = Lorenz96SDE.label_from_config(config_dynsys)
    param_abbrv_algo,param_label_algo = L96SDEDNS.label_from_config(config_algo)
    config_analysis = dict({
        'k_roll_step': 4, # step size for augmenting Lorenz96 with rotational symmetry 
        'spinup_phys': 50,
        'dns_duration_phys': 5.12e7, # TODO increase 
        'time_block_size_phys': 6,
        'observables_rotsym': dict({
            'xk': dict({
                'fun': lambda t,x: x,
                'abbrv': 'xk',
                'label': r'$x_k$',
                }),
            'Ek': dict({
                'fun': lambda t,x: x**2/2,
                'label': r'$\frac{1}{2}x_k^2$',
                'abbrv': 'Ek',
                }),
            }),
        # return statistics analysis
        # Other possible parameters: method used for fitting GEV, threshold for GPD, ...
        })

    dirdict = dict()
    dirdict['expt'] = join(expt_supdir, param_abbrv_dynsys, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    for dirname in list(dirdict.values()):
        makedirs(dirname, exist_ok=True)

    # List the quantities of interest
    obsprop = Lorenz96ODE.observable_props()
    obs_names = ['x0','x0sq','E0','E'] # Maybe we can do multivariate stuff on the second

    filedict = dict()
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    # Basic analysis
    return config_dynsys,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def dns_meta_workflow(idx_param,expt_supdir):
    num_expt = len(idx_param)
    workflow_tuple = tuple(dns_single_workflow(i_param,expt_supdir) for i_param in idx_param)
    workflows = dict()
    for i_key,key in enumerate(('configs_dynsys,configs_algo,configs_analysis,expt_labels,expt_abbrvs,dirdicts,filedicts').split(',')):
        workflows[key] = tuple(workflow_tuple[j][i_key] for j in range(len(workflow_tuple)))
    print(f'{workflows.keys() = }')
    meta_dirdict = dict()
    meta_dirdict['meta'] = join(expt_supdir,'meta')
    for subdir in ['data','analysis','plots']:
        meta_dirdict[subdir] = join(meta_dirdict['meta'],subdir)
        makedirs(meta_dirdict[subdir],exist_ok=True)
    config_meta_analysis = dict()
    for key in ['observables_rotsym','spinup_phys','time_block_size_phys']:
        config_meta_analysis[key] = workflows['configs_analysis'][0][key]
    return workflows,config_meta_analysis,meta_dirdict

def dns_meta_procedure(idx_expt,expt_supdir):
    tododict = dict({
        'compare_extreme_stats':           1,
        })
    workflows,config_meta_analysis,meta_dirdict = dns_meta_workflow(idx_expt,expt_supdir)
    if tododict['compare_extreme_stats']:
        compare_extreme_stats(workflows,config_meta_analysis,meta_dirdict)
    return

def run_dns(dirdict,filedict,config_dynsys,config_algo):
    root_dir = dirdict['data']
    obs_fun = lambda t,x: None
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'],'rb'))
        alg.ens.set_root_dir(root_dir)
        alg.set_capacity(config_algo['num_chunks_max'], config_algo['max_member_duration_phys'])
    else:
        sde = Lorenz96SDE(config_dynsys)
        ens = Ensemble(sde,root_dir=root_dir)
        alg = L96SDEDNS(config_algo, ens)
    nmem = alg.ens.get_nmem()
    print(f'{nmem = }')
    while not (alg.terminate):
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict(filename=f'mem{mem}.npz')
        alg.take_next_step(saveinfo)
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def measure_plot_extreme_stats(config_analysis, alg, dirdict, overwrite_extstats=False, plot_gev=True):
    nmem = alg.ens.get_nmem()
    tu = alg.ens.dynsys.dt_save
    spinup = int(config_analysis['spinup_phys']/tu)
    duration = int(config_analysis['dns_duration_phys']/tu)
    time_block_size = int(config_analysis['time_block_size_phys']/tu)
    # TODO need to distribute the block maxima method
    for obs_name,obs_props in config_analysis['observables_rotsym'].items():
        print(f'About to compute extreme stats for {obs_name}')
        returnstats_file = join(dirdict['analysis'],r'extstats_rotsym_%s.npz'%(obs_props['abbrv']))
        if overwrite_extstats or (not exists(returnstats_file)):
            alg.compute_extreme_stats_rotsym(obs_props['fun'], spinup, duration, time_block_size, returnstats_file)

        extstats = np.load(returnstats_file)
        bin_lows,hist,rlev,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale = (extstats[v] for v in 'bin_lows,hist,rlev,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale'.split(','))
        bins_bm,hist_bm,ccdf_bm,ccdf_bm_lower_cpi,ccdf_bm_upper_cpi,ccdf_bm_lower_bsi,ccdf_bm_upper_bsi = (extstats[v] for v in 'bins_bm,hist_bm,ccdf_bm,ccdf_bm_lower_cpi,ccdf_bm_upper_cpi,ccdf_bm_lower_bsi,ccdf_bm_upper_bsi'.split(','))
        print(f'{ccdf_bm = }')
        print(f'{ccdf_bm_lower_cpi = }')
        print(f'{ccdf_bm_upper_cpi = }')
        print(f'{ccdf_bm_lower_bsi = }')
        print(f'{ccdf_bm_upper_bsi = }')
        bin_width = bin_lows[1] - bin_lows[0]
        # Plot 
        bin_mids = bin_lows + 0.5*(bin_lows[1]-bin_lows[0])
        fig,axes = plt.subplots(ncols=2,figsize=(12,4),gridspec_kw={'wspace': 0.25})
        ax = axes[0]
        ax.plot(bin_mids,hist/(np.sum(hist)*bin_width),color='black',marker='.')
        ax.set_xlabel(obs_props['label'])
        ax.set_ylabel('Prob. density')
        ax.set_yscale('log')
        ax = axes[1]
        handles = []
        hemp, = ax.plot(rtime*tu,rlev,color='black',marker='.',label='Empirical')
        handles.append(hemp)
        if plot_gev:
            hgev, = ax.plot(rtime_gev*tu,rlev,color='cyan',marker='.',label='GEV fit')
            handles.append(hgev)

        # ----- Now plot the CIs etc ------
        sf2rt = lambda sf: utils.convert_sf_to_rtime(sf, time_block_size)
        for c in [ccdf_bm_lower_cpi,ccdf_bm_upper_cpi]:
            hcpi, = ax.plot(sf2rt(c)*tu, bins_bm[:-1], linestyle='--', color='black', label='Clopper-Pearson')
        for c in [ccdf_bm_lower_bsi,ccdf_bm_upper_bsi]:
            hbsi, = ax.plot(sf2rt(c)*tu, bins_bm[:-1], linestyle='--', color='red', label='Bootstrap')
        handles += [hcpi,hbsi]


        # ---------------------------------
        ax.legend(handles=handles)
        print(f'{rtime_gev = }')
        ax.set_ylim([rlev[np.argmax(rtime>0)],2*rlev[-1]-rlev[-2]])
        ax.set_xlabel('Return time')
        ax.set_ylabel('Return level %s'%(obs_props['label']))
        ax.set_xscale('log')
        fig.savefig(join(dirdict['plots'],r'extstats_%s.png'%(obs_props['abbrv'])),**pltkwargs)
        plt.close(fig)
    return

def compare_extreme_stats(workflows,config_meta_analysis, dirdict):
    # Pot the two curves on top of each other 
    fig,axes = plt.subplots(ncols=3, figsize=(24,4), gridspec_kw={'wspace': 0.25})
    # Left: PDFs of x 
    num_expts = len(workflows['dirdicts'])
    print(f'{num_expts = }')
    print(f'{workflows["expt_labels"] = }')
    handles = []
    time_block_size_phys = config_meta_analysis['time_block_size_phys']
    tu = workflows['configs_dynsys'][0]['ode']['dt_save']
    time_block_size = int(round(time_block_size_phys/tu))
    sf2rt = lambda sf: utils.convert_sf_to_rtime(sf, time_block_size)
    logsf2rt = lambda logsf: utils.convert_logsf_to_rtime(logsf, time_block_size)
    for i_expt in range(num_expts):
        extstats_x0 = np.load(join(workflows['dirdicts'][i_expt]['analysis'],r'extstats_rotsym_xk.npz'))
        extstats_E0 = np.load(join(workflows['dirdicts'][i_expt]['analysis'],r'extstats_rotsym_Ek.npz'))
        color = plt.cm.coolwarm(i_expt/num_expts)
        ax = axes[0]
        bin_mids = extstats_x0['bin_lows']
        bin_mids += (bin_mids[1]-bin_mids[0])/2
        h, = ax.plot(bin_mids, extstats_x0['hist']/np.sum(extstats_x0['hist'])/(bin_mids[1]-bin_mids[0]), color=color, label=workflows['expt_labels'][i_expt])
        handles.append(h)

        # Return levels for energy
        ax = axes[1]
        ax.plot(sf2rt(extstats_E0['ccdf_bm'])*tu, extstats_E0['bins_bm'][:-1], color=color, linewidth=2.5)
        ax.fill_betweenx(extstats_E0['bins_bm'][:-1], sf2rt(extstats_E0['ccdf_bm_lower_cpi'])*tu, sf2rt(extstats_E0['ccdf_bm_upper_cpi'])*tu, fc=color, ec='none', zorder=-1, alpha=0.3)
        ax.plot(sf2rt(extstats_E0['ccdf_bm_lower_bsi'])*tu, extstats_E0['bins_bm'][:-1], color=color, linestyle='dotted')
        ax.plot(sf2rt(extstats_E0['ccdf_bm_upper_bsi'])*tu, extstats_E0['bins_bm'][:-1], color=color, linestyle='dotted')
        # GEV fits
        #ax.plot(extstats_E0['rtime_gev']*tu, extstats_E0['rlev'], color=color, linestyle='--')
        # Invert for return period as function of return level
        ax = axes[2]
        logccdf_grid,rlev_inverted,rlev_inverted_lower,rlev_inverted_upper = utils.ccdf2rlev_of_rtime(extstats_E0['bins_bm'][:-1], extstats_E0['ccdf_bm'], extstats_E0['ccdf_bm_lower_cpi'], extstats_E0['ccdf_bm_upper_cpi'],) 
        print(f'{logccdf_grid = }')
        print(f'{logsf2rt(logccdf_grid) = }')
        print(f'{rlev_inverted = }')
        ax.plot(logsf2rt(logccdf_grid)*tu,rlev_inverted,color=color,linewidth=3)
        ax.fill_between(logsf2rt(logccdf_grid)*tu,rlev_inverted_lower,rlev_inverted_upper,fc=color,ec='none',zorder=-1,alpha=0.3)
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'$x_0$')
    axes[0].set_ylabel(r'Prob. dens.')
    axes[0].legend(handles=handles, bbox_to_anchor=(-0.25,0.5), loc='center right')
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'Return time $\pm0.95$ CI')
    axes[1].set_ylabel(r'$\frac{1}{2}x_0^2$ Return level')
    axes[2].set_xscale('log')
    axes[2].set_xlabel(r'Return time')
    axes[2].set_ylabel(r'$\frac{1}{2}x_0^2$ Return level $\pm0.95$ CI')
    fig.savefig(join(dirdict['plots'],'returnstats_x0_E0.png'),**pltkwargs)
    plt.close(fig)
    return



def dns_single_procedure(i_expt, expt_supdir):
    tododict = dict({
        'run':                   1,
        'plot_segment':          1,
        'return_stats':          1,
        })

    # Quantities of interest for statistics. These should be registered as observables under the system.
    print(f'Workflow setup...',end='')
    config_dynsys,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = dns_single_workflow(i_expt,expt_supdir)
    print('done')

    if tododict['run']:
        print(f'About to run DNS')
        run_dns(dirdict,filedict,config_dynsys,config_algo)
    alg = pickle.load(open(filedict['alg'],'rb'))
    if tododict['plot_segment']:
        outfile = join(dirdict['plots'],'dns_segment.png')
        alg.plot_dns_segment(outfile, tspan_phys=[500,515])
    if tododict['return_stats']:
        print(f'About to compute extreme stats')
        measure_plot_extreme_stats(config_analysis,alg,dirdict,overwrite_extstats=True)
    return

if __name__ == "__main__":
    procedure = sys.argv[1]
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-10-15"
    sub_date_str = "0"
    expt_supdir = join(scratch_dir,date_str,sub_date_str)
    idx_expt = [int(v) for v in sys.argv[2:]]
    if procedure == 'single':
        for i_expt in idx_expt:
            dns_single_procedure(i_expt,expt_supdir)
    elif procedure == 'meta':
        dns_meta_procedure(idx_expt,expt_supdir)
