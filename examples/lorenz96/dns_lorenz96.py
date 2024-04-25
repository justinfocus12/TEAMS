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
    seed_incs = [0]
    F4s = [0.0,0.25,0.5,1.0,3.0]
    return seed_incs,F4s

def dns_paramset(i_expt):
    # Organize the array of parameters as well as the output files 
    multiparams = dns_multiparams()
    idx_multiparam = np.unravel_index(i_expt, tuple(len(mp) for mp in multiparams))
    seed_inc,F4 = (multiparams[i][i_param] for (i,i_param) in enumerate(idx_multiparam))
    # Minimal labels to differentiate them 
    expt_label = r'$F_4=%g$, seed %d'%(F4,seed_inc)
    expt_abbrv = (r'F%g'%(F4)).replace('.','p') 
    config_dynsys = Lorenz96SDE.default_config()
    config_dynsys['frc']['white']['wavenumber_magnitudes'][0] = F4


    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_inc,
        'max_member_duration_phys': 1000.0,
        'num_chunks_max': 2500,
        })


    return config_dynsys,config_algo,expt_label,expt_abbrv

def dns_single_workflow(i_expt):
    config_dynsys,config_algo,expt_label,expt_abbrv = dns_paramset(i_expt)
    # Organize output directories
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-04-12"
    sub_date_str = "0"
    param_abbrv_dynsys,param_label_dynsys = Lorenz96SDE.label_from_config(config_dynsys)
    param_abbrv_algo,param_label_algo = L96SDEDNS.label_from_config(config_algo)
    config_analysis = dict({
        'time_block_size_phys': 6,
        'k_roll_step': 4, # step size for augmenting Lorenz96 with rotational symmetry 
        'spinup_phys': 50,
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
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_dynsys, param_abbrv_algo)
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

def meta_dns_workflow(idx_param):
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-03-19"
    sub_date_str = "0"
    mfd = dict() # meta-filedict
    mdd = dict() # meta-dirdict
    mdd['analysis'] = join(scratch_dir,date_str,sub_date_str,'meta_analysis')
    mdd['plots'] = join(scratch_dir,date_str,sub_date_str,'meta_plots')
    for meta_dir in list(mdd.values()):
        makedirs(meta_dir, exist_ok=True)


    expt_labels = []
    expt_abbrvs = []
    filedicts = []
    for i_param in idx_param:
        _,_,_,expt_label,expt_abbrv,_,filedict = dns_workflow(i_param)
        expt_labels.append(expt_label)
        expt_abbrvs.append(expt_abbrvs)
        filedicts.append(filedict)
    mfd['return_stats'] = dict()
    obs_names = list(filedicts[0]['return_stats'].keys())
    obsprop = Lorenz96ODE.observable_props()
    for obs_name in obs_names:
        mfd['return_stats'][obs_name] = dict({
            'plots': join(mdd['plots'], r'return_stats_%s.png'%(obsprop[obs_name]['abbrv'])),
            })
    return mdd,mfd,filedicts,expt_labels,expt_abbrvs

def meta_dns_procedure(idx_param):
    mdd,mfd,filedicts,expt_labels,expt_abbrvs = meta_dns_workflow(idx_param)
    # Plot return stats 
    obs_names = list(mfd['return_stats'].keys())
    obsprop = Lorenz96ODE.observable_props()
    for obs_name in obs_names:
        return_stats_filenames = [fd['return_stats'][obs_name]['analysis'] for fd in filedicts]
        L96SDEDNS.plot_return_stats_meta(return_stats_filenames, mfd['return_stats'][obs_name]['plots'], obsprop[obs_name], expt_labels)
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
    while not (alg.terminate):
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict(filename=f'mem{mem}.npz')
        alg.take_next_step(saveinfo)
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def compute_extreme_stats_new(config_analysis, alg, dirdict):
    # TODO fill in this function the same way it's done in TEAMS, just for synchrony
    pass

def compute_extreme_stats(config_analysis, alg, dirdict):
    nmem = alg.ens.get_nmem()
    tu = alg.ens.dynsys.dt_save
    spinup = int(config_analysis['spinup_phys']/tu)
    time_block_size = int(config_analysis['time_block_size_phys']/tu)
    all_starts,all_ends = alg.ens.get_all_timespans()
    mems2summarize = np.where((all_starts >= spinup)*(all_starts <= 1e7))[0]
    print(f'{len(mems2summarize) = }')
    # TODO need to distribute the block maxima method
    for obs_name,obs_props in config_analysis['observables_rotsym'].items():
        print(f'About to compute extreme stats for {obs_name}')
        blocks_per_k = int((all_ends[mems2summarize[-1]] - all_starts[mems2summarize[0]])/time_block_size)
        block_maxima = np.nan*np.ones((blocks_per_k, alg.ens.dynsys.ode.K))
        i_block = 0
        for i_mem,mem in enumerate(mems2summarize):
            fk = alg.ens.compute_observables([obs_props['fun']], mem)[0]
            if i_mem == 0:
                # Initialize a histogram, might have to extend it 
                bin_edges = np.linspace(np.min(fk)-1e-10,np.max(fk)+1e-10,40)
                bin_width = bin_edges[1] - bin_edges[0]
                hist = np.zeros(len(bin_edges)-1, dtype=int)
            elif np.max(fk) > bin_edges[-1]:
                num_new_bins = int(np.ceil((np.max(fk) - bin_edges[-1])/bin_width))
                bin_edges = np.concatenate((bin_edges, bin_edges[-1] + bin_width*np.arange(1,num_new_bins+1)))
                hist = np.concatenate((hist, np.zeros(num_new_bins, dtype=int)))
            elif np.min(fk) < bin_edges[0]:
                num_new_bins = int(np.ceil((bin_edges[0]-np.min(fk))/bin_width))
                bin_edges = np.concatenate((bin_edges[0]-bin_width*np.arange(1,num_new_bins+1)[::-1], bin_edges))
                hist = np.concatenate((np.zeros(num_new_bins,dtype=int), hist))

            hist_new,_ = np.histogram(fk.flat, bins=bin_edges)
            hist += hist_new

            for k in range(alg.ens.dynsys.ode.K):
                block_maxima_mem_k = utils.compute_block_maxima(fk[:,k],time_block_size)
                block_maxima[i_block:i_block+len(block_maxima_mem_k),k] = block_maxima_mem_k
            i_block += len(block_maxima_mem_k)

            if mem % 100 == 0: 
                print(f'{mem = }')
                memusage_GB = psutil.Process().memory_info().rss / 1e9
                print(f'Using {memusage_GB} GB')
        block_maxima = np.concatenate(block_maxima[:i_block,:], axis=0)
        rlev,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale = utils.compute_returnstats_preblocked(block_maxima, time_block_size)
        bin_lows = bin_edges[:-1]
        extstats = dict({'bin_lows': bin_lows, 'hist': hist, 'rlev': rlev, 'rtime': rtime, 'logsf': logsf, 'rtime_gev': rtime_gev, 'logsf_gev': logsf_gev, 'shape': shape, 'loc': loc, 'scale': scale})
        # TODO carefully compare different block sizes, both for GEV fitting and for return period validity 
        np.savez(join(dirdict['analysis'],r'extstats_rotsym_%s.npz'%(obs_props['abbrv'])), **extstats)
        # Plot 
        bin_mids = bin_lows + 0.5*(bin_lows[1]-bin_lows[0])
        fig,axes = plt.subplots(ncols=2,figsize=(12,4),gridspec_kw={'wspace': 0.25})
        ax = axes[0]
        ax.plot(bin_mids,hist/(np.sum(hist)*bin_width),color='black',marker='.')
        ax.set_xlabel(obs_props['label'])
        ax.set_ylabel('Prob. density')
        ax.set_yscale('log')
        ax = axes[1]
        hemp, = ax.plot(rtime,rlev,color='black',marker='.',label='Empirical')
        hgev, = ax.plot(rtime_gev,rlev,color='cyan',marker='.',label='GEV fit')
        ax.legend(handles=[hemp,hgev])
        print(f'{rtime_gev = }')
        ax.set_ylim([rlev[np.argmax(rtime>0)],2*rlev[-1]-rlev[-2]])
        ax.set_xlabel('Return time')
        ax.set_ylabel('Return level %s'%(obs_props['label']))
        ax.set_xscale('log')
        fig.savefig(join(dirdict['plots'],r'extstats_%s.png'%(obs_props['abbrv'])),**pltkwargs)
        plt.close(fig)
    return


def plot_dns(config_analysis,alg,dirdict):
    obs_names = list(filedict['return_stats'].keys())
    obsprop = Lorenz96ODE.observable_props()
    if tododict['basic_vis']:
        alg = pickle.load(open(filedict['alg'],'rb'))
        alg.plot_dns_segment(filedict['basic_vis']['dns_segment'])
    if tododict['return_stats']:
        for obs_name in obs_names:
            L96SDEDNS.plot_return_stats(filedict['return_stats'][obs_name]['analysis'], filedict['return_stats'][obs_name]['plots'], obsprop[obs_name])
    return


def dns_single_procedure(i_expt):
    tododict = dict({
        'run':                   0,
        'plot_segment':          0,
        'return_stats':          1,
        })

    # Quantities of interest for statistics. These should be registered as observables under the system.
    print(f'Workflow setup...',end='')
    config_dynsys,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = dns_single_workflow(i_expt)
    print('done')

    if tododict['run']:
        run_dns(dirdict,filedict,config_dynsys,config_algo)
    alg = pickle.load(open(filedict['alg'],'rb'))
    if tododict['plot_segment']:
        outfile = join(dirdict['plots'],'dns_segment.png')
        alg.plot_dns_segment(outfile, tspan_phys=[500,515])
    if tododict['return_stats']:
        print(f'About to compute extreme stats')
        memusage_GB = psutil.Process().memory_info().rss / 1e9
        print(f'Using {memusage_GB} GB')
        compute_extreme_stats(config_analysis,alg,dirdict)
        #L96SDEDNS.plot_return_stats(filedict['return_stats'][obs_name]['analysis'], filedict['return_stats'][obs_name]['plots'], obsprop[obs_name])
    return

if __name__ == "__main__":
    procedure = sys.argv[1]
    idx_expt = [int(v) for v in sys.argv[2:]]
    if procedure == 'single':
        for i_expt in idx_expt:
            dns_single_procedure(i_expt)
    elif procedure == 'meta':
        meta_dns_procedure(idx_expt)
