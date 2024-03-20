import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import sys
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

def dns_paramset(i_param):
    # Organize the array of parameters as well as the output files 
    F4s = [3.0,1.0,0.5,0.25,0.0]
    seed_incs = [0,0,0,0,0] # In theory we could make an unraveled array of (F4,seed)
    # Minimal labels to differentiate them 
    expt_labels = [r'$F_4=%g$'%(F4) for F4 in F4s]
    expt_abbrvs = [(r'F4eq%g'%(F4)).replace('.','p') for F4 in F4s]
    config_dynsys = Lorenz96SDE.default_config()
    config_dynsys['frc']['white']['wavenumber_magnitudes'][0] = F4s[i_param]

    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_incs[i_param], # add to seed_min
        'max_member_duration_phys': 50.0,
        'num_chunks_max': 100,
        })

    config_analysis = dict({
        # return statistics analysis
        'return_stats': dict({
            'time_block_size_phys': 12,
            'spinup_phys': 30,
            'k_roll_step': 4, # step size for augmenting Lorenz96 with rotational symmetry 
            })
        # Other possible parameters: method used for fitting GEV, threshold for GPD, ...
        })

    return config_dynsys,config_algo,config_analysis,expt_labels[i_param],expt_abbrvs[i_param]

def dns_workflow(i_param):
    config_dynsys,config_algo,config_analysis,expt_label,expt_abbrv = dns_paramset(i_param)
    # Organize output directories
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-03-19"
    sub_date_str = "0"
    param_abbrv_dynsys,param_label_dynsys = Lorenz96SDE.label_from_config(config_dynsys)
    param_abbrv_algo,param_label_algo = L96SDEDNS.label_from_config(config_algo)

    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_dynsys, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    for dirname in list(dirdict.values()):
        makedirs(dirname, exist_ok=True)

    # List the quantities of interest
    obsprop = Lorenz96ODE.observable_props()
    obs_names = ['x0','E0','E'] # Maybe we can do multivariate stuff on the second

    filedict = dict()
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    # Basic analysis
    filedict['basic_vis'] = dict()
    filedict['basic_vis']['dns_segment'] = join(dirdict['plots'], 'dns_segment_plot.png')
    # Return period analysis of scalar observables
    filedict['return_stats'] = dict()
    for obs_name in obs_names:
        filedict['return_stats'][obs_name] = dict({
            'analysis': join(dirdict['analysis'], r'return_stats_%s.npz'%(obsprop[obs_name]['abbrv'])),
            'plots': join(dirdict['plots'], r'return_stats_%s_plot.png'%(obsprop[obs_name]['abbrv'])),
            })
    # Autocorrelation analysis 
    filedict['autocorrelation'] = dict()
    for obs_name in obs_names:
        filedict['autocorrelation'][obs_name] = dict({
            'analysis': join(dirdict['analysis'], r'autocorrelation_%s.npz'%(obsprop[obs_name]['abbrv'])),
            'plot': join(dirdict['plots'], r'autocorrelation_%s.png'%(obsprop[obs_name]['abbrv'])),
            })
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
    else:
        sde = Lorenz96SDE(config_dynsys)
        ens = Ensemble(sde,root_dir=root_dir)
        alg = L96SDEDNS(config_algo, ens)
    nmem = alg.ens.get_nmem()
    alg.ens.set_root_dir(root_dir)
    alg.set_simulation_capacity(config_algo['num_chunks_max'], config_algo['max_member_duration_phys'])
    num_new_chunks = alg.num_chunks_max - nmem
    print(f'{num_new_chunks = }')
    if num_new_chunks > 0:
        alg.terminate = False
    while not (alg.terminate):
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict(filename=f'mem{mem}.npz')
        alg.take_next_step(saveinfo)
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def analyze_dns(config,tododict,dirdict,filedict):
    alg = pickle.load(open(filedict['alg'], 'rb'))
    tu = alg.ens.dynsys.dt_save
    obs_names = list(filedict['return_stats'].keys())
    if tododict['return_stats']:
        cfrs = config['return_stats']
        K = alg.ens.dynsys.ode.K
        for obs_name in obs_names:
            obs_funs2concat = []
            for kshift in np.arange(0,K,step=cfrs['k_roll_step'],dtype=int):
                print(f'{kshift = }')
                obs_funs2concat.append(
                        lambda t,x,kshift=kshift: getattr(alg.ens.dynsys.ode, obs_name)(t,np.roll(x,kshift,axis=1)))
            alg.compute_return_stats(
                    obs_funs2concat, int(cfrs['time_block_size_phys']/tu), int(cfrs['spinup_phys']/tu), filedict['return_stats'][obs_name]['analysis'])
    # TODO: do autocorrelation, cross-correlation analysis
    return

def plot_dns(tododict,dirdict,filedict):
    obs_names = list(filedict['return_stats'].keys())
    obsprop = Lorenz96ODE.observable_props()
    if tododict['basic_vis']:
        alg = pickle.load(open(filedict['alg'],'rb'))
        alg.plot_dns_segment(filedict['basic_vis']['dns_segment'])
    if tododict['return_stats']:
        for obs_name in obs_names:
            L96SDEDNS.plot_return_stats(filedict['return_stats'][obs_name]['analysis'], filedict['return_stats'][obs_name]['plots'], obsprop[obs_name])
    return

def dns_procedure(i_param):
    tododict = dict({
        'run':                   1,
        'analysis': dict({
            'return_stats': 1,
            'autocorrelation': 1,
            }),
        'plots': dict({
            'return_stats':  1,
            'autocorrelation': 1,
            'basic_vis':   1,
            }),
        })

    # Quantities of interest for statistics. These should be registered as observables under the system.
    config_dynsys,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = dns_workflow(i_param)

    if tododict['run']:
        run_dns(dirdict,filedict,config_dynsys,config_algo)
    if utils.find_true_in_dict(tododict['analysis']):
        analyze_dns(config_analysis,tododict['analysis'],dirdict,filedict)
    if utils.find_true_in_dict(tododict['plots']):
        plot_dns(tododict['plots'],dirdict,filedict)

    return

if __name__ == "__main__":
    procedure = sys.argv[1]
    idx_param = [int(v) for v in sys.argv[2:]]
    if procedure == 'single':
        for i_param in idx_param:
            dns_procedure(i_param)
    elif procedure == 'meta':
        meta_dns_procedure(idx_param)
