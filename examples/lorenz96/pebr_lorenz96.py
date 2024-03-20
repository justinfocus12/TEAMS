import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
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
sys.path.append('../..')
from lorenz96 import Lorenz96ODE,Lorenz96SDE
from ensemble import Ensemble
import forcing
from algorithms_lorenz96 import Lorenz96ODEPeriodicBranching as L96ODEPeBr, Lorenz96SDEPeriodicBranching as L96SDEPeBr
import utils

def pebr_paramset(i_param):
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
        'seed_inc_init': seed_incs[i_param], # added to seed_min to generate first seed
        'branches_per_group': 16, 
        'interbranch_interval_phys': 1.0,
        'branch_duration_phys': 15.0,
        'num_branch_groups': 4,
        'max_member_duration_phys': 20.0,
        })
    return config_dynsys,config_algo,expt_labels[i_param],expt_abbrvs[i_param]

def pebr_workflow(i_param):
    config_dynsys,config_algo,expt_label,expt_abbrv = pebr_paramset(i_param)
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-03-19"
    sub_date_str = "0"
    param_abbrv_dynsys,param_label_dynsys = Lorenz96SDE.label_from_config(config_dynsys)
    param_abbrv_algo,param_label_algo = L96SDEPeBr.label_from_config(config_algo)
    dirdict = dict()
    dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, param_abbrv_dynsys, param_abbrv_algo)
    dirdict['data'] = join(dirdict['expt'], 'data')
    dirdict['analysis'] = join(dirdict['expt'], 'analysis')
    dirdict['plots'] = join(dirdict['expt'], 'plots')
    for dirname in list(dirdict.values()):
        makedirs(dirname, exist_ok=True)

    # List the quantities of interest
    # observables
    obsprop = Lorenz96ODE.observable_props()
    observables = dict({
        'x0': dict({
            'fun_name': 'x0',
            'kwargs': {},
            'abbrv': 'x0',
            'label': r'$x_0$',
            }),
        'E0': dict({
            'fun_name': 'E0',
            'kwargs': {},
            'abbrv': 'E0',
            'label': r'$\frac{1}{2}x_0^2$',
            }),
        'E': dict({
            'fun_name': 'E',
            'kwargs': {},
            'abbrv': 'E',
            'label': r'$\frac{1}{2}\overline{x^2}$',
            }),
        })
    obs_names = list(observables.keys())
    # distance metrics
    dist_metrics = dict({
        'euclidean': dict({
            'fun_name': 'distance_euclidean',
            'abbrv': 'EUC',
            'label': 'Euclidean distance',
            'kwargs': {},
            }),
        'timedelay_xk': dict({
            'fun_name': 'distance_timedelay_xk',
            'abbrv': 'x0d2',
            'label': r'$x_0$ distance, $\Delta=2$',
            'kwargs': {'k': 0, 'timedelay_phys': 2.0},
            }),
        'timedelay_Ek': dict({
            'fun_name': 'distance_timedelay_Ek',
            'abbrv': 'E0d2',
            'label': r'$\frac{1}{2}x_0^2$ distance, $\Delta=2$',
            'kwargs': {'k': 0, 'timedelay_phys': 2.0},
            }),
        })
    dist_names = list(dist_metrics.keys())
    config_analysis = dict({
        'observables': observables,
        'dist_metrics': dist_metrics,
        })

    filedict = dict()
    # Algorithm manager
    filedict['alg'] = join(dirdict['data'], 'alg.pickle')
    # Basic visualization
    filedict['basic_vis'] = dict()
    filedict['basic_vis']['groupwise'] = dict({'observables': dict(), 'distances': dict()})
    for obs_name in obs_names:
        filedict['basic_vis']['groupwise']['observables'][obs_name] = []
        for group in range(4): 
            filedict['basic_vis']['groupwise']['observables'][obs_name].append(join(dirdict['plots'], r'groupwise_obs%s_bg%d.png'%(observables[obs_name]['abbrv'],group)))
    for dist_name in dist_names:
        filedict['basic_vis']['groupwise']['distances'][dist_name] = []
        for group in range(4): 
            filedict['basic_vis']['groupwise']['distances'][dist_name].append(join(dirdict['plots'], r'groupwise_dist%s_bg%d.png'%(dist_metrics[dist_name]['abbrv'],group)))
    # Quantitative analysis
    filedict['dispersion'] = dict() 
    filedict['dispersion']['distance'] = dict() # distance from control to perturbed, for every single ensemble member of every group
    filedict['dispersion']['satfractime'] = dict() # Time since split at which a given saturation fraction is reached
    for dist_name in dist_names:
        filedict['dispersion']['distance'][dist_name] = join(dirdict['analysis'], r'dispersion_distance_%s.npz'%(dist_metrics[dist_name]['abbrv']))
        filedict['dispersion']['satfractime'][dist_name] = join(dirdict['analysis'], r'dispersion_satfractime_%s.npz'%(dist_metrics[dist_name]['abbrv']))
    return config_dynsys,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict

def run_pebr(dirdict,filedict,config_dynsys,config_algo):
    root_dir = dirdict['data']
    obs_fun = lambda t,x: None
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'],'rb'))
        alg.ens.set_root_dir(root_dir)
    else:
        sde = Lorenz96SDE(config_dynsys)
        ens = Ensemble(sde,root_dir=root_dir)
        alg = L96SDEPeBr(config_algo, ens)
    nmem = alg.ens.get_nmem()
    alg.ens.set_root_dir(root_dir)
    print(f'{alg.ens.root_dir = }')
    # TODO expand capacity here to allow tree to grow if needed in different directions
    while not (alg.terminate):
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict(filename=f'mem{mem}.npz')
        alg.take_next_step(saveinfo)
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return

def analyze_pebr(config_analysis,tododict,dirdict,filedict):
    alg = pickle.load(open(filedict['alg'], 'rb'))
    if tododict['dispersion']:
        # Measure distances 
        for dist_name,dist_props in config_analysis['dist_metrics'].items():
            print(f'{dist_name = }')
            print(f'{dist_props = }')
            def dist_fun(t0,x0,t1,x1):
                print(f'{t0=},\n{t1=}')
                trange_full = np.arange(min(t0[0],t1[0]),max(t0[-1],t1[-1])+1)
                trange_valid = np.arange(max(t0[0],t1[0]),min(t0[-1],t1[-1])+1)
                tidx0 = trange_valid - t0[0]
                tidx1 = trange_valid - t1[0]
                fun = getattr(alg.ens.dynsys.ode, dist_props['fun_name'])
                dist = np.nan*np.ones_like(trange_full)
                rhs = fun(t0[tidx0],x0[tidx0],t1[tidx1],x1[tidx1],**dist_props['kwargs'])
                print(f'{rhs.shape = }')
                lhs = dist[trange_valid[0]-trange_full[0]:trange_valid[-1]+1-trange_full[0]] 
                print(f'{lhs.shape = }')
                dist[trange_valid[0]-trange_full[0]:trange_valid[-1]+1-trange_full[0]] = fun(t0[tidx0],x0[tidx0],t1[tidx1],x1[tidx1],**dist_props['kwargs'])
                return dist
            outfile = filedict['dispersion']['distance'][dist_name]
            alg.measure_pert_growth(dist_fun, outfile)
    return

def plot_dispersion(config_analysis,dirdict,filedict):
    alg = pickle.load(open(filedict['alg'], 'rb'))
    for dist_name,dist_props in config_analysis['dist_metrics'].items():
        disp_data = np.load(filedict['dispersion']['distance'][dist_name])
        split_times = disp_data['split_times']
        dists = disp_data['dists']
        ngroups,nbranches,ntimes = dists.shape
        for group in range(ngroups):
            fig,ax = plt.subplots()
            alg.plot_dispersion(split_times[group], dists[group], fig, ax)
            ax.set_xlabel(r'time since %g'%(split_times[group]*alg.ens.dynsys.dt_save))
            ax.set_title(r'Group %g'%(group))
            ax.set_ylabel(dist_props['label'])
            ax.set_yscale('log')
            fig.savefig(filedict['basic_vis']['groupwise']['distances'][dist_name][group], **pltkwargs)
            plt.close(fig)
    return

def visualize_pebr_basically(config_analysis,dirdict,filedict):
    alg = pickle.load(open(filedict['alg'], 'rb'))
    for obs_name,obs_props in config_analysis['observables'].items():
        obs_fun = lambda t,x: getattr(alg.ens.dynsys.ode, obs_props['fun_name'])(t,x,**obs_props['kwargs'])
        ylabel = obs_props['label']
        for group in range(min(4,alg.branching_state['next_branch_group']+1)):
            title = r'Group %d'%(group)
            outfile = filedict['basic_vis']['groupwise']['observables'][obs_name][group]
            alg.plot_obs_spaghetti(obs_fun,group,outfile,ylabel=ylabel,title=title,abbrv=obs_props['abbrv'])
    return

def pebr_procedure(i_param):
    tododict = dict({
        'run':                0,
        'analysis': dict({
            'dispersion':    0,
            }),
        'plots': dict({
            'observables':    1,
            'dispersion':     1,
            'response':       0,
            }),
        })
    config_dynsys,config_algo,config_analysis,expt_label,expt_abbrv,dirdict,filedict = pebr_workflow(i_param)

    if tododict['run']:
        run_pebr(dirdict,filedict,config_dynsys,config_algo)
    if utils.find_true_in_dict(tododict['analysis']):
        analyze_pebr(config_analysis,tododict['analysis'],dirdict,filedict)
    if utils.find_true_in_dict(tododict['plots']):
        if tododict['plots']['observables']:
            visualize_pebr_basically(config_analysis,dirdict,filedict)
        if tododict['plots']['dispersion']:
            plot_dispersion(config_analysis,dirdict,filedict)



    if False and utils.find_true_in_dict(tododict['plot_pebr']):
        alg = pickle.load(open(fndict['alg']['alg'], 'rb'))
        tu = alg.ens.dynsys.dt_save
        obsprop = alg.ens.dynsys.observable_props()
        if tododict['plot_pebr']['pert_growth']:
            pert_growth_dict = pickle.load(open(fndict['analysis']['pert_growth'],'rb'))
            lyap_dict = pickle.load(open(fndict['analysis']['lyap_exp'],'rb'))
            alg.plot_pert_growth(pert_growth_dict, lyap_dict, fndict['plots'], logscale=(systype=='ODE'))
        if tododict['plot_pebr']['observables']:
            print(f'plotting observables')
            obs_names = ['x0','E0','E','Emax']
            obs_funs = dict()
            obs_abbrvs = dict()
            obs_labels = dict()
            for obs_name in obs_names:
                obs_funs[obs_name] = getattr(alg.ens.dynsys, f'observable_{obs_name}')
                obs_abbrvs[obs_name] = obsprop[obs_name]['abbrv']
                obs_labels[obs_name] = obsprop[obs_name]['label']
            for branch_group in range(alg.branching_state['next_branch_group']+1):
                alg.plot_obs_spaghetti(obs_funs, branch_group, dirdict['plots'], labels=obs_labels, abbrvs=obs_abbrvs)

        # TODO implement methods for pairwise distances, maybe localized, and plot those divergences too 
    return



if __name__ == "__main__":
    procedure = sys.argv[1]
    idx_param = [int(v) for v in sys.argv[2:]]
    if procedure == 'single':
        for i_param in idx_param:
            pebr_procedure(i_param)
    elif procedure == 'meta':
        meta_pebr_procedure(idx_param)
