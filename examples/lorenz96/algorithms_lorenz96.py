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
import algorithms
import utils

class Lorenz96ODEPeriodicBranching(algorithms.ODEPeriodicBranching):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys, f'observable_{name}')(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict

class Lorenz96SDEPeriodicBranching(algorithms.SDEPeriodicBranching):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys.ode, f'observable_{name}')(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict


def periodic_branching(systype):
    tododict = dict({
        'run_pebr':                0,
        'analyze_pebr': dict({
            'measure_pert_growth':           0,
            'analyze_pert_growth':           1,
            }),
        'plot_pebr': dict({
            'observables':    0,
            'pert_growth':    1,
            'response':       0,
            }),
        })
    DynSysClass = {'ODE': Lorenz96ODE, 'SDE': Lorenz96SDE}[systype]
    AlgClass = {'ODE': Lorenz96ODEPeriodicBranching, 'SDE': Lorenz96SDEPeriodicBranching}[systype]
    config_dynsys = DynSysClass.default_config()
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-24"
    sub_date_str = "1"
    param_abbrv_dynsys,param_label_dynsys = DynSysClass.label_from_config(config_dynsys)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'branches_per_group': 16, 
        'interbranch_interval_phys': 5.0,
        'branch_duration_phys': 15.0,
        'num_branch_groups': 50,
        'max_member_duration_phys': 20.0,
        })
    param_abbrv_algo,param_label_algo = AlgClass.label_from_config(config_algo)
    seed = 849582 # TODO make this a command-line argument

    # Set up directories
    dirdict = dict({
        'alg': join(scratch_dir, date_str, sub_date_str, param_abbrv_dynsys, param_abbrv_algo)
        })
    dirdict['analysis'] = join(dirdict['alg'],'analysis')
    dirdict['plots'] = join(dirdict['alg'],'plots')
    for dirname in list(dirdict.values()):
        makedirs(dirname, exist_ok=True)

    # Enumerate filenames
    dist_names = ['euclidean']
    fndict = dict({
        'alg': dict({
            'alg': join(dirdict['alg'],'alg.pickle'),
            }),
        'analysis': dict({
            'pert_growth': join(dirdict['analysis'],'pert_growth.pickle'),
            'lyap_exp': join(dirdict['analysis'],'lyap_exp.pickle')
            })
        })
    fndict['plots'] = dict()
    for dist_name in dist_names:
        fndict['plots'][dist_name] = dict({'rmse': join(dirdict['plots'],f'rmse_dist{dist_name}')})
        fndict['plots'][dist_name]['lyap_exp'] = join(dirdict['plots'], f'lyap_exp_dist{dist_name}')
        for branch_group in range(config_algo['num_branch_groups']):
            fndict['plots'][dist_name][branch_group] = join(dirdict['plots'],f'pert_growth_bg{branch_group}_dist{dist_name}.png')


    root_dir = dirdict['alg']

    if tododict['run_pebr']:
        if exists(fndict['alg']['alg']):
            alg = pickle.load(open(fndict['alg']['alg'], 'rb'))
        else:
            dynsys = DynSysClass(config_dynsys)
            ens = Ensemble(dynsys,root_dir)
            alg = AlgClass(config_algo, ens, seed)

        mem = 0
        while not (alg.terminate):
            mem = alg.ens.get_nmem()
            print(f'----------- Starting member {mem} ----------------')
            saveinfo = dict(filename=f'mem{mem}.npz')
            alg.take_next_step(saveinfo)
            pickle.dump(alg, open(fndict['alg']['alg'], 'wb'))

    if utils.find_true_in_dict(tododict['analyze_pebr']):
        alg = pickle.load(open(fndict['alg']['alg'], 'rb'))
        if tododict['analyze_pebr']['measure_pert_growth']:
            def dist_euclidean_tdep(t0,x0,t1,x1):
                trange = np.array([max(t0[0],t1[0]),min(t0[-1],t1[-1])+1])
                tidx0 = np.arange(trange[0],trange[1])-t0[0]
                tidx1 = np.arange(trange[0],trange[1])-t1[0]
                return np.sqrt(np.sum((x0[tidx0] - x1[tidx1])**2, axis=1))
            def rmsd_euclidean(t0,x0,t1,x1):
                D2mat = np.add.outer(np.sum(x0**2, axis=1), np.sum(x1**2, axis=1)) - 2*x0.dot(x1.T)
                return np.sqrt(np.mean(D2mat))
            dist_funs = dict({
                'tdep': dict({
                    'euclidean': dist_euclidean_tdep,
                    }),
                'rmsd': dict({
                    'euclidean': rmsd_euclidean,
                    })
                })
            pert_growth = alg.measure_pert_growth(dist_funs, )
            pickle.dump(pert_growth, open(fndict['analysis']['pert_growth'], 'wb'))
        else:
            pert_growth = pickle.load(open(fndict['analysis']['pert_growth'], 'rb'))
        if tododict['analyze_pebr']['analyze_pert_growth']:
            lyapunov_exponents = alg.analyze_pert_growth(pert_growth)
            pickle.dump(lyapunov_exponents, open(fndict['analysis']['lyap_exp'], 'wb'))



    if utils.find_true_in_dict(tododict['plot_pebr']):
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


if __name__ == "__main__":
    systypes = ['ODE','SDE']
    systype = systypes[int(sys.argv[1])]
    print(f'{systype = }')
    periodic_branching(systype)
