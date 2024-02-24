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
            name: self.ens.dynsys.observable(t, x, name)
            for name in self.obs_dict_names()
            })
        return obs_dict


def periodic_branching_impulsive():
    tododict = dict({
        'run_pebr':                1,
        'plot_pebr': dict({
            'observables':    1,
            'divergence':     0,
            'response':       0,
            }),
        })
    config_ode = Lorenz96ODE.default_config()
    tu = config_ode['dt_save'],
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-24"
    sub_date_str = "0"
    param_abbrv_ode,param_label_ode = Lorenz96ODE.label_from_config(config_ode)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'branches_per_group': 8, 
        'interbranch_interval_phys': 10.0,
        'branch_duration_phys': 15.0,
        'num_branch_groups': 2,
        'max_member_duration_phys': 60.0,
        })
    seed = 849582 # TODO make this a command-line argument
    param_abbrv_algo,param_label_algo = Lorenz96ODEPeriodicBranching.label_from_config(config_algo)
    algdir = join(scratch_dir, date_str, sub_date_str, param_abbrv_ode, param_abbrv_algo)
    print(f'{algdir = }')
    makedirs(algdir, exist_ok=True)
    root_dir = algdir
    alg_filename = join(algdir,'alg.pickle')

    if tododict['run_pebr']:
        if exists(alg_filename):
            alg = pickle.load(open(alg_filename, 'rb'))
        else:
            ode = Lorenz96ODE(config_ode)
            ens = Ensemble(ode,root_dir)
            alg = Lorenz96ODEPeriodicBranching(config_algo, ens, seed)

        mem = 0
        while not (alg.terminate):
            mem = alg.ens.memgraph.number_of_nodes()
            print(f'----------- Starting member {mem} ----------------')
            saveinfo = dict(filename=f'mem{mem}.npz')
            alg.take_next_step(saveinfo)
            pickle.dump(alg, open(alg_filename, 'wb'))

    if utils.find_true_in_dict(tododict['plot_pebr']):
        plotdir = join(algdir, 'plots')
        makedirs(plotdir, exist_ok=1)
        alg = pickle.load(open(alg_filename, 'rb'))
        tu = alg.ens.dynsys.dt_save

        obsprop = alg.ens.dynsys.observable_props()

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
                alg.plot_obs_spaghetti(obs_funs, branch_group, plotdir, labels=obs_labels, abbrvs=obs_abbrvs)

        # TODO implement methods for pairwise distances, maybe localized, and plot those divergences too 

        if False:
            # Compute observables separately on branches and trunk 
            mems_branch = np.setdiff1d(np.arange(alg.ens.get_nmem()), alg.branching_state['trunk_lineage'])
            obs_dict_branch = alg.ens.compute_observables(obs_names, mems_branch)
            obs_dict_trunk = alg.ens.compute_observables_along_lineage(obs_names, alg.branching_state['trunk_lineage'][-1])
            for obs_name in obs_names:
                obs_dict_trunk[obs_name] = np.concatenate(obs_dict_trunk[obs_name], axis=0) # 0 is the time axis
                
            # Get all timespans
            all_timespans = np.array([np.array(alg.ens.get_member_timespan(mem)) for mem in range(alg.ens.memgraph.number_of_nodes())])
            all_init_times = all_timespans[:,0]
            all_fin_times = all_timespans[:,1]

        if tododict['plot_pebr']['divergence']:
            fig,ax = plt.subplots(figsize=(12,5))

            # Load the entire trunk
            t_trunk = []
            x_trunk = []
            for mem in alg.branching_state['trunk_lineage']:
                t,x = alg.ens.dynsys.load_trajectory(alg.ens.traj_metadata[mem], alg.ens.root_dir)
                t_trunk.append(t)
                x_trunk.append(x)
            t_trunk = np.concatenate(tuple(t_trunk))
            x_trunk = np.concatenate(tuple(x_trunk))


            for child in mems_branch:
                t_child,x_child = alg.ens.dynsys.load_trajectory(alg.ens.traj_metadata[child], alg.ens.root_dir)
                # Get the overlap time indices
                tmin,tmax = max(t_trunk[0],t_child[0]),min(t_trunk[-1],t_child[-1])
                tic,tfc = tmin-t_child[0], tmax-t_child[0]
                tip,tfp = tmin-t_trunk[0], tmax-t_trunk[0]
                dist = alg.ens.dynsys.distance(t_trunk[tip:tfp+1], x_trunk[tip:tfp+1], t_child[tic:tfc+1], x_child[tic:tfc+1], 'euclidean')
                ax.plot(t_trunk[tip:tfp+1]*tu, dist)
            ax.set_yscale('log')
            ax.set_xlabel('Time')
            ax.set_ylabel('Distance from parent')
            fig.savefig(join(plotdir, 'divergence.png'), **pltkwargs)
            plt.close(fig)

        if tododict['plot_pebr']['response']:
            # 3. Plot lagged observable vs. perturbation for each IC
            obs2plot = 'x0'
            timelags_phys = np.arange(10)
            timelags = (timelags_phys/alg.ens.dynsys.dt_save).astype(int)
            for i_group in range(alg.branching_state['next_branch_group']):
                branch_time = alg.init_time + alg.ens.dynsys.t_burnin + i_group*alg.interbranch_interval
                idx_branches = np.array([i_b for i_b in range(len(mems_branch)) if obs_dict_branch['t'][i_b][0] == branch_time])
                print(f'{branches = }')
                fig,ax = plt.subplots()
                impulses = np.nan*np.ones(len(idx_branches))
                obs_branch_lagged = np.nan*np.ones((len(idx_branches),len(timelags)))
                obs_trunk_lagged = np.nan*np.ones(len(timelags))
                for i_branch in range(len(idx_branches)):
                    mem = mems_branch[idx_branches[i_branch]]
                    imp = alg.ens.traj_metadata[mem]['icandf']['frc'].impulses[0]
                    print(f'{imp = }')
                    impulses[i_branch] = imp.item()
                    x0lagged[i_branch,:] = obs_dict_branch[i_mem][timelags]
                ord_imp = np.argsort(impulses)
                handles = []
                for i_timelag,timelag in enumerate(timelags):
                    h, = ax.plot(impulses[ord_imp], x0lagged[ord_imp,i_timelag] - obs_dict_trunk[branch_time+timelag-alg.init_time], color=plt.cm.coolwarm(timelag/timelags[-1]), label=r'$\Delta t=%g$'%(timelag))
                    handles.append(h)
                ax.set_xlabel('Impulse')
                ax.set_ylabel(r'Lagged response')
                ax.legend(handles=[handles[0],handles[-1]], loc=(1,0))
                imgname = (f'response_group{i_group}_{obs2plot}').replace('.','p')
                print(f'{imgname = }')
                fig.savefig(join(plotdir,imgname), **pltkwargs)
                plt.close()
            

def periodic_branching_white():
    tododict = dict({
        'run_pebr':                1,
        'plot_pebr':               1,
        })
    config_ode = Lorenz96ODE.default_config()
    for key in ['wavenumbers','wavenumber_magnitudes','sites','site_magnitudes']:
        config_ode['frc']['impulsive'][key] = []
    config_sde = Lorenz96SDE.default_config()
    tu = config_ode['dt_save'],
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-11"
    sub_date_str = "1"
    param_abbrv_sde,param_label_sde = Lorenz96SDE.label_from_config(config_ode,config_sde)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'branches_per_group': 8, 
        'interbranch_interval_phys': 12.0,
        'branch_duration_phys': 10.0,
        'num_branch_groups': 5,
        })
    seed = 849582 # TODO make this a command-line argument
    param_abbrv_algo,param_label_algo = Lorenz96SDEPeriodicBranching.label_from_config(config_algo)
    algdir = join(scratch_dir, date_str, sub_date_str, param_abbrv_sde, param_abbrv_algo)
    makedirs(algdir, exist_ok=True)
    alg_filename = join(algdir,'alg.pickle')

    if tododict['run_pebr']:
        if exists(alg_filename):
            print(f'Alg is continuing')
            alg = pickle.load(open(alg_filename, 'rb'))
        else:
            print(f'Alg is starting up')
            ode = Lorenz96ODE(config_ode)
            sde = Lorenz96SDE(ode,config_sde)
            ens = Ensemble(sde)
            alg = Lorenz96SDEPeriodicBranching(config_algo, ens, seed)

        while not alg.branching_state['terminate']:
            mem = alg.ens.memgraph.number_of_nodes()
            print(f'----------- Starting member {mem} ----------------')
            saveinfo = dict(filename=join(algdir,f'mem{mem}.npz'))
            alg.take_next_step(saveinfo)
            pickle.dump(alg, open(alg_filename, 'wb'))
        print(f'{alg.branching_state["terminate"] = }')

    if tododict['plot_pebr']:
        alg = pickle.load(open(alg_filename, 'rb'))
        tu = alg.ens.dynsys.dt_save
        fig,ax = plt.subplots(figsize=(12,5))
        parent = 0
        t_parent,x_parent = alg.ens.dynsys.load_trajectory(alg.ens.traj_metadata[parent])
        for child in range(1,alg.ens.memgraph.number_of_nodes()):
            t_child,x_child = alg.ens.dynsys.load_trajectory(alg.ens.traj_metadata[child])
            # Get the overlap time indices
            tmin,tmax = max(t_parent[0],t_child[0]),min(t_parent[-1],t_child[-1])
            tic,tfc = tmin-t_child[0], tmax-t_child[0]
            tip,tfp = tmin-t_parent[0], tmax-t_parent[0]
            dist = alg.ens.dynsys.distance(t_parent[tip:tfp+1], x_parent[tip:tfp+1], t_child[tic:tfc+1], x_child[tic:tfc+1], 'euclidean')
            ax.plot(t_parent[tip:tfp+1]*tu, dist)
        #ax.set_yscale('log')
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance from parent')
        fig.savefig(join(algdir, 'divergence.png'), **pltkwargs)
        plt.close(fig)

if __name__ == "__main__":
    periodic_branching_impulsive()
