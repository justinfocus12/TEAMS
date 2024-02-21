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
from lorenz96 import Lorenz96ODE,Lorenz96SDE
from ensemble import Ensemble
import forcing
import algorithms

class Lorenz96ODEPeriodicBranching(algorithms.ODEPeriodicBranching):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: self.ens.dynsys.observable(t, x, name)
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
        'plot_pebr':               1,
        })
    config_ode = Lorenz96ODE.default_config()
    tu = config_ode['dt_save'],
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-21"
    sub_date_str = "0"
    param_abbrv_ode,param_label_ode = Lorenz96ODE.label_from_config(config_ode)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'branches_per_group': 16, 
        'interbranch_interval_phys': 2.0,
        'branch_duration_phys': 15.0,
        'num_branch_groups': 10,
        'max_member_duration_phys': 20.0,
        })
    seed = 849582 # TODO make this a command-line argument
    param_abbrv_algo,param_label_algo = Lorenz96ODEPeriodicBranching.label_from_config(config_algo)
    algdir = join(scratch_dir, date_str, sub_date_str, param_abbrv_ode, param_abbrv_algo)
    makedirs(algdir, exist_ok=True)
    alg_filename = join(algdir,'alg.pickle')

    if tododict['run_pebr']:
        if exists(alg_filename):
            alg = pickle.load(open(alg_filename, 'rb'))
        else:
            ode = Lorenz96ODE(config_ode)
            ens = Ensemble(ode)
            alg = Lorenz96ODEPeriodicBranching(config_algo, ens, seed)

        mem = 0
        while not (alg.terminate):
            mem = alg.ens.memgraph.number_of_nodes()
            print(f'----------- Starting member {mem} ----------------')
            saveinfo = dict(filename=join(algdir,f'mem{mem}.npz'))
            alg.take_next_step(saveinfo)
            pickle.dump(alg, open(alg_filename, 'wb'))

    if tododict['plot_pebr']:
        alg = pickle.load(open(alg_filename, 'rb'))
        tu = alg.ens.dynsys.dt_save
        fig,ax = plt.subplots(figsize=(12,5))
        # Load the entire trunk
        t_trunk = []
        x_trunk = []
        for mem in alg.branching_state['trunk_lineage']:
            t,x = alg.ens.dynsys.load_trajectory(alg.ens.traj_metadata[mem])
            t_trunk.append(t)
            x_trunk.append(x)
        t_trunk = np.concatenate(tuple(t_trunk))
        x_trunk = np.concatenate(tuple(x_trunk))
        print(f'{t_trunk[[0,-1]] = }')
        for child in range(1,alg.ens.memgraph.number_of_nodes()):
            t_child,x_child = alg.ens.dynsys.load_trajectory(alg.ens.traj_metadata[child])
            print(f'{t_child[[0,-1]] = }')
            # Get the overlap time indices
            tmin,tmax = max(t_trunk[0],t_child[0]),min(t_trunk[-1],t_child[-1])
            tic,tfc = tmin-t_child[0], tmax-t_child[0]
            tip,tfp = tmin-t_trunk[0], tmax-t_trunk[0]
            dist = alg.ens.dynsys.distance(t_trunk[tip:tfp+1], x_trunk[tip:tfp+1], t_child[tic:tfc+1], x_child[tic:tfc+1], 'euclidean')
            ax.plot(t_trunk[tip:tfp+1]*tu, dist)
        ax.set_yscale('log')
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance from parent')
        fig.savefig(join(algdir, 'divergence.png'), **pltkwargs)
        plt.close(fig)

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
