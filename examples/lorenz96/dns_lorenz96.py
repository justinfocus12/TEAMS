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
from algorithms_lorenz96 import Lorenz96ODEDirectNumericalSimulation, Lorenz96SDEDirectNumericalSimulation
import utils


def dns(i_param,seed_inc):
    tododict = dict({
        'run':                   1,
        'summarize': dict({
            'return_stats':      1,
            }),
        'plot': dict({
            'timeseries':   1,
            'return_stats': 1,
            }),
        })

    # Quantities of interest for statistics
    qois = ['Ek','E']

    # Create a small ensemble
    # Run three trajectories, each one picking up where the previous one left off
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/lorenz96/"
    date_str = "2024-03-17"
    sub_date_str = "0"
    print(f'About to generate default config')

    config_dynsys = Lorenz96SDE.default_config()
    F4_list = [3.0,1.0,0.5,0.25,0.0]
    config_dynsys['frc']['white']['wavenumber_magnitudes'][0] = F4_list[i_param]

    param_abbrv_dynsys,param_label_dynsys = Lorenz96SDE.label_from_config(config_dynsys)
    config_algo = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'seed_inc_init': seed_inc, # add to seed_min
        'max_member_duration_phys': 50.0,
        'num_chunks_max': 50,
        })
    param_abbrv_algo,param_label_algo = Lorenz96SDEDirectNumericalSimulation.label_from_config(config_algo)
    dirdict = dict({
        'alg': join(scratch_dir, date_str, sub_date_str, param_abbrv_dynsys, param_abbrv_algo)
        })
    dirdict['analysis'] = join(dirdict['alg'],'analysis')
    dirdict['plots'] = join(dirdict['alg'],'plots')
    for dirname in list(dirdict.values()):
        makedirs(dirname, exist_ok=True)
    alg_filename = join(dirdict['alg'],'alg.pickle')
    root_dir = dirdict['alg']

    if tododict['run']:
        obs_fun = lambda t,x: None
        if exists(alg_filename):
            alg = pickle.load(open(alg_filename,'rb'))
            alg.ens.set_root_dir(root_dir)
        else:
            sde = Lorenz96SDE(config_dynsys)
            ens = Ensemble(sde,root_dir=root_dir)
            alg = Lorenz96SDEDirectNumericalSimulation(config_algo, ens)
        nmem = alg.ens.get_nmem()
        alg.ens.set_root_dir(root_dir)
        num_new_chunks = config_algo['num_chunks_max']-nmem
        alg.set_simulation_capacity(num_new_chunks, config_algo['max_member_duration_phys'])
        if num_new_chunks > 0:
            alg.terminate = False
        # TODO update capacity if needed
        while not (alg.terminate):
            mem = alg.ens.get_nmem()
            print(f'----------- Starting member {mem} ----------------')
            saveinfo = dict(filename=f'mem{mem}.npz')
            alg.take_next_step(saveinfo)
            pickle.dump(alg, open(alg_filename, 'wb'))
    # Load the ensemble for further analysis
    alg = pickle.load(open(alg_filename,'rb'))
    K = alg.ens.dynsys.ode.K
    tu = alg.ens.dynsys.dt_save
    obsprop = alg.ens.dynsys.ode.observable_props()

    # ------------ Measure return statistics ----------------------
    if utils.find_true_in_dict(tododict['summarize']):
        if tododict['summarize']['return_stats']:
            time_block_size_phys = 12
            spinup_phys = 30
            for obs_name in qois:
                obs_funs2concat = [lambda t,x: getattr(alg.ens.dynsys.ode, obs_name)(t,x)]
                alg.compute_return_stats(obs_funs2concat, int(time_block_size_phys/tu), int(spinup_phys/tu), dirdict['analysis'], abbrv=obsprop[obs_name]['abbrv'])

    # --------------- Plot timeseries, statistics -------------------
    if utils.find_true_in_dict(tododict['plot']):
        if tododict['plot']['timeseries']:
            alg.plot_dns_segment(dirdict['plots'])
        if tododict['plot']['return_stats']:
            for obs_name in qois:
                returnstats_filename = join(dirdict['analysis'], r'%s_returnstats.npz'%(obsprop[obs_name]['abbrv']))
                fig_filename = join(dirdict['plots'], r'%s_returnplot.png'%(obsprop[obs_name]['abbrv']))
                alg.plot_return_stats(returnstats_filename, fig_filename, obsprop[obs_name])
    return


def dns_meta_analysis(idx_param, seeds_inc):





if __name__ == "__main__":
    i_param = int(sys.argv[1])
    seed_inc = int(sys.argv[2])
    dns(i_param,seed_inc)
