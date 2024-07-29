import numpy as np

from numpy.random import default_rng
import xarray as xr
from matplotlib import pyplot as plt, rcParams 
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
from os.path import join, exists, basename, relpath
from os import mkdir, makedirs
import sys
import shutil
import glob
import subprocess
import resource
import time as timelib
import pickle
import copy as copylib
import pprint
from importlib import reload

sys.path.append("../..")
import utils; reload(utils)
import ensemble; reload(ensemble)
import forcing; reload(forcing)
import algorithms; reload(algorithms)
from algorithms_crommelin2004tracer import Crommelin2004TracerODEAncestorGenerator as AnGe
from crommelin2004tracer import Crommelin2004TracerODE as CromODE


todo = dict({
    'run':                    1,
    'plot_spaghetti':         1,
    'plot_runmax':            1,
    })

scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/crommelin2004accelerated"
date_str = "2024-07-29"
sub_date_str = "1"

config_ode = CromODE.default_config()
config_ode['Nxfv'] = 32
config_ode['Nyfv'] = 8
config_algo = AnGe.default_config()
config_algo['time_horizon_phys'] = 1000
config_algo['branches_per_buick'] = 15
abbrv_ode,label_ode = CromODE.label_from_config(config_ode)
abbrv_algo,label_algo = AnGe.label_from_config(config_algo)

dirdict = dict()
dirdict['expt'] = join(scratch_dir, date_str, sub_date_str, abbrv_ode, abbrv_algo)
dirdict['data'] = join(dirdict['expt'], 'data')
dirdict['analysis'] = join(dirdict['expt'], 'analysis')
dirdict['plots'] = join(dirdict['expt'], 'plots')
for dirname in list(dirdict.values()):
    makedirs(dirname, exist_ok=True)
filedict = dict()
filedict['alg'] = join(dirdict['data'], 'alg.pickle')

if todo['run']:
    
    root_dir = dirdict['data']
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'],'rb'))
        alg.ens.set_root_dir(root_dir)
        alg.set_capacity(config_algo['num_buicks'], config_algo['branches_per_buick'])
    else:
        ode = CromODE(config_ode)
        uic_time = 0
        uic = ode.generate_default_init_cond(uic_time)
        ens = ensemble.Ensemble(ode,root_dir=root_dir)
        alg = AnGe(uic_time, uic, config_algo, ens)
    nmem = alg.ens.get_nmem()
    
    # run basic algorithm
    nmem = alg.ens.get_nmem()
    print(f'{nmem = }')
    while not (alg.terminate):
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict(filename=f'mem{mem}.npz')
        t0 = timelib.time()
        alg.take_next_step(saveinfo)
        t1 = timelib.time()
        print(f'--------------------- DURATION ------------------')
        print(t1 - t0)
        print(f'------------------------------------------------')
        pickle.dump(alg, open(filedict['alg'], 'wb'))

alg = pickle.load(open(filedict['alg'],'rb'))
if todo['plot_spaghetti']:
    # Plot mode 0 
    obs_fun = lambda t,state: state[:,0]
    outfile = join(dirdict['plots'], 'mode0.png')
    for family in range(alg.num_buicks):
        outfile = join(dirdict['plots'], 'spaghetti_branching_mode0_fam%d.png'%(family))
        alg.plot_observable_spaghetti_branching(obs_fun, family, outfile, title='mode 0',)
    # Plot local concentrations
    locs = [(2*np.pi*a,1.2) for a in [1/8,3/8,5/8,7/8]]
    for i_loc,(x,y) in enumerate(locs):
        obs_fun = lambda t,state: alg.ens.dynsys.local_conc(t,state,x,y)
        obs_abbrv = (r'cx%.1fy%.1f'%(x,y)).replace('.','p')
        label = r'$c(%.1f,%.1f)$'%(x,y)
        for family in range(alg.num_buicks):
            # Basic spaghett
            filename = ('spaghetti_branching_conc%.1f_%.1f_fam%d'%(x,y,family)).replace('.','p')
            outfile = join(dirdict['plots'], filename+'.png')
            alg.plot_observable_spaghetti_branching(obs_fun, family, outfile, title=r'Family %d, c(%g,%g)'%(family,x,y),)

if todo['plot_runmax']:
    locs = [(2*np.pi*a,1.2) for a in [1/8,3/8,5/8,7/8]]
    for i_loc,(x,y) in enumerate(locs):
        obs_fun = lambda t,state: alg.ens.dynsys.local_conc(t,state,x,y)
        obs_abbrv = (r'cx%.1fy%.1f'%(x,y)).replace('.','p')
        label = r'$c(%.1f,%.1f)$'%(x,y)
        runmax_file = join(dirdict['analysis'], r'%s.npz'%(obs_abbrv))
        figfile_prefix = join(dirdict['plots'], r'%s.png'%(obs_abbrv))
        # Max score distribution
        filename = ('runmax_cx%.1fy%.1f'%(x,y)).replace('.','p')
        alg.measure_running_max(obs_fun, runmax_file, figfile_prefix, label=r'c(%g,%g)')

