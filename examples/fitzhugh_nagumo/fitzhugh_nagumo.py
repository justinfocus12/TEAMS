import numpy as np
from numpy.random import default_rng
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import sys
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
sys.path.append("../..")
from dynamicalsystem import ODESystem,SDESystem
import forcing
from ensemble import Ensemble
import utils

class FitzhughNagumoODE(ODESystem):
    def __init__(self,config):
        self.state_dim = 2
        super().__init__(config)
        return
    @staticmethod
    def default_config():
        config = dict({
            'epsilon': 0.01, # timescale separation
            'a': 1.05, # bias
            'dt_step': 0.001,
            'dt_save': 0.1,
            't_burnin_phys': 5.0,
            })
        return config
    @staticmethod
    def label_from_config(config):
        abbrv = r'FHN_e%g_a%g'%(
                config['epsilon'],
                config['a'],
                ).replace('.','p')
        label = r'$\varepsilon=%g, a=%g'%(
                config['epsilon'],
                config['a'],
                )
        return abbrv,label
    def derive_parameters(self, config):
        self.epsilon,self.a,self.D,self.dt_step,self.dt_save = (config[v] for v in 'epsilon,a,D,dt_step,dt_save'.split(' '))
        self.impulse_dim = 1
        self.impulse_matrix = np.array([[0.0],[1.0]])
        self.t_burnin = int(config['t_burnin_phys']/self.dt_save) # depends on whether to use a pre-seeded initial condition 
        return
    def tendency(self, t, x):
        xdot = np.zeros(2)
        xdot[0] = 1/self.epsilon * (x[0] - x[0]**3/3 - x[1])
        xdot[1] = x[0] + self.a
        return xdot
    def generate_default_init_cond(self, init_time):
        return np.array([self.a**3/3-self.a, -self.a])
    def compute_pairwise_observables(self, pairwise_funs, md0, md1list, root_dir):
        pairwise_fun_vals = [[] for pwf in pairwise_funs] # List of lists
        t0,x0 = self.load_trajectory(md0, root_dir)
        for i_md1,md1 in enumerate(md1list):
            t1,x1 = self.load_trajectory(md1, root_dir)
            for i_pwf,pwf in enumerate(pairwise_funs):
                pairwise_fun_vals[i_pwf].append(pwf(t0,x0,t1,x1))
        return pairwise_fun_vals
    def compute_observables(self, obs_funs, metadata, root_dir):
        t,x = FitzhughNagumoODE.load_trajectory(metadata, root_dir)
        obs = []
        for i_fun,fun in enumerate(obs_funs):
            obs.append(fun(t,x))
        return obs
    @staticmethod
    def observable_props():
        obslib = dict({
            'obs_x': dict({
                'abbrv': 'x',
                'label': r'$x$',
                'cmap': 'coolwarm',
                }),
            'obs_y': dict({
                'abbrv': 'y',
                'label': r'$y$',
                'cmap': 'coolwarm',
                }),
            })
        return obslib
    def obs_x(self, t, x):
        return x[:,0]
    def obs_y(self, t, x):
        return x[:,1]

class FitzhughNagumoSDE(SDESystem):
    def __init__(self, config):
        ode = FitzhughNagumoODE(config['ode'])
        super().__init__(ode, config)
        return
    @staticmethod
    def default_config():
        config = dict({'ode': FitzhughNagumoODE.default_config()})
        config['seed_min'] = 1000
        config['seed_max'] = 100000
        config['D'] = 0.25
        return config
    @staticmethod
    def label_from_config(config):
        abbrv_ode,label_ode = FitzhughNagumoODE.label_from_config(config['ode'])
        abbrv_noise = (r'D%g'%(config['D'])).replace('.','p')
        label_noise = r'$D=%g$'%(config['D'])
        abbrv = r'%s_%s'%(abbrv_ode,abbrv_noise)
        label = r'%s, %s'%(label_ode,label_noise)
        return abbrv,label
    def derive_parameters(self, config):
        self.config = config
        self.sqrt_dt_step = np.sqrt(self.ode.dt_step)
        self.dt_step = self.ode.dt_step
        self.dt_save = self.ode.dt_save
        self.seed_min = config['seed_min']
        self.seed_max = config['seed_max']
        self.t_burnin = self.ode.t_burnin
        self.white_noise_dim = 1
        self.diffusion_matrix = np.array([[0.0],[1.0]])
        return
    def diffusion(self, t, x):
        return self.diffusion_matrix
    # ------------- Forwarded methods (can this be automated?) ---------------
    def compute_observables(self, obs_funs, metadata, root_dir):
        return self.ode.compute_observables(obs_funs, metadata, root_dir)
    def compute_pairwise_observables(self, pair_funs, md0, md1list, root_dir):
        return self.ode.compute_pairwise_observables(pair_funs, md0, md1list, root_dir)
    @staticmethod
    def observable_props():
        return FitzhughNagumoODE.observable_props()


