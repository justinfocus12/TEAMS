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


class Lorenz96ODE(ODESystem): # TODO make a superclass Lorenz96, and a sibling subclass Lorenz96SDE
    def __init__(self, config):
        self.state_dim = config['K']
        super().__init__(config)
    @staticmethod
    def default_config():
        config = dict({'K': 40, 'F': 6.0, 'dt_step': 0.001, 'dt_save': 0.05,})
        config['t_burnin'] = 0 #int(10/config['dt_save'])
        config['frc'] = dict({
            'type': 'impulsive',
            'impulsive': dict({
                'wavenumbers': [4],
                'wavenumber_magnitudes': [0.01],
                'sites': [],
                'site_magnitudes': [],
                }),
            })
        return config
    @staticmethod
    def label_from_config(config):
        abbrv_kf = f"L96_K{config['K']:g}F{config['F']:g}".replace(".","p")
        label_kf = r"$K=%g,\ F=%g$"%(config['K'],config['F'])
        if config['frc']['type'] == 'impulsive':
            abbrv_noise_type = "impnoise"
            label_noise_type = "Impulsive noise: "
        w = config['frc'][config['frc']['type']]
        abbrv_noise_wave = ""
        label_noise_wave = ""
        abbrv_noise_site = ""
        label_noise_site = ""
        if len(w['wavenumbers']) > 0:
            abbrv_noise_wave = "wv"
            abbrv_noise_wave += "-".join([f"{wn:g}" for wn in w['wavenumbers']]) + "_"
            abbrv_noise_wave += "-".join([f"{mag:g}" for mag in w['wavenumber_magnitudes']])
            label_noise_wave = ", ".join(["$F_{%g}=%g$"%(wn,mag) for (wn,mag) in zip(w['wavenumbers'],w['wavenumber_magnitudes'])])
        else:
            abbrv_noise_wave = "wvnil"
        if len(w['sites']) > 0:
            abbrv_noise_site = "site"
            abbrv_noise_site += "-".join([f"{site:g}" for site in w['sites']]) + "_"
            abbrv_noise_site += "-".join([f"{mag:g}" for mag in w['site_magnitudes']])
            label_noise_site += ", ".join(["$\mathcal{F}_{%g}=%g$"%(site,mag) for (site,mag) in zip(w['sites'],w['site_magnitudes'])])
        else:
            abbrv_noise_site = "sitenil"
        abbrv = "_".join([abbrv_kf,abbrv_noise_type,abbrv_noise_wave,abbrv_noise_site]).replace('.','p')
        label = "\n".join([label_kf,label_noise_type,label_noise_wave,label_noise_site])

        return abbrv,label
                
    def derive_parameters(self, config):
        self.K = config['K']
        self.F = config['F']
        self.dt_step = config['dt_step']
        self.dt_save = config['dt_save'] 
        self.t_burnin = config['t_burnin']
        # Forcing
        fpar = config['frc']['impulsive']
        self.impulse_dim = 2*len(fpar['wavenumbers']) + len(fpar['sites'])
        impmat = np.zeros((self.K, self.impulse_dim))
        i_noise = 0
        for i_wn,wn in enumerate(fpar['wavenumbers']):
            impmat[:,i_noise] = fpar['wavenumber_magnitudes'][i_wn] * np.cos(2*np.pi*wn*np.arange(self.K)/self.K)
            i_noise += 1
            impmat[:,i_noise] = fpar['wavenumber_magnitudes'][i_wn] * np.sin(2*np.pi*wn*np.arange(self.K)/self.K)
            i_noise += 1
        for i_site,site in enumerate(fpar['sites']):
            impmat[site,i_noise] = fpar['site_magnitudes'][i_site]
            i_noise += 1
        self.impulse_matrix = sps.csr_matrix(impmat)
        return
    def tendency(self, t, x):
        return np.roll(x,1) * (np.roll(x, -1) - np.roll(x,2)) - x + self.F
    def generate_default_init_cond(self, init_time):
        return self.F + 0.001*np.sin(2*np.pi*np.arange(self.K)/self.K)
    # --------------- Pairwise functions (e.g., distance) -----------------
    def compute_pairwise_observables(self, pair_funs, md0, md1list, root_dir):
        # Compute distances between one trajectory and many others
        pair_names = list(pair_funs.keys())
        pair_dict = dict({pn: [] for pn in pair_names})
        t0,x0 = self.load_trajectory(md0, root_dir)
        for i_md1,md1 in enumerate(md1list):
            t1,x1 = self.load_trajectory(md1, root_dir)
            for i_pn,pn in enumerate(pair_names):
                pair_dict[pn].append(pair_funs[pn](t0,x0,t1,x1))
        return pair_dict
            
    # --------------- Common observable functions --------
    def compute_observables(self, obs_funs, metadata, root_dir):
        t,x = Lorenz96ODE.load_trajectory(metadata, root_dir)
        obs = []
        for i_fun,fun in enumerate(obs_funs):
            obs.append(fun(t,x))
        return obs
    def compute_stats_dns_rotsym(self, fk, k_roll_step, time_block_size, bounds=None):
        # Given a physical input field f(k), augment it by rotations to compute return periods
        # constant parameters to adjust 
        time_block_size = 10
        # Concatenate a long array of timeseries at different longitudes
        ksubset = np.arange(0, self.K, step=k_roll_step)
        # Clip the time axis to contain exactly an integer multiple of the block size
        ntimes = fk.shape[0]
        clip_size = np.mod(ntimes, time_block_size)
        fconcat = np.concatenate(tuple(fk[:,k] for k in ksubset))
        return utils.compute_returnstats_and_histogram(fconcat, time_block_size, bounds=bounds)
    @staticmethod
    def observable_props():
        obslib = dict({
            'x0': dict({
                'abbrv': 'x0',
                'label': r'$x_0$',
                'cmap': 'coolwarm',
                }),
            'E0': dict({
                'abbrv': 'E0',
                'label': r'$\frac{1}{2}x_0^2$',
                'cmap': 'coolwarm',
                }),
            'E': dict({
                'abbrv': 'E',
                'label': r'$\frac{1}{2}\overline{x_k^2}$',
                'cmap': 'coolwarm',
                }),
            'Emax': dict({
                'abbrv': 'Emax',
                'label': r'$\mathrm{max}_k \{\frac{1}{2}x_k^2\}$',
                'cmap': 'coolwarm',
                }),
            })
        return obslib
    def x0(self, t, x):
        return x[:,0]
    def E0(self, t, x):
        return x[:,0]**2/2
    def E(self, t, x):
        return np.mean(x**2, axis=1)/2
    def Emax(self, t, x):
        return np.max(x**2, axis=1)/2
    # -------------- Distance functions --------------
    def distance(self, t0, x0, t1, x1, dist_name):
        name2func = dict({
            'euclidean': self.distance_euclidean,
            })
        return name2func[dist_name](t0,x0,t1,x1)
    def distance_euclidean(self, t0, x0, t1, x1):
        return np.sqrt(np.sum((x0 - x1)**2, axis=1))
    # --------------- plotting functions -----------------
    def check_fig_ax(self, fig=None, ax=None):
        if fig is None:
            if ax is None:
                fig,ax = plt.subplots()
        elif ax is None:
            raise Exception("You can't just give me a fig without an axis")
        return fig,ax
    def plot_hovmoller(self, t, x, fig, ax):
        im = ax.pcolormesh(t*self.dt_save, np.arange(self.K), x.T, shading='nearest', cmap='BrBG')
        return im
    def plot_site_timeseries(self, t, x, k, linekw, fig=None, ax=None, ):
        fig,ax = self.check_fig_ax(fig,ax)
        h, = ax.plot(t*self.dt_save, x[:,k], **linekw)
        ax.set_xlabel("Time")
        return fig,ax,h


class Lorenz96SDE(SDESystem):
    def __init__(self, config):
        ode = Lorenz96ODE(config['ode'])
        super().__init__(ode, config)
        return
    @staticmethod
    def default_config():
        config = dict({'ode': Lorenz96ODE.default_config()})
        config['seed_min'] = 1000
        config['seed_max'] = 100000
        config['frc'] = dict({
            'type': 'white',
            'white': dict({
                'wavenumbers': [4],
                'wavenumber_magnitudes': [0.25],
                'sites': [],
                'site_magnitudes': [],
                }),
            })
        return config
    @staticmethod
    def label_from_config(config):
        # config needs a separate sub-config dictionary pertaining to the ODE
        abbrv_ode,label_ode = Lorenz96ODE.label_from_config(config['ode'])
        # Now append any new things
        if config['frc']['type'] == 'white':
            abbrv_noise_type = "whitenoise"
            label_noise_type = "White noise: "
        w = config['frc'][config['frc']['type']]
        abbrv_noise_wave = ""
        label_noise_wave = ""
        abbrv_noise_site = ""
        label_noise_site = ""
        if len(w['wavenumbers']) > 0:
            abbrv_noise_wave = "wv"
            abbrv_noise_wave += "-".join([f"{wn:g}" for wn in w['wavenumbers']]) + "_"
            abbrv_noise_wave += "-".join([f"{mag:g}" for mag in w['wavenumber_magnitudes']])
            label_noise_wave = ", ".join(["$F_{%g}=%g"%(wn,mag) for (wn,mag) in zip(w['wavenumbers'],w['wavenumber_magnitudes'])])
        else:
            abbrv_noise_wave = "wvnil"
        if len(w['sites']) > 0:
            abbrv_noise_site = "site"
            abbrv_noise_site += "-".join([f"{site:g}" for site in w['sites']]) + "_"
            abbrv_noise_site += "-".join([f"{mag:g}" for mag in w['site_magnitudes']])
            label_noise_site += ", ".join(["$\mathcal{F}_{%g}=%g"%(site,mag) for (site,mag) in zip(w['sites'],w['site_magnitudes'])])
        else:
            abbrv_noise_site = "sitenil"
        abbrv_sde = "_".join([abbrv_noise_type,abbrv_noise_wave,abbrv_noise_site]).replace('.','p')
        label_sde = "\n".join([label_noise_type,label_noise_wave,label_noise_site])

        abbrv = f'{abbrv_ode}_{abbrv_sde}'
        label = f'{label_ode}\n{label_sde}'

        return abbrv,label

    def derive_parameters(self, config):
        # These config parameters are specific to the SDE 
        self.config = config
        self.sqrt_dt_step = np.sqrt(self.ode.dt_step)
        self.dt_step = self.ode.dt_step
        self.dt_save = self.ode.dt_save
        self.seed_min = config['seed_min']
        self.seed_max = config['seed_max']
        self.t_burnin = self.ode.t_burnin
        # White noie forcing
        fpar = config['frc']['white']
        self.white_noise_dim = 2*len(fpar['wavenumbers']) + len(fpar['sites'])
        diffmat = np.zeros((self.ode.K, self.white_noise_dim))
        i_noise = 0
        for i_wn,wn in enumerate(fpar['wavenumbers']):
            diffmat[:,i_noise] = fpar['wavenumber_magnitudes'][i_wn] * np.cos(2*np.pi*wn*np.arange(self.ode.K)/self.ode.K)
            i_noise += 1
            diffmat[:,i_noise] = fpar['wavenumber_magnitudes'][i_wn] * np.sin(2*np.pi*wn*np.arange(self.ode.K)/self.ode.K)
            i_noise += 1
        for i_site,site in enumerate(fpar['sites']):
            diffmat[site,i_noise] = fpar['site_magnitudes'][i_site]
            i_noise += 1
        self.diffusion_matrix = sps.csr_matrix(diffmat)
        return
    def diffusion(self, t, x):
        return self.diffusion_matrix
    # ------------- Forwarded methods (can this be automated?) ---------------
    def compute_observables(self, obs_funs, metadata, root_dir):
        return self.ode.compute_observables(obs_funs, metadata, root_dir)
    def compute_pairwise_observables(self, pair_funs, md0, md1list, root_dir):
        return self.ode.compute_pairwise_observables(self, pair_funs, md0, md1list, root_dir)
    @staticmethod
    def observable_props():
        return Lorenz96ODE.observable_props()

