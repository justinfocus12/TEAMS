import numpy as np
from numpy.random import default_rng
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import sys
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

class Lorenz96ODE(ODESystem): # TODO make a superclass Lorenz96, and a sibling subclass Lorenz96SDE
    def __init__(self, config):
        state_dim = config['K']
        super().__init__(state_dim, config)
    @staticmethod
    def label_from_config(config):
        abbrv_kf = f"K{config['K']:g}F{config['F']:g}".replace(".","p")
        label_kf = r"$K=%g,\ F=%g$"%(config['K'],config['F'])
        if config['frc']['type'] == 'impulsive':
            abbrv_noise = "impnoise_"
            label_noise = "Impulsive noise: "
        w = config['frc'][config['frc']['type']]
        abbrv_noise_wave = ""
        label_noise_wave = ""
        abbrv_noise_site = ""
        label_noise_site = ""
        if len(w['wavenumbers']) > 0:
            abbrv_noise_wave = "wvnum"
            abbrv_noise_wave += "-".join([f"{wn:g}" for wn in w['wavenumbers']]) + "_"
            abbrv_noise_wave += "-".join([f"{mag:g}" for mag in w['wavenumber_magnitudes']])
            label_noise_wave = ", ".join(["$F_{%g}=%g"%(wn,mag) for (wn,mag) in zip(w['wavenumbers'],w['wavenumber_magnitudes'])])
        if len(w['sites']) > 0:
            abbrv_noise_site = "site"
            abbrv_noise_site += "-".join([f"{site:g}" for site in w['sites']]) + "_"
            abbrv_noise_site += "-".join([f"{mag:g}" for mag in w['site_magnitudes']])
            label_noise_site += ", ".join(["$\mathcal{F}_{%g}=%g"%(site,mag) for (site,mag) in zip(w['sites'],w['site_magnitudes'])])
        abbrv = "_".join([abbrv_kf,abbrv_noise_wave,abbrv_noise_site]).replace('.','p')
        label = "\n".join([label_kf,label_noise_wave,label_noise_site])

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
    # --------------- Common observable functions --------
    def observable(self, t, x, obs_name):
        if obs_name == 't':
            return t
        name2func = dict({
            'x0': self.observable_x0,
            'E0': self.observable_E0,
            'E': self.observable_E,
            'Emax': self.observable_Emax,
            })
        return name2func[obs_name](t,x)
    def observable_x0(self, t, x):
        return x[:,0]
    def observable_E0(self, t, x):
        return x[:,0]**2/2
    def observable_E(self, t, x):
        return np.sum(x**2, axis=1)/2
    def observable_Emax(self, t, x):
        return np.max(x**2, axis=1)/2
    # --------------- plotting functions -----------------
    def check_fig_ax(self, fig=None, ax=None):
        if fig is None:
            if ax is None:
                fig,ax = plt.subplots()
        elif ax is None:
            raise Exception("You can't just give me a fig without an axis")
        return fig,ax
    def plot_hovmoller(self, t, x, fig=None, ax=None):
        fig,ax = self.check_fig_ax(fig,ax)
        im = ax.pcolormesh(t*self.dt_save, np.arange(self.K), x.T, shading='nearest', cmap='BrBG')
        ax.set_xlabel('Time')
        ax.set_ylabel('Longitude $k$')
        return fig,ax,im
    def plot_site_timeseries(self, t, x, k, linekw, fig=None, ax=None, ):
        fig,ax = self.check_fig_ax(fig,ax)
        h, = ax.plot(t*self.dt_save, x[:,k], **linekw)
        ax.set_xlabel("Time")
        return fig,ax,h


class Lorenz96SDE(SDESystem):
    @staticmethod
    def label_from_config(config_ode, config):
        abbrv_ode,label_ode = Lorenz96ODE.label_from_config(config_ode)
        # Now append any new things
        if config['frc']['type'] == 'white':
            abbrv_noise = "whitenoise_"
            label_noise = "White noise: "
        w = config['frc'][config['frc']['type']]
        abbrv_noise_wave = ""
        label_noise_wave = ""
        abbrv_noise_site = ""
        label_noise_site = ""
        if len(w['wavenumbers']) > 0:
            abbrv_noise_wave = "wvnum"
            abbrv_noise_wave += "-".join([f"{wn:g}" for wn in w['wavenumbers']]) + "_"
            abbrv_noise_wave += "-".join([f"{mag:g}" for mag in w['wavenumber_magnitudes']])
            label_noise_wave = ", ".join(["$F_{%g}=%g"%(wn,mag) for (wn,mag) in zip(w['wavenumbers'],w['wavenumber_magnitudes'])])
        if len(w['sites']) > 0:
            abbrv_noise_site = "site"
            abbrv_noise_site += "-".join([f"{site:g}" for site in w['sites']]) + "_"
            abbrv_noise_site += "-".join([f"{mag:g}" for mag in w['site_magnitudes']])
            label_noise_site += ", ".join(["$\mathcal{F}_{%g}=%g"%(site,mag) for (site,mag) in zip(w['sites'],w['site_magnitudes'])])
        abbrv_sde = "_".join([abbrv_noise_wave,abbrv_noise_site]).replace('.','p')
        label_sde = "\n".join([label_noise_wave,label_noise_site])

        abbrv = f'{abbrv_ode}_{abbrv_sde}'
        label = f'{label_ode}\n{label_sde}'

        return abbrv,label

    def derive_parameters(self, config):
        # These config parameters are specific to the SDE 
        self.sqrt_dt_step = np.sqrt(self.ode.dt_step)
        self.seed_min = config['seed_min']
        self.seed_max = config['seed_max']
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




