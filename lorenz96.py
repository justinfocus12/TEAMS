from abc import ABC, abstractmethod
import numpy as np
from numpy.random import default_rng
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from dynamicalsystem import ODESystem

class Lorenz96(ODESystem):
    def __init__(self, config):
        state_dim = config['K']
        super().__init__(state_dim, config)
    @staticmethod
    def label_from_config(config):
        abbrv_kf = f"K{config['K']:g}F{config['F']:g}".replace(".","p")
        label_kf = r"$K=%g,\ F=%g$"%(config['K'],config['F'])
        if config['frc']['type'] == 'white':
            w = config['frc']['white']
            abbrv_noise = "white_"
            label_noise = "White noise: "
            if len(w['wavenumbers']) > 0:
                abbrv_noise += "-".join([f"{wn:g}" for wn in w['wavenumbers']])
                abbrv_noise += "-".join([f"{mag:g}" for mag in w['wavenumber_magnitudes']])
                label_noise += ", ".join(["$F_{%g}=%g"%(wn,mag) for (wn,mag) in zip(w['wavenumbers'],w['wavenumber_magnitudes'])])
            if len(w['sites']) > 0:
                abbrv_noise += "-".join([f"{site:g}" for site in w['sites']])
                abbrv_noise += "-".join([f"{mag:g}" for mag in w['site_magnitudes']])
                label_noise += ", ".join(["$\mathcal{F}_{%g}=%g"%(site,mag) for (site,mag) in zip(w['sites'],w['site_magnitudes'])])
        elif config['frc']['type'] == 'impulsive':
            label_noise = "Impulsive noise: "
            w = config['frc']['impulsive']
            abbrv_noise = "impulsive_"
            if len(w['wavenumbers']) > 0:
                abbrv_noise += "-".join([f"{wn:g}" for wn in w['wavenumbers']])
                abbrv_noise += "-".join([f"{mag:g}" for mag in w['wavenumber_magnitudes']])
                label_noise += ", ".join(["$\mathcal{F}_{%g}=%g"%(wn,mag) for (wn,mag) in zip(w['wavenumbers'],w['wavenumber_magnitudes'])])
            if len(w['sites']) > 0:
                abbrv_noise += "-".join([f"{site:g}" for site in w['sites']])
                abbrv_noise += "-".join([f"{mag:g}" for mag in w['site_magnitudes']])
                label_noise += ", ".join(["$\mathcal{F}_{%g}=%g"%(site,mag) for (site,mag) in zip(w['sites'],w['site_magnitudes'])])
        abbrv = "_".join([abbrv_kf,abbrv_noise])
        label = "\n".join([label_kf,label_noise])

        return abbrv,label

        
                
    def derive_parameters(self, config):
        self.K = config['K']
        self.F = config['F']
        self.dt_step = config['dt_step']
        self.dt_save = config['dt_save'] 
        if config['frc']['type'] == 'white':
            fpar = config['frc']['white']
            self.white_noise_dim = 2*len(fpar['wavenumbers']) + len(fpar['sites'])
            diffmat = np.zeros((self.K, self.white_noise_dim))
            i_noise = 0
            for i_wn,wn in enumerate(fpar['wavenumbers']):
                diffmat[:,i_noise] = fpar['wavenumber_magnitudes'][i_wn] * np.cos(2*np.pi*wn*np.arange(self.K)/self.K)
                i_noise += 1
                diffmat[:,i_noise] = fpar['wavenumber_magnitudes'][i_wn] * np.sin(2*np.pi*wn*np.arange(self.K)/self.K)
                i_noise += 1
            for i_site,site in enumerate(fpar['sites']):
                diffmat[site,i_noise] = fpar['site_magnitudes'][i_site]
                i_noise += 1
            self.diffusion_matrix = sps.csr_matrix(diffmat)
        elif config['frc']['type'] == 'impulsive':
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
    def diffusion(self, t, x):
        return self.diffusion_matrix
    def apply_impulse(self, t, x, imp):
        print(f'{imp = }')
        return x + self.impulse_matrix @ imp #imp[0]*np.cos(2*np.pi*4*np.arange(self.K)/self.K) + imp[1]*np.sin(2*np.pi*4*np.arange(self.K)/self.K)
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




