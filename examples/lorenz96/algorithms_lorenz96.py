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

class Lorenz96ODEDirectNumericalSimulation(algorithms.ODEDirectNumericalSimulation):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys, f'observable_{name}')(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict

class Lorenz96SDEDirectNumericalSimulation(algorithms.SDEDirectNumericalSimulation):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys.ode, f'observable_{name}')(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict
    def plot_dns_segment(self, plotdir, tspan_phys=None):
        tu = self.ens.dynsys.dt_save
        K = self.ens.dynsys.ode.K
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(15/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]

        fig,axes = plt.subplots(ncols=2, figsize=(16,4))
        # Left: timeseries
        ax = axes[0]
        handles = []
        for k in [K//2-1,K//2,K//2+1]:
            obs_fun = lambda t,x: x[:,k]
            h = self.plot_obs_segment(obs_fun, tspan, fig, ax, label=r'$x_{%g}$'%(k))
            handles.append(h)
        ax.legend(handles=handles)

        # Right: Hovmoller
        ax = axes[1]
        time,memset,tidx = self.get_member_subset(tspan)
        obs_fun = lambda t,x: x
        x_seg = np.concatenate(tuple(self.ens.compute_observables([obs_fun], mem)[0] for mem in memset), axis=0)[tidx,:]
        im = ax.pcolormesh(time*tu, np.arange(K), x_seg.T, shading='nearest', cmap='BrBG')
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$k$')
        abbrv_tspan = (r'%g-%g'%(tspan[0]*tu, tspan[1]*tu)).replace('.','p')
        filename = join(plotdir,r'x_%s.png'%(abbrv_tspan))
        fig.savefig(filename, **pltkwargs)
        plt.close(fig)
        return
    def plot_return_stats(self, returnstats_filename, output_filename, obsprop):
        rst = np.load(returnstats_filename)
        bin_lows,hist,logsf,rtime = rst['bin_lows'],rst['hist'],rst['logsf'],rst['rtime']
        fig,axes = plt.subplots(ncols=2,figsize=(12,4),gridspec_kw={'wspace': 0.25})
        ax = axes[0]
        self.plot_return_curves(bin_lows, rtime, fig, ax)
        ax.set_xlabel(r'Return time')
        ax.set_ylabel(r'%s Return level'%(obsprop['label']))
        ax = axes[1]
        self.plot_histogram(bin_lows, hist, fig, ax)
        ax.set_ylabel('Probability density')
        ax.set_xlabel(obsprop['label'])
        fig.savefig(output_filename, **pltkwargs)
        plt.close(fig)
        return


def experimental_params():
    # TODO


        

