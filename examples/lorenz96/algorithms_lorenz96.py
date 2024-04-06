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
            name: getattr(self.ens.dynsys, name)(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict

class Lorenz96SDEPeriodicBranching(algorithms.SDEPeriodicBranching):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys.ode, name)(t,x)
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
    def plot_dns_segment(self, outfile, tspan_phys=None):
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
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    @classmethod
    def plot_return_stats(cls, return_stats_filename, output_filename, obsprop):
        fig,axes = plt.subplots(ncols=2,figsize=(10,5),sharey=True)
        ax = axes[0]
        cls.plot_return_curves(return_stats_filename, fig, ax)
        ax.set_xlabel(r'Return time')
        ax.set_ylabel(r'%s Return level'%(obsprop['label']))
        ax = axes[1]
        cls.plot_histogram(return_stats_filename, fig, ax, orientation='horizontal')
        ax.set_xlabel(r'Counts')
        ax.set_ylabel('')
        ax.yaxis.set_tick_params(which='both',labelbottom=True)
        fig.savefig(output_filename, **pltkwargs)
        plt.close(fig)
        return
    @classmethod
    def plot_return_stats_meta(cls, return_stats_filenames, output_filename, obsprop, labels):
        fig,axes = plt.subplots(ncols=2,figsize=(12,4),gridspec_kw={'wspace': 0.25}, sharey=True)
        handles = []
        for i_param,rsf in enumerate(return_stats_filenames):
            color = plt.cm.Set1(i_param)
            ax = axes[0]
            h = cls.plot_return_curves(rsf, fig, ax, color=color, marker='.', label=labels[i_param])
            handles.append(h)
            ax.set_xlabel(r'Return time')
            ax.set_ylabel(r'%s Return level'%(obsprop['label']))
            ax = axes[1]
            cls.plot_histogram(rsf, fig, ax, orientation='horizontal', color=color, marker='.')
            ax.set_xlabel(r'Counts')
            ax.set_ylabel('')
            ax.yaxis.set_tick_params(which='both', labelbottom=True)
            ax.set_xlabel(obsprop['label'])
        axes[0].legend(handles=handles)
        fig.savefig(output_filename, **pltkwargs)
        plt.close(fig)
        return

class Lorenz96AncestorGenerator(algorithms.SDEAncestorGenerator):
    def do_something():
        return


class Lorenz96SDEITEAMS(algorithms.SDEITEAMS):
    def derive_parameters(self, config):
        sc = config['score']
        self.score_params = dict({
            'ks2avg': sc['ks'], # List of sites of interest to sum over
            'kweights': sc['kweights'],
            'tavg': max(1,int(round(sc['tavg_phys']/self.ens.dynsys.dt_save))),
            })
        super().derive_parameters(config)
        return
    def score_components(self, t, x):
        scores = list((x[:,self.score_params['ks2avg']]**2).T/2)
        return scores
    def score_combined(self, sccomps):
        score = np.mean(np.array([sccomps[i]*self.score_params['kweights'][i] for i in range(len(sccomps))]), axis=0)
        score[:self.advance_split_time] = np.nan
        return score
    @staticmethod
    def label_from_config(config):
        abbrv_population,label_population = algorithms.ITEAMS.label_from_config(config)
        abbrv_k = 'score'+'_'.join([
            r'%gx%g'%(
                config['score']['kweights'][i],
                config['score']['ks'][i]) 
                for i in range(len(config['score']['ks']))
            ])
        abbrv_t = r'tavg%g'%(config['score']['tavg_phys'])
        abbrv = r'%s_%s_%s'%(abbrv_population,abbrv_k,abbrv_t)
        abbrv = abbrv.replace('.','p')
        return abbrv,label_population


class Lorenz96SDETEAMS(algorithms.SDETEAMS):
    def derive_parameters(self, config):
        sc = config['score']
        self.score_params = dict({
            'ks2avg': sc['ks'], # List of sites of interest to sum over
            'kweights': sc['kweights'],
            'tavg': max(1,int(round(sc['tavg_phys']/self.ens.dynsys.dt_save))),
            })
        super().derive_parameters(config)
        return
    def score_components(self, t, x):
        scores = list((x[:,self.score_params['ks2avg']]**2).T/2)
        return scores
    def score_combined(self, sccomps):
        score = np.mean(np.array([sccomps[i]*self.score_params['kweights'][i] for i in range(len(sccomps))]), axis=0)
        score[:self.advance_split_time] = np.nan
        return score
    def merge_score_components(self, mem_leaf, score_components_leaf): #comps0, comps1, nsteps2prepend):
        init_time,fin_time = self.ens.get_member_timespan(mem_leaf)
        parent = next(self.ens.memgraph.predecessors(mem_leaf))
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        while init_time_parent > init_time:
            parent = next(self.ens.memgraph.predecessors(parent))
            init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        nsteps2prepend = init_time - init_time_parent
        score_components_parent = self.branching_state['score_components_tdep'][parent]
        return [np.concatenate((c0[:nsteps2prepend], c1)) for (c0,c1) in zip(score_components_parent,score_components_leaf)]
    @staticmethod
    def label_from_config(config):
        abbrv_population,label_population = algorithms.TEAMS.label_from_config(config)
        abbrv_k = 'score'+'_'.join([
            r'%gx%g'%(
                config['score']['kweights'][i],
                config['score']['ks'][i]) 
                for i in range(len(config['score']['ks']))
            ])
        abbrv_t = r'tavg%g'%(config['score']['tavg_phys'])
        abbrv = r'%s_%s_%s'%(abbrv_population,abbrv_k,abbrv_t)
        abbrv = abbrv.replace('.','p')
        return abbrv,label_population



        

