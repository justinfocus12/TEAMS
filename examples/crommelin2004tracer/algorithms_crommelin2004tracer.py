import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
import networkx as nx
from os.path import join, exists
from os import makedirs
import sys
import copy as copylib
import psutil
import time as timelib
from matplotlib import pyplot as plt, animation, rcParams
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
sys.path.append('../..')
from crommelin2004tracer import Crommelin2004TracerODE
from ensemble import Ensemble
import forcing
import algorithms
import utils

class Crommelin2004ODEPeriodicBranching(algorithms.ODEPeriodicBranching):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys, name)(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict

class Crommelin2004SDEPeriodicBranching(algorithms.SDEPeriodicBranching):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys.ode, name)(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict

class Crommelin2004TracerODEDirectNumericalSimulation(algorithms.ODEDirectNumericalSimulation):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys, f'observable_{name}')(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict
    def plot_dns_segment(self, outfile, tspan_phys=None):
        tu = self.ens.dynsys.dt_save
        obsprops = self.ens.dynsys.observable_props()
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(400/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]
        print(f'{tspan = }')
        fig,axes = plt.subplots(ncols=1, figsize=(6,6))
        ax = axes
        handles = []
        modes = ['c1y','c2y','c1xs1y','s1xs1y','c1xs2y','s1xs2y']
        for i_mode,mode in enumerate(modes):
            obs_fun = lambda t,x: getattr(self.ens.dynsys, mode)(t,x)
            label = obsprops[mode]['label']
            h = self.plot_obs_segment(obs_fun, tspan, fig, ax, label=obsprops[mode]['label'])
            handles.append(h)
        ax.set_xlabel('Time')
        ax.legend(handles=handles, bbox_to_anchor=(1,1), loc='upper left')

        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    def animate_dns_segment(self, outfile_prefix, tspan_phys=None):
        fig,ax = plt.subplots(figsize=(8,4))
        b = self.ens.dynsys.config['b']
        tu = self.ens.dynsys.dt_save
        dt_plot = 1.0 #self.ens.dynsys.dt_plot

        Nx = 64
        Ny = 32
        psi = np.zeros((Nx,Ny))
        # Warmup: plot the first snapshot
        comps,xspat,yspat,xspat_e,yspat_e = self.ens.dynsys.basis_functions(Nx, Ny)
        def psi_fun(t,x):
            Nt = len(t)
            psi = np.zeros((Nt,Nx,Ny))
            for i_comp in range(len(comps)):
                for i_t in range(Nt):
                    psi[i_t] += comps[i_comp,:,:]*x[i_t,i_comp]
            return psi
        def trposns_fun(t,x):
            Nt = len(t)
            Ntr = self.ens.dynsys.config["Ntr"]
            b = self.ens.dynsys.config['b']
            flowdim = self.ens.dynsys.timestep_constants["flowdim"]
            trposns = np.zeros((Nt,Ntr,2))
            trposns[:,:,0] = np.mod(x[:,flowdim:flowdim+Ntr], 2*np.pi)
            trposns[:,:,1] = np.mod(x[:,flowdim+Ntr:flowdim+2*Ntr], np.pi*b)
            return trposns
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(400/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]
        time,memset,tidx = self.get_member_subset(tspan)
        psi,trposns = self.ens.compute_observables([psi_fun,trposns_fun], memset[0])
        print(f'{time.shape = }')
        print(f'{psi.shape = }')
        fig,ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        Xe,Ye = np.meshgrid(xspat_e,yspat_e,indexing='ij')
        X,Y = np.meshgrid(xspat,yspat,indexing='ij')
        levels_pos = np.linspace(0,np.max(np.abs(psi)),9)[1:]
        levels_neg = np.linspace(-np.max(np.abs(psi)),0,9)[:-1]
        print(f'{psi.shape = }')
        print(f'{np.min(psi) = }, {np.max(psi) = }')
        artists = []
        for i in range(0,len(time),int(round(dt_plot/tu))):
            contours_pos = ax.contour(X,Y,psi[i],levels=levels_pos,colors='black',linestyles='solid')
            contours_neg = ax.contour(X,Y,psi[i],levels=levels_neg,colors='black',linestyles='dashed')
            scat = ax.scatter(trposns[i,:,0],trposns[i,:,1],color='black',marker='o')
            title = ax.set_title(r'$\psi(t=%g)$'%(time[i]))
            artists.append([contours_pos,contours_neg,scat,title])
            if i == 0:
                fig.savefig(outfile_prefix+'.png',**pltkwargs)
        ani = animation.ArtistAnimation(fig, artists, interval=50, blit=True, repeat_delay=1000)
        print(f'made the ani')
        ani.save(outfile_prefix+'.gif', writer="pillow") #**pltkwargs)
        return



class Crommelin2004SDEDirectNumericalSimulation(algorithms.SDEDirectNumericalSimulation):
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

        print(f'{tspan = }')
        fig,axes = plt.subplots(ncols=2, figsize=(16,4))
        # Left: timeseries
        ax = axes[0]
        handles = []
        modes = ['c1y','c2y','c1xs1y','s1xs1y','c1xs2y','s1xs2y']
        for i_mode,mode in enumerate(modes):
            obs_fun = lambda t,x: getattr(self, mode)(t,x)
            h = self.plot_obs_segment(obs_fun, tspan, fig, ax, label=r'$x_{%g}(t)$'%(k),color=colors[i_k])
            handles.append(h)
        ax.set_xlabel('Time')
        ax.legend(handles=handles)

        # Right: Hovmoller
        ax = axes[1]
        time,memset,tidx = self.get_member_subset(tspan)
        obs_fun = lambda t,x: x
        x_seg = np.concatenate(tuple(self.ens.compute_observables([obs_fun], mem)[0] for mem in memset), axis=0)[tidx,:]
        # Roll x 
        x_seg = np.roll(x_seg, K//2, axis=1)
        im = ax.pcolormesh(time*tu, np.arange(-K//2,K/2), x_seg.T, shading='nearest', cmap='BrBG')
        ax.set_ylabel(r'Longitude $k$')
        ax.set_xlabel('Time')
        
        m = self.ens.dynsys.config['frc']['white']['wavenumbers'][0]
        Fm = self.ens.dynsys.config['frc']['white']['wavenumber_magnitudes'][0]
        fig.suptitle(r'$F_{%d}=%g$'%(m,Fm))
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    def compute_extreme_stats_rotsym(self, obs_fun, spinup, duration, time_block_size, returnstats_file):
        tu = self.ens.dynsys.dt_save
        K = self.ens.dynsys.ode.K
        all_starts,all_ends = self.ens.get_all_timespans()
        mems2summarize = np.where((all_starts >= spinup)*(all_ends <= spinup + duration/K))[0]
        print(f'{len(mems2summarize) = }; {all_starts[-1] = }; {all_ends[-1] = }; {np.min(all_ends - all_starts)*tu = }; {np.max(all_ends - all_starts)*tu = }')
        blocks_per_k = int((all_ends[mems2summarize[-1]] - all_starts[mems2summarize[0]])/time_block_size)
        block_maxima = np.nan*np.ones((blocks_per_k,K))
        i_block = 0
        time_comp_obs = 0.0
        time_block_max = 0.0
        time_hist = 0.0
        time_minmax = 0.0
        for i_mem,mem in enumerate(mems2summarize):
            now0 = timelib.time()
            fk = self.ens.compute_observables([obs_fun], mem)[0]
            now1 = timelib.time()
            time_comp_obs += now1 - now0
            now0 = timelib.time()
            fkmax,fkmin = np.max(fk),np.min(fk)
            now1 = timelib.time()
            time_minmax += now1 - now0
            if i_mem == 0:
                # Initialize a histogram, might have to extend it 
                bin_edges = np.linspace(np.min(fk)-1e-10,np.max(fk)+1e-10,40)
                bin_width = bin_edges[1] - bin_edges[0]
                hist = np.zeros(len(bin_edges)-1, dtype=int)
            elif np.max(fk) > bin_edges[-1]:
                num_new_bins = int(np.ceil((np.max(fk) - bin_edges[-1])/bin_width))
                bin_edges = np.concatenate((bin_edges, bin_edges[-1] + bin_width*np.arange(1,num_new_bins+1)))
                hist = np.concatenate((hist, np.zeros(num_new_bins, dtype=int)))
            elif np.min(fk) < bin_edges[0]:
                num_new_bins = int(np.ceil((bin_edges[0]-np.min(fk))/bin_width))
                bin_edges = np.concatenate((bin_edges[0]-bin_width*np.arange(1,num_new_bins+1)[::-1], bin_edges))
                hist = np.concatenate((np.zeros(num_new_bins,dtype=int), hist))

            if all_ends[i_mem] < 10*spinup:
                now0 = timelib.time()
                hist_new,_ = np.histogram(fk.flat, bins=bin_edges)
                now1 = timelib.time()
                time_hist += now1 - now0
                hist += hist_new
            now0 = timelib.time()
            block_maxima_mem = utils.compute_block_maxima(fk,time_block_size)
            now1 = timelib.time()
            time_block_max += now1 - now0
            block_maxima[i_block:i_block+len(block_maxima_mem)] = block_maxima_mem
            i_block += len(block_maxima_mem)

            if mem % 100 == 0: 
                print(f'{mem = }')
                memusage_GB = psutil.Process().memory_info().rss / 1e9
                print(f'Using {memusage_GB} GB')
                print(f'{time_comp_obs = }; {time_block_max = }; {time_hist = }')
                print(f'{time_minmax = }')
        block_maxima = block_maxima[:i_block].flatten()
        rlev,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale = utils.compute_returnstats_preblocked(block_maxima, time_block_size)
        bin_lows = bin_edges[:-1]
        # Now the block maxima-centric method, with error bars
        bins_bm = np.linspace(np.min(block_maxima)-1e-10, np.max(block_maxima)+1e-10, 40)
        hist_bm,_ = np.histogram(block_maxima, bins=bins_bm)
        alpha = 0.05
        ccdf_bm,ccdf_bm_lower_cpi,ccdf_bm_upper_cpi = utils.pmf2ccdf(hist_bm,bins_bm,return_errbars=True,alpha=alpha) 
        ccdf_bm_2,ccdf_bm_boot = utils.compute_ccdf_errbars_bootstrap(block_maxima,bins_bm)
        assert np.all(ccdf_bm == ccdf_bm_2)
        print(f'{ccdf_bm_boot.shape = }')
        ccdf_bm_lower_bsi = np.quantile(ccdf_bm_boot,alpha/2,axis=0)
        ccdf_bm_upper_bsi = np.quantile(ccdf_bm_boot,1-alpha/2,axis=0)
        extstats = dict({
            'block_maxima': block_maxima,
            'time_block_size': time_block_size,
            'bin_lows': bin_lows, 
            'hist': hist, 
            'rlev': rlev, 
            'rtime': rtime, 
            'logsf': logsf, 
            'rtime_gev': rtime_gev, 
            'logsf_gev': logsf_gev, 
            'shape': shape, 
            'loc': loc, 
            'scale': scale,
            # Below the more basic method results
            'bins_bm': bins_bm, 
            'hist_bm': hist_bm, 
            'ccdf_bm': ccdf_bm,
            'ccdf_bm_lower_cpi': ccdf_bm_lower_cpi, 
            'ccdf_bm_upper_cpi': ccdf_bm_upper_cpi, 
            'ccdf_bm_lower_bsi': ccdf_bm_lower_bsi, 
            'ccdf_bm_upper_bsi': ccdf_bm_upper_bsi
            })
        np.savez(returnstats_file, **extstats)
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

class Crommelin2004AncestorGenerator(algorithms.SDEAncestorGenerator):
    def do_something():
        return


class Crommelin2004SDEITEAMS(algorithms.SDEITEAMS):
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
        scores = list((x[:,self.score_params['ks2avg']]**2/2).T)
        return scores
    def score_combined(self, sccomps):
        score = np.mean(np.array([sccomps[i]*self.score_params['kweights'][i] for i in range(len(sccomps))]), axis=0)
        #score[:self.advance_split_time] = np.nan
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


class Crommelin2004SDETEAMS(algorithms.SDETEAMS):
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
        #score[:self.advance_split_time] = np.nan
        return score
    def merge_score_components(self, mem_leaf, score_components_leaf): #comps0, comps1, nsteps2prepend):
        init_time,fin_time = self.ens.get_member_timespan(mem_leaf)
        parent = next(self.ens.memgraph.predecessors(mem_leaf))
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        while init_time_parent > init_time:
            parent = next(self.ens.memgraph.predecessors(parent))
            init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        sccp = self.branching_state['score_components_tdep'][parent]
        nsteps2prepend = len(sccp[0]) - len(score_components_leaf[0])
        return [np.concatenate((c0[:nsteps2prepend], c1)) for (c0,c1) in zip(sccp,score_components_leaf)]
    def get_block_maxima_from_dns_rotsym(self, t_dns, x_dns, time_block_size):
        # NOTE this is specific to the version of the score used in Finkel & O'Gorman 2024, namely x0**2/2
        pass


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
        if 'buick_choices' in config.keys():
            abbrv_buick = r'buicks%d-%d'%(config['buick_choices'][0],config['buick_choices'][-1])
        else:
            abbrv_buick = r'buicksNA'
        abbrv = r'%s_%s_%s_%s'%(abbrv_population,abbrv_k,abbrv_t,abbrv_buick)
        abbrv = abbrv.replace('.','p')
        return abbrv,label_population
    # ----------- post analysis ---------------
    def plot_hovmoller_lineage(self, ancestor, special_descendant, outfile):
        # For a single lineage from ancestor to descendant, plot the Hovmoller diagrams 
        tu = self.ens.dynsys.dt_save
        nmem = self.ens.get_nmem()
        N = self.population_size
        lineage = list(sorted(nx.ancestors(self.ens.memgraph,special_descendant) | {special_descendant}))
        if len(lineage) == 1:
            return
        tinits,tfins = self.ens.get_all_timespans()
        tinits = tinits[lineage]
        tfins = tfins[lineage]
        t0 = tinits[0]
        obs_fun = lambda t,x: (t,x)
        fig,axes = plt.subplots(ncols=2, nrows=len(lineage), figsize=(10,3*len(lineage)), gridspec_kw={'hspace': 0.2}, sharex=True, sharey=True)
        xs = []
        ts = []
        vmin,vmax = np.inf,-np.inf
        for i_mem,mem in enumerate(lineage):
            t,x = self.ens.compute_observables([obs_fun], mem)[0]
            xs.append(x)
            ts.append(t)
            vmin = min(vmin,np.min(x))
            vmax = max(vmax,np.max(x))
        vmax_abs = max(abs(vmin),abs(vmax))
        for i_mem,mem in enumerate(lineage):
            ax = axes[i_mem,0]
            self.ens.dynsys.ode.plot_hovmoller(ts[i_mem]-t0,xs[i_mem],fig,ax,vmin=vmin,vmax=vmax)
            tmx = self.branching_state['scores_max_timing'][mem]
            ax.axvline((tmx-t0)*tu, color='black', linestyle='--')
            ax.set_ylabel(r'Longitude $k$')
            ax.set_title(r'$\mathbf{x}^{(%d)}(t)$'%(mem))
            ax = axes[i_mem,1]
            if i_mem == 0:
                ax.axis('off')
            else:
                i_adopted_ancestor = i_mem - 1 #np.where(tinits[:i_mem] <= tinits[i_mem])[0][-1]
                overlap_duration = min(len(ts[i_mem]),len(ts[i_adopted_ancestor]))
                if overlap_duration > 0:
                    t_overlap = ts[i_mem][-overlap_duration:]
                    diff = xs[i_mem][-overlap_duration:]-xs[i_adopted_ancestor][-overlap_duration:]
                    # normalized
                    diff = sps.diags(1/np.sqrt(np.sum(diff**2, axis=1))) @ diff
                    self.ens.dynsys.ode.plot_hovmoller(t_overlap-t0,diff,fig,ax)
                    ax.axvline((tmx-t0)*tu, color='black', linestyle='--')
                ax.set_title(r'$\mathbf{x}^{(%d)}(t)-\mathbf{x}^{(%d)}(t)$ normalized'%(mem,lineage[i_adopted_ancestor]))

        for i_col in range(axes.shape[1]):
            axes[-1,i_col].set_xlabel(r'Time')
        if outfile is not None:
            fig.savefig(outfile, **pltkwargs)
            plt.close(fig)
        return fig,axes









        

