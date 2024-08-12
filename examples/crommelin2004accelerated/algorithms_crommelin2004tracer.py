import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from scipy.stats import skew as spskew
import networkx as nx
from os.path import join, exists
from os import makedirs
import sys
import copy as copylib
import psutil
import time as timelib
from matplotlib import pyplot as plt, animation, rcParams, colors as mplcolors
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


class Crommelin2004TracerODEDirectNumericalSimulation(algorithms.ODEDirectNumericalSimulation):
    def obs_dict_names(self):
        return ['x0','E0','E','Emax']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys, f'observable_{name}')(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict
    def plot_tracer_traj(self, outfile, tspan_phys=None):
        tu = self.ens.dynsys.dt_save
        obsprops = self.ens.dynsys.observable_props()
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(400/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]
        print(f'{tspan = }')
        fig,axes = plt.subplots(ncols=1, nrows=2, figsize=(12,12))
        ax = axes
        handles = []
        ntr2plot = 2
        flowdim = self.ens.dynsys.timestep_constants["flowdim"]
        Nparticles = self.ens.dynsys.config["Nparticles"]
        Nx,Ny = (self.ens.dynsys.timestep_constants[key] for key in ['Nx','Ny'])
        for i_tr,tr in enumerate(range(ntr2plot)):
            x_tr_fun = lambda t,x: x[:,flowdim+Nx*Ny+i_tr]
            self.plot_obs_segment(x_tr_fun, tspan, fig, axes[0], label=r'$x$')
            y_tr_fun = lambda t,x: x[:,flowdim+Nx*Ny+Nparticles+i_tr]
            self.plot_obs_segment(y_tr_fun, tspan, fig, axes[1], label=r'$y$')
        axes[1].set_xlabel('Time')
        axes[0].set_ylabel('x')
        axes[1].set_ylabel('y')
        #ax.legend(handles=handles, bbox_to_anchor=(1,1), loc='upper left')

        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    def plot_dns_particle_counts(self, outfile, tspan_phys=None):
        tu = self.ens.dynsys.dt_save
        b = self.ens.dynsys.config['b']
        flowdim,Nx,Ny = (self.ens.dynsys.timestep_constants[key] for key in ['flowdim','Nx','Ny'])
        Nparticles = self.ens.dynsys.config['Nparticles']
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(400/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]
        print(f'{tspan = }')
        ncells_x = 2
        ncells_y = 2
        Lx = 2*np.pi
        Ly = np.pi*b
        cell_size_x = Lx/ncells_x
        cell_size_y = Ly/ncells_y
        ncells = ncells_x * ncells_y
        xlo = np.linspace(0,2*np.pi,ncells_x+1)[:-1]
        ylo = np.linspace(0,np.pi*b,ncells_y+1)[:-1]
        idx_x_part = flowdim + Nx*Ny + np.arange(Nparticles, dtype=int)
        idx_y_part = idx_x_part + Nparticles
        def conc_in_each_cell(t, x):
            i_cell_x = np.floor(x[:,idx_x_part]/cell_size_x).astype(int)
            i_cell_y = np.floor(x[:,idx_y_part]/cell_size_y).astype(int)
            i_cell_flat = np.ravel_multi_index((i_cell_x,i_cell_y), (ncells_x,ncells_y))
            cell_counts = np.zeros((len(t),ncells))
            for i_cell in range(ncells):
                cell_counts[:,i_cell] = np.sum(i_cell_flat == i_cell, axis=1)
            return cell_counts
        time,memset,tidx = self.get_member_subset(tspan)
        concs = np.concatenate(tuple(self.ens.compute_observables([conc_in_each_cell], mem)[0] for mem in memset), axis=0)[tidx]
        fig,axes = plt.subplots(nrows=ncells_y, ncols=ncells_x, figsize=(12*ncells_x, 4*ncells_y), sharey=True)
        for i_cell in range(ncells):
            i_cell_x,i_cell_y = np.unravel_index(i_cell,(ncells_x,ncells_y))
            ax = axes[ncells_x-1-i_cell_x,i_cell_y]
            h, = ax.plot(time, concs[:,i_cell])
            ax.set_xlabel('time')
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    def plot_dns_concs_hovmoller(self, outfile, tspan_phys=None):
        # Hovmoller plots
        tu = self.ens.dynsys.dt_save
        q = self.ens.dynsys.timestep_constants
        (Nx,Ny,dx,dy,Lx,Ly) = (q[key] for key in 'Nx,Ny,dx,dy,Lx,Ly'.split(','))
        lats = np.linspace(0,Ly,5)[1:-1]

        x_s,y_s,basis_s,x_u,y_u,basis_u,x_v,y_v,basis_v,x_c,y_c = self.ens.dynsys.basis_functions(Nx, Ny)
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(400/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]
        time,memset,tidx = self.get_member_subset(tspan)
        t_s = np.arange(tspan[0], tspan[1]+1)
        Xe,Te = np.meshgrid(x_s, t_s, indexing='ij')
        fig,axes = plt.subplots(figsize=(18,18), ncols=3, nrows=1, sharex=True, sharey=True)
        for i_lat,lat in enumerate(lats):
            ax = axes[i_lat]
            # Make an observable for the slice
            i_y = int(round(lat/dy))
            idx_flat = np.ravel_multi_index((np.arange(Nx, dtype=int), i_y*np.ones(Nx,dtype=int)), (Nx,Ny))
            conc_hov_fun = lambda t,state: state[:,idx_flat]
            hov = np.concatenate(tuple(self.ens.compute_observables([conc_hov_fun], mem)[0] for mem in memset), axis=0)[tidx,:].T

            print(f'{hov.shape = }')
            print(f'{Xe.shape = }')
            print(f'{Te.shape = }')
            img = ax.pcolormesh(Xe,Te,hov,cmap='YlOrRd')
            ax.set_xlabel("x")
            fig.colorbar(img, ax=ax, orientation='horizontal')
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    def plot_dns_local_concs(self, outfile, tspan_phys_display=None, tspan_phys_stats=None):
        tu = self.ens.dynsys.dt_save
        b = self.ens.dynsys.config['b']
        obslib = self.ens.dynsys.observable_props()
        nmem = self.ens.get_nmem()
        if tspan_phys_display is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan_display = [fin_time-int(800/tu),fin_time]
            tspan_stats = [fin_time-int(800/tu),fin_time]
        else:
            tspan_display = [int(t/tu) for t in tspan_phys_display]
            tspan_stats = [int(t/tu) for t in tspan_phys_stats]

        # --------- Plot some local concentrations ------
        #locs = [(5.0,1.0),(3.0,0.4),(3.0,1.0)]
        lons = [2*np.pi*frac for frac in [1/8,3/8,5/8,7/8]]
        lats = [np.pi*b*frac for frac in [1/8,3/8,5/8/7/8]]
        lon_colors = plt.cm.Set1(np.arange(len(lons)))
        fig,axes = plt.subplots(ncols=2,nrows=len(lats), figsize=(12,3*len(lats)), width_ratios=[3,1])
        for i_lat,lat in enumerate(lats):
            handles = []
            for i_lon,lon in enumerate(lons):
                ax = axes[len(lats)-1-i_lat, 0]
                obs_fun = lambda t,state: self.ens.dynsys.local_conc(t,state,lon,lat)
                label = r'$c(%.1f,%.1f)$'%(lon,lat)
                h = self.plot_obs_segment(obs_fun, tspan_display, fig, ax, label=label, color=lon_colors[i_lon])
                handles.append(h)
                ax = axes[len(lats)-1-i_lat, 1]
                time,memset,tidx = self.get_member_subset(tspan_stats)
                obs_vals = np.concatenate(tuple(self.ens.compute_observables([obs_fun], mem)[0] for mem in memset), axis=0)[tidx]
                hist,bin_edges = np.histogram(obs_vals, bins=20)
                bin_mids = (bin_edges[:-1]+bin_edges[1:])/2
                ax.plot(hist, bin_mids, color=lon_colors[i_lon])
                ax.set_xscale('log')

        axes[-1,0].set_xlabel('Time')
        axes[-1,1].set_xlabel('Prob. dens.')
        fig.legend(handles=handles, bbox_to_anchor=(0.5,0), loc='upper center')
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return
    def plot_dns_spatial_stats(self, outfile, tspan_phys=None):
        tu = self.ens.dynsys.dt_save
        obslib = self.ens.dynsys.observable_props()
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(400/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]
        print(f'{tspan = }')
        # Load the files one by one and keep track of various moments 
        n_moments = 4
        q = self.ens.dynsys.timestep_constants
        flowdim,Nx,Ny,dx,dy = (q[key] for key in 'flowdim,Nx,Ny,dx,dy'.split(','))
        def moment_fun(t,state):
            moments_map = np.zeros((n_moments+1, Nx, Ny))
            for k in range(0,n_moments+1):
                moments_map[k] = np.reshape(np.sum(state[:,flowdim:flowdim+Nx*Ny]**k, axis=0), (Nx,Ny))
            return moments_map
        time,memset,tidx = self.get_member_subset(tspan)
        moments_obs = []
        for i_mem,mem in enumerate(memset):
            moments_obs.append(self.ens.compute_observables([moment_fun], mem)[0])
        moments = np.zeros((n_moments+1, Nx, Ny))
        Nt = 0
        for i_mem,mem in enumerate(memset):
            moments += moments_obs[i_mem]
        for i_moment in range(1,n_moments+1):
            moments[i_moment] *= 1/moments[0]


        # Calculate the standard things 
        mean = moments[1]
        variance = moments[2] - moments[1]**2
        std = np.where(variance > 0, np.sqrt(np.maximum(variance, 0)), np.nan)
        skewness = (moments[3] - 3*moments[2]*moments[1] + 2*moments[1]**3)/std**3
        print(f'{np.min(skewness) = }, {np.max(skewness) = }')
        kurtosis = (moments[4] - 4*moments[3]*moments[1] + 6*moments[2]*moments[1]**2 - 3*moments[1]**4)/std**4
        print(f'{np.min(kurtosis) = }, {np.max(kurtosis) = }')


        fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(12,12),sharex=True,sharey=True)
        x_s,y_s,basis_s,x_u,y_u,basis_u,x_v,y_v,basis_v,x_c,y_c = self.ens.dynsys.basis_functions(Nx, Ny)
        Xe,Ye = np.meshgrid(x_s,y_s,indexing='ij')
        # mean 
        ax = axes[0,0]
        img = ax.pcolormesh(Xe, Ye, mean, cmap=plt.cm.coolwarm)
        fig.colorbar(img, ax=ax)
        ax.set_title('mean')
        # std 
        ax = axes[0,1]
        img = ax.pcolormesh(Xe, Ye, std, cmap=plt.cm.YlOrRd)
        fig.colorbar(img, ax=ax)
        ax.set_title('std. dev.')
        # skewness
        ax = axes[1,0]
        vmax = np.nanmax(np.abs(skewness))
        img = ax.pcolormesh(Xe, Ye, skewness, cmap=plt.cm.coolwarm, vmin=-vmax, vmax=vmax)
        fig.colorbar(img, ax=ax)
        ax.set_title('skewness')
        # excess kurtosis
        ax = axes[1,1]
        max_excess = np.nanmax(np.abs(kurtosis-3))
        img = ax.pcolormesh(Xe, Ye, kurtosis, cmap=plt.cm.coolwarm, vmin=3-max_excess, vmax=3+max_excess)
        fig.colorbar(img, ax=ax)
        ax.set_title('kurtosis')

        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)
        return


    def plot_dns_segment(self, outfile, tspan_phys=None):
        tu = self.ens.dynsys.dt_save
        obslib = self.ens.dynsys.observable_props()
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(400/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]
        print(f'{tspan = }')

        # ---------- Plot modes ----------
        fig,axes = plt.subplots(ncols=1, figsize=(6,6))
        ax = axes
        handles = []
        modes = ['c1y','c2y','c1xs1y','s1xs1y','c1xs2y','s1xs2y']
        for i_mode,mode in enumerate(modes):
            obs_fun = lambda t,x: getattr(self.ens.dynsys, mode)(t,x)
            label = obslib[mode]['label']
            h = self.plot_obs_segment(obs_fun, tspan, fig, ax, label=obslib[mode]['label'])
            handles.append(h)
        ax.set_xlabel('Time')
        ax.legend(handles=handles, bbox_to_anchor=(1,1), loc='upper left')
        fig.savefig(outfile, **pltkwargs)
        plt.close(fig)

        return
    def animate_dns_segment(self, outfile_prefix, tspan_phys=None):
        b = self.ens.dynsys.config['b']
        Nx,Ny = (self.ens.dynsys.timestep_constants[key] for key in ('Nx','Ny'))
        tu = self.ens.dynsys.dt_save
        dt_plot = self.ens.dynsys.dt_plot
        x_s,y_s,basis_s,x_u,y_u,basis_u,x_v,y_v,basis_v,x_c,y_c = self.ens.dynsys.basis_functions(Nx, Ny)
        def psi_fun(t,x):
            Nt = len(t)
            psi = np.zeros((Nt,Nx+1,Ny+1))
            for i_comp in range(len(basis_s)):
                for i_t in range(Nt):
                    psi[i_t] += basis_s[i_comp,:,:]*x[i_t,i_comp]
            return psi
        def trposns_fun(t,x):
            Nt = len(t)
            Nparticles = self.ens.dynsys.config["Nparticles"]
            b = self.ens.dynsys.config['b']
            flowdim,Nx,Ny,Lx,Ly = (self.ens.dynsys.timestep_constants[key] for key in ('flowdim','Nx','Ny','Lx','Ly'))
            print(f'{Lx = }, {Ly = }')
            idx_x_part = flowdim + (Nx*Ny) + np.arange(Nparticles, dtype=int)
            idx_y_part = idx_x_part + Nparticles
            trposns = np.zeros((Nt,Nparticles,2))
            trposns[:,:,0] = np.mod(x[:,idx_x_part], Lx)
            trposns[:,:,1] = np.mod(x[:,idx_y_part], Ly)
            return trposns
        def conc_fun(t,x):
            Nx,Ny,Nparticles,flowdim = (self.ens.dynsys.timestep_constants[key] for key in ['Nx','Ny','Nparticles','flowdim'])
            Nt = len(t)
            concs = x[:,flowdim:flowdim+(Nx*Ny)].reshape((Nt,Nx,Ny))
            return concs
        nmem = self.ens.get_nmem()
        if tspan_phys is None:
            _,fin_time = self.ens.get_member_timespan(nmem-1)
            tspan = [fin_time-int(400/tu),fin_time]
        else:
            tspan = [int(t/tu) for t in tspan_phys]
        print(f'{tspan_phys = }')
        print(f'{tspan = }')
        time,memset,tidx = self.get_member_subset(tspan)
        print(f'{tidx = }')
        print(f'{memset = }')
        print(f'{time = }')
        print(f'{memset = }')

        psi_mems,trposns_mems,concs_mems = [],[],[]
        for mem in memset:
            psi_mem,trposns_mem,concs_mem = self.ens.compute_observables([psi_fun,trposns_fun,conc_fun], mem)
            psi_mems.append(psi_mem)
            trposns_mems.append(trposns_mem)
            concs_mems.append(concs_mem)
        psi = np.concatenate(tuple(psi_mems), axis=0)
        trposns = np.concatenate(tuple(trposns_mems), axis=0)
        concs = np.concatenate(tuple(concs_mems), axis=0)
        print(f'{time.shape = }')
        print(f'{psi.shape = }')
        print(f'{concs.shape = }')
        print(f'{concs.min() = }, {concs.max() = }')
        fig,axes = plt.subplots(figsize=(24,8), ncols=4, nrows=1, sharey=True, width_ratios=[5,1,1,1])
        ax = axes[0]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        axes[1].set_xlabel('Zonal mean')
        axes[2].set_xlabel('Zonal std')
        axes[3].set_xlabel('Zonal skew')
        Xe,Ye = np.meshgrid(x_s,y_s,indexing='ij')
        X,Y = np.meshgrid(x_c,y_c,indexing='ij')
        levels_pos = np.linspace(0,np.max(np.abs(psi)),9)[1:]
        levels_neg = np.linspace(-np.max(np.abs(psi)),0,9)[:-1]
        print(f'{psi.shape = }')
        print(f'{np.min(psi) = }, {np.max(psi) = }')

        # ------------- ArtistAnimation -----------------
        conc_mean_x = np.mean(concs, axis=1)
        conc_std_x = np.std(concs, axis=1)
        conc_skew_x = spskew(concs, axis=1)
        conc_mean_xt = np.mean(concs, axis=(0,1))
        conc_std_xt = np.std(concs, axis=(0,1)) 
        conc_skew_xt = spskew(concs, axis=(0,1)) 
        handles = []
        h, = axes[1].plot(conc_mean_xt, y_c, color='black', linestyle='--', label='Mean')
        handles.append(h)
        h, = axes[2].plot(conc_std_xt, y_c, color='dodgerblue', linestyle='--', label='Std')
        handles.append(h)
        h, = axes[3].plot(conc_skew_xt, y_c, color='red', linestyle='--', label='Skew')
        handles.append(h)

        artists = []
        ntr2plot = self.ens.dynsys.config["Nparticles"]
        print(f'{len(time) = }')
        print(f'{dt_plot = }')
        print(f'{tu = }')
        for i in range(0,len(time),int(round(dt_plot/tu))):
            ax = axes[0]
            conc_pcm = ax.pcolormesh(Xe,Ye,concs[i],cmap='YlOrBr',vmin=0,vmax=concs.max()) #,norm=mplcolors.LogNorm(vmin=concs.max()/1e6,vmax=concs.max()))
            if i == 0: cbar = fig.colorbar(conc_pcm, ax=ax)
            contours_pos = ax.contour(Xe,Ye,psi[i],levels=levels_pos,colors='black',linestyles='solid')
            contours_neg = ax.contour(Xe,Ye,psi[i],levels=levels_neg,colors='black',linestyles='dashed')
            scat = ax.scatter(trposns[i,:ntr2plot,0],trposns[i,:ntr2plot,1],color='black',marker='.')
            title = ax.text(0.5, 1.0, r'$\psi(t=%.2f)$'%(time[i]*tu), ha='center', va='bottom', transform=ax.transAxes)
            ax = axes[1]
            h_mean, = ax.plot(conc_mean_x[i,:], y_c, color='black')
            ax = axes[2]
            h_std, = ax.plot(conc_std_x[i,:], y_c, color='dodgerblue')
            ax = axes[3]
            h_skew, = ax.plot(conc_skew_x[i,:], y_c, color='red')
            artists.append([conc_pcm,contours_pos,contours_neg,scat,title,h_mean,h_std,h_skew,])
        print(f'{len(artists) = }')
        ani = animation.ArtistAnimation(fig, artists, interval=80, blit=True, repeat=False) #repeat_delay=5000)
        print(f'made the ani')
        ani.save(outfile_prefix+'.gif', writer="pillow") #**pltkwargs)

        return




class Crommelin2004TracerODEAncestorGenerator(algorithms.ODEAncestorGenerator):
    @classmethod
    def default_config(cls):
        cfg = dict({
            'seed_min': 1000,
            'seed_max': 10000,
            'seed_inc_init': 0,
            'burnin_time_phys': 500,
            'time_horizon_phys': 500,
            'num_buicks': 5,
            'branches_per_buick': 3,
            })
        return cfg








        

