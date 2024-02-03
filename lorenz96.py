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
from ensemble import Ensemble
import forcing

class Lorenz96(ODESystem):
    def __init__(self, config):
        state_dim = config['K']
        super().__init__(state_dim, config)
    def derive_parameters(self, config):
        self.K = config['K']
        self.F = config['F']
        self.dt_step = config['dt_step']
        self.dt_save = config['dt_save'] 
        if config['forcing']['type'] == 'white':
            fpar = config['forcing']['white']
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
        elif config['forcing']['type'] == 'impulsive':
            fpar = config['forcing']['impulsive']
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
    def plot_site_timeseries(self, t, x, k=0, color=None, fig=None, ax=None):
        fig,ax = self.check_fig_ax(fig,ax)
        h, = ax.plot(t*self.dt_save, x[:,k], label=r"$x_{%d}(t)$"%(k), color=color)
        ax.set_xlabel("Time")
        return fig,ax,h

def test_Lorenz96_white():
    config = dict({'K': 40, 'F': 8.0, 'dt_step': 0.001, 'dt_save': 0.05})
    config['forcing'] = dict({
        'type': 'white',
        'impulsive': dict({
            'wavenumbers': [1,4],
            'wavenumber_magnitudes': [0.1,0.1],
            'sites': [20,30],
            'site_magnitudes': [0.5, 0.5],
            }),
        'white': dict({
            'wavenumbers': [1,4],
            'wavenumber_magnitudes': [0.25,0.25],
            'sites': [20,30],
            'site_magnitudes': [0.01, 0.01],
            }),
        })
    ode = Lorenz96(config)
    tu = ode.dt_save

    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-02"
    sub_date_str = "0"
    ensdir = join(scratch_dir, date_str, sub_date_str)

    ens = Ensemble(ensdir, ode)

    init_time_phys = -4.0
    fin_time_phys = 30.0
    #obs_fun = lambda t,x: (t,x[:,0]**2,np.sum(x**2, axis=1)) # local and global energy
    obs_fun = lambda t,x: (t,x)


    # 0
    reseed_times = [init_time_phys/tu]
    seeds = [8765]
    f = forcing.WhiteNoiseForcing(reseed_times, seeds, fin_time_phys/tu)
    icandf = dict({'init_cond': 0.001*np.sin(2*np.pi*np.arange(config['K'])/config['K']), 'forcing': f})
    saveinfo = dict(filename = join(ensdir, 'mem0.npz'))
    observables = ens.branch_or_plant(icandf, obs_fun, saveinfo)
    t,x = observables

    fig,axes = plt.subplots(figsize=(10,10),nrows=2)
    ode.plot_hovmoller(t, x, fig=fig, ax=axes[0])
    ode.plot_site_timeseries(t, x, k=0, color='tomato', fig=fig, ax=axes[1])
    ode.plot_site_timeseries(t, x, k=1, color='cyan', fig=fig, ax=axes[1])
    fig.suptitle("Member 0")
    fig.savefig(join(ensdir,'plot_hov_mem0'),**pltkwargs)
    plt.close(fig)



    # 1
    if False:
        # 1
        reseed_times.append(5.0/tu)
        seeds.append(9853)
        f = forcing.WhiteNoiseForcing(reseed_times, seeds, (fin_time_phys+3)/tu)
        t_save_1,x_save_1 = ode.run_trajectory(init_cond, f, None, method)
        # 2
        reseed_times.append(20.0/tu)
        seeds.append(2124)
        f = forcing.WhiteNoiseForcing(reseed_times, seeds, (fin_time_phys+3)/tu)
        t_save_2,x_save_2 = ode.run_trajectory(init_cond, f, None, method)

        # Plot them all 
        fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)

        fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
        ax = axes[0]
        handles = []
        for (k,color) in zip([0,1,2],['tomato','cyan','black']):
            _,_,h = ode.plot_site_timeseries(t_save_0, x_save_0, k=k, color=color, fig=fig, ax=ax)
            handles.append[h]
        ax.legend(handles=handles)
        ax = axes[1]
        _,_,im = ode.plot_hovmoller(t_save_0, x_save_0, fig=fig, ax=ax)
        fig.savefig('L96_white_x0', **pltkwargs)
        plt.close(fig)

        fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
        ax = axes[0]
        ax.plot(t_save_0*tu, x_save_0[:,0], color='black',linestyle='--')
        ax.plot(t_save_1*tu, x_save_1[:,0], color='red',linestyle='-')
        ax.set_xlabel('time')
        ax.set_ylabel('x0')
        ax = axes[1]
        im = ax.pcolormesh(t_save_1*tu, np.arange(ode.K), x_save_1.T, shading='nearest', cmap='BrBG')
        ax.set_xlabel('time')
        ax.set_ylabel('Longitude $k$')
        fig.savefig('L96_white_x1', **pltkwargs)
        plt.close(fig)

        ax = axes[0]
        ax.plot(t_save_0*tu, x_save_0[:,0], color='black',linestyle='--')
        ax.plot(t_save_1*tu, x_save_1[:,0], color='red',linestyle='-')
        ax.plot(t_save_2*tu, x_save_2[:,0], color='dodgerblue',linestyle='-')
        ax.set_xlabel('time')
        ax.set_ylabel('x2')
        ax = axes[1]
        im = ax.pcolormesh(t_save_1*tu, np.arange(ode.K), x_save_1.T, shading='nearest', cmap='BrBG')
        ax.set_xlabel('time')
        ax.set_ylabel('Longitude $k$')
        fig.savefig('L96_white_x2', **pltkwargs)
        plt.close(fig)

        # TODO add some asserts to verify the trajectories are equal exactly where they're supposed to be

    return
def test_Lorenz96_impulsive():
    config = dict({'K': 40, 'F': 6.0, 'dt_step': 0.001, 'dt_save': 0.05})
    config['forcing'] = dict({
        'type': 'impulsive',
        'impulsive': dict({
            'wavenumbers': [1,4],
            'wavenumber_magnitudes': [0.1,0.1],
            'sites': [20,30],
            'site_magnitudes': [0.5, 0.5],
            }),
        'white': dict({
            'wavenumbers': [1,4],
            'wavenumber_magnitudes': [3.0, 3.0],
            'sites': [],
            'site_magnitudes': [],
            }),
        })
    ode = Lorenz96(config)
    tu = ode.dt_save

    # Make a small set of simulations with branching perturbations
    # 0 ---------------------
    #        |
    # 1      o----x--------------
    #                      | 
    # 2                    x--------------
    # 
    # (In the above, 1 has to replicate trajectory of 0 from the beginning, since the state is not available at the time of splitting. Later, the Ensemble object will have methods for building this in)
    rng0 = default_rng(8888)
    init_cond = 0.001*rng0.normal(size=(config['K'],))
    init_time_phys = -4.0
    fin_time_phys = 30.0
    method = 'rk4'
    t_save_0,x_save_0 = ode.run_trajectory_unperturbed(init_cond, init_time_phys/tu, fin_time_phys/tu, method)

    fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t_save_0*tu, x_save_0[:,0])
    ax.set_xlabel('time')
    ax.set_ylabel('x0')
    ax = axes[1]
    im = ax.pcolormesh(t_save_0*tu, np.arange(ode.K), x_save_0.T, shading='nearest', cmap='BrBG')
    ax.set_xlabel('time')
    ax.set_ylabel('Longitude $k$')
    fig.savefig('L96_x0', **pltkwargs)
    plt.close(fig)

    # now make a perturbation
    rng1 = default_rng(1928)
    impulse_times = [init_time_phys/tu,5.0/tu]
    impulses = [np.zeros(ode.impulse_dim),0.1*rng1.normal(size=(ode.impulse_dim,))]
    f = forcing.ImpulsiveForcing(impulse_times, impulses, (fin_time_phys+3)/tu)
    t_save_1,x_save_1 = ode.run_trajectory(init_cond, f, None, method)
    fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t_save_0*tu, x_save_0[:,0], color='black',linestyle='--')
    ax.plot(t_save_1*tu, x_save_1[:,0], color='red',linestyle='-')
    ax.set_xlabel('time')
    ax.set_ylabel('x0')
    ax = axes[1]
    im = ax.pcolormesh(t_save_1*tu, np.arange(ode.K), x_save_1.T, shading='nearest', cmap='BrBG')
    ax.set_xlabel('time')
    ax.set_ylabel('Longitude $k$')
    fig.savefig('L96_x1', **pltkwargs)
    plt.close(fig)

    # now make a perturbation
    rng2 = default_rng(1928)
    impulse_times = [init_time_phys/tu,5.0/tu,20.0/tu]
    impulses = [np.zeros(ode.impulse_dim),0.1*rng2.normal(size=(ode.impulse_dim,)),0.2*rng2.normal(size=(ode.impulse_dim,))]
    f = forcing.ImpulsiveForcing(impulse_times, impulses, (fin_time_phys+3)/tu)
    t_save_2,x_save_2 = ode.run_trajectory(init_cond, f, None, method)
    fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t_save_0*tu, x_save_0[:,0], color='black',linestyle='--')
    ax.plot(t_save_1*tu, x_save_1[:,0], color='red',linestyle='-')
    ax.plot(t_save_2*tu, x_save_2[:,0], color='dodgerblue',linestyle='-')
    ax.set_xlabel('time')
    ax.set_ylabel('x2')
    ax = axes[1]
    im = ax.pcolormesh(t_save_1*tu, np.arange(ode.K), x_save_1.T, shading='nearest', cmap='BrBG')
    ax.set_xlabel('time')
    ax.set_ylabel('Longitude $k$')
    fig.savefig('L96_x2', **pltkwargs)
    plt.close(fig)

    # TODO add some asserts to verify the trajectories are equal exactly where they're supposed to be

    return

if __name__ == "__main__":
    test_Lorenz96_white()



