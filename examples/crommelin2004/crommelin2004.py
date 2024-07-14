
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


class Crommelin2004ODE(ODESystem): 
    def __init__(self, config):
        self.state_dim = config['K']
        super().__init__(config)
    @staticmethod
    def default_config():
        config = dict({
            "b": 0.5, "beta": 1.25, "gamma_limits": [0.2, 0.2], 
            "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": 400.0,
            })
        config['t_burnin_phys'] = 10.0
        config['frc'] = dict({
            'type': 'impulsive',
            'impulsive': dict({
                'modes': [0],
                'magnitudes': [0.01],
                }),
            })
        return config
    @staticmethod
    def label_from_config(config):
        abbrv_x1str = (r"x1st%g_r%g"%(config['x1star'],config['r'])).replace(".","p")
        abbrv_gam = (r'gam%g-%g'%(config['gamma_limits'][0],config['gamma_limits'][1])).replace('.','p')
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
    def derive_parameters(cls, fpd):
        n_max = 1
        m_max = 2
        xdim = 7
        q = dict({"fpd": fpd})
        q["year_length"] = fpd["year_length"]
        q["epsilon"] = 16*np.sqrt(2)/(5*np.pi)
        q["C"] = fpd["C"]
        q["b"] = fpd["b"]
        q["gamma_limits_fpd"] = fpd["gamma_limits"]
        q["xstar"] = np.array([fpd["x1star"],0,0,fpd["r"]*fpd["x1star"],0,0])
        q["alpha"] = np.zeros(m_max)
        q["beta"] = np.zeros(m_max)
        q["gamma_limits"] = np.zeros((2,m_max))
        q["gamma_tilde_limits"] = np.zeros((2,m_max))
        q["delta"] = np.zeros(m_max)
        for i_m in range(m_max):
            m = i_m + 1
            q["alpha"][i_m] = 8*np.sqrt(2)/np.pi*m**2/(4*m**2 - 1) * (fpd["b"]**2 + m**2 - 1)/(fpd["b"]**2 + m**2)
            q["beta"][i_m] = fpd["beta"]*fpd["b"]**2/(fpd["b"]**2 + m**2)
            q["delta"][i_m] = 64*np.sqrt(2)/(15*np.pi) * (fpd["b"]**2 - m**2 + 1)/(fpd["b"]**2 + m**2)
            for j in range(2):
                q["gamma_tilde_limits"][j,i_m] = fpd["gamma_limits"][j]*4*m/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/np.pi
                q["gamma_limits"][j,i_m] = fpd["gamma_limits"][j]*4*m**3/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/(np.pi*(fpd["b"]**2 + m**2))
        # Construct the forcing, linear matrix, and quadratic matrix, but leaving out the time-dependent parts 
        # 1. Constant term
        F = q["C"]*q["xstar"]
        # 2. Matrix for linear term
        L = -q["C"]*np.eye(xdim-1)
        #L[0,2] = q["gamma_tilde"][0]
        L[1,2] = q["beta"][0]
        #L[2,0] = -q["gamma"][0]
        L[2,1] = -q["beta"][0]
        #L[3,5] = q["gamma_tilde"][1]
        L[4,5] = q["beta"][1]
        L[5,4] = -q["beta"][1]
        #L[5,3] = -q["gamma"][1]
        # 3. Matrix for bilinear term
        B = np.zeros((xdim-1,xdim-1,xdim-1))
        B[1,0,2] = -q["alpha"][0]
        B[1,3,5] = -q["delta"][0]
        B[2,0,1] = q["alpha"][0]
        B[2,3,4] = q["delta"][0]
        B[3,1,5] = q["epsilon"]
        B[3,2,4] = -q["epsilon"]
        B[4,0,5] = -q["alpha"][1]
        B[4,3,2] = -q["delta"][1]
        B[5,0,4] = q["alpha"][1]
        B[5,3,1] = q["delta"][1]
        q["forcing_term"] = F
        q["linear_term"] = L
        q["bilinear_term"] = B
        q["state_dim"] = 7
        return q
    @classmethod
    def default_init(cls, expt_dir, model_params_patch, ensemble_size_limit):
        dirs_ens = dict({
            "work": join(expt_dir, "work"),
            "output": join(expt_dir, "output"),
            })
        fundamental_param_dict = dict({
            "b": 0.5, "beta": 1.25, "gamma_limits": [0.2, 0.2], 
            "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": 400.0,})
        model_params = cls.derive_parameters(fundamental_param_dict)
        model_params["dt_sim"] = 0.1
        model_params["dt_save"] = 0.5 
        model_params["dt_print"] = 500.0
        model_params.update(model_params_patch)
        ensemble_size_limit = ensemble_size_limit
        ens = cls(dirs_ens, model_params, ensemble_size_limit)
        return ens
                
    def derive_parameters(self, config):
        self.K = config['K']
        self.F = config['F']
        self.dt_step = config['dt_step']
        self.dt_save = config['dt_save'] 
        self.t_burnin = int(config['t_burnin_phys']/self.dt_save) # depends on whether to use a pre-seeded initial condition 
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
    def compute_pairwise_observables(self, pairwise_funs, md0, md1list, root_dir):
        # Compute distances between one trajectory and many others
        pairwise_fun_vals = [[] for pwf in pairwise_funs] # List of lists
        t0,x0 = self.load_trajectory(md0, root_dir)
        for i_md1,md1 in enumerate(md1list):
            t1,x1 = self.load_trajectory(md1, root_dir)
            for i_pwf,pwf in enumerate(pairwise_funs):
                pairwise_fun_vals[i_pwf].append(pwf(t0,x0,t1,x1))
        return pairwise_fun_vals
            
    # --------------- Common observable functions --------
    def compute_observables(self, obs_funs, metadata, root_dir):
        t,x = Lorenz96ODE.load_trajectory(metadata, root_dir)
        obs = []
        for i_fun,fun in enumerate(obs_funs):
            obs.append(fun(t,x))
        return obs
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
    def x0sq(self, t, x):
        return x[:,0]**2
    def E0(self, t, x):
        return x[:,0]**2/2
    def E(self, t, x):
        return np.mean(x**2, axis=1)/2
    def Emax(self, t, x):
        return np.max(x**2, axis=1)/2
    # -------------- Distance functions --------------
    def distance_euclidean(self, t0, x0, t1, x1):
        return np.sqrt(np.mean((x0 - x1)**2, axis=1))
    def distance_timedelay_xk(self, t0, x0, t1, x1, k=0, timedelay_phys=1):
        timedelay = max(1, min(len(t0), int(timedelay_phys/self.dt_save)))
        conv = np.convolve(np.ones(timedelay), (x0[:,k]-x1[:,k])**2, mode='full')
        return conv[:len(conv)-(timedelay-1)]
    def distance_timedelay_Ek(self, t0, x0, t1, x1, k=0, timedelay_phys=0.0):
        timedelay = max(1, min(len(t0), int(timedelay_phys/self.dt_save)))
        conv = np.convolve(np.ones(timedelay), (x0[:,k]**2/2-x1[:,k]**2/2)**2, mode='full')
        return conv[:len(conv)-(timedelay-1)]


    # --------------- plotting functions -----------------
    def check_fig_ax(self, fig=None, ax=None):
        if fig is None:
            if ax is None:
                fig,ax = plt.subplots()
        elif ax is None:
            raise Exception("You can't just give me a fig without an axis")
        return fig,ax
    def plot_hovmoller(self, t, x, fig, ax, **kwargs):
        im = ax.pcolormesh(t*self.dt_save, np.arange(self.K)-self.K//2, np.roll(x,self.K//2,axis=1).T, shading='nearest', cmap='BrBG', **kwargs)
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
                'coefs': [],
                'magnitudes': [0.25],
                }),
            })
        # TODO decide on the full config, whether tracers should be part of the state space, and how to perturb them
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
        return self.ode.compute_pairwise_observables(pair_funs, md0, md1list, root_dir)
    def compute_stats_dns_rotsym(self, *args, **kwargs):
        return self.ode.compute_stats_dns_rotsym(*args, **kwargs)
    @staticmethod
    def observable_props():
        return Lorenz96ODE.observable_props()
