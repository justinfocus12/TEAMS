
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


class Crommelin2004TracerODE(ODESystem): 
    def __init__(self, cfg):
        self.state_dim = 6 + 2*cfg["Ntr"]
        super().__init__(cfg)
    @staticmethod
    def default_config():
        cfg = dict({
            "b": 0.5, "beta": 1.25, "gamma_limits": [0.2, 0.2], 
            "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": 400.0,
            "Ntr": 64, 
            })
        cfg['t_burnin_phys'] = 10.0
        cfg['dt_step'] = 0.025
        cfg['dt_save'] = 0.05
        cfg["dt_plot"] = 0.05
        cfg['frc'] = dict({
            'type': 'impulsive',
            'impulsive': dict({
                'modes': [0],
                'magnitudes': [0.01],
                }),
            })
        return cfg
    @staticmethod
    def label_from_config(cfg):
        abbrv_x1star = (r"x1st%g_r%g"%(cfg['x1star'],cfg['r'])).replace(".","p")
        abbrv_gamma = (r'gam%gto%g'%(cfg['gamma_limits'][0],cfg['gamma_limits'][1])).replace('.','p')
        label_physpar = r"$x_1^*=%g,\ r=%g \n \gamma\in[%g,%g],\ \beta=%g$"%(
                cfg['x1star'],
                cfg['r'],
                cfg['gamma_limits'][0],cfg['gamma_limits'][1],
                cfg['beta']
                )
        if cfg['frc']['type'] == 'impulsive':
            abbrv_noise_type = "frcimp"
            label_noise_type = "Impulse: "
        w = cfg['frc'][cfg['frc']['type']]
        abbrv_noise_mode = ""
        label_noise_mode = ""
        if len(w['modes']) > 0:
            abbrv_noise_mode = "mode"
            abbrv_noise_mode += "-".join([f"{m:g}" for m in w['modes']]) + "_"
        else:
            abbrv_noise_mode= "modenil"
        abbrv = "_".join([abbrv_x1star,abbrv_gamma,abbrv_noise_type,abbrv_noise_mode]).replace('.','p')
        label = "\n".join([label_physpar,label_noise_type,label_noise_mode])

        return abbrv,label
    def derive_parameters(self, cfg):
        n_max = 1
        m_max = 2
        flowdim = 6
        self.config = cfg
        self.dt_step = cfg['dt_step']
        self.dt_save = cfg['dt_save'] 
        self.dt_plot = cfg['dt_plot']
        self.t_burnin = int(cfg['t_burnin_phys']/self.dt_save) # depends on whether to use a pre-seeded initial condition 
        q = dict()
        q["year_length"] = cfg["year_length"]
        q["epsilon"] = 16*np.sqrt(2)/(5*np.pi)
        q["C"] = cfg["C"]
        q["b"] = cfg["b"]
        q["gamma_limits_cfg"] = cfg["gamma_limits"]
        q["xstar"] = np.array([cfg["x1star"],0,0,cfg["r"]*cfg["x1star"],0,0])
        q["alpha"] = np.zeros(m_max)
        q["beta"] = np.zeros(m_max)
        q["gamma_limits"] = np.zeros((2,m_max))
        q["gamma_tilde_limits"] = np.zeros((2,m_max))
        q["delta"] = np.zeros(m_max)
        for i_m in range(m_max):
            m = i_m + 1
            q["alpha"][i_m] = 8*np.sqrt(2)/np.pi*m**2/(4*m**2 - 1) * (cfg["b"]**2 + m**2 - 1)/(cfg["b"]**2 + m**2)
            q["beta"][i_m] = cfg["beta"]*cfg["b"]**2/(cfg["b"]**2 + m**2)
            q["delta"][i_m] = 64*np.sqrt(2)/(15*np.pi) * (cfg["b"]**2 - m**2 + 1)/(cfg["b"]**2 + m**2)
            for j in range(2):
                q["gamma_tilde_limits"][j,i_m] = cfg["gamma_limits"][j]*4*m/(4*m**2 - 1)*np.sqrt(2)*cfg["b"]/np.pi
                q["gamma_limits"][j,i_m] = cfg["gamma_limits"][j]*4*m**3/(4*m**2 - 1)*np.sqrt(2)*cfg["b"]/(np.pi*(cfg["b"]**2 + m**2))
        # Construct the forcing, linear matrix, and quadratic matrix, but leaving out the time-dependent parts 
        # 1. Constant term
        F = q["C"]*q["xstar"]
        # 2. Matrix for linear term
        L = -q["C"]*np.eye(flowdim)
        #L[0,2] = q["gamma_tilde"][0]
        L[1,2] = q["beta"][0]
        #L[2,0] = -q["gamma"][0]
        L[2,1] = -q["beta"][0]
        #L[3,5] = q["gamma_tilde"][1]
        L[4,5] = q["beta"][1]
        L[5,4] = -q["beta"][1]
        #L[5,3] = -q["gamma"][1]
        # 3. Matrix for bilinear term
        B = np.zeros((flowdim,flowdim,flowdim))
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
        q["flowdim"] = flowdim
        q["state_dim"] = flowdim + 2*cfg["Ntr"]
        self.timestep_constants = q
        # Impulse matrix
        imp_modes = cfg['frc']['impulsive']['modes']
        imp_mags = cfg['frc']['impulsive']['magnitudes']
        self.impulse_dim = len(imp_modes)
        print(f'{imp_mags = }')
        print(f'{imp_modes = }')
        self.impulse_matrix = np.zeros((q['state_dim'],self.impulse_dim))
        for i,mode in enumerate(imp_modes):
            self.impulse_matrix[mode,0] += imp_mags[i]
        return 
    def basis_functions(self, Nx, Ny):
        b = self.config['b']
        xspat_e = np.linspace(0,2*np.pi,Nx+1)
        xspat = xspat_e[:-1]
        yspat_e = np.linspace(0,np.pi*b,Ny+1)
        yspat = yspat_e[:-1]
        sqrt2 = np.sqrt(2)
        c1x = np.cos(xspat)
        s1x = np.sin(xspat)
        c1y = np.cos(yspat/b)
        c2y = np.cos(2*yspat/b)
        s1y = np.sin(yspat/b)
        s2y = np.sin(2*yspat/b)
        comps = np.zeros((6,Nx,Ny))
        comps[0,:,:] = b*sqrt2 * np.outer(np.ones(Nx), c1y)
        comps[3,:,:] = b*sqrt2 * np.outer(np.ones(Nx), c2y)
        comps[1,:,:] = 2*b*np.outer(c1x, s1y)
        comps[2,:,:] = 2*b*np.outer(s1x, s1y)
        comps[4,:,:] = 2*b*np.outer(c1x, s2y)
        comps[5,:,:] = 2*b*np.outer(s1x, s2y)
        return comps,xspat,yspat,xspat_e,yspat_e
    def streamfunction_at_particles(self, t, x, xspat, yspat):
        # Don't confuse xspat (a spatial position between 0 and 2pi) with x
        Ntr = len(xspat)
        sqrt2 = np.sqrt(2)
        b = self.config['b']
        c1x = np.cos(xspat)
        s1x = np.sin(xspat)
        c1y = np.cos(yspat/b)
        s1y = np.sin(yspat/b)
        c2y = np.cos(2*yspat/b)
        s2y = np.sin(2*yspat/b)
        psi = np.zeros(Ntr)
        psi_x = np.zeros(Ntr)
        psi_y = np.zeros(Ntr)

        psi += b*sqrt2  *        (x[0]*c1y + x[3]*c2y)
        psi_y += b*sqrt2 *       (x[0]*(-s1y/b) + x[3]*(-s2y*2/b))

        psi += 2*b     *  s1y * (x[1]*c1x + x[2]*s1x)
        psi_x += 2*b   *  s1y * (x[1]*(-s1x) + x[2]*c1x)
        psi_y += 2*b   *  c1y/b * (x[1]*c1x + x[2]*s1x)

        psi += 2*b     *  s2y * (x[4]*c1x + x[5]*s1x)
        psi_x += 2*b   *  s2y * (x[4]*(-s1x) + x[5]*c1x)
        psi_y += 2*b   *  c2y*(2/b) * (x[4]*c1x + x[5]*s1x)

        return psi,psi_x,psi_y
    def orography_cycle(self,t_abs):
        """
        Parameters
        ----------
        t_abs: numpy.ndarray
            The absolute time. Shape should be (Nx,) where Nx is the number of ensemble members running in parallel. 

        Returns 
        -------
        gamma_t: numpy.ndarray
            The gamma parameter corresponding to the given time of year. It varies sinusoidaly. Same is (Nx,m_max) where m_max is the maximum zonal wavenumber.
        gamma_tilde_t: numpy.ndarray
            The gamma_tilde parameter corresponding to the given time of year. Same shape as gamma_t.
        """
        q = self.timestep_constants
        cosine = np.cos(2*np.pi*t_abs/q["year_length"])
        sine = np.sin(2*np.pi*t_abs/q["year_length"])
        gamma_t = cosine * (q["gamma_limits"][1,:] - q["gamma_limits"][0,:])/2 + (q["gamma_limits"][0,:] + q["gamma_limits"][1,:])/2
        gammadot_t = -sine * (q["gamma_limits"][1,:] - q["gamma_limits"][0,:])/2
        gamma_tilde_t = cosine * (q["gamma_tilde_limits"][1,:] - q["gamma_tilde_limits"][0,:])/2 + (q["gamma_tilde_limits"][0,:] + q["gamma_tilde_limits"][1,:])/2
        gammadot_tilde_t = -sine * (q["gamma_tilde_limits"][1,:] - q["gamma_tilde_limits"][0,:])/2 + (q["gamma_tilde_limits"][0,:] + q["gamma_tilde_limits"][1,:])/2
        gamma_cfg_t = cosine * (q["gamma_limits_cfg"][1] - q["gamma_limits_cfg"][0])/2 
        gammadot_cfg_t = -sine * (q["gamma_limits_cfg"][1] - q["gamma_limits_cfg"][0])/2
        return gamma_t,gamma_tilde_t,gamma_cfg_t,gammadot_t,gammadot_tilde_t,gammadot_cfg_t
    def tendency_forcing(self,t,x):
        return self.timestep_constants["forcing_term"]
    def tendency_dissipation(self,t,x):
        diss = self.timestep_constants["linear_term"] @ x
        # Modify the time-dependent components
        gamma_t,gamma_tilde_t,gamma_cfg_t,gammadot_t,gammadot_tilde_t,gammadot_cfg_t = self.orography_cycle(t)
        diss[0] += gamma_tilde_t[0]*x[2]
        diss[2] -= gamma_t[0]*x[0]
        diss[3] += gamma_tilde_t[1]*x[5]
        diss[5] -= gamma_t[1]*x[3]
        return diss
    def tendency_advection(self,t,x):
        """
        Compute the tendency according to only the nonlinear terms, in order to check conservation of energy and enstrophy.
        """
        flowdim = self.timestep_constants["flowdim"]
        adv = np.zeros(flowdim)
        for j in range(flowdim):
            adv[j] += np.sum(x * (self.timestep_constants["bilinear_term"][j] @ x))
        return adv
    def tendency(self, t, x):
        flowdim = self.timestep_constants["flowdim"]
        Ntr = self.config["Ntr"]
        x_flow = x[:flowdim]
        x_tr = x[flowdim:flowdim+Ntr]
        y_tr = x[flowdim+Ntr:flowdim+2*Ntr]
        tendency_flow = (
                self.tendency_advection(t,x_flow) 
                + self.tendency_dissipation(t,x_flow) 
                + self.tendency_forcing(t,x_flow)
                )
        psi,psi_x,psi_y = self.streamfunction_at_particles(t, x_flow, x_tr, y_tr)
        tendency = np.concatenate((tendency_flow, -psi_y, psi_x))
        return tendency
    def correct_timestep(self, t, x):
        flowdim = self.timestep_constants["flowdim"]
        Ntr = self.config["Ntr"]
        b = self.config["b"]
        x[flowdim:flowdim+Ntr] = np.mod(x[flowdim:flowdim+Ntr], 2*np.pi)
        x[flowdim+Ntr:flowdim+2*Ntr] = np.mod(x[flowdim+Ntr:flowdim+2*Ntr], np.pi*b)
        return x
    def generate_default_init_cond(self, init_time):
        xstar = self.timestep_constants["xstar"]
        x_tr = np.linspace(0,2*np.pi,self.config["Ntr"]+1)[:-1]
        y_tr = np.pi*self.config["b"] * np.random.rand(self.config["Ntr"])
        x_init = np.concatenate((xstar,x_tr,y_tr))
        return x_init
    def compute_observables(self, obs_funs, metadata, root_dir):
        t,x = Crommelin2004TracerODE.load_trajectory(metadata, root_dir)
        obs = []
        for i_fun,fun in enumerate(obs_funs):
            obs.append(fun(t,x))
        return obs
    def compute_pairwise_observables(self, pairwise_funs, md0, md1list, root_dir):
        # Compute distances between one trajectory and many others
        pairwise_fun_vals = [[] for pwf in pairwise_funs] # List of lists
        t0,x0 = self.load_trajectory(md0, root_dir)
        for i_md1,md1 in enumerate(md1list):
            t1,x1 = self.load_trajectory(md1, root_dir)
            for i_pwf,pwf in enumerate(pairwise_funs):
                pairwise_fun_vals[i_pwf].append(pwf(t0,x0,t1,x1))
        return pairwise_fun_vals
    @staticmethod
    def observable_props():
        obslib = dict({
            'c1y': dict({
                'abbrv': 'c1y',
                'label': r'$\langle\psi,\cos(y/b)\rangle$',
                'cmap': 'coolwarm',
                }),
            'c2y': dict({
                'abbrv': 'c2y',
                'label': r'$\langle\psi,\cos(2y/b)\rangle$',
                'cmap': 'coolwarm',
                }),
            'c1xs1y': dict({
                'abbrv': 'c1xs1y',
                'label': r'$\langle\psi,\cos(x)\sin(y/b)\rangle$',
                'cmap': 'coolwarm',
                }),
            's1xs1y': dict({
                'abbrv': 's1xs1y',
                'label': r'$\langle\psi,\sin(x)\sin(y/b)\rangle$',
                'cmap': 'coolwarm',
                }),
            'c1xs2y': dict({
                'abbrv': 'c1xs2y',
                'label': r'$\langle\psi,\cos(x)\sin(2y/b)\rangle$',
                'cmap': 'coolwarm',
                }),
            's1xs2y': dict({
                'abbrv': 's1xs2y',
                'label': r'$\langle\psi,\sin(x)\sin(2y/b)\rangle$',
                'cmap': 'coolwarm',
                }),
            })
        return obslib
    @staticmethod
    def c1y(t, x):
        return x[:,0]
    @staticmethod
    def c2y(t, x):
        return x[:,3]
    @staticmethod
    def c1xs1y(t,x):
        return x[:,1]
    @staticmethod
    def s1xs1y(t,x):
        return x[:,2]
    @staticmethod
    def c1xs2y(t,x):
        return x[:,4]
    @staticmethod
    def s1xs2y(t,x):
        return x[:,5]

    # --------------- plotting functions -----------------
    def check_fig_ax(self, fig=None, ax=None):
        if fig is None:
            if ax is None:
                fig,ax = plt.subplots()
        elif ax is None:
            raise Exception("You can't just give me a fig without an axis")
        return fig,ax
    def plot_mode_energy_timeseries(self, t, x, k, linekw, fig=None, ax=None, ):
        fig,ax = self.check_fig_ax(fig,ax)
        h, = ax.plot(t*self.dt_save, x[:,k], **linekw)
        ax.set_xlabel("Time")
        return fig,ax,h




