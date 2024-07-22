
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
        self.state_dim = 6 + 2*cfg["Nparticles"] + cfg["Nxfv"]*cfg["Nyfv"]
        super().__init__(cfg)
    @staticmethod
    def default_config():
        cfg = dict({
            "b": 0.5, "beta": 1.25, "gamma_limits": [0.2, 0.2], 
            "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": 400.0,
            "Nparticles": 64,
            "Nxfv": 12, "Nyfv": 6, 
            })
        cfg['t_burnin_phys'] = 10.0
        cfg['dt_step'] = 0.025
        cfg['dt_save'] = 0.05
        cfg["dt_plot"] = 0.05
        cfg['timestepper'] = 'finvol'
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
        self.dt_step = cfg['dt_step']
        self.dt_save = cfg['dt_save'] 
        self.dt_plot = cfg['dt_plot']
        self.t_burnin = int(cfg['t_burnin_phys']/self.dt_save) # depends on whether to use a pre-seeded initial condition 
        q = dict()
        flowdim = 6
        q["flowdim"] = flowdim
        q["Nparticles"] = cfg['Nparticles']
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
        # 4. Grid for finite volume
        q['Lx'] = 2*np.pi
        q['Ly'] = np.pi*cfg['b']
        q['Nx'] = cfg['Nxfv']
        q['Ny'] = cfg['Nyfv']
        q['dx'] = q['Lx']/cfg['Nxfv']
        q['dy'] = q['Ly']/cfg['Nyfv']
        q['x_s'],q['y_s'],q['basis_s'],q['x_u'],q['y_u'],q['basis_u'],q['x_v'],q['y_v'],q['basis_v'],q['x_c'],q['y_c'] = self.basis_functions(q['Nx'],q['Ny'])
        # source and sink
        q['source_flag'] = np.zeros((q['Nx'],q['Ny']), dtype=bool)
        q['source_flag'][:,[0,q['Ny']-1]] = True
        q['source_conc'] = np.zeros((q['Nx'],q['Ny']))
        q['source_conc'][:,0] = 1.0
        self.timestep_constants = q
        # Impulse matrix
        imp_modes = cfg['frc']['impulsive']['modes']
        imp_mags = cfg['frc']['impulsive']['magnitudes']
        self.impulse_dim = len(imp_modes)
        print(f'{imp_mags = }')
        print(f'{imp_modes = }')
        self.impulse_matrix = np.zeros((self.state_dim,self.impulse_dim))
        for i,mode in enumerate(imp_modes):
            self.impulse_matrix[mode,0] += imp_mags[i]
        return 
    def timestep_finvol(self, t, state):
        q = self.timestep_constants
        Nx,Ny,dx,dy = (q[key] for key in ('Nx','Ny','dx','dy'))
        Nparticles = self.config['Nparticles']
        # Runge-Kutta for flow field
        flowdim = self.timestep_constants["flowdim"]
        strfn = state[:flowdim]
        k1 = self.dt_step * self.tendency_flow(t,strfn)
        k2 = self.dt_step * self.tendency_flow(t+self.dt_step/2, strfn+k1/2)
        k3 = self.dt_step * self.tendency_flow(t+self.dt_step/2, strfn+k2/2)
        k4 = self.dt_step * self.tendency_flow(t+self.dt_step, strfn+k3)
        strfn_next = strfn + (k1 + 2*(k2 + k3) + k4)/6
        # Finite-voloume step for tracer
        conc = state[flowdim:flowdim+(Nx*Ny)]
        conc_next = np.zeros(Nx*Ny)
        u = np.zeros((Nx+1,Ny))
        v = np.zeros((Nx,Ny+1))
        for i in range(flowdim):
            u += 0.5*(strfn[i] + strfn_next[i]) * q["basis_u"][i,:,:]
            v += 0.5*(strfn[i] + strfn_next[i]) * q["basis_v"][i,:,:]
        for iflat in range(Nx*Ny):
            ix,iy = np.unravel_index(iflat,(Nx,Ny))
            if q['source_flag'][ix,iy]:
                continue
            iflat_nbs = np.zeros(4,dtype=int)
            outflows = np.zeros(4)
            inflows = np.zeros(4)
            # right
            i_nb = 0
            ix_nb = (ix+1) % Nx
            iy_nb = iy
            iflat_nb = np.ravel_multi_index((ix_nb,iy_nb),(Nx,Ny))
            iflat_nbs[i_nb] = iflat_nb
            velocity = u[ix+1,iy]
            if velocity > 0:
                outflows[i_nb] = conc[iflat]*dy*self.dt_step*velocity
            elif q['source_flag'][ix_nb,iy_nb]:
                inflows[i_nb] = q['source_conc'][ix_nb,iy_nb]*dy*self.dt_step*(-velocity)
            # left 
            i_nb = 1
            ix_nb = (ix-1) % Nx
            iy_nb = iy
            #print(f'{ix_nb = }, {Nx = }, {iy_nb = }, {Ny = }')
            iflat_nb = np.ravel_multi_index((ix_nb,iy_nb),(Nx,Ny))
            iflat_nbs[i_nb] = iflat_nb
            velocity = u[ix,iy]
            if velocity < 0:
                outflows[i_nb] = conc[iflat]*dy*self.dt_step*(-velocity)
            elif q['source_flag'][ix_nb,iy_nb]:
                inflows[i_nb] = q['source_conc'][ix_nb,iy_nb]*dy*self.dt_step*velocity
            # top
            i_nb = 2
            ix_nb = ix
            iy_nb = iy+1
            iflat_nb = np.ravel_multi_index((ix_nb,iy_nb),(Nx,Ny))
            iflat_nbs[i_nb] = iflat_nb
            velocity = v[ix,iy+1]
            if velocity > 0:
                outflows[i_nb] = conc[iflat]*dx*self.dt_step*velocity
            elif q['source_flag'][ix_nb,iy_nb]:
                inflows[i_nb] = q['source_conc'][ix_nb,iy_nb]*dx*self.dt_step*(-velocity)
            # bottom
            i_nb = 3
            ix_nb = ix
            iy_nb = iy-1
            iflat_nb = np.ravel_multi_index((ix_nb,iy_nb),(Nx,Ny))
            iflat_nbs[i_nb] = iflat_nb
            velocity = v[ix,iy]
            if velocity > 0:
                outflows[i_nb] = conc[iflat]*dx*self.dt_step*velocity
            elif q['source_flag'][ix_nb,iy_nb]:
                inflows[i_nb] = q['source_conc'][ix_nb,iy_nb]*dx*self.dt_step*(-velocity)
            # Sum up all in- and out-flows
            conc_next[iflat] = np.sum(inflows)/(dx*dy)
            sum_outflows = np.sum(outflows)
            if np.sum(outflows) > conc[iflat]*dx*dy:
                outflows *= conc[iflat]*dx*dy/sum_outflows
            conc_next[iflat] -= np.sum(outflows)/(dx*dy)
            for i_nb in range(4):
                conc_next[iflat_nbs[i_nb]] += outflows[i_nb]/(dx*dy)
        # Forward Euler for particles
        idx_x_part = flowdim + (Nx*Ny) + np.arange(q['Nparticles'])
        idx_y_part = idx_x_part + q['Nparticles']
        x_part = state[flowdim+(Nx*Ny):flowdim+(Nx*Ny)+q['Nparticles']]
        y_part = state[flowdim+(Nx*Ny)+q['Nparticles']:flowdim+(Nx*Ny)+2*q['Nparticles']]

        s_part,s_x_part,s_y_part =  self.streamfunction_at_particles(t, strfn, x_part, y_part)
        x_part_next = state[idx_x_part] - self.dt_step*s_y_part
        y_part_next = state[idx_y_part] + self.dt_step*s_x_part

        return t+self.dt_step, np.concatenate((strfn_next, conc_next, x_part_next, y_part_next))




                




    def basis_functions(self, Nx, Ny):
        # Compute streamfunction and derivatives at cell corners and face centers (C-grid)
        sqrt2 = np.sqrt(2)
        Lx = 2*np.pi
        Ly = np.pi*self.config['b']
        dx = Lx/Nx
        dy = Ly/Ny
        b = self.config['b']
        # vertical faces where zonal velocity is evaluated
        x_u = np.linspace(0,Lx,Nx+1)
        y_u = np.linspace(dy/2,Ly-dy/2,Ny)
        # horizontal faces where meridional velocity is evaluated
        x_v = np.linspace(dx/2,Lx-dx/2,Nx)
        y_v = np.linspace(0,Ly,Ny+1)
        # vertices where streamfunction is evaluated
        x_s = np.linspace(0,Lx,Nx+1)
        y_s = np.linspace(0,Ly,Ny+1)
        # cell centers where concentrations are evaluated
        x_c = np.linspace(dx/2,Lx-dx/2,Nx)
        y_c = np.linspace(dy/2,Ly-dy/2,Ny)
        # 
        c1x_u,c1x_v,c1x_s,c1x_c = (np.cos(x) for x in (x_u,x_v,x_s,x_c))
        s1x_u,s1x_v,s1x_s,s1x_c = (np.sin(x) for x in (x_u,x_v,x_s,x_c))
        c1y_u,c1y_v,c1y_s,c1y_c = (np.cos(y/b) for y in (y_u,y_v,y_s,y_c))
        s1y_u,s1y_v,s1y_s,s1y_c = (np.sin(y/b) for y in (y_u,y_v,y_s,y_c))
        c2y_u,c2y_v,c2y_s,c2y_c = (np.cos(2*y/b) for y in (y_u,y_v,y_s,y_c))
        s2y_u,s2y_v,s2y_s,s2y_c = (np.sin(2*y/b) for y in (y_u,y_v,y_s,y_c))
        # Streamfunction
        basis_s = np.zeros((6,Nx+1,Ny+1)) 
        basis_s[0,:,:] = b*sqrt2 * np.outer(np.ones_like(x_s), c1y_s)
        basis_s[3,:,:] = b*sqrt2 * np.outer(np.ones_like(x_s), c2y_s)
        basis_s[1,:,:] = 2*b     * np.outer(c1x_s, s1y_s)
        basis_s[2,:,:] = 2*b     * np.outer(s1x_s, s1y_s)
        basis_s[4,:,:] = 2*b     * np.outer(c1x_s, s2y_s)
        basis_s[5,:,:] = 2*b     * np.outer(s1x_s, s2y_s)
        # zonal velocity 
        basis_u = np.zeros((6,Nx+1,Ny)) 
        basis_u[0,:,:] = -b*sqrt2 * np.outer(np.ones_like(x_u), -s1y_u/b)
        basis_u[3,:,:] = -b*sqrt2 * np.outer(np.ones_like(x_u), -s2y_u*2/b)
        basis_u[1,:,:] = -2*b    * np.outer(c1x_u, c1y_u/b)
        basis_u[2,:,:] = -2*b    * np.outer(s1x_u, c1y_u/b)
        basis_u[4,:,:] = -2*b    * np.outer(c1x_u, c2y_u*2/b)
        basis_u[5,:,:] = -2*b    * np.outer(s1x_u, c2y_u*2/b)
        # meridional velocity 
        basis_v = np.zeros((6,Nx,Ny+1)) 
        basis_v[1,:,:] = 2*b     * np.outer(-s1x_v/b, s1y_v)
        basis_v[2,:,:] = 2*b     * np.outer(c1x_v/b, s1y_v)
        basis_v[4,:,:] = 2*b     * np.outer(-s1x_v/b, s2y_v)
        basis_v[5,:,:] = 2*b     * np.outer(c1x_v/b, s2y_v)
        return (
                x_s,y_s,basis_s,
                x_u,y_u,basis_u,
                x_v,y_v,basis_v,
                x_c,y_c,
                )
    def streamfunction_at_particles(self, t, x, xspat, yspat):
        # Don't confuse xspat (a spatial position between 0 and 2pi) with x
        Nparticles = len(xspat)
        sqrt2 = np.sqrt(2)
        b = self.config['b']
        c1x = np.cos(xspat)
        s1x = np.sin(xspat)
        c1y = np.cos(yspat/b)
        s1y = np.sin(yspat/b)
        c2y = np.cos(2*yspat/b)
        s2y = np.sin(2*yspat/b)
        s = np.zeros(Nparticles)
        s_x = np.zeros(Nparticles)
        s_y = np.zeros(Nparticles)

        s += b*sqrt2  *        (x[0]*c1y + x[3]*c2y)
        s_y += b*sqrt2 *       (x[0]*(-s1y/b) + x[3]*(-s2y*2/b))

        s += 2*b     *  s1y * (x[1]*c1x + x[2]*s1x)
        s_x += 2*b   *  s1y * (x[1]*(-s1x) + x[2]*c1x)
        s_y += 2*b   *  c1y/b * (x[1]*c1x + x[2]*s1x)

        s += 2*b     *  s2y * (x[4]*c1x + x[5]*s1x)
        s_x += 2*b   *  s2y * (x[4]*(-s1x) + x[5]*c1x)
        s_y += 2*b   *  c2y*(2/b) * (x[4]*c1x + x[5]*s1x)

        return s,s_x,s_y
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
    def tendency_dissipation(self,t,x_flow):
        diss = self.timestep_constants["linear_term"] @ x_flow
        # Modify the time-dependent components
        gamma_t,gamma_tilde_t,gamma_cfg_t,gammadot_t,gammadot_tilde_t,gammadot_cfg_t = self.orography_cycle(t)
        diss[0] += gamma_tilde_t[0]*x_flow[2]
        diss[2] -= gamma_t[0]*x_flow[0]
        diss[3] += gamma_tilde_t[1]*x_flow[5]
        diss[5] -= gamma_t[1]*x_flow[3]
        return diss
    def tendency_advection(self,t,x_flow):
        """
        Compute the tendency according to only the nonlinear terms, in order to check conservation of energy and enstrophy.
        """
        flowdim = self.timestep_constants["flowdim"]
        adv = np.zeros(flowdim)
        for j in range(flowdim):
            adv[j] += np.sum(x_flow * (self.timestep_constants["bilinear_term"][j] @ x_flow))
        return adv
    def tendency_flow(self, t, x_flow):
        return (
                self.tendency_advection(t,x_flow) 
                + self.tendency_dissipation(t,x_flow) 
                + self.tendency_forcing(t,x_flow)
                )
    def correct_timestep(self, t, x):
        flowdim = self.timestep_constants["flowdim"]
        Nparticles = self.config["Nparticles"]
        b = self.config["b"]
        x[flowdim:flowdim+Nparticles] = np.mod(x[flowdim:flowdim+Nparticles], 2*np.pi)
        x[flowdim+Nparticles:flowdim+2*Nparticles] = np.mod(x[flowdim+Nparticles:flowdim+2*Nparticles], np.pi*b)
        return x
    def generate_default_init_cond(self, init_time):
        # Flow
        s_star = self.timestep_constants["xstar"]
        # Concentrations
        Nx,Ny = self.config["Nxfv"],self.config["Nyfv"]
        conc = np.zeros(Nx*Ny)
        # Tracer positions
        x_tr = np.linspace(0,2*np.pi,self.config["Nparticles"]+1)[:-1]
        y_tr = np.pi*self.config["b"] * np.random.rand(self.config["Nparticles"])
        state_init = np.concatenate((s_star,conc,x_tr,y_tr))
        return state_init
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




