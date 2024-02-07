from abc import ABC, abstractmethod
import numpy as np
from numpy.random import default_rng
from scipy import sparse as sps
import forcing
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)

class DynamicalSystem(ABC):
    def __init__(self):
        return
    @abstractmethod
    def default_icandf(self):
        pass
    @abstractmethod
    def run_trajectory(self, icandf, obs_fun, saveinfo):
        # return some metadata sufficient to reconstruct the output, for example (1) a filename, (2) full numpy array of the output of an ODE solver. 
        # Optionally, return some observables passed as a dictionary of function handles
        # icandf stands for "initial conditions and forcing." It: could be e.g. 
        # 1. (a full state vector, a few impulses) (for a small ODESystem) 
        # 2. (a full state vector, a few reseeds) (for a small stochastic ODESystem) 
        # 3. (A filename containing a full state vector, a namelist) (for a big PDE system)
        pass 
    @abstractmethod
    def assemble_metadata(self):
        pass
    @abstractmethod
    def load_trajectory(self, metadata):
        pass



class ODESystem(DynamicalSystem):
    # This is only SMALL systems --- small enough to fit a full-state trajectory in memory, avoiding the need for restart files. 
    def __init__(self, state_dim, config, *args, **kwargs):
        self.state_dim = state_dim
        self.config = config
        self.derive_parameters(config) # This includes both physical and simulation parameters
        return
    @staticmethod
    @abstractmethod
    def label_from_config(config):
        pass
    @abstractmethod
    def derive_parameters(self, config):
        # convert raw configuration into class attributes for efficient integration of dynamics
        pass
    @abstractmethod
    def tendency(self, t, x): # aka 'drift' for an SDE
        pass
    @staticmethod
    @abstractmethod
    def apply_impulse(t, x, imp):
        # apply the impulse perturbation from imp to the instantaneous state x, to get a perturbed state xpert
        pass
    @staticmethod
    def timestep_rk4(t, x, dt, tendency): # physical time units
        k1 = dt * tendency(t,x)
        k2 = dt * tendency(t+dt/2, x+k1/2)
        k3 = dt * tendency(t+dt/2, x+k2/2)
        k4 = dt * tendency(t+dt, x+k3)
        xnew = x + (k1 + 2*(k2 + k3) + k4)/6
        return t+dt, xnew
    @staticmethod
    def timestep_euler(t, x, dt, tendency): # physical time units
        k1 = dt * tendency(t,x)
        xnew = x + k1
        return t+dt, xnew
    @staticmethod
    def timestep_euler_maruyama(t, x, dt, drift, diffusion, rng): # physical time units
        k1 = dt * drift(t,x)
        sigma = diffusion(t,x)
        sdw = np.sqrt(dt) * sigma @ rng.normal(size=(sigma.shape[1],))
        xnew = x + k1 + sdw
        return t+dt, xnew
    def assemble_metadata(self, init_cond, f, method, filename):
        md = dict({
            'init_cond': init_cond, 
            'frc': f, 
            'method': method,
            'filename': filename,
            })
        return md
    def run_trajectory(self, icandf, obs_fun, saveinfo):
        init_cond_nopert,f = icandf['init_cond'],icandf['frc']
        assert(isinstance(f.init_time,int) and isinstance(f.fin_time,int))
        t = np.arange(f.init_time, f.fin_time+1)
        Nt = len(t)
        x = np.zeros((Nt, self.state_dim))
        # Need to set up three things: init_time, init_cond, fin_time
        init_time_temp = f.init_time
        if isinstance(f, forcing.ImpulsiveForcing):
            method = 'rk4'
            # run one segment at a time, undisturbed, from one impulse to the next
            init_cond_temp = init_cond_nopert.copy()
            i_save = 0
            nimp = len(f.impulse_times)
            for i_imp in range(nimp):
                init_cond_temp = self.apply_impulse(init_time_temp, init_cond_temp, f.impulses[i_imp])
                if i_imp+1 < len(f.impulse_times):
                    fin_time_temp = f.impulse_times[i_imp+1]
                else:
                    fin_time_temp = f.fin_time
                print(f'{init_time_temp = }; {fin_time_temp = }')
                t_temp,x_temp = self.run_trajectory_unperturbed(init_cond_temp, init_time_temp, fin_time_temp, method)
                x[i_save:i_save+len(t_temp)] = x_temp
                init_time_temp = fin_time_temp
                init_cond_temp = x_temp[-1]
                i_save += len(t_temp) - 1
        elif isinstance(f, forcing.WhiteNoiseForcing):
            method = 'euler_maruyama'
            init_cond_temp = init_cond_nopert.copy()
            i_save = 0
            for i_seed in range(len(f.reseed_times)):
                if i_seed+1 < len(f.reseed_times):
                    fin_time_temp = f.reseed_times[i_seed+1]
                else:
                    fin_time_temp = f.fin_time
                rng = default_rng(f.seeds[i_seed])
                t_temp,x_temp = self.run_trajectory_unperturbed(init_cond_temp, init_time_temp, fin_time_temp, method, rng)
                x[i_save:i_save+len(t_temp)] = x_temp
                init_time_temp = fin_time_temp
                init_cond_temp = x_temp[-1]
                i_save += len(t_temp) - 1

        # Return metadata and observables
        metadata = self.assemble_metadata(init_cond_nopert, f, method, saveinfo['filename'])
        observables = obs_fun(t,x)
        # save full state out to saveinfo
        np.savez(saveinfo['filename'], t=t, x=x)
        return metadata,observables
    def load_trajectory(self, metadata, tspan=None):
        # TODO optionally specify a single time , or write a new method for time slice
        traj = dict(np.load(metadata['filename']))
        if tspan is not None:
            idx0 = np.where(traj['t'] == tspan[0])[0][0]
            idx1 = np.where(traj['t'] == tspan[1])[0][0]
            traj['t'] = traj['t'][idx0:idx1+1]
            traj['x'] = traj['x'][idx0:idx1+1]
        return traj['t'],traj['x']
    def run_trajectory_unperturbed(self, init_cond, init_time, fin_time, method, rng=None):
        assert(isinstance(init_time,int) and isinstance(fin_time,int))
        t_save = np.arange(init_time, fin_time+1, 1) # unitless
        tp_save = t_save * self.dt_save # physical (unitful)
        Nt_save = len(t_save)
        # Initialize the solution array
        x_save = np.zeros((Nt_save, self.state_dim))
        # special cases: endpoints
        if init_time % 1 == 0:
            x_save[0] = init_cond
            i_save = 1
        else:
            i_save = 0
        tp_save_next = tp_save[i_save]
        x = init_cond
        tp = init_time * self.dt_save # physical units
        if method == 'rk4':
            timestep_fun = self.timestep_rk4
            args = ()
        elif method == 'euler':
            timestep_fun = self.timestep_euler
            args = ()
        elif method == 'euler_maruyama':
            assert rng is not None
            timestep_fun = self.timestep_euler_maruyama
            args = (self.diffusion,rng,)
        while tp < tp_save[-1]:
            tpnew,xnew = timestep_fun(tp, x, self.dt_step, self.tendency, *args)
            if tpnew > tp_save_next:
                new_weight = (tp_save_next - tp)/self.dt_step 
                # TODO: save out an observable instead of the full state? Depends on a specified frequency
                x_save[i_save] = (1-new_weight)*x + new_weight*xnew 
                i_save += 1
                if i_save < Nt_save:
                    tp_save_next = tp_save[i_save]
            x = xnew
            tp = tpnew
        return t_save,x_save




