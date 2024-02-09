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
    def list_required_instance_variables(self):
        return []
    @abstractmethod
    def generate_default_icandf(self): # for spinup
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


# TODO make subclasses for ODESystem and SDESystem, each of which will have their own way to incorporate perturbations (apply_impulse vs run with white noise) and have their own way to generate a random sequence of standard inputs. 

class ODESystem(DynamicalSystem):
    # This is only SMALL systems --- small enough to fit a full-state trajectory in memory, avoiding the need for restart files. 
    def __init__(self, state_dim, config):
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
    @staticmethod
    def list_required_instance_variables():
        varreq = super().list_required_instance_variables()
        varreq += ['dt_step','dt_save','impulse_matrix','impulse_dim']
        return varreq
    @abstractmethod
    def tendency(self, t, x): # aka 'drift' for an SDE
        pass
    @staticmethod
    def apply_impulse(t, x, imp):
        # apply the impulse perturbation from imp to the instantaneous state x, to get a perturbed state xpert
        return x + self.impulse_matrix @ imp
    @abstractmethod
    def generate_default_init_cond(self,init_time): # for spinup
        pass
    def generate_default_forcing_sequence(self,init_time,fin_time): 
        f = forcing.ImpulsiveForcing([init_time], np.zeros((1,self.impulse_dim)), fin_time)
        return f
    def generate_default_icandf(self,init_time,fin_time):
        init_cond = self.generate_default_init_cond(init_time)
        f = self.generate_default_init_cond(init_time)
        icandf = dict({'init_cond': init_cond, 'frc': f})
        return icandf
    def assemble_metadata(self, init_cond, f, method, filename):
        md = dict({
            'init_cond': init_cond, 
            'frc': f, 
            'method': method,
            'filename': filename,
            })
        return md
        
    def load_trajectory(self, metadata, tspan=None):
        # TODO optionally specify a single time , or write a new method for time slice
        traj = dict(np.load(metadata['filename']))
        if tspan is not None:
            idx0 = np.where(traj['t'] == tspan[0])[0][0]
            idx1 = np.where(traj['t'] == tspan[1])[0][0]
            traj['t'] = traj['t'][idx0:idx1+1]
            traj['x'] = traj['x'][idx0:idx1+1]
        return traj['t'],traj['x']
    # These are driven by impulses only
    def timestep_rk4(t, x): # physical time units
        k1 = self.dt_step * self.tendency(t,x)
        k2 = self.dt_step * self.tendency(t+self.dt_step/2, x+k1/2)
        k3 = self.dt_step * self.tendency(t+self.dt_step/2, x+k2/2)
        k4 = self.dt_step * self.tendency(t+self.dt_step, x+k3)
        xnew = x + (k1 + 2*(k2 + k3) + k4)/6
        return t+self.dt_step, xnew
    def timestep_euler(t, x): # physical time units
        k1 = self.dt_step * self.tendency(t,x)
        xnew = x + k1
        return t+self.dt_step, xnew
    def run_trajectory_unperturbed(self, init_cond, init_time, fin_time, method):
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
        timestep_fun_dict = {'rk4': self.timestep_rk4, 'euler': self.timestep_euler}
        timestep_fun = timestep_fun_dict[method]
        while tp < tp_save[-1]:
            tpnew,xnew = timestep_fun(tp, x)
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
    def run_trajectory(self, icandf, obs_fun, saveinfo):
        init_cond_nopert,f = icandf['init_cond'],icandf['frc']
        assert(isinstance(f.init_time,int) and isinstance(f.fin_time,int))
        t = np.arange(f.init_time, f.fin_time+1)
        Nt = len(t)
        x = np.zeros((Nt, self.state_dim))
        # Need to set up three things: init_time, init_cond, fin_time
        init_time_temp = f.init_time
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
        # Return metadata and observables
        metadata = self.assemble_metadata(init_cond_nopert, f, method, saveinfo['filename'])
        observables = obs_fun(t,x)
        # save full state out to saveinfo
        np.savez(saveinfo['filename'], t=t, x=x)
        return metadata,observables

class SDESystem(DESystem):
    def __init__(self, config, ode):
        self.ode = ode
        super().__init__(self.ode.state_dim, config)
        return

    @staticmethod
    def list_required_instance_variables():
        varreq = super().list_required_instance_variables()
        varreq += self.ode.list_required_instance_variables()
        varreq += ['sqrt_dt_step','seed_min','seed_max','white_noise_dim']
        return varreq
    @abstractmethod
    def diffusion(self, t, x):
        pass
    def timestep_euler_maruyama(t, x, rng): # physical time units
        k1 = self.dt_step * self.ode.tendency(t,x)
        sdw = self.sqrt_dt_step * self.diffusion(t,x) @ rng.normal(size=(self.white_noise_dim,))
        return t+self.dt_step, x+k1+sdw
    def run_trajectory_unperturbed(self, init_cond, init_time, fin_time, method, rng):
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
        if method == 'euler_maruyama':
            timestep_fun = self.timestep_euler_maruyama
        while tp < tp_save[-1]:
            tpnew,xnew = timestep_fun(tp, x, rng)
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
    def run_trajectory(self, icandf, obs_fun, saveinfo):
        init_cond_nopert,f_white,f_imp = icandf['init_cond'],icandf['frc_white'],icandf['frc_impulsive']
        assert(np.all([isinstance(tinst,int) for tinst in [f_white.init_time,f_imp.init_time, f_white.fin_time, f_imp.fin_time]]))
        assert((f_white.init_time == f_imp.init_time) and (f_white.fin_time == f_imp.fin_time))
        t = np.arange(f_imp.init_time, f_imp.fin_time+1)
        Nt = len(t)
        x = np.zeros((Nt, self.state_dim))
        # Need to set up three things: init_time, init_cond, fin_time
        init_time_temp = f_imp.init_time
        method = 'euler_maruyama'
        init_cond_temp = init_cond_nopert.copy()
        i_save = 0
        frc_change_times = np.sort(np.union1d(f_imp.impulse_times, f_white.reseed_times))
        i_imp = 0
        i_seed = 0
        for i_df in range(len(frc_change_times)):
            if i_df+1 < len(frc_change_times):
                fin_time_temp = frc_change_times[i_df+1]
            else:
                fin_time_temp = f_imp.fin_time
            if frc_change_times[i_df] in f_imp.impulse_times:
                init_cond_temp = self.apply_impulse(init_time_temp, init_cond_temp, f_imp.impulses[i_imp])
                i_imp += 1
            if frc_change_times[i_df] in f_white.seed_times:
                rng = default_rng(f_white.seeds[i_seed])
                i_seed += 1
            t_temp,x_temp = self.run_trajectory_unperturbed(init_cond_temp, init_time_temp, fin_time_temp, method, rng)
            x[i_save:i_save+len(t_temp)] = x_temp
            init_time_temp = fin_time_temp
            init_cond_temp = x_temp[-1]
            i_save += len(t_temp) - 1
        metadata = self.assemble_metadata(init_cond_nopert, f, method, saveinfo['filename'])
        observables = obs_fun(t,x)
        # save full state out to saveinfo
        np.savez(saveinfo['filename'], t=t, x=x)
        return metadata,observables
    def generate_default_forcing_sequence(self,init_time,fin_time):
        f = forcing.WhiteNoiseForcing([init_time], [self.seed_min], fin_time)
        return f
    def generate_default_icandf(self,init_time,fin_time):
        f_white = self.generate_default_forcing_sequence(init_time, fin_time)
        icandf_imp = self.ode.generate_default_icandf(init_time,fin_time)
        icandf = dict({'init_cond': icandf_imp['init_cond'], 'frc_white': f_white, 'frc_impulsive': icandf_imp['frc']})
        return icandf



